# Rental Churn Prediction Project: Comprehensive Technical Deep Dive

## 1. Executive Summary

This project implements a state-of-the-art deep learning system for predicting **rental payment behavior** (churn/delays). It treats the problem as a **temporal sequence regression** task: predicting the `target_days_to_payment` for a specific invoice given the history of previous invoices and the metadata of the current one.

The architecture is a hybrid Hierarchical Transformer that combines:
1.  **Fourier Features** for high-fidelity continuous variable encoding.
2.  **Transformer Encoders** for intra-invoice feature mixing.
3.  **Causal Sequence Encoders** with Rotary Positional Embeddings (RoPE) for temporal history modeling.
4.  **Temporal Cross-Attention** to dynamically link current static features with historical behavioral patterns.

This document details the exact mechanisms, mathematics, and engineering decisions behind every component.

---

## 2. Mathematical Foundation & Advanced Concepts

### 2.1 Fourier Features for Continuous Embeddings
Standard neural networks often struggle to learn high-frequency functions (the "spectral bias" problem). To mitigate this, we project scalar continuous inputs (like `amount`, `days_since_last`) into a higher-dimensional space using random Fourier features.

**Mechanism:**
For a scalar input $v \in \mathbb{R}$ and an embedding dimension $D$:
1.  We initialize a random frequency matrix $\mathbf{B} \in \mathbb{R}^{D/2}$ drawn from a normal distribution $\mathcal{N}(0, \sigma^2)$, where $\sigma$ is `periodic_sigma`.
2.  The projection is computed as:
    $$ \gamma(v) = \text{concat}\left[ \sin(2\pi \mathbf{B}v), \cos(2\pi \mathbf{B}v) \right] \in \mathbb{R}^D $$
3.  This vector is then passed through a learnable MLP (Linear $\to$ Sigmoid $\to$ Gated Linear) to allow the model to tune the representation.

**Code Reference:** `core.model.FourierFeatures`

### 2.2 Rotary Positional Embeddings (RoPE)
In the Sequence Encoder, we use RoPE instead of absolute positional embeddings. RoPE encodes position information by **rotating** the Query and Key vectors in the complex plane. This has the property that the inner product (attention score) between two positions $m$ and $n$ depends only on their relative distance $m-n$.

**Formulation:**
For a feature vector $x$ at position $m$, we group elements into pairs. For a pair $(x_1, x_2)$, the rotation is:
$$ \begin{pmatrix} x'_1 \\ x'_2 \end{pmatrix} = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix} $$
where $\theta_i = 10000^{-2i/d}$.
This effectively injects relative position information explicitly into the attention mechanism without adding learnable parameters.

**Code Reference:** `core.model.RoPE`

### 2.3 Gated Residual Networks (GRN)
Used in the prediction head and cross-attention, GRNs allow the model to flexibly control information flow. They can behave as a simple linear skip connection (identity) or a complex non-linear transformation.

**Architecture:**
$$ \begin{aligned}
\eta_1 &= \text{ELU}(W_1 x + b_1) \\
\eta_2 &= W_2 \eta_1 + b_2 \\
\text{gate} &= \sigma(W_3 \eta_2 + b_3) \\
\text{out} &= \text{LayerNorm}\left( \text{gate} \odot \eta_2 + (1 - \text{gate}) \odot W_s x \right)
\end{aligned} $$
This gating mechanism stabilizes training and allows the network to "skip" depth where it's not needed.

**Code Reference:** `core.model.GRN`

### 2.4 SwiGLU Activation
In the Transformer Feed-Forward Network (FFN), we use the SwiGLU variant, which has been shown to outperform standard ReLU/GELU.

$$ \text{SwiGLU}(x) = (x W_G) \odot \text{SiLU}(x W_1) \cdot W_2 $$
It combines the Swish activation function ($\text{SiLU}(x) = x \cdot \sigma(x)$) with a Gated Linear Unit, providing a smoother optimization landscape.

---

## 3. Data Processing Pipeline (Detailed)

### 3.1 Data Origin & Structure
The fundamental unit of the dataset is a **single payment event** (an invoice/parcela) associated with a rental contract.
-   **Granularity**: Each row represents one specific bill that was issued to a user.
-   **Hierarchy**:
    1.  **User (`usuarioId`)**: The top-level entity. A user creates an account.
    2.  **Contract (`contratoId`)**: A user can have multiple rental contracts over time (or simultaneously).
    3.  **Payment (`parcela_order`)**: Each contract generates a sequence of payments (e.g., monthly rent).
-   **Sequence Implication**: This hierarchy allows the model to learn intra-contract dynamics (e.g., "users often delay the first payment of a new contract but stabilize later") as well as inter-contract behavior (e.g., "a user who defaulted on Contract A is likely to default on Contract B").

### 3.2 Data Loading & Sampling
-   **Source**: Parquet files (`data/training_data.parquet`).
-   **Sampling**: To handle large datasets or for debugging, we sample subsets of users. If `user_sample_count` is set, we select exactly that many unique users arbitrarily. Otherwise, we sample a fraction (`load_sample_fraction`). This ensures we always get *complete* histories for the users we select, rather than random rows.

### 3.3 Feature Engineering
The raw data is enriched with specific features tailored for churn prediction:
-   **Cyclical Time**: `sin`/`cos` transforms for DayOfWeek (7-day cycle) and Month (12-day cycle).
-   **Lag Features**: `days_since_last_invoice` is critical for understanding payment gaps.
-   **Rolling Metrics**: `rolling_mean_delay_3`, `rolling_max_delay_5` etc., capture the recent trend of the user.
-   **Target Transformation**: `Use_log1p_transform = True`. The raw target (days) is transformed via $y' = \log(1 + y)$. This compresses the range of delays (e.g., difference between 1 and 5 days is more significant than 100 vs 105 days).

### 3.4 The "Expanding Window" Strategy (Sequence Generation)
To create a supervision signal for *every* possible invoice while maintaining temporal causality, we use an **expanding window** approach (also known as a causal sliding window). This allows us to train on every invoice in a user's history, provided there is enough history to form a valid sequence.

**Algorithm (`_create_expanding_indices`):**
1.  **Group & Sort**: Data is grouped by `usuarioId` and strictly sorted by `vencimentoData` (Due Date).
2.  **Iterate**: For a user with $N$ total invoices, we iterate through indices $i$ from `min_seq_len - 1` to $N - 1$.
    -   $i$ represents the index of the **Target Invoice** (the one we want to predict).
3.  **Window Definition**: For each target invoice $i$:
    -   **Sequence End**: $i + 1$. This means the sequence slice stops *after* $i$, so it **includes** $i$ as the last element.
    -   **Sequence Start**: $\max(0, \text{End} - \text{max\_seq\_len})$. We take up to `max_seq_len` recent items.
    -   **Slice**: `data[Start : End]`.

**Visual Example:**
Imagine a user with 5 invoices: `[Inv0, Inv1, Inv2, Inv3, Inv4]`.
Assume `min_seq_len=2` and `max_seq_len=3`.

-   **Step 1 (Target = Inv1):**
    -   Window: `[Inv0, Inv1]`
    -   Input to Model: Features of Inv0 + Features of Inv1.
    -   Prediction: Delay of Inv1.
-   **Step 2 (Target = Inv2):**
    -   Window: `[Inv0, Inv1, Inv2]`
    -   Input to Model: Features of Inv0 + Features of Inv1 + Features of Inv2.
    -   Prediction: Delay of Inv2.
-   **Step 3 (Target = Inv3):**
    -   Window: `[Inv1, Inv2, Inv3]` (Shifted, as max length is 3)
    -   Input to Model: Features of Inv1 + Features of Inv2 + Features of Inv3.
    -   Prediction: Delay of Inv3.

**Key Insight**: The **last element** of the input sequence is always the target invoice itself. We include it because we need its "static" properties (Due Date, Billed Amount) to make the prediction. However, we must hide its "answer" (Delay), which leads to the masking strategy below.

### 3.5 Masking & Collating
This section details how we handle variable lengths and prevent data leakage.

#### A. Batch Padding (`collate_sequences`)
Deep learning frameworks require rectangular tensors. Since each sequence has a different length (up to `max_seq_len`), we use dynamic padding per batch.
-   **Mechanism**: The `collate_fn` finds the longest sequence in the current batch.
-   **Padding Value**: All shorter sequences are padded with `0` (for categorical) or `0.0` (for continuous) to match the longest one.
-   **Masking**: The model uses a `padding_mask` to ensure attention mechanisms completely ignore these padded positions (they contribute 0 to the gradient).

#### B. Target Masking (`mask_target` method)
**This is the most critical step for correctness.**
As noted in 3.3, the input sequence *contains* the target invoice at the very last position `t`. The dataset contains the ground-truth delay for this invoice at that position. If we feed this directly, the model will simply "read" the answer and achieve 0 loss without learning to predict.

To prevent this **Data Leakage**, we apply a specific mask to the feature vector at the last time step `t`:

1.  **Identify Features**: We identify columns that carry leakage info, specifically:
    -   `delay_clipped`: The actual delay value.
    -   `delay_is_known`: A binary flag indicating if delay is observable.
2.  **Zero Out**: `features[t, delay_idx] = 0.0`.
3.  **Result**:
    -   At steps `0` to `t-1` (History): The model sees real delays. "In the past, you delayed 5 days."
    -   At step `t` (Target): The model sees 0. "For this current invoice, the delay is unknown. Predict it based on the history and current invoice details (Amount, Date)."

### 3.6 Augmentation
During training (`augment=True`), we apply probabilistic augmentations to make the model robust against data quality issues:
1.  **Temporal Cutout**: Randomly zeroes out *all* features for a subset of time steps. Simulates missing history records.
2.  **Feature Dropout**: Randomly zeroes out specific feature columns (e.g., `amount`) for the *entire* sequence. Simulates missing data fields/sensor failure.
3.  **Gaussian Noise**: Adds $\mathcal{N}(0, 0.05)$ to continuous features. Simulates measurement noise or input jitter.
4.  **Time Warp**: Randomly duplicates or removes a time step. Simulates irregular data ingestion or duplicate logs.

---

## 4. Detailed Architecture Breakdown

The model flow is: `Input -> Tokenizer -> InvoiceEncoder -> SequenceEncoder -> CrossAttention -> PredictionHead`.

### 4.1 Feature Tokenizer
-   **Role**: Homogenize inputs into a standard vector space.
-   **Inputs**: $C$ categorical columns, $N$ continuous columns.
-   **Categorical**: Each has its own `nn.Embedding(cardinality, hidden_dim)`.
-   **Continuous**: Concatenated and projected via `FourierFeatures` to `hidden_dim`.
-   **Output**: A tensor of shape `(Batch, Time, Num_Features, Hidden_Dim)`. Note that `Num_Features` is the sum of categorical columns + 1 (for the combined continuous block).

### 4.2 Invoice Encoder (Intra-Sample)
-   **Role**: Summarize the "row" of data for a single invoice.
-   **Operation**: A standard Transformer Encoder block that attends over the `Num_Features` dimension.
-   **Pooling**: Global Average Pooling over the feature dimension.
-   **Result**: `(Batch, Time, Hidden_Dim)`. We now have one vector per invoice.

### 4.3 Sequence Encoder (Inter-Sample)
-   **Role**: Understand the story over time.
-   **Operation**: A **Causal** Transformer Masked Attention.
    -   `is_causal=True`: Ensures step $t$ cannot see step $t+1$.
    -   `RoPE`: Injects relative position info.
-   **Output**:
    -   `all_hidden`: The history states `(Batch, Time, Hidden_Dim)`.
    -   `context`: The state at the *last* valid time step (representing the target invoice context).

### 4.4 Temporal Cross-Attention
-   **Role**: Explicitly query the history based on the current context.
-   **Query**: The Invoice Representation of the *current* target invoice.
-   **Key/Value**: The `all_hidden` output from the Sequence Encoder (the history).
-   **Intuition**: "I am a high-value invoice due on a Monday (Query). Show me similar situations in the past (Keys/Values)."

### 4.5 Prediction Head
-   **Input**: Concatenation of `[Current_Invoice_Rep, Sequence_Context, Cross_Attention_Result]`.
    -   Total Dim: `3 * Hidden_Dim`.
-   **Layers**: `GRN -> GRN -> Linear(1)`.
-   **Output**: Scalar prediction (log-space days).

---

## 5. Training Strategy & Metrics

### 5.1 Loss Function Details
-   **Base Loss**: `SmoothL1Loss` (Huber).
    $$ L = \begin{cases} 0.5 x^2 & \text{if } |x| < 1 \\ |x| - 0.5 & \text{otherwise} \end{cases} $$
-   **Weighting Strategy**: We apply a custom weight to "high target" values.
    $$ w_i = 1.0 + \text{weight\_factor} \times \frac{\text{target}_i}{\text{mean\_target}} $$
    This penalizes errors on high-churn/high-delay customers significantly more than errors on punctual payers.

### 5.2 Optimization
-   **Optimizer**: AdamW (`lr=1e-3`, `weight_decay=1e-4`).
-   **Scheduler**: ReduceLROnPlateau. If validation RMSE stops improving for 3 epochs, reduce LR by factor 0.5.
-   **Gradient Clipping**: Norm clipped to 3.0 to prevent exploding gradients in the LSTM/Transformer structures.
-   **Mixed Precision**: FP16 training enabled for GPU efficiency.

### 5.3 Exponential Moving Average (EMA)
-   We maintain a "Shadow Model" whose weights are an exponential moving average of the training model's weights.
-   Decay: $0.9999$.
-   **Benefit**: The EMA model is much more stable and generalizes better. All validation/test metrics are reported using the EMA weights, not the raw training weights.

### 5.4 Ablation Study Capability (`ablate.py`)
The system includes a fully automated ablation engine. It can sequentially disable features or groups of features (by zeroing them out) and re-evaluating the pre-trained model to determine feature importance.

---

## 6. Configuration Parameters (Defaults)
Defined in `core/config.py`:
-   **Input Window**: `max_seq_len = 50`.
-   **Model Size**: `hidden_dim = 128`, `heads = 4`.
-   **Layers**: 1 Invoice Encoder Layer, 1 Sequence Encoder Layer (lightweight but effective).
-   **Training**: Batch size 128, 20 Epochs.
-   **Augmentation**: `cutout_ratio=0.15`, `dropout_ratio=0.2`.

---

## 7. Critical Nuances & Implementation Details

This section highlights subtle engineering choices that significantly impact model performance and stability.

### 7.1 Weighted Loss Strategy ("Business-Aware Loss")
In churn/delay prediction, not all errors are equal. Underestimating a large delay (predicting 2 days when actual is 30) is far worse than overestimating a small one.
-   **Mechanism**: We apply a dynamic weight $w_i$ to the loss for each sample:
    $$ w_i = 1.0 + \text{weight\_factor} \times \frac{\text{target}_i}{\text{mean\_target}} $$
-   **Impact**: The model pays significantly more attention to "problematic" invoices (long delays). A sample with 5x the average delay contributes ~5x more to the gradient updates (depending on `weight_factor`).

### 7.2 Stochastic Depth (Drop Path)
To train deep Transformer models effectively on this dataset without overfitting:
-   **Implementation**: Inside `InvoiceEncoder` and `SequenceEncoder`, entire residual branches are randomly dropped during training with probability `drop_path_rate`.
-   **Effect**: This acts as an implicit ensemble of many shallower sub-networks. It prevents co-adaptation of layers and forces the model to learn robust features that don't rely on every single layer being present.

### 7.3 Target Decoding & Stability
The target variable flows through a rigorous transformation pipeline to ensure numerical stability:
1.  **Training**: $y' = \log(1 + y)$.
2.  **Inference**:
    -   Raw output: $\hat{y}'$.
    -   Inverse transform: $\hat{y} = \exp(\hat{y}') - 1$.
    -   **clipping**: $\hat{y}_{final} = \max(0, \hat{y})$.
-   **Nuance**: The explicit clip at the end is crucial. Without it, the model might predict negative delays (physically impossible) for very punctual users, which distorts metrics like MAPE.

### 7.4 Fourier Feature Gating
We don't just project continuous features into high dimensions; we **gate** them.
-   **Code**: `gated = sigmoid(GateLayer(fourier_feats))`
-   **Why**: Some continuous features might be noise for specific invoices. The gating mechanism allows the network to suppress (multiply by ~0) specific frequency components or entire features dynamically, acting as a learned feature selector.
