# Especificação da Arquitetura de Rede Neural

Este documento fornece uma especificação técnica abrangente da arquitetura do modelo, incluindo descrições em nível de componentes, formulações matemáticas, diagramas de fluxo de dados e justificativas de design fundamentadas em avanços recentes em deep learning para dados sequenciais.

---

## Resumo

A arquitetura proposta é uma **rede neural sequencial baseada em Transformer** especificamente projetada para previsão de inadimplência de pagamentos em múltiplos horizontes. O modelo processa históricos de pagamento de usuários de comprimento variável e produz estimativas de probabilidade calibradas para três níveis de severidade de atraso em faturas. As principais inovações arquiteturais incluem tokenização hierárquica de features, embeddings posicionais rotativos, redes residuais com gates e estratégias de regularização por target.

---

## Visão Geral da Arquitetura

```
┌─────────────────────────────────────────────────────────────────────┐
│                            ENTRADA                                   │
│    Histórico de Pagamentos: [Fatura₁, Fatura₂, ..., Faturaₙ]        │
│    Features Categóricas: (batch, seq_len, n_cat)                    │
│    Features Contínuas: (batch, seq_len, n_cont)                     │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     1. TOKENIZAÇÃO DE FEATURES                       │
│         Features heterogêneas → Vetores de token uniformes (d=128)   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     2. ENCODER DE FATURA                             │
│         Agrega tokens de features dentro de cada fatura              │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     3. ENCODER DE SEQUÊNCIA                          │
│         Modela dependências temporais através da sequência           │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     4. ATENÇÃO TEMPORAL                              │
│         Foco seletivo em momentos historicamente significativos      │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     5. CABEÇAS DE PREVISÃO                           │
│         Cabeças específicas: P(curto), P(médio), P(longo)            │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                            SAÍDA                                     │
│              Probabilidades: [P(>3d), P(>7d), P(>14d)]              │
│              Calibradas via Temperature Scaling                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Componente 1: Tokenização de Features

### Objetivo

Transformar features de entrada heterogêneas (categóricas e contínuas) em um espaço de representação denso unificado adequado para processamento baseado em Transformer.

### Embedding de Features Categóricas

Cada feature categórica é mapeada através de uma matriz de embedding aprendível:

$$\mathbf{e}_{\text{cat}}^{(j)} = \mathbf{E}^{(j)}[\text{id}_j] \in \mathbb{R}^{d}$$

onde $\mathbf{E}^{(j)} \in \mathbb{R}^{V_j \times d}$ é a matriz de embedding para a feature $j$ com tamanho de vocabulário $V_j$, e $\text{id}_j$ é o índice da categoria codificada.

**Propriedades Principais:**
- Matriz de embedding separada por feature categórica
- Índice de padding = 0 reservado para padding de sequência
- Dimensão de embedding $d = 128$ (configurável)

### Embedding de Features Contínuas

Features contínuas passam por uma transformação em múltiplos estágios:

$$\text{Valor Bruto} \xrightarrow{\text{StandardScaler}} \text{Normalizado} \xrightarrow{\text{Periódico}} \text{Multi-frequência} \xrightarrow{\text{Proj. com Gate}} \mathbf{e}_{\text{cont}} \in \mathbb{R}^d$$

**Embedding Periódico:**

O embedding periódico expande um valor escalar em uma base sinusoidal multi-frequência:

$$\text{PE}(x) = \left[\sin(2\pi f_1 x), \cos(2\pi f_1 x), \ldots, \sin(2\pi f_K x), \cos(2\pi f_K x)\right]$$

onde as frequências $f_k$ são aprendíveis ou fixas de acordo com uma escala geométrica.

**Projeção com Gate:**

$$\mathbf{e}_{\text{cont}} = \sigma(\mathbf{W}_g \cdot \text{PE}(x) + \mathbf{b}_g) \odot (\mathbf{W}_v \cdot \text{PE}(x) + \mathbf{b}_v)$$

onde $\sigma$ denota a ativação sigmoide e $\odot$ representa multiplicação elemento a elemento. O mecanismo de gate permite fluxo seletivo de informação.

### Diagrama Arquitetural

```
┌───────────────────────────────────────────────────────────────┐
│                     Tokenizador de Features                    │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│   Features Categóricas          Features Contínuas            │
│   ┌─────────────────┐           ┌─────────────────┐          │
│   │ formaPagamento  │           │ hist_mean_delay │          │
│   │ regiao          │           │ value_ratio     │          │
│   │ venc_dayofweek  │           │ seq_position    │          │
│   └────────┬────────┘           └────────┬────────┘          │
│            │                             │                    │
│            ▼                             ▼                    │
│   ┌─────────────────┐           ┌─────────────────┐          │
│   │    Embedding    │           │    Embedding    │          │
│   │  Lookup Matriz  │           │    Periódico    │          │
│   └────────┬────────┘           └────────┬────────┘          │
│            │                             │                    │
│            ▼                             ▼                    │
│   e_cat ∈ ℝ^{n_cat × d}         ┌─────────────────┐          │
│            │                    │ Projeção Linear │          │
│            │                    │    com Gate     │          │
│            │                    └────────┬────────┘          │
│            │                             │                    │
│            │                    e_cont ∈ ℝ^{n_cont × d}       │
│            │                             │                    │
│            └──────────┬──────────────────┘                   │
│                       ▼                                       │
│              Sequência de Tokens: [t₁, t₂, ..., tₖ]          │
│              Shape: (batch, seq_len, n_features, d)           │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

---

## Componente 2: Encoder de Fatura

### Objetivo

Agregar todos os tokens de features de uma única fatura (posição temporal) em uma representação unificada de nível de fatura que captura interações intra-fatura entre features.

### Arquitetura

O Encoder de Fatura emprega um Transformer superficial para modelar interações de features dentro de cada fatura:

```
┌───────────────────────────────────────────────────────────────┐
│                     Encoder de Fatura                          │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│   Tokens de Features para Fatura t:                           │
│   [formaPagamento, valor, regiao, venc_day, hist_delay, ...]  │
│                     │                                         │
│                     ▼                                         │
│   ┌─────────────────────────────────────────┐                │
│   │    Bloco de Self-Attention Multi-Head   │                │
│   │    (features atendem umas às outras)    │                │
│   └──────────────────┬──────────────────────┘                │
│                      │                                        │
│                      ▼                                        │
│   ┌─────────────────────────────────────────┐                │
│   │    Rede Feed-Forward (SwiGLU)           │                │
│   └──────────────────┬──────────────────────┘                │
│                      │                                        │
│                      ▼                                        │
│   ┌─────────────────────────────────────────┐                │
│   │          Mean Pooling                   │                │
│   │    h_t = (1/K) Σᵢ zᵢ                    │                │
│   └──────────────────┬──────────────────────┘                │
│                      │                                        │
│                      ▼                                        │
│        Representação da Fatura: h_t ∈ ℝ^d                     │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

### Formulação do Mean Pooling

$$\mathbf{h}_t = \frac{1}{K} \sum_{i=1}^{K} \mathbf{z}_i^{(t)}$$

onde $\mathbf{z}_i^{(t)}$ são as representações de saída do Transformer para a feature $i$ no passo temporal $t$, e $K$ é o número total de features.

**Justificativa do Design:** O mean pooling fornece uma agregação invariante à permutação que é robusta à ordenação de features enquanto preserva informação de todos os tokens de features.

---

## Componente 3: Encoder de Sequência

### Objetivo

Modelar dependências temporais através da sequência de faturas, capturando evolução comportamental e padrões de longo alcance no histórico de pagamentos.

### Decisões de Design Principais

| Decisão | Implementação | Justificativa |
|---------|---------------|---------------|
| Tipo de Atenção | Causal (mascarada) | Garante validade temporal—sem vazamento futuro |
| Encoding Posicional | Rotativo (RoPE) | Generalização de comprimento superior |
| Ativação | SwiGLU | Estado da arte para Transformers |
| Normalização | Pre-LayerNorm | Estabilidade de treinamento melhorada |

### Arquitetura

```
┌───────────────────────────────────────────────────────────────┐
│                    Encoder de Sequência                        │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│   Sequência de Faturas:                                       │
│   [h₁] → [h₂] → [h₃] → ... → [hₙ]                            │
│                                                               │
│       │      │      │              │                         │
│       ▼      ▼      ▼              ▼                         │
│   ┌──────────────────────────────────────────────────┐       │
│   │       Embedding Posicional Rotativo (RoPE)       │       │
│   │       (codifica posição temporal relativa)       │       │
│   └──────────────────────────────────────────────────┘       │
│                          ▼                                    │
│   ┌──────────────────────────────────────────────────┐       │
│   │          Bloco Transformer 1 (Causal)            │       │
│   │     Posição 3 atende apenas a 1, 2, 3            │       │
│   └──────────────────────────────────────────────────┘       │
│                          │                                    │
│                          ▼                                    │
│   ┌──────────────────────────────────────────────────┐       │
│   │          Bloco Transformer 2 (Causal)            │       │
│   └──────────────────────────────────────────────────┘       │
│                          │                                    │
│                          ▼                                    │
│   ┌──────────────────────────────────────────────────┐       │
│   │          Bloco Transformer 3 (Causal)            │       │
│   └──────────────────────────────────────────────────┘       │
│                          │                                    │
│                          ▼                                    │
│   Saídas:                                                     │
│   • context: h̃ₙ ∈ ℝ^d (representação última posição)        │
│   • all_hidden: [h̃₁, h̃₂, ..., h̃ₙ] ∈ ℝ^{n×d}               │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

### Máscara de Atenção Causal

A máscara de atenção garante causalidade temporal:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

onde $M_{ij} = -\infty$ se $j > i$ (posições futuras mascaradas).

---

## Componente 4: Atenção Temporal

### Propósito

Permitir que o modelo foque seletivamente em momentos historicamente significativos ao fazer previsões.

### Intuição

Nem todas as faturas passadas são igualmente informativas. Um atraso significativo há seis meses pode ser mais preditivo do que um pagamento pontual recente. O mecanismo de atenção aprende a ponderar posições históricas de acordo.

### Arquitetura

```
┌───────────────────────────────────────────────────────────────┐
│                   Atenção ao Histórico                        │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│   Query: Representação da Fatura Atual                        │
│          │                                                    │
│          ▼                                                    │
│   ┌─────────────────────────────────────────────────┐         │
│   │  Cálculo dos Pesos de Atenção:                  │         │
│   │                                                 │         │
│   │  α = softmax(h_current · W_q · W_k^T · H^T)     │         │
│   │                                                 │         │
│   │  Fatura 1: ████░░░░░░ (10% atenção)             │         │
│   │  Fatura 2: ██████████ (50% atenção) ← atraso!   │         │
│   │  Fatura 3: ████████░░ (30% atenção)             │         │
│   │  Fatura 4: ██░░░░░░░░ (10% atenção)             │         │
│   │                                                 │         │
│   └─────────────────────────────────────────────────┘         │
│                          │                                    │
│                          ▼                                    │
│   ┌─────────────────────────────────────────────────┐         │
│   │     Rede Residual com Gate (GRN)                │         │
│   │     (combina informação atendida)               │         │
│   └─────────────────────────────────────────────────┘         │
│                          │                                    │
│                          ▼                                    │
│        Representação Atendida: a ∈ ℝ^d                        │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

---

## Componente 5: Cabeças de Previsão

### Propósito

Transformar a representação combinada em saídas de probabilidade para cada variável alvo.

### Arquitetura

Cada target tem uma cabeça de previsão dedicada com regularização específica da tarefa:

```
┌───────────────────────────────────────────────────────────────┐
│                   Cabeças de Previsão                         │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│   Representação Combinada (3d dimensões):                     │
│   [fatura_atual | contexto_sequência | histórico_atendido]    │
│                                                               │
│            ┌──────────────┬──────────────┬──────────────┐     │
│            ▼              ▼              ▼              │     │
│   ┌────────────────┐ ┌────────────────┐ ┌────────────────┐    │
│   │ Cabeça: Curto  │ │ Cabeça: Médio  │ │ Cabeça: Longo  │    │
│   │  dropout=10%   │ │ dropout=12.5%  │ │  dropout=15%   │    │
│   ├────────────────┤ ├────────────────┤ ├────────────────┤    │
│   │     GRN 1      │ │     GRN 1      │ │     GRN 1      │    │
│   │     GRN 2      │ │     GRN 2      │ │     GRN 2      │    │
│   │    Linear(1)   │ │    Linear(1)   │ │    Linear(1)   │    │
│   └───────┬────────┘ └───────┬────────┘ └───────┬────────┘    │
│           │                  │                  │             │
│           ▼                  ▼                  ▼             │
│       logit_curto       logit_médio       logit_longo         │
│           │                  │                  │             │
│           └─────────────────┼───────────────────┘             │
│                             ▼                                 │
│   ┌─────────────────────────────────────────────────────┐     │
│   │            Temperature Scaling (τ)                  │     │
│   │         logits_calibrados = logits / τ              │     │
│   └─────────────────────────────────────────────────────┘     │
│                             │                                 │
│                             ▼                                 │
│                          Sigmoid                              │
│                             │                                 │
│                             ▼                                 │
│         [P(>3 dias), P(>7 dias), P(>14 dias)]                 │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

### Justificativa do Dropout por Target

Targets mais raros recebem regularização mais forte para prevenir overfitting:

| Target | Prevalência | Taxa de Dropout |
|--------|-------------|-----------------|
| `target_short` | ~10% | 10% |
| `target_medium` | ~5% | 12.5% |
| `target_long` | ~3% | 15% |

---

## Componentes Especializados

### Rede Residual com Gate (GRN)

A GRN permite fluxo de informação adaptativo com gate aprendido:

$$\text{GRN}(x) = \text{LayerNorm}\left(x + g \odot \eta_2\right)$$

onde:

$$\eta_1 = \text{SiLU}(W_1 x + b_1)$$
$$\eta_2 = W_2 \eta_1 + b_2$$
$$g = \sigma(W_g \eta_2 + b_g)$$

**Propriedade Chave**: O gate $g$ permite que a rede contorne seletivamente transformações, facilitando o fluxo de gradientes durante o treinamento.

---

### Ativação SwiGLU

SwiGLU é uma unidade linear com gate que supera ReLU em arquiteturas Transformer:

$$\text{SwiGLU}(x) = \text{Swish}(W_1 x) \odot (W_2 x)$$

onde $\text{Swish}(x) = x \cdot \sigma(x)$.

É usado nas camadas feedforward de todos os blocos Transformer.

---

### Stochastic Depth (DropPath)

Durante o treinamento, blocos residuais inteiros são removidos aleatoriamente:

$$\text{Saída} = \begin{cases} x + f(x) & \text{com probabilidade } 1-p \\ x & \text{com probabilidade } p \end{cases}$$

**Benefícios**:
- Efeito de regularização similar ao dropout
- Reduz co-adaptação entre camadas
- Permite treinamento de redes mais profundas

---

### Temperature Scaling

Calibração pós-treinamento ajusta a confiança das previsões:

$$\text{logits}_{\text{calibrados}} = \frac{\text{logits}}{\tau}$$

| Temperatura | Efeito |
|-------------|--------|
| $\tau > 1$ | Probabilidades mais suaves (menos confiante) |
| $\tau < 1$ | Probabilidades mais agudas (mais confiante) |
| $\tau = 1$ | Inalterado (não calibrado) |

A temperatura é otimizada no conjunto de validação para minimizar o Erro de Calibração Esperado (ECE).

---

## Especificações de Hiperparâmetros

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| `hidden_dim` | 128 | Dimensão do token/representação oculta |
| `n_heads` | 4 | Número de cabeças de atenção |
| `n_blocks` | 3 | Profundidade do encoder de sequência |
| `num_invoice_layers` | 1-2 | Profundidade do encoder de fatura |
| `dropout` | 0.1 | Taxa base de dropout |
| `drop_path_rate` | 0.1 | Taxa de stochastic depth |
| `ff_mult` | 4 | Fator de expansão feedforward |

### Contagem de Parâmetros

Com configuração padrão: **~500K–1M parâmetros treináveis**

---

## Fluxo de Dados Completo

```
Entrada: categorical (batch, seq_len, n_cat), continuous (batch, seq_len, n_cont), lengths
                                    │
                                    ▼
                          ┌─────────────────┐
                          │  Tokenização    │
                          └────────┬────────┘
                                   │
                    tokens: (batch, seq_len, n_tokens, 128)
                                   │
                                   ▼
                          ┌─────────────────┐
                          │Encoder Fatura   │
                          └────────┬────────┘
                                   │
                  repr_fatura: (batch, seq_len, 128)
                                   │
                                   ▼
                          ┌─────────────────┐
                          │Encoder Sequência│
                          └────────┬────────┘
                                   │
                 ┌─────────────────┴─────────────────┐
                                                     
        context: (batch, 128)           all_hidden: (batch, seq_len, 128)
                                                     
                 │                                   │
                 │                    ┌──────────────────────┐
                 │                    │  Atenção Temporal    │
                 │                    └──────────┬───────────┘
                 │                               │
                 │                    attended: (batch, 128)
                 │                               │
                 └───────────────┬───────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │ current = repr_fatura   │
                    │         [:, -1, :]      │
                    └────────────┬────────────┘
                                 │
                   current: (batch, 128)
                                 │
                    ┌────────────┴────────────┐
                    │  Concatenar:            │
                    │  [current, context,     │
                    │   attended]             │
                    └────────────┬────────────┘
                                 │
                   combined: (batch, 384)
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │  Cabeças de Previsão    │
                    └────────────┬────────────┘
                                 │
                    logits: (batch, 3)
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │  Temperature Scaling    │
                    │       + Sigmoid         │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    probabilidades: (batch, 3)
```

---

## Justificativa do Design

| Escolha de Design | Justificativa |
|-------------------|---------------|
| Arquitetura Transformer | Captura dependências de longo alcance em dados sequenciais |
| Atenção Causal | Mantém validade temporal (sem vazamento futuro) |
| Embedding Posicional Rotativo | Generalização de comprimento superior vs. posições absolutas |
| Redes Residuais com Gate | Permite computação adaptativa e fluxo de gradientes |
| Cabeças de Previsão Separadas | Regularização específica por tarefa para diferentes prevalências |
| Temperature Scaling | Calibração pós-hoc para probabilidades confiáveis |
| Embeddings Periódicos | Representação natural para features temporais cíclicas |

---

## Referências

- Vaswani, A., et al. (2017). "Attention Is All You Need." *NeurIPS*.
- Su, J., et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding." *arXiv*.
- Shazeer, N. (2020). "GLU Variants Improve Transformer." *arXiv*.
- Huang, G., et al. (2016). "Deep Networks with Stochastic Depth." *ECCV*.
- Guo, C., et al. (2017). "On Calibration of Modern Neural Networks." *ICML*.
