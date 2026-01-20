# Modelagem de Sequências e Organização Temporal de Dados

Este documento fornece um tratamento completo da metodologia empregada para organizar dados de pagamento em sequências temporais adequadas para consumo por modelos de deep learning.

---

## Framework Teórico

O modelo conceitualiza o histórico de pagamento de cada usuário como uma **série temporal multivariada**, onde cada fatura constitui uma observação temporal discreta. A suposição fundamental de modelagem é que o comportamento futuro de pagamento pode ser previsto a partir de padrões observados em sequências históricas de pagamento.

### Estrutura Hierárquica de Dados

```
Usuário (usuarioId)
  └── Contrato 1 (locacaoId)
  │     ├── Parcela 1 (t₁)
  │     ├── Parcela 2 (t₂)
  │     └── Parcela 3 (t₃)
  └── Contrato 2 (locacaoId)
        ├── Parcela 1 (t₄)
        └── Parcela 2 (t₅)
```

Esta organização hierárquica captura múltiplos níveis de dependência temporal:

| Nível | Tipo de Padrão | Descrição |
|-------|----------------|-----------|
| **Nível de Usuário** | Consistência entre contratos | Traços comportamentais que persistem através dos contratos |
| **Nível de Contrato** | Dinâmicas intra-contrato | Padrões de pagamento específicos ao ciclo de vida do contrato |
| **Nível de Parcela** | Progressão sequencial | Dependências temporais imediatas entre pagamentos consecutivos |

---

## Metodologia de Janela Expansiva

O modelo emprega uma estratégia de **janela expansiva** para construção de amostras de treinamento, simulando o cenário real de inferência onde previsões são geradas usando toda informação histórica disponível.

### Definição Formal

Para um usuário com $N$ faturas ordenadas cronologicamente como $\{I_1, I_2, \ldots, I_N\}$, o sistema gera $N-1$ amostras de treinamento:

$$\text{Amostra}_k: \{I_1, I_2, \ldots, I_k\} \rightarrow y_{k+1}, \quad k \in \{1, 2, \ldots, N-1\}$$

onde $I_i$ denota o vetor de features para a $i$-ésima fatura e $y_{k+1} \in \{0,1\}^3$ representa o vetor alvo multi-label para a fatura $k+1$.

### Exemplo Ilustrativo

```
Usuário com 5 faturas:

Amostra 1: [I₁]             → Prever labels de atraso para I₂
Amostra 2: [I₁, I₂]         → Prever labels de atraso para I₃
Amostra 3: [I₁, I₂, I₃]     → Prever labels de atraso para I₄
Amostra 4: [I₁, I₂, I₃, I₄] → Prever labels de atraso para I₅
```

### Propriedades das Janelas Expansivas

| Propriedade | Descrição |
|-------------|-----------|
| **Integridade causal** | Sem vazamento de informação futura—previsões usam apenas dados passados |
| **Utilização máxima de dados** | Múltiplas amostras de treinamento extraídas por usuário |
| **Contexto progressivo** | Previsões posteriores beneficiam-se de contexto histórico mais rico |
| **Currículo natural** | Previsões iniciais (menos amostras) são inerentemente mais difíceis |

---

## Configuração de Comprimento de Sequência

### Parâmetros

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `max_sequence_length` | 50 | Número máximo de faturas na sequência de entrada |
| `min_sequence_length` | 2 | Mínimo de faturas necessárias para amostra válida |

### Janela Deslizante para Históricos Extensos

Quando o histórico do usuário excede `max_sequence_length`, um mecanismo de janela deslizante é ativado para manter a tratabilidade computacional:

```
Usuário com 60 faturas (max_length = 50):

Amostra 50: [I₁,  I₂,  ..., I₅₀]  → Prever: I₅₁
Amostra 51: [I₂,  I₃,  ..., I₅₁]  → Prever: I₅₂  ← janela desliza para frente
Amostra 52: [I₃,  I₄,  ..., I₅₂]  → Prever: I₅₃
⋮
```

Este mecanismo mantém uma janela de contexto fixa das **50 faturas mais recentes**, balanceando:
- **Eficiência computacional**: Comprimento de sequência limitado permite processamento em lote consistente
- **Relevância temporal**: Histórico recente é mais preditivo que passado distante
- **Restrições de memória**: Memória GPU escala linearmente com comprimento da sequência

---

## Algoritmo de Construção de Índices

O método `_create_expanding_indices()` gera tuplas de índice especificando padrões de acesso aos dados:

### Formato de Saída

```python
(start_index: int, end_index: int, target_index: int)
```

| Campo | Descrição |
|-------|-----------|
| `start_index` | Primeiro índice de linha da sequência de entrada (inclusivo) |
| `end_index` | Último índice de linha da sequência de entrada (exclusivo) |
| `target_index` | Índice de linha contendo labels alvo |

### Pseudocódigo do Algoritmo

```python
def _create_expanding_indices(
    dataframe: pd.DataFrame, 
    group_cols: List[str], 
    max_seq_len: int
) -> List[Tuple[int, int, int]]:
    """
    Gera índices de janela expansiva para construção de sequência.
    
    Complexidade de Tempo: O(N) onde N é a contagem total de faturas
    Complexidade de Espaço: O(M) onde M é a contagem total de amostras
    """
    indices = []
    
    # Computa índices de linha contíguos para cada grupo usuário-contrato
    group_offsets = compute_group_offsets(dataframe, group_cols)
    
    for user_id, user_indices in group_offsets.items():
        n_invoices = len(user_indices)
        
        for k in range(n_invoices - 1):
            target_idx = user_indices[k + 1]
            seq_end = user_indices[k + 1]  # Limite superior exclusivo
            
            # Aplica janela deslizante se sequência excede comprimento máximo
            seq_start = max(seq_end - max_seq_len, user_indices[0])
            
            indices.append((seq_start, seq_end, target_idx))
    
    return indices
```

### Análise de Complexidade

| Métrica | Complexidade | Descrição |
|---------|--------------|-----------|
| Tempo | $O(N)$ | Passagem única por todas as faturas |
| Espaço | $O(M)$ | Armazena $M = \sum_{u} (n_u - 1)$ tuplas de índice |

onde $N$ é o total de faturas e $n_u$ é a contagem de faturas para o usuário $u$.

---

## Especificação de Ordenação e Agrupamento

### Ordenação Temporal

Os dados devem ser ordenados para garantir consistência cronológica dentro do histórico de cada usuário:

```python
# Especificação de ordenação primária
sort_cols = ['usuarioId', 'locacaoId', 'ordem_parcela']

# Fallback quando ordem_parcela não disponível
sort_cols_fallback = ['usuarioId', 'locacaoId', 'vencimentoData']
```

### Especificação de Agrupamento

```python
group_cols = ['usuarioId', 'locacaoId']
```

Este agrupamento garante:
1. Armazenamento contíguo de faturas do mesmo par usuário-contrato
2. Preservação da ordenação temporal dentro dos grupos
3. Captura implícita de dependências entre contratos através de agregação no nível de usuário

---

## Padding de Sequência e Colação de Batch

Sequências de comprimento variável requerem padding para formar tensores uniformes para computação eficiente em GPU.

### Função de Colação

```python
def collate_sequences(batch: List[Tuple]) -> Tuple[Tensor, ...]:
    """
    Cola sequências de comprimento variável em tensores de batch com padding.
    
    Retorna:
        categorical_padded: (batch_size, max_len, n_cat_features)
        continuous_padded: (batch_size, max_len, n_cont_features)
        lengths: (batch_size,) - comprimentos originais das sequências
        targets: (batch_size, n_targets)
    """
    categorical_padded = pad_sequence(
        categorical_tensors, 
        batch_first=True, 
        padding_value=0      # Índice de padding para embeddings
    )
    continuous_padded = pad_sequence(
        continuous_tensors, 
        batch_first=True, 
        padding_value=0.0    # Zero-padding para features normalizadas
    )
    
    lengths = torch.tensor([len(seq) for seq in categorical_tensors])
    targets = torch.stack(target_tensors)
    
    return categorical_padded, continuous_padded, lengths, targets
```

### Ilustração do Padding

```
Batch com 3 sequências:
  Sequência A: [I₁, I₂, I₃]     (comprimento = 3)
  Sequência B: [I₁, I₂, I₃, I₄] (comprimento = 4)
  Sequência C: [I₁, I₂]         (comprimento = 2)

Após padding (batch_first=True, pad para max_len=4):
  Sequência A: [I₁, I₂, I₃, PAD]
  Sequência B: [I₁, I₂, I₃, I₄ ]
  Sequência C: [I₁, I₂, PAD, PAD]

Tensor de comprimentos: [3, 4, 2]
```

O modelo utiliza o tensor `lengths` para construir **máscaras de atenção** apropriadas, garantindo que posições de padding recebam peso de atenção zero e não contribuam para o cálculo da loss.

---

## Otimização do DataLoader

### Configuração de Alto Desempenho

```python
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=256,
    shuffle=True,
    num_workers=8,              # Processos paralelos de carregamento de dados
    pin_memory=True,            # Transferência acelerada CPU→GPU
    persistent_workers=True,    # Reutiliza workers entre epochs
    prefetch_factor=4           # Pré-carrega múltiplos batches por worker
)
```

### Justificativa da Otimização

| Parâmetro | Benefício |
|-----------|-----------|
| `num_workers > 0` | Sobrepõe pré-processamento de dados com computação GPU |
| `pin_memory=True` | Habilita transferência assíncrona de memória CPU para GPU |
| `persistent_workers=True` | Elimina overhead de inicialização de workers entre epochs |
| `prefetch_factor` | Reduz tempo ocioso da GPU mantendo fila de batches |

---

## Vantagens Metodológicas

| Vantagem | Descrição |
|----------|-----------|
| **Contexto Histórico Completo** | Modelo observa histórico de pagamento relevante completo até o ponto de previsão |
| **Previsão Causal** | Cada fatura prevista antes de ocorrer (sem vazamento temporal) |
| **Sem Vazamento de Informação** | Informação alvo estritamente excluída das sequências de entrada |
| **Eficiência de Dados** | Múltiplas amostras de treinamento geradas por usuário |
| **Dependências de Longo Alcance** | Atenção do Transformer captura relações temporais distantes |

---

## Estatísticas Típicas

Contagens representativas de amostras após construção de sequência:

```
[Sequências] Treinamento:  ~150.000 amostras
[Sequências] Validação:     ~20.000 amostras
[Sequências] Teste:         ~20.000 amostras
[Sequências] Total:        ~190.000 amostras

Configuração: max_seq_len=50, min_seq_len=2
```

Cada "amostra" representa uma subsequência de janela expansiva pareada com labels alvo correspondentes.

---

## Fundamentação Teórica

A abordagem de janela expansiva é fundamentada no paradigma de **modelagem autoregressiva**, amplamente empregado em:

| Domínio | Exemplo |
|---------|---------|
| Language Modeling | GPT, LLaMA, Claude |
| Time Series Forecasting | Temporal Fusion Transformer, Informer |
| Sequential Recommendation | SASRec, BERT4Rec |
| Speech Recognition | Conformer, Whisper |

The fundamental insight is that predicting the $(k+1)$-th element conditioned on elements $1$ through $k$ naturally captures sequential dependencies while maintaining causal validity:

$$P(y_{k+1} | I_1, I_2, \ldots, I_k) = f_\theta(I_{1:k})$$

where $f_\theta$ is the neural network parameterized by $\theta$.

---

## References

- Vaswani, A., et al. (2017). "Attention Is All You Need." *NeurIPS*.
- Lim, B., et al. (2021). "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting." *International Journal of Forecasting*.
- Brown, T., et al. (2020). "Language Models are Few-Shot Learners." *NeurIPS*.
