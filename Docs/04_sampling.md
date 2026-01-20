# Metodologia de Amostragem e Particionamento de Dados

Este documento fornece uma especificação completa das estratégias de amostragem de dados, procedimentos de particionamento train/validation/test e técnicas de data augmentation empregadas no pipeline de treinamento.

---

## Visão Geral do Pipeline

O pipeline de preparação de dados consiste em cinco estágios sequenciais:

```
Dados Brutos
    │
    ▼
[1] Limpeza e Validação de Dados
    │
    ▼
[2] Subamostragem Estratificada (Opcional)
    │
    ▼
[3] Divisão Train/Validation/Test Baseada em Grupos
    │
    ▼
[4] Normalização de Features (Ajuste no Conjunto de Treinamento)
    │
    ▼
[5] Construção de Índices de Sequência
    │
    ▼
Datasets Prontos para DataLoader
```

---

## Estágio 1: Limpeza e Validação de Dados

### Remoção de Usuários Inválidos

Usuários com informações de data de pagamento corrompidas ou ausentes são excluídos inteiramente:

```python
# Identificar usuários com qualquer data de pagamento ausente (NaT)
invalid_users = dataframe.loc[
    dataframe[payment_date_col].isna(), 
    user_col
].unique()

# Remover todos os registros destes usuários
dataframe = dataframe[~dataframe[user_col].isin(invalid_users)]
```

**Justificativa:** Históricos parciais de usuário com datas ausentes comprometem a integridade da sequência. Remover o usuário inteiro previne vazamento de dados de informação temporal incompleta.

---

## Estágio 2: Subamostragem Estratificada

### Configuração

Controlada via o parâmetro `sample_frac` em `config.py`:

```python
sample_frac: float = 0.08  # Utilizar 8% dos usuários para treinamento
```

Quando `sample_frac < 1.0`, a subamostragem estratificada é ativada para reduzir o tamanho do dataset enquanto preserva a representação das classes minoritárias.

### Amostragem com Prioridade Hierárquica

O algoritmo de amostragem implementa uma **fila de prioridade** baseada na raridade da variável alvo:

```python
# Ordem de prioridade (do mais raro ao mais comum):
Prioridade 1: Todos usuários com target_long = 1    (atraso > 14 dias)
Prioridade 2: Todos usuários com target_medium = 1  (7 < atraso ≤ 14 dias apenas)
Prioridade 3: Todos usuários com target_short = 1   (3 < atraso ≤ 7 dias apenas)
Prioridade 4: Amostra aleatória de usuários sem inadimplências
```

### Especificação do Algoritmo

```python
def stratified_subsample(user_targets: Dict, sample_frac: float) -> List:
    total_users = len(user_targets)
    target_count = int(total_users * sample_frac)
    
    # Incluir todos usuários com inadimplências raras
    selected = list(users_with_target_long)
    selected += list(users_with_target_medium_only)
    
    # Preencher quota restante das classes majoritárias
    remaining_quota = target_count - len(selected)
    if remaining_quota > 0:
        majority_pool = np.concatenate([
            users_with_target_short_only, 
            users_no_default
        ])
        sampled_majority = np.random.choice(
            majority_pool, 
            size=min(remaining_quota, len(majority_pool)), 
            replace=False
        )
        selected.extend(sampled_majority)
    
    return selected
```

### Exemplo de Saída de Log

```
[Amostragem Estratificada] Usuários com target_long (>14d): 2.500
[Amostragem Estratificada] Usuários com target_medium apenas (7-14d): 3.200
[Amostragem Estratificada] Usuários com target_short apenas (3-7d): 5.800
[Amostragem Estratificada] Usuários sem inadimplências (≤3d): 88.500

[Subamostragem] Amostragem estratificada: 10.000/100.000 usuários (10,0%)
[Subamostragem] Quota alvo: 8.000 | Selecionados: 10.000 (prioridade classe minoritária)
```

**Propriedade Chave:** Esta abordagem garante que **100% dos usuários com padrões raros de inadimplência** sejam incluídos, mesmo quando a fração de amostragem os excluiria normalmente.

---

## Estágio 3: Particionamento Train/Validation/Test

### Configuração

```python
@dataclass
class DataParams:
    test_size: float = 0.10    # 10% para conjunto de teste
    val_size: float = 0.10     # 10% para conjunto de validação
    random_state: int = 42     # Semente para reprodutibilidade
```

### Divisão Baseada em Grupos

O particionamento é realizado no **nível de usuário**, não no nível de amostra:

$$\text{Todas faturas do usuário } u \in \{\text{Train}, \text{Val}, \text{Test}\} \text{ exclusivamente}$$

### Justificativa para Divisão por Grupos

| Benefício | Explicação |
|-----------|------------|
| **Previne vazamento de dados** | Padrões comportamentais de usuário não vazam entre divisões |
| **Simula produção** | Novos usuários no momento de inferência não têm histórico de treinamento |
| **Previne overfitting** | Modelo não pode memorizar padrões específicos de usuário |
| **Permite avaliação adequada** | Desempenho de teste reflete generalização para usuários não vistos |

### Procedimento de Divisão em Dois Estágios

```python
# Estágio 1: Separar conjunto de teste (10% dos usuários)
group_split_test = GroupShuffleSplit(
    n_splits=1, 
    test_size=0.10, 
    random_state=42
)
train_val_idx, test_idx = next(group_split_test.split(X, groups=user_ids))

# Estágio 2: Separar validação do restante (10% do original = 11.1% do resto)
adjusted_val_size = 0.10 / (1.0 - 0.10)  # ≈ 0.111
group_split_val = GroupShuffleSplit(
    n_splits=1, 
    test_size=adjusted_val_size, 
    random_state=42
)
train_idx, val_idx = next(group_split_val.split(X_train_val, groups=train_val_user_ids))
```

### Distribuição Resultante

```
[Divisão] Conjunto de Treinamento:  8.000 usuários (80%)
[Divisão] Conjunto de Validação:    1.000 usuários (10%)
[Divisão] Conjunto de Teste:        1.000 usuários (10%)
```

---

## Estágio 4: Normalização de Features

### Aplicação do StandardScaler

A normalização é aplicada exclusivamente a features contínuas:

```python
from sklearn.preprocessing import StandardScaler

# Ajustar APENAS nos dados de treinamento
scaler = StandardScaler()
scaler.fit(train_df[continuous_columns])

# Transformar todas divisões usando estatísticas de treinamento
train_df[continuous_columns] = scaler.transform(train_df[continuous_columns])
val_df[continuous_columns] = scaler.transform(val_df[continuous_columns])
test_df[continuous_columns] = scaler.transform(test_df[continuous_columns])
```

### Restrição Crítica

> **Aviso:** O scaler deve ser ajustado **exclusivamente no conjunto de treinamento**. Ajustar em dados de validação ou teste constitui vazamento de dados, pois incorpora informação futura nos parâmetros de normalização.

---

## Estágio 5: Construção de Índices de Sequência

Após o particionamento, cada subconjunto passa pela geração de índices de janela expansiva:

```python
train_indices = _create_expanding_indices(train_df, group_cols, max_seq_len)
val_indices = _create_expanding_indices(val_df, group_cols, max_seq_len)
test_indices = _create_expanding_indices(test_df, group_cols, max_seq_len)
```

### Contagens de Amostras Resultantes

```
[Sequências] Conjunto de Treinamento:  45.000 amostras
[Sequências] Conjunto de Validação:     5.500 amostras
[Sequências] Conjunto de Teste:         5.500 amostras
[Sequências] Total:                    56.000 amostras
```

Note que a contagem de amostras excede a contagem de usuários porque a metodologia de janela expansiva gera múltiplas amostras por usuário (uma por posição de fatura previsível).

---

## Data Augmentation

### Escopo de Aplicação

Data augmentation é aplicado **exclusivamente ao conjunto de treinamento** para prevenir contaminação da avaliação:

```python
train_dataset = SequentialDataset(..., augment=True, augment_probability=0.1)
val_dataset = SequentialDataset(..., augment=False)
test_dataset = SequentialDataset(..., augment=False)
```

### Técnicas de Augmentation

| Técnica | Descrição | Probabilidade | Efeito |
|---------|-----------|---------------|--------|
| **Temporal Cutout** | Zerar posições temporais aleatórias | 10% | Simula dados históricos ausentes |
| **Feature Dropout** | Zerar dimensões de feature aleatórias | 10% | Encoraja redundância de features |
| **Ruído Gaussiano** | Adicionar $\mathcal{N}(0, \sigma^2)$ a features contínuas | 10% | Melhora robustez a ruído de entrada |
| **Time Warp** | Duplicar ou remover um passo temporal | 5% | Simula irregularidades temporais |

### Implementação de Augmentation

```python
def apply_augmentation(sequence: Tensor, probability: float) -> Tensor:
    if random.random() < probability:
        augmentation = random.choice([
            temporal_cutout,
            feature_dropout,
            gaussian_noise,
            time_warp
        ])
        return augmentation(sequence)
    return sequence
```

---

## Análise de Desbalanceamento de Classes

Estatísticas pós-processamento são computadas para monitoramento:

```python
def compute_class_weights(targets: np.ndarray) -> Dict[str, float]:
    weights = {}
    for i, name in enumerate(target_names):
        n_positive = targets[:, i].sum()
        n_negative = len(targets) - n_positive
        weights[name] = n_negative / n_positive
    return weights
```

### Exemplo de Saída

```
┌─────────────────┬────────────┬────────────┬────────────┬─────────────────────┐
│ Alvo            │ Positivos  │ Prevalência│ Negativos  │ Peso Recomendado    │
├─────────────────┼────────────┼────────────┼────────────┼─────────────────────┤
│ target_short    │ 4.500      │ 10,00%     │ 40.500     │ 9,00                │
│ target_medium   │ 2.400      │ 5,33%      │ 42.600     │ 17,75               │
│ target_long     │ 1.200      │ 2,67%      │ 43.800     │ 36,50               │
└─────────────────┴────────────┴────────────┴────────────┴─────────────────────┘
```

---

## Referência de Configuração

```python
@dataclass
class DataParams:
    test_size: float = 0.10           # Proporção do conjunto de teste
    val_size: float = 0.10            # Proporção do conjunto de validação
    random_state: int = 42            # Semente aleatória para reprodutibilidade
    sample_frac: float = 0.08         # Fração de amostragem de usuários
    min_sequence_length: int = 2      # Mínimo de faturas por sequência
    max_sequence_length: int = 50     # Comprimento máximo da sequência
```

---

## Checklist de Validação

- [x] Sem sobreposição de usuários entre conjuntos train/validation/test
- [x] Classes minoritárias adequadamente representadas em todas divisões
- [x] Normalização ajustada exclusivamente em dados de treinamento
- [x] Data augmentation aplicado apenas ao conjunto de treinamento
- [x] Semente aleatória fixada para reprodutibilidade experimental
- [x] Construção de sequência respeita ordenação temporal

---

## Referências

- Pedregosa, F., et al. (2011). "Scikit-learn: Machine Learning in Python." *JMLR*.
- Lemaitre, G., et al. (2017). "Imbalanced-learn: A Python Toolbox to Tackle the Curse of Imbalanced Datasets." *JMLR*.
