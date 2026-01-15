# Modelo de Predicao de Inadimplencia

## Visao Geral

Este documento descreve a arquitetura do modelo de predicao de inadimplencia. O modelo utiliza uma arquitetura baseada em Transformer com componentes especializados para dados tabulares.

---

## Arquitetura

O modelo segue a estrutura:

```
Entrada -> Tokenizador de Features -> GRN -> Cross-Attention -> Transformer -> Pooling -> Classificador
```

---

## Componentes

### 1. Tokenizador de Features (Feature Tokenizer)

Converte features brutas em representacoes vetoriais uniformes.

#### Variaveis Categoricas
- Cada variavel categorica recebe um Embedding dedicado
- Dimensao do token: igual ao `hidden_dim` (32)
- Dropout de 10% aplicado apos embedding

#### Variaveis Continuas
- Utilizacao de PLR Embedding (Piecewise Linear Encoding)
- Numero de bins: 8
- O PLR cria representacoes nao-lineares para variaveis numericas

#### Token CLS
- Token especial adicionado no inicio da sequencia
- Utilizado para agregacao da informacao global

### 2. PLR Embedding (Piecewise Linear Encoding)

Tecnica para codificacao de variaveis continuas:
- Divisao do espaco numerico em bins
- Ativacao ReLU baseada na distancia aos limites dos bins
- Projecao linear para dimensao do token

Parametros:
- `n_bins`: 8
- Limites iniciais: distribuidos entre -3 e 3

### 3. Gated Residual Network (GRN)

Rede com mecanismo de gate para selecao de features:

```
h = GELU(Linear(x))
h = Dropout(h)
output = LayerNorm(x + sigmoid(Gate(h)) * Linear(h))
```

- Dimensao oculta: 2x dimensao do modelo
- Dropout: 25%

### 4. Cross-Feature Attention

Mecanismo de atencao cruzada entre features categoricas e continuas:
- Features continuas atendem as features categoricas
- Numero de heads: 2
- Permite captura de interacoes entre tipos de features

### 5. Transformer Encoder

Encoder padrao do Transformer com modificacoes:
- Pre-Layer Normalization (Pre-LN)
- Ativacao GELU
- Dimensao do feedforward: 4x hidden_dim

Parametros:
| Parametro | Valor |
|-----------|-------|
| Camadas (n_blocks) | 2 |
| Dimensao (hidden_dim) | 32 |
| Attention heads | 2 |
| Dropout | 25% |

### 6. Multi-Scale Pooling

Agregacao multi-escala das representacoes:
1. Token CLS (representacao global)
2. Media dos tokens de features
3. Maximo dos tokens de features

Concatenacao resulta em dimensao: `hidden_dim * 3 = 96`

### 7. Classificador

Rede MLP para classificacao binaria:

```
LayerNorm -> Linear(96, 32) -> GELU -> Dropout -> Linear(32, 1)
```

---

## Parametros do Modelo

| Parametro | Valor |
|-----------|-------|
| hidden_dim | 32 |
| n_blocks | 2 |
| n_heads | 2 |
| dropout | 0.25 |
| Total de parametros | ~100k |

---

## Funcao de Perda

### Asymmetric Loss

Funcao de perda para dados desbalanceados:

```
loss_pos = -weight_pos * log(p) * (1-p)^gamma_pos
loss_neg = -log(1-p_clipped) * p^gamma_neg
```

Parametros:
| Parametro | Valor | Descricao |
|-----------|-------|-----------|
| gamma_neg | 4 | Foco em negativos dificeis |
| gamma_pos | 0 | Preserva todos os positivos |
| clip | 0.05 | Margem para negativos |
| pos_weight | 15.0 | Peso da classe positiva |

### Focal Loss (Alternativa)

```
FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
```

Parametros:
| Parametro | Valor |
|-----------|-------|
| alpha | 0.75 |
| gamma | 2.0 |

---

## Otimizacao

### Otimizador
- AdamW
- Learning rate: 5e-4
- Weight decay: 1e-3

### Scheduler
- ReduceLROnPlateau
- Fator de reducao: 0.5
- Paciencia: 8 epocas
- Learning rate minimo: 1e-6

### Regularizacao
- Dropout: 25%
- Weight decay: 1e-3
- Gradient clipping: max_norm = 1.0

---

## Treinamento

| Parametro | Valor |
|-----------|-------|
| Batch size | 256 |
| Epocas maximas | 150 |
| Early stopping patience | 25 |
| Mixed precision | Habilitado (FP16) |

### Metrica de Validacao
- ROC-AUC utilizado para early stopping
- Melhor modelo salvo com base no AUC de validacao

---

## Inicializacao de Pesos

| Tipo de Camada | Metodo |
|----------------|--------|
| Linear | Truncated Normal (std=0.02) |
| Embedding | Normal (std=0.02) |
| LayerNorm | weight=1.0, bias=0.0 |

---

## Inferencia

1. Entrada: features categoricas e continuas
2. Tokenizacao das features
3. Processamento pelo Transformer
4. Pooling multi-escala
5. Classificacao
6. Saida: logit (sigmoidear para probabilidade)

---

## Arquivos

| Arquivo | Descricao |
|---------|-----------|
| `Model/network.py` | Definicao da arquitetura |
| `Model/trainer.py` | Logica de treinamento |
| `Model/data.py` | Carregamento e preprocessamento |
| `Configs/config.py` | Parametros do modelo |
