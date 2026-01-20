# Especificação de Variáveis Alvo

Este documento fornece uma especificação rigorosa das variáveis alvo empregadas no framework de classificação multi-label, incluindo suas relações hierárquicas, características de desbalanceamento de classes e estratégias de tratamento.

---

## Formulação do Problema

O modelo realiza **classificação binária multi-label**, prevendo simultaneamente três resultados binários que correspondem a diferentes níveis de severidade de atraso no pagamento:

| Variável Alvo | Label | Definição | Interpretação |
|---------------|-------|-----------|---------------|
| `target_short` | Inadimplência Curto Prazo | atraso > 3 dias | Indicador de alerta precoce |
| `target_medium` | Inadimplência Médio Prazo | atraso > 7 dias | Inadimplência moderada |
| `target_long` | Inadimplência Longo Prazo | atraso > 14 dias | Inadimplência severa |

---

## Configuração de Limiares

Os limiares de atraso são parâmetros configuráveis definidos em `Configs/config.py`:

```python
@dataclass
class DataParams:
    delay_threshold_1: int = 3    # Limiar curto prazo (dias)
    delay_threshold_2: int = 7    # Limiar médio prazo (dias)
    delay_threshold_3: int = 14   # Limiar longo prazo (dias)
```

Estes limiares podem ser ajustados para alinhar com definições específicas de negócio sobre severidade de inadimplência.

---

## Estrutura Hierárquica de Labels

As variáveis alvo exibem uma **dependência hierárquica estrita** decorrente da natureza aninhada das definições de limiar:

$$\text{target\_long} = 1 \implies \text{target\_medium} = 1 \implies \text{target\_short} = 1$$

Esta relação hierárquica pode ser formalizada como:

```
Se atraso > 14 dias (target_long = 1):
   → Necessariamente atraso > 7 dias  (target_medium = 1)
   → Necessariamente atraso > 3 dias  (target_short = 1)

Se atraso > 7 dias (target_medium = 1):
   → Necessariamente atraso > 3 dias  (target_short = 1)
   → Possivelmente atraso ≤ 14 dias   (target_long pode ser 0 ou 1)
```

### Exemplos Ilustrativos

| Atraso (dias) | `target_short` | `target_medium` | `target_long` | Classificação |
|---------------|----------------|-----------------|---------------|---------------|
| 0 | 0 | 0 | 0 | Pagamento em dia |
| 2 | 0 | 0 | 0 | Atraso menor (dentro da tolerância) |
| 5 | 1 | 0 | 0 | Apenas inadimplência curto prazo |
| 10 | 1 | 1 | 0 | Inadimplência médio prazo |
| 20 | 1 | 1 | 1 | Inadimplência longo prazo/severa |

---

## Análise de Desbalanceamento de Classes

As variáveis alvo exibem **severo desbalanceamento de classes**, com proporções de classe positiva decrescendo monotonicamente através da hierarquia:

```
Distribuição de Targets (Exemplo Representativo):
┌─────────────────┬────────────┬────────────┬────────────┬─────────────────────┐
│ Alvo            │ Positivos  │ Prevalência│ Negativos  │ Peso de Classe      │
├─────────────────┼────────────┼────────────┼────────────┼─────────────────────┤
│ target_short    │ 15.000     │ 10,00%     │ 135.000    │ 9,00                │
│ target_medium   │ 8.000      │ 5,33%      │ 142.000    │ 17,75               │
│ target_long     │ 4.000      │ 2,67%      │ 146.000    │ 36,50               │
└─────────────────┴────────────┴────────────┴────────────┴─────────────────────┘
```

### Implicações do Desbalanceamento de Classes

1. **Funções de loss padrão falham**: Cross-entropy sem ajuste leva a modelos que preveem a classe majoritária
2. **Sensibilidade ao threshold**: Thresholds ótimos de classificação diferem significativamente de 0.5
3. **Seleção de métricas**: Acurácia é enganosa; métricas baseadas em precision-recall são essenciais
4. **Classes mais raras são mais difíceis**: `target_long` requer tratamento especializado

---

## Estratégias de Mitigação do Desbalanceamento de Classes

### 1. Função de Loss Assimétrica (ASL)

O modelo emprega **Asymmetric Loss** (Ben-Baruch et al., 2020), que pondera diferencialmente amostras positivas e negativas:

$$\mathcal{L}_{\text{ASL}} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i (1-p_i)^{\gamma_+} \log(p_i) + (1-y_i) p_{m,i}^{\gamma_-} \log(1-p_{m,i}) \right]$$

onde $p_{m,i} = \max(p_i - m, 0)$ aplica clipping de margem de probabilidade.

**Configuração de Hiperparâmetros:**
```python
AsymmetricLoss(
    gamma_negative=4,   # Forte redução de peso de negativos fáceis
    gamma_positive=1,   # Foco moderado em positivos difíceis
    clip=0.05           # Margem de probabilidade para clipping de gradiente
)
```

**Mecanismo:**
- Alto $\gamma_-$ suprime gradientes de negativos corretamente classificados
- Clipping de margem previne gradientes que desaparecem para previsões muito confiantes
- Efeito líquido: modelo foca capacidade de aprendizado em amostras positivas

### 2. Amostragem Estratificada

Durante a subamostragem de dados, um esquema de prioridade hierárquica garante representação de classes raras:

```python
# Prioridade de seleção (do mais raro ao mais comum):
1. Todos usuários com target_long = 1    (inadimplências severas)
2. Todos usuários com target_medium = 1  (sem target_long)
3. Todos usuários com target_short = 1   (sem medium/long)
4. Amostra aleatória dos usuários restantes  (sem inadimplências)
```

Esta estratégia garante que **todos os usuários com padrões raros de inadimplência sejam incluídos** antes de amostrar da classe majoritária.

### 3. Ponderação Dinâmica de Positivos

Pesos de classe são automaticamente computados baseados em frequências empíricas de classe:

$$w_{\text{positivo}} = \frac{N_{\text{negativo}}}{N_{\text{positivo}}}$$

Estes pesos informam o escalonamento da função de loss e estratégias de amostragem.

---

## Interpretação da Saída do Modelo

### Saídas de Probabilidade

O modelo produz **estimativas de probabilidade calibradas** no intervalo $[0, 1]$ para cada alvo:

```python
logits = model(categorical, continuous, lengths)
probabilities = torch.sigmoid(logits)
# probabilities.shape = [batch_size, 3]
# Colunas: [P(target_short), P(target_medium), P(target_long)]
```

### Exemplo de Interpretação

```
Previsões para Fatura X:
  P(target_short)  = 0.72  → 72% de probabilidade de atraso > 3 dias
  P(target_medium) = 0.45  → 45% de probabilidade de atraso > 7 dias
  P(target_long)   = 0.18  → 18% de probabilidade de atraso > 14 dias
```

A consistência hierárquica destas probabilidades não é explicitamente imposta, mas modelos bem treinados tipicamente produzem saídas satisfazendo:

$$P(\text{target\_long}) \leq P(\text{target\_medium}) \leq P(\text{target\_short})$$

---

## Thresholds Ótimos de Classificação

Cada alvo possui um **threshold individualmente otimizado** $\tau^*$ que maximiza o F1-Score no conjunto de validação:

### Algoritmo de Cálculo do Threshold

```python
def compute_optimal_threshold(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred)
    
    # Computa F1 em cada threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    
    # Seleciona threshold que maximiza F1
    optimal_idx = np.argmax(f1_scores[:-1])
    return thresholds[optimal_idx]
```

### Valores Típicos de Threshold

| Alvo | Threshold Ótimo | Justificativa |
|------|-----------------|---------------|
| `target_short` | ~0.45 | Maior prevalência permite threshold mais alto |
| `target_medium` | ~0.38 | Prevalência intermediária |
| `target_long` | ~0.25 | Baixa prevalência requer threshold menor para recall |

**Insight Chave:** Alvos mais raros requerem thresholds de classificação mais baixos para alcançar recall adequado, refletindo o trade-off precision-recall inerente à classificação desbalanceada.

---

## Temperature Scaling (Calibração de Probabilidade)

Após treinamento, as probabilidades são **calibradas** via temperature scaling para melhorar a confiabilidade:

$$\text{logits}_{\text{calibrados}} = \frac{\text{logits}}{\tau}$$

```python
class TemperatureScaling(nn.Module):
    def __init__(self, num_outputs: int = 3):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(num_outputs) * 1.5)
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature
```

**Efeito:** 
- $\tau > 1$: Suaviza probabilidades (reduz excesso de confiança)
- $\tau < 1$: Aguça probabilidades (aumenta confiança)
- $\tau = 1$: Sem alteração (não calibrado)

A calibração garante que probabilidades previstas reflitam com precisão as frequências empíricas—uma previsão de 80% deve corresponder a aproximadamente 80% dos casos sendo positivos.

---

## Boas Práticas

1. **Evite threshold padrão (0.5)**: Sempre use thresholds otimizados específicos por alvo derivados dos dados de validação

2. **Priorize AUC-PR sobre AUC-ROC**: Para classificação desbalanceada, curvas Precision-Recall fornecem avaliação mais informativa que curvas ROC

3. **Utilize scores de probabilidade**: Para decisões operacionais, use probabilidades brutas ao invés de classificações binárias para permitir estratificação de risco nuançada

4. **Monitore desempenho por alvo**: Cada alvo exibe características distintas; métricas agregadas podem mascarar deficiências de alvos individuais

5. **Valide consistência hierárquica**: Garanta que probabilidades previstas respeitem a ordenação natural $P(\text{long}) \leq P(\text{medium}) \leq P(\text{short})$

---

## Referências

- Ben-Baruch, E., et al. (2020). "Asymmetric Loss For Multi-Label Classification." *arXiv preprint arXiv:2009.14119*.
- Guo, C., et al. (2017). "On Calibration of Modern Neural Networks." *ICML*.
- Saito, T., & Rehmsmeier, M. (2015). "The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets." *PLOS ONE*.
