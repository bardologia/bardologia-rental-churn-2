# Métricas de Avaliação para Classificação Multi-Label Desbalanceada

Este documento fornece um tratamento rigoroso das métricas de avaliação empregadas para avaliar o desempenho do modelo no contexto de classificação multi-label severamente desbalanceada. Cada métrica é apresentada com definições formais, diretrizes de interpretação e detalhes de implementação.

---

## Resumo

Avaliar classificadores em datasets desbalanceados requer seleção cuidadosa de métricas, já que medidas tradicionais baseadas em acurácia podem ser enganosas quando as distribuições de classes são distorcidas. Este documento apresenta um framework abrangente para avaliação de modelos, enfatizando métricas que fornecem sinal significativo para detecção de classes positivas raras.

---

## Métricas de Avaliação Primárias

### 1. Área Sob a Curva ROC (AUC-ROC)

**Definição Formal**: A área sob a curva Receiver Operating Characteristic, que plota a Taxa de Verdadeiros Positivos (Sensibilidade) contra a Taxa de Falsos Positivos (1 - Especificidade) através de todos os limiares de classificação.

$$\text{AUC-ROC} = \int_0^1 \text{TPR}(t) \, d\text{FPR}(t) = P(\hat{y}_+ > \hat{y}_-)$$

A segunda igualdade fornece uma interpretação probabilística: AUC-ROC é igual à probabilidade de que uma instância positiva escolhida aleatoriamente receba um score maior que uma instância negativa escolhida aleatoriamente.

**Escala de Interpretação**:

| AUC-ROC | Interpretação | Significado Prático |
|---------|---------------|---------------------|
| 0.50 | Classificador aleatório | Sem poder discriminativo |
| 0.60–0.70 | Discriminação pobre | Utilidade prática limitada |
| 0.70–0.80 | Discriminação aceitável | Útil com ajuste cuidadoso de threshold |
| 0.80–0.90 | Boa discriminação | Confiável para maioria das aplicações |
| 0.90–1.00 | Excelente discriminação | Previsões de alta confiança |

**Casos de Uso Apropriados**:
- Datasets balanceados ou moderadamente desbalanceados
- Quando falsos positivos e falsos negativos incorrem em custos similares
- Comparação rápida de modelos durante iterações de desenvolvimento

**Limitação Crítica**: AUC-ROC pode ser enganosamente otimista em datasets severamente desbalanceados. A abundância de verdadeiros negativos domina o denominador da FPR, permitindo alta AUC-ROC mesmo quando a detecção da classe positiva é pobre.

```python
from sklearn.metrics import roc_auc_score

auc_roc = roc_auc_score(y_true, y_pred_proba)
```

---

### 2. Área Sob a Curva Precision-Recall (AUC-PR)

**Definição Formal**: A área sob a curva Precision-Recall, computada como a média ponderada das precisões alcançadas em cada limiar de recall. Também conhecida como Average Precision (AP).

$$\text{AUC-PR} = \sum_{k=1}^{n} (R_k - R_{k-1}) \cdot P_k$$

onde $P_k$ e $R_k$ são a precisão e recall no $k$-ésimo limiar.

**Escala de Interpretação (Dependente do Contexto)**:

| AUC-PR | Interpretação | Notas |
|--------|---------------|-------|
| < $\pi$ | Abaixo da baseline | Pior que aleatório (onde $\pi$ = prevalência) |
| $\pi$ – 0.30 | Pobre | Melhoria mínima sobre baseline |
| 0.30–0.50 | Aceitável | Utilidade prática para alvos de alta prevalência |
| 0.50–0.70 | Bom | Desempenho forte |
| > 0.70 | Excelente | Desempenho estado da arte |

> **Importante**: A baseline aleatória para AUC-PR é igual à prevalência da classe positiva $\pi$. Para `target_long` com 3% de prevalência, um classificador aleatório atinge AUC-PR ≈ 0.03, tornando qualquer valor acima de 0.10 potencialmente significativo.

**Por que AUC-PR é a Métrica Primária Recomendada**:

1. **Foco na classe positiva**: Avalia desempenho exclusivamente na classe minoritária de interesse
2. **Robusta ao desbalanceamento**: Não é inflacionada pela abundância de verdadeiros negativos
3. **Alinhamento com negócio**: Mede diretamente o trade-off precision-recall relevante para previsão de inadimplência
4. **Sensível a melhorias**: Pequenas mudanças na detecção da classe positiva produzem mudanças visíveis na métrica

```python
from sklearn.metrics import average_precision_score

auc_pr = average_precision_score(y_true, y_pred_proba)
```

---

### 3. F1-Score

**Definição Formal**: A média harmônica de Precision e Recall, fornecendo um resumo balanceado do trade-off precision-recall em um limiar operacional específico.

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

A média harmônica penaliza desequilíbrios extremos entre precision e recall mais severamente que a média aritmética.

**Escala de Interpretação**:

| F1-Score | Interpretação | Orientação Operacional |
|----------|---------------|------------------------|
| < 0.30 | Pobre | Modelo requer melhoria significativa |
| 0.30–0.50 | Aceitável | Utilizável com ajustes de restrições de negócio |
| 0.50–0.70 | Bom | Adequado para deploy em produção |
| > 0.70 | Excelente | Previsões de alta qualidade |

**Dependência de Threshold**: Diferentemente das métricas AUC, F1-Score requer binarização das saídas de probabilidade usando um limiar de classificação. O F1 reportado emprega o **threshold ótimo** $\tau^*$ que maximiza F1 no conjunto de validação.

```python
from sklearn.metrics import f1_score

y_pred_binary = (y_pred_proba >= optimal_threshold).astype(int)
f1 = f1_score(y_true, y_pred_binary)
```

---

### 4. Threshold Ótimo de Classificação ($\tau^*$)

**Definição Formal**: O limiar de probabilidade que maximiza o F1-Score no conjunto de validação.

$$\tau^* = \arg\max_{\tau \in [0,1]} F_1(\tau)$$

**Algoritmo de Cálculo**:

```python
from sklearn.metrics import precision_recall_curve

def compute_optimal_threshold(y_true, y_pred_proba):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    # Computar F1 em cada threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    
    # Encontrar threshold que maximiza F1
    optimal_idx = np.argmax(f1_scores[:-1])
    optimal_threshold = thresholds[optimal_idx]
    
    return optimal_threshold
```

**Valores Típicos**:

| Alvo | Threshold | Justificativa |
|------|-----------|---------------|
| `target_short` | ~0.45 | Classe positiva mais comum |
| `target_medium` | ~0.38 | Prevalência intermediária |
| `target_long` | ~0.25 | Classe mais rara, threshold mais baixo para recall |

---

### 5. Brier Score (Métrica de Calibração)

**Definição**: O erro quadrático médio entre probabilidades previstas e resultados binários reais. Mede tanto discriminação quanto calibração.

$$\text{Brier} = \frac{1}{N} \sum_{i=1}^{N} (p_i - y_i)^2$$

**Interpretação**:

| Brier Score | Interpretação |
|-------------|---------------|
| > 0.25 | Muito pobre (pior que prever prevalência) |
| 0.10–0.25 | Pobre |
| 0.05–0.10 | Aceitável |
| < 0.05 | Bom |
| 0 | Perfeito |

**Decomposição**: O Brier Score pode ser decomposto em:
- **Confiabilidade**: Quão bem as probabilidades correspondem às frequências empíricas
- **Resolução**: Quanto as previsões variam da taxa base
- **Incerteza**: Variância irredutível dos dados

```python
from sklearn.metrics import brier_score_loss
brier = brier_score_loss(y_true, y_pred_proba)
```

---

### 6. Loss de Treinamento

**Definição**: O valor da Asymmetric Loss (ASL) computado durante o treinamento.

**Formulação da Asymmetric Loss**:

$$\mathcal{L}_{\text{ASL}} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i (1-p_i)^{\gamma_+} \log(p_i) + (1-y_i) p_{m,i}^{\gamma_-} \log(1-p_{m,i}) \right]$$

onde $p_{m,i} = \max(p_i - m, 0)$ aplica clipping de margem.

**Hiperparâmetros**:
- $\gamma_+ = 1$: Fator focal para amostras positivas
- $\gamma_- = 4$: Redução agressiva de peso de negativos fáceis
- $m = 0.05$: Margem de probabilidade para clipping

**Uso**: Compare tendências de loss através das epochs, não entre diferentes modelos ou configurações de hiperparâmetros.

---

## Relatório Abrangente de Métricas

Exemplo de saída de avaliação do modelo:

```
┌─────────────────┬──────────┬──────────┬──────────┬───────────┬───────────┐
│ Alvo            │ AUC-ROC  │ AUC-PR   │ F1-Score │ Threshold │ Brier     │
├─────────────────┼──────────┼──────────┼──────────┼───────────┼───────────┤
│ target_short    │ 0.8234   │ 0.4521   │ 0.5123   │ 0.450     │ 0.0823    │
│ target_medium   │ 0.8567   │ 0.3892   │ 0.4567   │ 0.380     │ 0.0534    │
│ target_long     │ 0.8901   │ 0.3245   │ 0.4012   │ 0.250     │ 0.0312    │
├─────────────────┼──────────┼──────────┼──────────┼───────────┼───────────┤
│ Média Macro     │ 0.8567   │ 0.3886   │ 0.4567   │ —         │ 0.0556    │
└─────────────────┴──────────┴──────────┴──────────┴───────────┴───────────┘

Resumo do Treinamento:
  Total de Epochs:  50
  Melhor Epoch:     42
  Loss Train Final: 0.0823
  Loss Val Final:   0.0912
```

---

## Métricas Auxiliares

### Precision

**Definição**: A proporção de positivos previstos que são verdadeiros positivos.

$$\text{Precision} = \frac{TP}{TP + FP}$$

**Interpretação**: Alta precision indica poucos alarmes falsos. Priorize quando custos de falso positivo são altos (ex.: ações de cobrança desnecessárias).

---

### Recall (Sensibilidade)

**Definição**: A proporção de positivos reais corretamente identificados.

$$\text{Recall} = \frac{TP}{TP + FN}$$

**Interpretação**: Alto recall indica poucos casos perdidos. Priorize quando custos de falso negativo são altos (ex.: falha em identificar inadimplentes).

---

## Trade-off Precision-Recall

A relação entre precision e recall é inversamente relacionada em um dado nível de desempenho do modelo:

```
Threshold ↑ (ex.: 0.8):
  → Precision ↑ (menos alarmes falsos)
  → Recall ↓ (mais casos perdidos)

Threshold ↓ (ex.: 0.2):
  → Precision ↓ (mais alarmes falsos)
  → Recall ↑ (menos casos perdidos)
```

### Seleção de Threshold Orientada pelo Negócio

| Cenário de Negócio | Abordagem Recomendada |
|-------------------|----------------------|
| Ações de cobrança caras | Otimizar para Precision |
| Alto custo de inadimplências perdidas | Otimizar para Recall |
| Custos operacionais balanceados | Otimizar para F1-Score |
| Aplicação de score de risco | Usar probabilidades brutas |

---

## Logging no TensorBoard

O pipeline de treinamento automaticamente registra métricas para monitoramento em tempo real:

```python
# Métricas de treinamento
logger.log_scalar("Loss/train", train_loss, epoch)
logger.log_scalar("LR", learning_rate, epoch)

# Métricas de validação (por alvo)
for target_name in ['target_short', 'target_medium', 'target_long']:
    logger.log_scalar(f"AUC-ROC/val_{target_name}", auc_roc, epoch)
    logger.log_scalar(f"AUC-PR/val_{target_name}", auc_pr, epoch)
    logger.log_scalar(f"F1/val_{target_name}", f1_score, epoch)
    logger.log_scalar(f"Threshold/val_{target_name}", threshold, epoch)
```

---

## Recomendações de Seleção de Métricas

1. **Métrica Primária**: Use **AUC-PR** para seleção de modelo e ajuste de hiperparâmetros devido ao desbalanceamento de classes

2. **Comparação de Thresholds**: Note que thresholds ótimos diferem entre alvos; alvos mais raros requerem thresholds mais baixos

3. **Monitoramento de Calibração**: Acompanhe o Brier Score após aplicar Temperature Scaling para garantir confiabilidade das probabilidades

4. **Análise Por Alvo**: Avalie cada alvo independentemente, pois exibem características e taxas de prevalência diferentes

5. **Alinhamento com Negócio**: A seleção final de threshold deve incorporar custos operacionais e restrições de negócio

---

## Referências

- Davis, J., & Goadrich, M. (2006). "The Relationship Between Precision-Recall and ROC Curves." *ICML*.
- Saito, T., & Rehmsmeier, M. (2015). "The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets." *PLOS ONE*.
- Brier, G. W. (1950). "Verification of Forecasts Expressed in Terms of Probability." *Monthly Weather Review*.
