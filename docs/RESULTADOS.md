# Resultados do Modelo

## Visao Geral

Este documento apresenta os resultados da avaliacao do modelo de predicao de inadimplencia no conjunto de teste.

---

## Metricas de Avaliacao

### Metrica Principal

| Metrica | Valor |
|---------|-------|
| ROC-AUC | Calculado no conjunto de teste |

O ROC-AUC mede a capacidade do modelo de distinguir entre classes, independente do threshold de classificacao.

---

## Distribuicao de Classes

O problema apresenta desbalanceamento de classes:

| Classe | Descricao |
|--------|-----------|
| Negativa (0) | Pagamento realizado |
| Positiva (1) | Inadimplencia |

A taxa de inadimplencia no conjunto de teste e reportada apos execucao.

---

## Matriz de Confusao

A matriz de confusao apresenta quatro categorias:

| Categoria | Sigla | Descricao |
|-----------|-------|-----------|
| Verdadeiro Negativo | TN | Pagamento previsto e realizado |
| Falso Positivo | FP | Inadimplencia prevista, pagamento realizado |
| Falso Negativo | FN | Pagamento previsto, inadimplencia ocorreu |
| Verdadeiro Positivo | TP | Inadimplencia prevista e ocorrida |

### Interpretacao

- **FN (Falso Negativo):** Representa risco financeiro. O modelo nao detectou a inadimplencia.
- **FP (Falso Positivo):** Representa custo operacional. Acao preventiva desnecessaria.

---

## Analise de Threshold

O threshold de classificacao afeta o balanco entre deteccao e alarmes falsos:

| Threshold | Comportamento |
|-----------|---------------|
| 30% | Maior deteccao de inadimplencia, mais alarmes falsos |
| 50% | Balanceado |
| 70% | Menor taxa de alarmes falsos, mais inadimplencias nao detectadas |

### Metricas por Threshold

Para cada threshold sao calculados:
- Defaults detectados (Recall)
- Alarmes falsos (FP)
- Defaults nao detectados (FN)
- Precisao

---

## Casos de Analise

### Verdadeiros Positivos (TP)
- Casos de inadimplencia corretamente identificados
- Alta probabilidade de default atribuida pelo modelo
- Representam sucesso na deteccao

### Falsos Negativos (FN)
- Casos de inadimplencia nao detectados
- Baixa probabilidade de default atribuida pelo modelo
- Representam risco financeiro

### Falsos Positivos (FP)
- Casos de pagamento com previsao de inadimplencia
- Representam custo operacional de acoes preventivas desnecessarias

---

## Conjunto de Dados

| Conjunto | Proporcao |
|----------|-----------|
| Treino | 70% |
| Validacao | 15% |
| Teste | 15% |

A divisao e aleatoria com seed fixo para reproducibilidade.

---

## Consideracoes

### Desbalanceamento de Classes

O dataset apresenta desbalanceamento significativo. Tecnicas aplicadas:
1. Asymmetric Loss com peso elevado para classe positiva
2. Amostragem balanceada durante treinamento
3. Analise de multiplos thresholds na avaliacao

### Limitacoes

1. **Analise por usuario:** Com divisao aleatoria, os identificadores de usuario nao sao preservados no conjunto de teste
2. **Generalização temporal:** O modelo foi validado com divisao aleatoria, nao temporal

### Aplicacao

Para uso em producao, recomenda-se:
1. Definir threshold com base no custo-beneficio entre FN e FP
2. Monitorar distribuicao de probabilidades ao longo do tempo
3. Retreinar periodicamente com dados atualizados

---

## Arquivos de Saida

| Arquivo | Descricao |
|---------|-----------|
| `runs/[timestamp]/checkpoints/best_model.pth` | Modelo com melhor AUC de validacao |
| `runs/[timestamp]/checkpoints/last_checkpoint.pth` | Ultimo checkpoint de treinamento |
| `runs/[timestamp]/tensorboard/` | Logs de treinamento para visualizacao |

---

## Reproducibilidade

| Parametro | Valor |
|-----------|-------|
| Random seed | 42 |
| Framework | PyTorch |
| Mixed precision | FP16 (CUDA) |

Para reproduzir os resultados:
1. Utilizar o mesmo seed de divisao de dados
2. Carregar o checkpoint salvo
3. Executar inferencia no conjunto de teste
