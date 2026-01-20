# Introdução ao Projeto

## Sistema de Previsão de Inadimplência em Faturas

---

## Objetivo

Desenvolver um modelo de **deep learning baseado em Transformer** capaz de prever a probabilidade de atraso no pagamento de faturas em múltiplos horizontes temporais. O sistema analisa o histórico de pagamentos de clientes para identificar padrões comportamentais que precedem inadimplências.

### Objetivos Específicos

| Objetivo | Descrição |
|----------|-----------|
| **Previsão Multi-Horizonte** | Estimar probabilidades de atraso em 3, 7 e 14 dias |
| **Modelagem Temporal** | Capturar dependências de longo prazo no comportamento de pagamento |
| **Calibração de Probabilidades** | Produzir estimativas confiáveis para tomada de decisão |
| **Interpretabilidade** | Identificar fatores que contribuem para o risco de inadimplência |

---

## Dados

O modelo é treinado com dados históricos de faturas contendo informações transacionais e comportamentais.

### Estrutura dos Dados

```
Dados de Entrada
├── Features Categóricas
│   ├── Forma de pagamento
│   ├── Região do cliente
│   ├── Dia da semana do vencimento
│   └── Tipo de fatura
│
├── Features Contínuas
│   ├── Valor da fatura
│   ├── Histórico médio de atraso
│   ├── Razão valor/média
│   └── Posição na sequência
│
└── Variáveis Alvo (Multi-Label)
    ├── target_short  → Atraso > 3 dias
    ├── target_medium → Atraso > 7 dias
    └── target_long   → Atraso > 14 dias
```

### Características do Dataset

| Aspecto | Descrição |
|---------|-----------|
| **Formato** | Sequências temporais de faturas por cliente |
| **Comprimento** | Até 50 faturas por sequência |
| **Desbalanceamento** | Classes positivas representam 3-10% dos casos |
| **Particionamento** | 70% treino / 15% validação / 15% teste |

---

## Arquitetura

O modelo emprega uma arquitetura **Transformer hierárquica** com dois estágios:

1. **Encoder de Fatura**: Processa features dentro de cada timestep
2. **Encoder de Sequência**: Modela dependências temporais entre faturas

```
Entrada → Tokenização → Encoder Fatura → Encoder Sequência → Atenção → Previsão
```

---

## Métricas de Avaliação

Devido ao desbalanceamento de classes, priorizamos métricas robustas:

- **AUC-ROC**: Capacidade de discriminação geral
- **AUC-PR**: Desempenho na classe minoritária (inadimplentes)
- **F1-Score**: Equilíbrio precision-recall com threshold otimizado
- **Brier Score**: Qualidade da calibração de probabilidades

---

## Documentação Relacionada

| Documento | Conteúdo |
|-----------|----------|
| [01_features.md](01_features.md) | Especificação detalhada das features |
| [02_sequences.md](02_sequences.md) | Organização temporal dos dados |
| [03_targets.md](03_targets.md) | Definição das variáveis alvo |
| [04_sampling.md](04_sampling.md) | Estratégia de amostragem e particionamento |
| [05_metrics.md](05_metrics.md) | Métricas de avaliação |
| [06_results_analysis.md](06_results_analysis.md) | Guia de análise de resultados |
| [07_architecture.md](07_architecture.md) | Especificação da arquitetura neural |
