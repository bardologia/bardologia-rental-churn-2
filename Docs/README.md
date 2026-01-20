# Documentação do Projeto

## Visão Geral

Bem-vindo à documentação completa do **Sistema de Previsão de Inadimplência de Faturas**. Esta documentação fornece especificações técnicas detalhadas, explicações metodológicas e orientações práticas para compreensão, treinamento e avaliação do modelo de deep learning projetado para previsão probabilística de atrasos de pagamento em múltiplos horizontes temporais.

O sistema emprega arquiteturas de última geração baseadas em Transformers para modelar o comportamento sequencial de pagamentos, possibilitando previsões probabilísticas precisas de inadimplência em diferentes níveis de severidade.

---

## Índice de Documentos

| Capítulo | Documento | Descrição |
|----------|-----------|-----------|
| 01 | [Engenharia de Features](01_features.md) | Especificação completa das features de entrada, pipelines de pré-processamento e transformações |
| 02 | [Modelagem de Sequências](02_sequences.md) | Organização temporal dos dados, metodologia de janela expansiva e algoritmos de construção de sequências |
| 03 | [Variáveis Alvo](03_targets.md) | Targets de classificação multi-label, relações hierárquicas e tratamento de desbalanceamento |
| 04 | [Amostragem de Dados](04_sampling.md) | Estratégias de amostragem estratificada, particionamento train/val/test e técnicas de data augmentation |
| 05 | [Métricas de Avaliação](05_metrics.md) | Métricas de desempenho para classificação desbalanceada com fundamentos teóricos e diretrizes de interpretação |
| 06 | [Análise de Resultados](06_results_analysis.md) | Interpretação do modelo, receitas de visualização e metodologias de comparação experimental |
| 07 | [Arquitetura Neural](07_architecture.md) | Especificações arquiteturais detalhadas, descrição de componentes e fundamentação de design |

---

## Guia de Início Rápido

### Passo 1: Compreendendo os Dados

Comece pelo documento [01_features.md](01_features.md) para compreender o espaço de features que serve como entrada para o modelo. Este documento detalha embeddings categóricos, normalização de features contínuas e estratégias de codificação temporal.

### Passo 2: Compreendendo o Problema

Revise [03_targets.md](03_targets.md) para entender a formulação do problema de classificação multi-label, incluindo a natureza hierárquica dos limiares de atraso e considerações sobre desbalanceamento de classes.

### Passo 3: Execução do Treinamento

Execute o pipeline de treinamento:

```bash
python train.py
```

O script de treinamento gerencia automaticamente o carregamento de dados, inicialização do modelo, otimização e gerenciamento de checkpoints.

### Passo 4: Interpretação dos Resultados

Siga o guia completo em [06_results_analysis.md](06_results_analysis.md) para interpretar métricas de avaliação, gerar visualizações e conduzir análise de erros.

---

## Arquitetura do Projeto

```
project - 2/
├── Configs/
│   └── config.py              # Gerenciamento centralizado de configurações
├── Model/
│   ├── core.py                # Orquestração do modelo e inferência
│   ├── data.py                # DataModule, Datasets e pipelines de dados
│   ├── network.py             # Definições da arquitetura neural
│   └── trainer.py             # Loop de treinamento e procedimentos de avaliação
├── Utils/
│   └── logger.py              # Logging e integração com TensorBoard
├── Test/
│   └── test_*.py              # Suite completa de testes unitários
├── Docs/                      # 📍 Localização atual
│   └── *.md                   # Documentação técnica
├── runs/                      # Artefatos de experimentos e checkpoints
└── train.py                   # Ponto de entrada principal para treinamento
```

---

## Perguntas Frequentes

### Qual métrica de avaliação deve ser priorizada?

**Área Sob a Curva Precision-Recall (AUC-PR)** é a métrica primária recomendada para este domínio de problema devido ao severo desbalanceamento de classes. Diferentemente da AUC-ROC, a AUC-PR não é inflacionada pela abundância de verdadeiros negativos e mede diretamente o trade-off precision-recall relevante para previsão de inadimplência. Consulte [05_metrics.md](05_metrics.md) para análise detalhada das métricas.

### O que constitui um threshold de classificação ótimo?

O threshold ótimo é o ponto de corte de probabilidade que maximiza o F1-Score no conjunto de validação. Cada variável alvo possui seu próprio threshold otimizado, tipicamente inversamente relacionado à prevalência da classe (classes mais raras requerem thresholds mais baixos para recall adequado). Consulte [03_targets.md](03_targets.md) para a metodologia de cálculo do threshold.

### Por que empregar modelagem sequencial?

O comportamento de pagamento exibe fortes dependências temporais—os padrões históricos de pagamento de um usuário são altamente preditivos do comportamento futuro. A abordagem sequencial permite que o modelo capture:
- Dinâmicas de tendência (comportamento melhorando/piorando)
- Padrões sazonais
- Dependências de longo alcance via mecanismos de atenção

Consulte [02_sequences.md](02_sequences.md) para a fundamentação teórica.

### Como a amostragem estratificada preserva classes minoritárias?

A estratégia de amostragem prioriza usuários que exibem comportamentos raros de inadimplência, garantindo representação adequada das classes minoritárias no treinamento. Isso é alcançado através de um processo de seleção hierárquica que garante a inclusão de todos os usuários com inadimplências severas antes de amostrar da classe majoritária. Consulte [04_sampling.md](04_sampling.md) para detalhes de implementação.

---

## Referências

Para dúvidas sobre a documentação ou implementação, consulte os arquivos de código fonte ou submeta issues através do sistema de rastreamento de issues do repositório.
