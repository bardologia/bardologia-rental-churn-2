# Especificação de Engenharia de Features

Este documento fornece uma especificação completa das features de entrada utilizadas pelo modelo de previsão de inadimplência de faturas, incluindo pipelines de pré-processamento, estratégias de embedding e tratamento de casos especiais.

---

## Visão Geral

O modelo processa features heterogêneas compreendendo variáveis categóricas e contínuas. Estas features passam por pipelines de transformação distintos para produzir representações vetoriais densas adequadas para processamento baseado em Transformers.

---

## Features Categóricas

Features categóricas são transformadas em **vetores de embedding** aprendíveis—representações densas de baixa dimensão que capturam relações semânticas entre valores de categorias.

### Demografia do Usuário

| Feature | Descrição | Valores Exemplo |
|---------|-----------|-----------------|
| `sexo` | Gênero do usuário | Masculino, Feminino, Desconhecido |
| `faixa_idade_resumida` | Classificação por faixa etária | 18-25, 26-35, 36-45, 46-55, 56+ |
| `lugar` | Localização geográfica (nível cidade) | São Paulo, Rio de Janeiro, Curitiba |
| `regiao` | Região geográfica | Sul, Sudeste, Norte, Nordeste, Centro-Oeste |

### Características de Produto e Serviço

| Feature | Descrição | Valores Exemplo |
|---------|-----------|-----------------|
| `recorrencia_pagamento` | Tipo de recorrência de pagamento | Mensal, Semanal, Quinzenal |
| `veiculo_modelo` | Modelo do veículo (para contratos de aluguel) | Sedan, SUV, Compacto |
| `pacoteNome` | Nome do pacote de serviço contratado | Básico, Premium, Enterprise |
| `formaPagamento` | Método de pagamento | Cartão de Crédito, Boleto, PIX |
| `produto_categoria` | Classificação da categoria do produto | Standard, Especial, Promocional |

### Características Temporais (Codificação Categórica)

| Feature | Descrição | Intervalo de Valores |
|---------|-----------|---------------------|
| `venc_dayofweek` | Dia da semana do vencimento da fatura | 0 (Segunda) a 6 (Domingo) |
| `venc_quarter` | Trimestre do ano | 1, 2, 3, 4 |
| `venc_is_weekend` | Indicador de fim de semana para vencimento | 0 (Dia útil), 1 (Fim de semana) |
| `venc_is_month_start` | Indicador de início do mês (dias 1-5) | 0, 1 |
| `venc_is_month_end` | Indicador de fim do mês (dias 26-31) | 0, 1 |

### Indicadores de Posição na Sequência

| Feature | Descrição | Intervalo de Valores |
|---------|-----------|---------------------|
| `is_first_invoice` | Indicador de primeira fatura do usuário | 0, 1 |
| `is_improving` | Tendência de melhoria no comportamento de pagamento | 0, 1 |
| `is_first_contract` | Indicador de primeiro contrato do usuário | 0, 1 |

---

## Features Contínuas

Features contínuas passam por normalização via **StandardScaler** (média zero, variância unitária) seguida de transformação através de **Embeddings Periódicos** para capturar padrões cíclicos inerentes a dados temporais.

### Valores Monetários e Contratuais

| Feature | Descrição | Intervalo Típico |
|---------|-----------|------------------|
| `quantidadeDiarias` | Número de diárias no contrato | 1–365 |
| `valor_caucao_brl` | Valor do depósito em Reais | 0–10.000+ |

### Features Temporais Cíclicas

Estas features empregam codificação senoidal para preservar a natureza cíclica do tempo:

| Par de Features | Descrição | Período |
|-----------------|-----------|---------|
| `venc_dayofweek_sin` / `venc_dayofweek_cos` | Dia da semana (codificação cíclica) | 7 dias |
| `venc_day_sin` / `venc_day_cos` | Dia do mês (codificação cíclica) | ~30 dias |
| `venc_month_sin` / `venc_month_cos` | Mês do ano (codificação cíclica) | 12 meses |

A codificação cíclica é computada como:

$$x_{\sin} = \sin\left(\frac{2\pi \cdot x}{\text{período}}\right), \quad x_{\cos} = \cos\left(\frac{2\pi \cdot x}{\text{período}}\right)$$

### Comportamento Histórico do Usuário

Features derivadas do histórico de pagamento do usuário anterior à fatura atual:

| Feature | Descrição | Interpretação |
|---------|-----------|---------------|
| `hist_mean_delay` | Atraso médio histórico (dias) | Latência média de pagamento |
| `hist_std_delay` | Desvio padrão histórico de atrasos | Consistência do comportamento de pagamento |
| `hist_max_delay` | Atraso máximo histórico (dias) | Pior caso de comportamento de pagamento |
| `hist_default_rate` | Taxa de inadimplência histórica (0–1) | Proporção de pagamentos em atraso |
| `hist_payment_count` | Número de pagamentos históricos | Tempo/experiência do usuário |
| `last_delay` | Atraso de pagamento mais recente (dias) | Estado comportamental atual |
| `delay_trend` | Coeficiente de tendência de atrasos | Positivo = piorando, Negativo = melhorando |
| `days_since_last_default` | Dias desde a última inadimplência | Recência de comportamento problemático |

### Features de Posição na Sequência

| Feature | Descrição | Intervalo de Valores |
|---------|-----------|---------------------|
| `seq_position` | Posição absoluta na sequência de faturas | 1, 2, 3, ... |
| `seq_position_norm` | Posição normalizada na sequência | [0, 1] |
| `days_since_last_invoice` | Dias desde a fatura anterior | Contínuo positivo |
| `rolling_mean_delay_3` | Média móvel de atraso (3 faturas) | Contínuo |
| `rolling_max_delay_3` | Máximo móvel de atraso (3 faturas) | Contínuo positivo |

### Features de Nível de Contrato

| Feature | Descrição | Interpretação |
|---------|-----------|---------------|
| `parcela_position` | Posição da parcela dentro do contrato | 1, 2, 3, ... |
| `parcela_position_norm` | Posição normalizada da parcela | [0, 1] |
| `n_contratos_anteriores` | Número de contratos anteriores | Histórico de contratos do usuário |
| `contract_mean_delay` | Atraso médio no contrato atual | Comportamento específico do contrato |

### Razões de Valor

| Feature | Descrição | Intervalo Típico |
|---------|-----------|------------------|
| `value_ratio` | Razão do valor atual sobre a média histórica | Limitado a [0.1, 10.0] |

---

## Pipelines de Processamento de Features

### Pipeline de Features Categóricas

```
Valor Bruto → LabelEncoder → Lookup na Matriz de Embedding → Vetor Token (d=128)
```

**Etapas de Processamento:**
1. Valores string/categoria são codificados em índices inteiros via `LabelEncoder`
2. Índices inteiros servem como chaves de lookup em matrizes de embedding aprendíveis
3. Cada coluna categórica mantém uma matriz de embedding dedicada $\mathbf{E} \in \mathbb{R}^{V \times d}$
4. Índice de padding = 0 é reservado para padding de sequência

### Pipeline de Features Contínuas

```
Valor Bruto → StandardScaler → Embedding Periódico → Projeção Gated → Vetor Token (d=128)
```

**Etapas de Processamento:**
1. Valores são normalizados para média zero e variância unitária
2. Embedding periódico expande o escalar para funções base multi-frequência
3. Projeção gated aplica transformação aprendida com gating elemento a elemento:

$$\mathbf{e} = \sigma(\mathbf{W}_g \cdot \text{PE}(x)) \odot (\mathbf{W}_v \cdot \text{PE}(x))$$

---

## Tratamento de Valores Especiais

| Cenário | Tratamento | Justificativa |
|---------|------------|---------------|
| Valores categóricos ausentes | Codificados como string `"nan"` | Preserva ausência como categoria aprendível |
| Valores contínuos ausentes | Mantidos após ajuste do scaler | Permite imputação via embedding |
| `days_since_last_default` = NaN | Preenchido com 999 | Indica que o usuário nunca ficou inadimplente |
| Extremos de `value_ratio` | Limitados a [0.1, 10.0] | Previne distorção por outliers |

---

## Referência de Configuração

Colunas de features são especificadas em `Configs/config.py`:

```python
@dataclass
class DataParams:
    cat_cols: List[str]      # Nomes das colunas de features categóricas
    cont_cols: List[str]     # Nomes das colunas de features contínuas
    target_cols: List[str]   # Nomes das colunas de variáveis alvo
```

---

## Fundamentos Teóricos

### Representações por Embedding

Embeddings categóricos transformam símbolos discretos em espaços vetoriais contínuos onde a similaridade semântica é preservada através de proximidade aprendida. Esta abordagem é superior à codificação one-hot para features de alta cardinalidade, pois:
- Reduz dimensionalidade de $O(V)$ para $O(d)$ onde $d \ll V$
- Permite generalização entre categorias similares
- Suporta aprendizado baseado em gradiente das relações entre categorias

### Embeddings Periódicos para Features Contínuas

Projeções lineares tradicionais falham em capturar padrões cíclicos (ex.: segunda-feira é adjacente a domingo). Embeddings periódicos abordam isso projetando escalares em uma base de funções senoidais em múltiplas frequências, permitindo que o modelo aprenda relações periódicas naturalmente.

---

## Referências

- Guo, C., & Berkhahn, F. (2016). "Entity Embeddings of Categorical Variables." *arXiv preprint*.
- Kazemi, S. M., et al. (2019). "Time2Vec: Learning a Unified Representation of Time." *arXiv preprint*.
