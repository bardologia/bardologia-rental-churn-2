# Tratamento de Dados

## Visao Geral

Este documento descreve o pipeline de processamento de dados para o modelo de predicao de inadimplencia. Os dados brutos passam por etapas de limpeza, transformacao e engenharia de features antes de serem utilizados no treinamento.

---

## Fonte de Dados

- **Arquivo de entrada:** `Data/raw_data.csv`
- **Arquivo de saida:** `Data/training_data.csv`
- **Separador:** ponto e virgula (`;`)

---

## Etapas do Pipeline

### 1. Filtragem Inicial

- Filtragem por categoria: apenas registros com categoria "aluguel"
- Remocao de colunas irrelevantes:
  - `emissaoNotaFiscalData`
  - `emissaoNotaFiscalDia`
  - `codigoOSOmie`
  - `numeroNFSeOmie`
  - `criacaoDataAcrescimoDesconto`
  - `adyenPspReferencePagamento`

### 2. Tratamento de Datas

Colunas convertidas para formato datetime:
- `vencimentoData`
- `pagamentoData`
- `pagamentoData_UTC`
- `criacaoData`
- `competenciaInicioData`
- `competenciaFimData`
- `atualizacao_dt`

Filtro temporal: apenas faturas com vencimento anterior a data atual.

### 3. Tratamento de Valores Ausentes

| Tipo de Coluna | Tratamento |
|----------------|------------|
| Numericas (valores e dias de atraso) | Preenchimento com 0 |
| Categoricas | Preenchimento com "Unknown" |

### 4. Ordenacao

Os dados sao ordenados por:
1. `usuarioId`
2. `vencimentoData`
3. `criacaoData`

---

## Variaveis Criadas

### Variaveis de Identificacao

| Variavel | Descricao |
|----------|-----------|
| `invoice_sequence` | Numero sequencial da fatura por usuario |
| `is_delayed` | Indicador binario de atraso (dias_atraso > 0) |
| `is_default` | Indicador de inadimplencia (pagamento ausente apos 90 dias) |
| `days_since_due` | Dias desde o vencimento |

### Variaveis de Sequencia (Streak)

| Variavel | Descricao |
|----------|-----------|
| `current_streak` | Numero consecutivo de faturas atrasadas |
| `current_streak_days` | Soma acumulada de dias de atraso na sequencia atual |

### Variaveis Historicas (Shifted)

Todas as variaveis historicas utilizam shift para evitar vazamento de dados (data leakage).

| Variavel | Descricao |
|----------|-----------|
| `hist_count_delays` | Quantidade acumulada de atrasos anteriores |
| `hist_sum_val_delayed` | Soma dos valores atrasados anteriores |
| `hist_total_penalty` | Total de multas acumuladas |
| `hist_total_paid` | Total pago acumulado |
| `hist_max_streak` | Maior sequencia de atrasos historica |
| `hist_max_delay_days` | Maior numero de dias de atraso historico |
| `hist_delay_rate` | Taxa de atraso historica |
| `hist_avg_val_delayed` | Media de valores atrasados |
| `hist_avg_penalty` | Media de multas |

### Variaveis de Recencia

| Variavel | Descricao |
|----------|-----------|
| `recent_delay_rate` | Taxa de atraso nas ultimas 3 faturas |
| `recent_avg_delay_days` | Media de dias de atraso nas ultimas 3 faturas |

### Variaveis de Tendencia

| Variavel | Descricao |
|----------|-----------|
| `trend_delay_rate` | Diferenca entre taxa recente e historica |

### Score de Risco

```
risk_score = hist_delay_rate * 0.3 + recent_delay_rate * 0.4 + (current_streak / 5) * 0.3
```

Valores limitados ao intervalo [0, 1].

### Variavel Alvo

| Variavel | Descricao |
|----------|-----------|
| `target_default_1` | Indicador se a proxima fatura sera inadimplente |

---

## Divisao dos Dados

| Conjunto | Proporcao |
|----------|-----------|
| Treino | 70% |
| Validacao | 15% |
| Teste | 15% |

- Seed de reproducibilidade: 42
- Divisao aleatoria estratificada

---

## Normalizacao

- Variaveis continuas normalizadas com `StandardScaler`
- Scaler ajustado apenas nos dados de treino
- Transformacao aplicada em validacao e teste

---

## Tratamento de Desbalanceamento

O dataset apresenta desbalanceamento de classes. Tecnicas aplicadas:

1. **Amostragem Balanceada (Weighted Random Sampler)**
   - Razao alvo de positivos por batch: 30%
   - Pesos calculados para atingir a razao desejada

2. **Codificacao de Variaveis Categoricas**
   - Label Encoding para todas as variaveis categoricas
   - Dimensao de embedding calculada: `min(50, (cardinalidade + 1) / 2)`

---

## Variaveis Categoricas

| Variavel | Descricao |
|----------|-----------|
| `recorrencia_pagamento` | Tipo de recorrencia |
| `sexo` | Genero do usuario |
| `faixa_idade_resumida` | Faixa etaria |
| `veiculo_modelo` | Modelo do veiculo |
| `pacoteNome` | Nome do pacote contratado |
| `formaPagamento` | Forma de pagamento |
| `lugar` | Local |
| `regiao` | Regiao geografica |
| `produto_categoria` | Categoria do produto |

---

## Variaveis Continuas

| Variavel | Descricao |
|----------|-----------|
| `invoice_sequence` | Sequencia da fatura |
| `current_streak` | Sequencia atual de atrasos |
| `current_streak_days` | Dias acumulados na sequencia |
| `hist_count_delays` | Contagem historica de atrasos |
| `hist_sum_val_delayed` | Soma historica de valores atrasados |
| `hist_total_penalty` | Total historico de multas |
| `hist_max_streak` | Maior sequencia historica |
| `hist_max_delay_days` | Maior atraso historico |
| `hist_delay_rate` | Taxa historica de atraso |
| `hist_avg_val_delayed` | Media historica de valores atrasados |
| `hist_avg_penalty` | Media historica de multas |
| `recent_delay_rate` | Taxa de atraso recente |
| `recent_avg_delay_days` | Media de dias de atraso recente |
| `trend_delay_rate` | Tendencia da taxa de atraso |
| `risk_score` | Score de risco composto |
| `quantidadeDiarias` | Quantidade de diarias |
| `valor_caucao_brl` | Valor da caucao |

---

## Tratamento de Valores Invalidos

- Divisoes por zero tratadas com funcao `_safe_divide`
- Valores infinitos substituidos por 0
- Valores NaN substituidos por 0 apos calculo das features
