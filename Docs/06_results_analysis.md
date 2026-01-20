# Guia Abrangente de Análise de Resultados e Interpretação do Modelo

Este documento fornece uma metodologia sistemática para analisar saídas do modelo, gerar visualizações de qualidade para publicação, interpretar resultados experimentais e conduzir análise rigorosa de erros.

---

## Resumo

A avaliação efetiva de modelos vai além do cálculo de métricas agregadas. Este guia apresenta um framework abrangente para entender o comportamento do modelo através de visualização, quantificação de incerteza, análise de erros e comparação entre experimentos.

---

## Estrutura de Artefatos do Experimento

Após a conclusão do treinamento, o diretório `runs/` contém artefatos estruturados organizados por timestamp:

```
runs/
└── 20260119_203110/                    # Identificador do experimento baseado em timestamp
    ├── checkpoints/
    │   ├── best_model.pth              # Pesos com melhor desempenho de validação
    │   ├── best_model_ema.pth          # Pesos do melhor modelo EMA
    │   └── last_checkpoint.pth         # Checkpoint da epoch final
    ├── tensorboard/
    │   └── events.out.tfevents.*       # Logs de eventos do TensorBoard
    ├── profile_results.prof            # Dados de profiling de desempenho
    └── test_probs_targets.npz          # Previsões e ground truth do conjunto de teste
```

### Descrições dos Artefatos

| Artefato | Propósito | Uso |
|----------|-----------|-----|
| `best_model.pth` | Pesos ótimos para inferência | Deploy em produção |
| `best_model_ema.pth` | Pesos de média móvel exponencial | Frequentemente mais estáveis |
| `last_checkpoint.pth` | Capacidade de retomar treinamento | Inclui estado do otimizador |
| `events.out.tfevents.*` | Visualização de dinâmicas de treinamento | Análise no TensorBoard |
| `test_probs_targets.npz` | Avaliação e análise offline | Cálculo de métricas pós-hoc |

---

## Visualização no TensorBoard

### Inicialização

Inicie o TensorBoard para visualizar dinâmicas de treinamento:

```bash
tensorboard --logdir runs/20260119_203110/tensorboard --port 6006
```

Acesse o dashboard interativo em `http://localhost:6006`.

### Painéis de Visualização Principais

#### 1. Análise de Curva de Loss

| Painel | Comportamento Esperado | Valor Diagnóstico |
|--------|------------------------|-------------------|
| `Loss/train` | Decréscimo monotônico com ruído estocástico | Convergência do treinamento |
| `Loss/val` | Acompanha loss de treino; gap indica generalização | Detecção de overfitting |

**Padrões Diagnósticos**:

| Padrão | Interpretação | Ação Recomendada |
|--------|---------------|------------------|
| Val acompanha train com gap pequeno | Treinamento saudável | Continuar abordagem atual |
| Val aumenta enquanto train diminui | Overfitting | Aumentar regularização, early stopping |
| Ambos estabilizam em valores altos | Underfitting | Aumentar capacidade, reduzir regularização |
| Alta variância em ambas curvas | Taxa de aprendizado muito alta | Reduzir taxa de aprendizado |

#### 2. Cronograma de Taxa de Aprendizado

| Painel | Comportamento Esperado |
|--------|------------------------|
| `LR` | Cosine annealing do LR inicial até próximo de zero |

#### 3. Painéis de Métricas Por Alvo

```
Métricas de Validação:
├── AUC-ROC/val_{target_short, target_medium, target_long}
├── AUC-PR/val_{target_short, target_medium, target_long}
├── F1/val_{target_short, target_medium, target_long}
└── Threshold/val_{target_short, target_medium, target_long}
```

---

## Framework de Análise do Conjunto de Teste

### Carregamento de Previsões

```python
import numpy as np
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    precision_recall_curve,
    f1_score,
    roc_curve,
    confusion_matrix,
    brier_score_loss
)
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar padrões de visualização
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.family'] = 'serif'

# Carregar previsões
data = np.load('runs/20260119_203110/test_probs_targets.npz')
probabilities = data['probabilities']  # Shape: (N_samples, 3)
targets = data['targets']              # Shape: (N_samples, 3)

target_names = ['target_short', 'target_medium', 'target_long']
```

### Cálculo Abrangente de Métricas

```python
def compute_comprehensive_metrics(probabilities, targets, target_names):
    """Calcula todas as métricas de avaliação para cada alvo."""
    
    print("=" * 70)
    print("RESUMO DE DESEMPENHO NO CONJUNTO DE TESTE")
    print("=" * 70)
    
    results = []
    
    for i, name in enumerate(target_names):
        probs = probabilities[:, i]
        targs = targets[:, i]
        
        # Métricas independentes de threshold
        auc_roc = roc_auc_score(targs, probs)
        auc_pr = average_precision_score(targs, probs)
        brier = brier_score_loss(targs, probs)
        
        # Calcular threshold ótimo
        precisions, recalls, thresholds = precision_recall_curve(targs, probs)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        optimal_idx = np.argmax(f1_scores[:-1])
        optimal_threshold = thresholds[optimal_idx]
        optimal_f1 = f1_scores[optimal_idx]
        optimal_precision = precisions[optimal_idx]
        optimal_recall = recalls[optimal_idx]
        
        # Distribuição de classes
        n_positive = targs.sum()
        prevalence = targs.mean()
        
        print(f"\n{name.upper()}")
        print("-" * 40)
        print(f"  AUC-ROC:      {auc_roc:.4f}")
        print(f"  AUC-PR:       {auc_pr:.4f}")
        print(f"  Brier Score:  {brier:.4f}")
        print(f"  F1-Score:     {optimal_f1:.4f} (τ = {optimal_threshold:.3f})")
        print(f"  Precision:    {optimal_precision:.4f}")
        print(f"  Recall:       {optimal_recall:.4f}")
        print(f"  Positivos:    {n_positive:,} ({prevalence*100:.2f}%)")
        
        results.append({
            'target': name,
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'f1': optimal_f1,
            'threshold': optimal_threshold,
            'brier': brier
        })
    
    return results

metrics = compute_comprehensive_metrics(probabilities, targets, target_names)
```

---

## Receitas de Visualização para Publicação

### 1. Curvas ROC com Bandas de Confiança

```python
def plot_roc_curves(probabilities, targets, target_names, save_path=None):
    """Gera curvas ROC de qualidade para publicação."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, (name, color) in enumerate(zip(target_names, colors)):
        fpr, tpr, _ = roc_curve(targets[:, i], probabilities[:, i])
        auc = roc_auc_score(targets[:, i], probabilities[:, i])
        
        axes[i].plot(fpr, tpr, linewidth=2.5, color=color, 
                     label=f'AUC = {auc:.3f}')
        axes[i].plot([0, 1], [0, 1], 'k--', linewidth=1.5, 
                     alpha=0.7, label='Baseline Aleatória')
        axes[i].fill_between(fpr, tpr, alpha=0.15, color=color)
        
        axes[i].set_xlabel('Taxa de Falsos Positivos', fontsize=12, fontweight='medium')
        axes[i].set_ylabel('Taxa de Verdadeiros Positivos', fontsize=12, fontweight='medium')
        axes[i].set_title(f'Curva ROC: {name}', fontsize=14, fontweight='bold')
        axes[i].legend(loc='lower right', fontsize=11, framealpha=0.9)
        axes[i].set_xlim([-0.02, 1.02])
        axes[i].set_ylim([-0.02, 1.02])
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
    plt.show()

plot_roc_curves(probabilities, targets, target_names, 'curvas_roc.png')
```

### 2. Curvas Precision-Recall

```python
def plot_pr_curves(probabilities, targets, target_names, save_path=None):
    """Gera curvas Precision-Recall com indicadores de baseline."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, (name, color) in enumerate(zip(target_names, colors)):
        precision, recall, _ = precision_recall_curve(
            targets[:, i], probabilities[:, i]
        )
        ap = average_precision_score(targets[:, i], probabilities[:, i])
        baseline = targets[:, i].mean()
        
        axes[i].plot(recall, precision, linewidth=2.5, color=color, 
                     label=f'AP = {ap:.3f}')
        axes[i].axhline(y=baseline, color='red', linestyle='--', linewidth=1.5,
                        alpha=0.7, label=f'Baseline = {baseline:.3f}')
        axes[i].fill_between(recall, precision, alpha=0.15, color=color)
        
        axes[i].set_xlabel('Recall', fontsize=12, fontweight='medium')
        axes[i].set_ylabel('Precision', fontsize=12, fontweight='medium')
        axes[i].set_title(f'Precision-Recall: {name}', fontsize=14, fontweight='bold')
        axes[i].legend(loc='upper right', fontsize=11, framealpha=0.9)
        axes[i].set_xlim([-0.02, 1.02])
        axes[i].set_ylim([-0.02, 1.02])
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    plt.show()

plot_pr_curves(probabilities, targets, target_names, 'curvas_pr.png')
```

### 3. Análise de Distribuição de Scores

```python
def plot_score_distributions(probabilities, targets, target_names, save_path=None):
    """Visualiza distribuições de probabilidade prevista por classe real."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, name in enumerate(target_names):
        pos_probs = probabilities[targets[:, i] == 1, i]
        neg_probs = probabilities[targets[:, i] == 0, i]
        
        axes[i].hist(neg_probs, bins=50, alpha=0.6, density=True,
                     label=f'Negativo (n={len(neg_probs):,})', color='#1f77b4')
        axes[i].hist(pos_probs, bins=50, alpha=0.6, density=True,
                     label=f'Positivo (n={len(pos_probs):,})', color='#d62728')
        
        axes[i].set_xlabel('Probabilidade Prevista', fontsize=12, fontweight='medium')
        axes[i].set_ylabel('Densidade', fontsize=12, fontweight='medium')
        axes[i].set_title(f'Distribuição de Scores: {name}', fontsize=14, fontweight='bold')
        axes[i].legend(fontsize=11, framealpha=0.9)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    plt.show()

plot_score_distributions(probabilities, targets, target_names, 'distribuicoes.png')
```

### 4. Matrizes de Confusão

```python
from sklearn.metrics import confusion_matrix

# Thresholds ótimos (calcular ou usar valores salvos)
thresholds = [0.45, 0.38, 0.25]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, name in enumerate(target_names):
    preds_binary = (probabilities[:, i] >= thresholds[i]).astype(int)
    cm = confusion_matrix(targets[:, i], preds_binary)
    
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        ax=axes[i], 
        cmap='Blues',
        xticklabels=['Previsto 0', 'Previsto 1'],
        yticklabels=['Real 0', 'Real 1']
    )
    axes[i].set_title(f'Matriz de Confusão: {name}\n(τ = {thresholds[i]:.2f})', 
                      fontsize=14)

plt.tight_layout()
plt.savefig('matrizes_confusao.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 5. Gráficos de Calibração

```python
from sklearn.calibration import calibration_curve

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, name in enumerate(target_names):
    prob_true, prob_pred = calibration_curve(
        targets[:, i], 
        probabilities[:, i], 
        n_bins=10,
        strategy='uniform'
    )
    
    axes[i].plot(prob_pred, prob_true, 's-', linewidth=2, 
                 markersize=8, label='Modelo')
    axes[i].plot([0, 1], [0, 1], 'k--', linewidth=1, 
                 label='Perfeitamente Calibrado')
    axes[i].set_xlabel('Probabilidade Média Prevista', fontsize=12)
    axes[i].set_ylabel('Fração de Positivos', fontsize=12)
    axes[i].set_title(f'Gráfico de Calibração: {name}', fontsize=14)
    axes[i].legend(loc='lower right', fontsize=10)
    axes[i].grid(True, alpha=0.3)
    axes[i].set_xlim([0, 1])
    axes[i].set_ylim([0, 1])

plt.tight_layout()
plt.savefig('curvas_calibracao.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## Análise de Erros

### Identificação de Casos Difíceis

```python
# Falsos Negativos (inadimplências perdidas)
print("Análise de Falsos Negativos:")
for i, name in enumerate(target_names):
    fn_mask = (targets[:, i] == 1) & (probabilities[:, i] < thresholds[i])
    fn_count = fn_mask.sum()
    fn_rate = fn_count / targets[:, i].sum() * 100
    print(f"  {name}: {fn_count:,} FN ({fn_rate:.1f}% dos positivos perdidos)")

print("\nAnálise de Falsos Positivos:")
for i, name in enumerate(target_names):
    fp_mask = (targets[:, i] == 0) & (probabilities[:, i] >= thresholds[i])
    fp_count = fp_mask.sum()
    fp_rate = fp_count / (targets[:, i] == 0).sum() * 100
    print(f"  {name}: {fp_count:,} FP ({fp_rate:.1f}% dos negativos mal classificados)")
```

### Análise de Incerteza

```python
# Casos com maior incerteza de previsão
mean_probs = probabilities.mean(axis=1)
uncertainty = np.abs(mean_probs - 0.5)  # Distância do limite de decisão

# Previsões mais incertas
most_uncertain_idx = np.argsort(uncertainty)[:100]
print(f"Casos mais incertos (n=100):")
print(f"  Probabilidade média: {mean_probs[most_uncertain_idx].mean():.3f}")
print(f"  Taxa de positivos real: {targets[most_uncertain_idx].mean():.3f}")
```

---

## Comparação Entre Experimentos

### Comparando Múltiplas Execuções

```python
import pandas as pd

experiment_runs = [
    'runs/20260119_203110',
    'runs/20260119_222804',
    'runs/20260120_002347'
]

results = []
for run_path in experiment_runs:
    run_id = run_path.split('/')[-1]
    data = np.load(f'{run_path}/test_probs_targets.npz')
    probs = data['probabilities']
    targs = data['targets']
    
    for i, name in enumerate(target_names):
        results.append({
            'Experimento': run_id,
            'Alvo': name,
            'AUC-ROC': roc_auc_score(targs[:, i], probs[:, i]),
            'AUC-PR': average_precision_score(targs[:, i], probs[:, i])
        })

df = pd.DataFrame(results)

# Pivô para comparação fácil
print("\nComparação de AUC-ROC:")
print(df.pivot(index='Experimento', columns='Alvo', values='AUC-ROC').round(4))

print("\nComparação de AUC-PR:")
print(df.pivot(index='Experimento', columns='Alvo', values='AUC-PR').round(4))
```

---

## Checklist de Análise

- [ ] Verificar curvas de loss para indicadores de overfitting
- [ ] Comparar AUC-ROC e AUC-PR em todos os alvos
- [ ] Analisar distribuições de probabilidade para separação de classes
- [ ] Verificar qualidade de calibração do modelo
- [ ] Identificar e analisar falsos positivos/negativos
- [ ] Comparar desempenho contra experimentos anteriores
- [ ] Validar thresholds ótimos para uso operacional
- [ ] Documentar principais descobertas e recomendações

---

## Próximos Passos Recomendados

Com base nos resultados da análise, considere as seguintes ações:

| Observação | Ação Recomendada |
|------------|------------------|
| Baixa AUC-PR | Aumentar augmentation, treinar mais, ou modificar arquitetura |
| Overfitting detectado | Aumentar dropout/weight decay, usar early stopping |
| Calibração pobre | Verificar Temperature Scaling, considerar Platt scaling |
| Alta taxa de falso negativo | Diminuir threshold ou ajustar loss para priorizar recall |
| Alta taxa de falso positivo | Aumentar threshold ou ajustar loss para priorizar precision |
| Treinamento instável | Reduzir taxa de aprendizado, aumentar passos de warmup |

---

## Template de Relatório

Para documentação de experimentos, use a seguinte estrutura:

```markdown
## Relatório de Experimento: [TIMESTAMP]

### Configuração
- Modelo: [detalhes da arquitetura]
- Epochs: [N]
- Batch Size: [B]
- Taxa de Aprendizado: [LR]

### Resultados
| Métrica | target_short | target_medium | target_long | Média |
|---------|--------------|---------------|-------------|-------|
| AUC-ROC | X.XXXX | X.XXXX | X.XXXX | X.XXXX |
| AUC-PR | X.XXXX | X.XXXX | X.XXXX | X.XXXX |
| F1 | X.XXXX | X.XXXX | X.XXXX | X.XXXX |

### Principais Observações
1. [Descoberta 1]
2. [Descoberta 2]

### Próximos Passos
1. [Ação 1]
2. [Ação 2]
```
