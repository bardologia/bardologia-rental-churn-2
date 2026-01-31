import sys
import os
from pathlib import Path
import json
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.metrics import mean_absolute_percentage_error, median_absolute_error


class Results:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model
        self.model.to(self.device)
        self.target_scaler = model.target_scaler
    
    def evaluate(self, model, loader, target_scaler):
        all_preds, all_targets, all_lengths = [], [], []
        model.eval()
        with torch.no_grad():
            for x_cat, x_cont, y, lengths in loader:
                x_cat   = x_cat.to(self.device)
                x_cont  = x_cont.to(self.device)
                lengths = lengths.to(self.device)
                
                preds = model(x_cat, x_cont, lengths)
                
                all_preds.append(preds.cpu())
                all_targets.append(y.cpu())
                all_lengths.extend(lengths.cpu().numpy())

        preds_tensor   = torch.cat(all_preds)
        targets_tensor = torch.cat(all_targets)
        
        seq_lengths = np.array(all_lengths)
        den_targets = np.expm1(target_scaler.inverse_transform(targets_tensor.reshape(-1, 1)))
        den_preds   = np.expm1(target_scaler.inverse_transform(preds_tensor.reshape(-1, 1)))
        
        den_preds   = np.clip(den_preds, 0, None)
        den_targets = np.clip(den_targets, 0, None)
        
        df_preds = pd.DataFrame({'pred': den_preds.flatten(), 'target': den_targets.flatten()})
        return den_preds, den_targets, seq_lengths, df_preds

    @staticmethod
    def compute_metrics(den_preds: np.ndarray, den_targets: np.ndarray) -> dict:
        den_preds   = np.asarray(den_preds).flatten()
        den_targets = np.asarray(den_targets).flatten()
        
        mae = float(np.mean(np.abs(den_preds - den_targets)))
        rmse = float(np.sqrt(np.mean((den_preds - den_targets) ** 2)))
        ss_res = float(np.sum((den_targets - den_preds) ** 2))
        ss_tot = float(np.sum((den_targets - np.mean(den_targets)) ** 2))
        r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else float('nan')
        std = float(np.std(den_targets - den_preds))
        medae = float(median_absolute_error(den_targets, den_preds))
        max_error = float(np.max(np.abs(den_targets - den_preds)))
        p50 = np.percentile(np.abs(den_targets - den_preds), 50)
        p90 = np.percentile(np.abs(den_targets - den_preds), 90)
        p95 = np.percentile(np.abs(den_targets - den_preds), 95)

        abs_err = np.abs(den_targets - den_preds)
        error_bin_0_5 = float(np.mean(abs_err <= 5) * 100)
        error_bin_5_10 = float(np.mean((abs_err > 5) & (abs_err <= 10)) * 100)
        error_bin_10_15 = float(np.mean((abs_err > 10) & (abs_err <= 15)) * 100)
        error_bin_15_20 = float(np.mean((abs_err > 15) & (abs_err <= 20)) * 100)
        error_bin_20_25 = float(np.mean((abs_err > 20) & (abs_err <= 25)) * 100)
        error_bin_above_25 = float(np.mean(abs_err > 25) * 100)

        def mean_target_range(low, high=None):
            mask = (den_targets > low) & (den_targets <= high)
            vals = abs_err[mask]
            return float(np.mean(vals)) if len(vals) > 0 else float('nan')

        metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'std': std,
            'p50': p50,
            'p90': p90,
            'p95': p95,
            'medae': medae,
            'max_error': max_error,
            
            'error_0_5': error_bin_0_5,
            'error_5_10': error_bin_5_10,
            'error_10_15': error_bin_10_15,
            'error_15_20': error_bin_15_20,
            'error_20_25': error_bin_20_25,
            'error_above_25': error_bin_above_25,
    
            'target_0_5': mean_target_range(0, 5),
            'target_5_10': mean_target_range(5, 10),
            'target_10_15': mean_target_range(10, 15),
            'target_15_20': mean_target_range(15, 20),
            'target_20_25': mean_target_range(20, 25),
            'target_above_25': mean_target_range(25, 30),
            
        }
        
        return metrics

    @staticmethod
    def _make_plots(den_preds: np.ndarray, den_targets: np.ndarray, seq_lengths: np.ndarray) -> dict:
        errors = np.ravel(den_preds - den_targets)
        figs = {}

        fig = plt.figure(figsize=(8, 5))
        ax = fig.gca()
        ax.hist(errors, bins='auto', color='skyblue', edgecolor='black')
        ax.set_title('Histogram of Prediction Errors (Residuals)')
        ax.set_xlabel('Prediction Error')
        ax.set_ylabel('Frequency')
        ax.grid(True, linestyle='--', alpha=0.6)
        fig.tight_layout()
        figs['errors_hist'] = fig

        fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].scatter(seq_lengths, errors, alpha=0.5, color='royalblue', edgecolor='k')
        axes[0].set_title('Error vs. Sequence Length')
        axes[0].set_xlabel('Sequence Length')
        axes[0].set_ylabel('Prediction Error')
        axes[0].grid(True, linestyle='--', alpha=0.6)

        axes[1].scatter(den_targets, errors, alpha=0.5, color='darkorange', edgecolor='k')
        axes[1].set_title('Error vs. Target Value')
        axes[1].set_xlabel('True Target Value')
        axes[1].set_ylabel('Prediction Error')
        axes[1].grid(True, linestyle='--', alpha=0.6)

        fig2.tight_layout()
        figs['error_vs_length_and_target'] = fig2

        fig3 = plt.figure(figsize=(7, 6))
        ax3 = fig3.gca()
        ax3.scatter(den_targets, den_preds, alpha=0.5, color='mediumseagreen', edgecolor='k')
        ax3.plot([den_targets.min(), den_targets.max()], [den_targets.min(), den_targets.max()], 'r--', lw=2, label='Ideal Fit')
        ax3.set_title('Predicted vs. True Values')
        ax3.set_xlabel('True Target Value')
        ax3.set_ylabel('Predicted Value')
        ax3.legend()
        ax3.grid(True, linestyle='--', alpha=0.6)
        fig3.tight_layout()
        figs['pred_vs_true'] = fig3

        fig4 = plt.figure(figsize=(7, 5))
        ax4 = fig4.gca()
        ax4.scatter(den_preds, errors, alpha=0.5, color='slateblue', edgecolor='k')
        ax4.axhline(0, color='red', linestyle='--', lw=2)
        ax4.set_title('Residuals vs. Predicted Values')
        ax4.set_xlabel('Predicted Value')
        ax4.set_ylabel('Residual (Error)')
        ax4.grid(True, linestyle='--', alpha=0.6)
        fig4.tight_layout()
        figs['residuals_vs_predicted'] = fig4

        fig5 = plt.figure(figsize=(6, 6))
        ax5 = fig5.gca()
        stats.probplot(errors, dist="norm", plot=ax5)
        ax5.set_title('QQ-plot of Residuals')
        ax5.grid(True, linestyle='--', alpha=0.6)
        fig5.tight_layout()
        figs['qqplot_residuals'] = fig5

        sorted_abs_errors = np.sort(np.abs(errors))
        cum_abs_error = np.cumsum(sorted_abs_errors)
        fig6 = plt.figure(figsize=(8, 5))
        ax6 = fig6.gca()
        ax6.plot(np.arange(1, len(cum_abs_error) + 1), cum_abs_error, color='teal')
        ax6.set_title('Cumulative Absolute Error')
        ax6.set_xlabel('Sample (sorted by error)')
        ax6.set_ylabel('Cumulative Absolute Error')
        ax6.grid(True, linestyle='--', alpha=0.6)
        fig6.tight_layout()
        figs['cumulative_abs_error'] = fig6

        return figs

    def run(self, loader, make_plots: bool = False):
        den_preds, den_targets, seq_lengths, df_preds = self.evaluate(self.model, loader, self.target_scaler)
        metrics = self.compute_metrics(den_preds, den_targets)

        plots = {}
        if make_plots:
            plots = self._make_plots(den_preds, den_targets, seq_lengths)

        return {
            'metrics': metrics,
            'preds_df': df_preds,
            'plots': plots
        }


