import logging
import os
import sys
import torch
import optuna
from datetime import datetime
from typing import Dict, Optional, Tuple
import json
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from data import DataPipeline
from model import Model
from trainer import Trainer
from config import config
from logger import Logger


class OptunaTuner:
    def __init__(self, data_path, project_root, train_loader, val_loader, target_scaler=None, feature_scaler=None):
        
        self.logger = Logger(name="OptunaTuner", level=logging.INFO, log_dir=None)
        self.data_path = data_path
        self.study_name = config.optuna.study_name
        self.storage = config.optuna.storage
        self.n_trials = config.optuna.n_trials
        self.timeout = config.optuna.timeout
        self.device = config.model.device 
        self.pruning_warmup_epochs = config.optuna.pruning_warmup_epochs
        self.early_stopping_patience = config.optuna.early_stopping_patience
        self.max_epochs = config.optuna.max_epochs
        self.project_root = project_root if project_root else os.path.dirname(current_dir)
        
        self.sampler = TPESampler(n_startup_trials=config.optuna.n_startup_trials, seed=config.data.random_state)
        self.pruner = MedianPruner(
            n_startup_trials=config.optuna.n_startup_trials,
            n_warmup_steps=config.optuna.pruning_warmup_epochs,
            interval_steps=1
        )
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.target_scaler = target_scaler
        self.feature_scaler = feature_scaler
        self.best_params = None
        self.best_value = None

    def _suggest_hyperparameters(self, trial: optuna.Trial) -> Dict:
        params = {
            'learning_rate': trial.suggest_float('learning_rate', config.optuna.learning_rate_min, config.optuna.learning_rate_max, log=True),
            'hidden_dim': trial.suggest_categorical('hidden_dim', config.optuna.hidden_dim_options),
            'num_invoice_layers': trial.suggest_int('num_invoice_layers', config.optuna.num_invoice_layers_min, config.optuna.num_invoice_layers_max),
            'num_sequence_layers': trial.suggest_int('num_sequence_layers', config.optuna.num_sequence_layers_min, config.optuna.num_sequence_layers_max),
            'num_heads': trial.suggest_categorical('num_heads', config.optuna.num_heads_options),
            'dropout': trial.suggest_float('dropout', config.optuna.dropout_min, config.optuna.dropout_max),
            'drop_path_rate': trial.suggest_float('drop_path_rate', config.optuna.drop_path_rate_min, config.optuna.drop_path_rate_max),
            'weight_decay': trial.suggest_float('weight_decay', config.optuna.weight_decay_min, config.optuna.weight_decay_max, log=True),
            'periodic_sigma': trial.suggest_float('periodic_sigma', config.optuna.periodic_sigma_min, config.optuna.periodic_sigma_max),
            'embedding_dropout': trial.suggest_float('embedding_dropout', config.optuna.embedding_dropout_min, config.optuna.embedding_dropout_max),
            'ema_decay': trial.suggest_float('ema_decay', config.optuna.ema_decay_min, config.optuna.ema_decay_max),
            'warmup_epochs': trial.suggest_int('warmup_epochs', config.optuna.warmup_epochs_min, config.optuna.warmup_epochs_max),
            'scheduler_t0': trial.suggest_int('scheduler_t0', config.optuna.scheduler_t0_min, config.optuna.scheduler_t0_max),
            'max_grad_norm': trial.suggest_float('max_grad_norm', config.optuna.max_grad_norm_min, config.optuna.max_grad_norm_max),
            'head_dropout_multiplier_medium': trial.suggest_float('head_dropout_multiplier_medium', config.optuna.head_dropout_multiplier_medium_min, config.optuna.head_dropout_multiplier_medium_max),
            'head_dropout_multiplier_long': trial.suggest_float('head_dropout_multiplier_long', config.optuna.head_dropout_multiplier_long_min, config.optuna.head_dropout_multiplier_long_max),
            'asymmetric_gamma_negative': trial.suggest_int('asymmetric_gamma_negative', config.optuna.asymmetric_gamma_negative_min, config.optuna.asymmetric_gamma_negative_max),
            'asymmetric_gamma_positive': trial.suggest_int('asymmetric_gamma_positive', config.optuna.asymmetric_gamma_positive_min, config.optuna.asymmetric_gamma_positive_max),
            'label_smoothing': trial.suggest_float('label_smoothing', config.optuna.label_smoothing_min, config.optuna.label_smoothing_max),
            'temperature_init': trial.suggest_float('temperature_init', config.optuna.temperature_init_min, config.optuna.temperature_init_max),
        }
        return params
    
    def _objective(self, trial: optuna.Trial) -> float:
        self._prepare_data()
        params = self._suggest_hyperparameters(trial)
        run_id = f"optuna_{datetime.now().strftime('%Y%m%d_%H%M%S')}_trial{trial.number}"
        run_dir = os.path.join(self.project_root, "Runs", run_id)
        checkpoint_dir = os.path.join(run_dir, "checkpoints")
        logger = Logger(log_dir=run_dir, name=f"trial_{trial.number}")
        logger.info(f"Trial {trial.number} started")
        logger.info(f"Parameters: {params}")
        
        try:
            original_config_model = {
                'periodic_sigma': config.model.periodic_sigma,
                'embedding_dropout': config.model.embedding_dropout,
                'head_dropout_multiplier_medium': config.model.head_dropout_multiplier_medium,
                'head_dropout_multiplier_long': config.model.head_dropout_multiplier_long,
                'num_sequence_layers': config.model.num_sequence_layers,
            }
            
            config.model.periodic_sigma = params['periodic_sigma']
            config.model.embedding_dropout = params['embedding_dropout']
            config.model.head_dropout_multiplier_medium = params['head_dropout_multiplier_medium']
            config.model.head_dropout_multiplier_long = params['head_dropout_multiplier_long']
            config.model.num_sequence_layers = params['num_sequence_layers']
            
            model = Model(embedding_dimensions=self.data_module.embedding_dimensions, num_continuous=len(self.data_module.continuous_columns))
            
            trainer = Trainer(
                model=model,
                train_loader=self.train_loader,
                validation_loader=self.val_loader,
                checkpoint_dir=checkpoint_dir,
                logger=logger,
                target_scaler=self.target_scaler,
                feature_scaler=self.feature_scaler
            )
            
            trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                trainer.optimizer,
                T_0=params['scheduler_t0'],
                T_mult=config.model.scheduler_t_mult,
                eta_min=config.model.min_lr
            )
            
            best_metric = float('inf')
            
            epochs_without_improvement = 0
            for epoch in range(1, self.max_epochs + 1):
                train_loss = trainer.train_epoch(epoch)
                val_metrics = trainer.evaluate(self.val_loader)
                trainer.scheduler.step()
                current_metric = val_metrics['rmse']
                
                if current_metric < best_metric:
                    best_metric = current_metric
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                
                if epoch >= self.pruning_warmup_epochs:
                    trial.report(current_metric, epoch)
                    if trial.should_prune():
                        logger.info(f"Trial {trial.number} pruned at epoch {epoch}")
                        raise optuna.TrialPruned()
                
                if epochs_without_improvement >= self.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            config.model.periodic_sigma = original_config_model['periodic_sigma']
            config.model.embedding_dropout = original_config_model['embedding_dropout']
            config.model.head_dropout_multiplier_medium = original_config_model['head_dropout_multiplier_medium']
            config.model.head_dropout_multiplier_long = original_config_model['head_dropout_multiplier_long']
            config.model.num_sequence_layers = original_config_model['num_sequence_layers']
            logger.info(f"Trial {trial.number} completed with best RMSE: {best_metric:.4f}")
            logger.close()
            return best_metric
        
        except optuna.TrialPruned:
            config.model.periodic_sigma = original_config_model['periodic_sigma']
            config.model.embedding_dropout = original_config_model['embedding_dropout']
            config.model.head_dropout_multiplier_medium = original_config_model['head_dropout_multiplier_medium']
            config.model.head_dropout_multiplier_long = original_config_model['head_dropout_multiplier_long']
            config.model.num_sequence_layers = original_config_model['num_sequence_layers']
            logger.close()
            raise
        
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {str(e)}")
            logger.close()
            raise
    
    def optimize(self) -> Tuple[Dict, float]:
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction='maximize',
            sampler=self.sampler,
            pruner=self.pruner,
            load_if_exists=True
        )
        
        study.optimize(
            self._objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        self.best_params = study.best_params
        self.best_value = study.best_value
        
        results_logger = Logger(log_dir=os.path.join(self.project_root, config.paths.runs_dir, config.paths.optuna_results_dir), name="optimization_summary")
        results_logger.info("Optimization completed!")
        results_logger.info(f"\nBest hyperparameters:")
        for key, value in self.best_params.items():
            results_logger.info(f"  {key}: {value}")

        results_dir = os.path.join(self.project_root, config.paths.runs_dir, config.paths.optuna_results_dir)
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(results_dir, f"best_params_{timestamp}.json")
        
        results = {
            'best_params': self.best_params,
            'best_value': self.best_value,
            'best_trial': study.best_trial.number,
            'n_trials': len(study.trials),
            'timestamp': timestamp
        }
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        results_logger.info(f"\nResults saved to: {results_file}")
        results_logger.close()
        
        return self.best_params, self.best_value
    
    def get_statistics(self, study_name: Optional[str] = None) -> Dict:
        if study_name is None:
            study_name = self.study_name
            
        study = optuna.load_study(
            study_name=study_name,
            storage=self.storage
        )
        
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        
        stats = {
            'n_trials': len(study.trials),
            'n_completed': len(completed_trials),
            'n_pruned': len(pruned_trials),
            'best_value': study.best_value,
            'best_trial': study.best_trial.number,
            'best_params': study.best_params
        }
        
        return stats


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, config.paths.train_data)
    
    setup_logger = Logger(log_dir=os.path.join(project_root, "runs", "setup"), name="tuning_setup")
    
    if not os.path.exists(data_path):
        setup_logger.info(f"Data file not found: {data_path}")
        setup_logger.info("Generating training data...")
        raw_path = os.path.join(project_root, config.paths.raw_data)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        pipeline = DataPipeline()
        pipeline.run(raw_path, data_path)
        setup_logger.info(f"Training data generated: {data_path}")
    
    setup_logger.close()

    tuner = OptunaTuner(data_path=data_path)
    best_params, best_value = tuner.optimize()
    
