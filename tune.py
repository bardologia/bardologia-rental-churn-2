import sys
import os
import torch

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(enabled=False)
torch.autograd.profiler.emit_nvtx(enabled=False)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))

from model.core import DataPipeline
from model.tuner import OptunaTuner
from configs.config import config
from utils.logger import Logger


def main():
    project_root = current_dir
    
    run_dir = os.path.join(project_root, "Runs", "optuna_tuning")
    logger = Logger(log_dir=run_dir, name="OptunaTuning")
    
    raw_path = os.path.join(project_root, config.paths.raw_data)
    data_path = os.path.join(project_root, config.paths.train_data)
    
    if not os.path.exists(data_path):
        logger.info(f"[Data] File not found: {data_path}")
        logger.info("[Data] Generating training data...")
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        pipeline = DataPipeline()
        pipeline.run(raw_path, data_path)
        logger.info(f"[Data] Training data generated: {data_path}")
    
    logger.section("Hyperparameter Tuning with Optuna")
    logger.info(f"[Config] Study name: {config.optuna.study_name}")
    logger.info(f"[Config] Number of trials: {config.optuna.n_trials}")
    logger.info(f"[Config] Max epochs per trial: {config.optuna.max_epochs}")
    logger.info(f"[Config] Early stopping patience: {config.optuna.early_stopping_patience}")
    
    tuner = OptunaTuner(data_path=data_path, project_root=project_root)
    
    best_params, best_value = tuner.optimize()
    
    logger.section("Tuning Results")
    logger.info(f"[Results] Best AUC achieved: {best_value:.4f}")
    logger.info("[Results] Best parameters saved to: Runs/optuna_results/best_params_*.json")
    logger.info("[Results] Use these parameters in config.py for final training")
    logger.close()


if __name__ == "__main__":
    main()
