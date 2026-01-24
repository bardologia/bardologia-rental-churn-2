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

from core.data import DataPipeline
from core.tuner import OptunaTuner
from core.config import config
from core.logger import Logger
from core.dataset import DatasetLoader


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
    
    data_module = DatasetLoader(data_path)
        
    train_loader, validation_loader, test_loader = data_module.dataloader_pipeline()
    
    target_scaler = data_module.target_scalers["target_days_to_payment"]
    continuous_scalers = data_module.continuous_scalers

    tuner = OptunaTuner(data_path=data_path, project_root=project_root, train_loader=train_loader, val_loader=validation_loader, target_scaler=target_scaler, feature_scaler=continuous_scalers)
    
    best_params, best_value = tuner.optimize()
    
    logger.section("Tuning Results")
    logger.info(f"[Tuning] Best Parameters: {best_params}")
    logger.info(f"[Tuning] Best Validation Value: {best_value}")
    logger.close()


if __name__ == "__main__":
    main()
