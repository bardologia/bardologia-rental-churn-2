import sys
import os
from unittest import loader
import torch
import numpy as np
from datetime import datetime
import argparse

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(enabled=False)
torch.autograd.profiler.emit_nvtx(enabled=False)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))

from core.data import DataPipeline
from core.dataset import DatasetLoader
from core.model import Model
from core.trainer import Trainer
from core.config import config
from core.logger import Logger


def main(overfit_single_batch=False):
    if overfit_single_batch:
        config.model.overfit_single_batch = True
        print("Overfit single batch mode enabled")
    project_root = current_dir 
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(project_root, "Runs", run_id)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    model_save_path = os.path.join(checkpoint_dir, "best_model.pth")
    
    raw_path = os.path.join(project_root, config.paths.raw_data)
    data_path = os.path.join(project_root, config.paths.train_data)
    
    if not os.path.exists(data_path):
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        pipeline = DataPipeline()
        pipeline.run(raw_path, data_path)
        print(f"Generated new training data: {data_path}")
    
    logger = Logger(log_dir=run_dir, name="PaymentDefaultPrediction")
    logger.section("Experiment Initialization")
    logger.info(f"[Experiment] Run ID: {run_id}")
    logger.info(f"[Experiment] Run directory: {run_dir}")
    
    try:
        logger.subsection("Data Module Setup")
        data_module = DatasetLoader(data_path)
        
        train_loader, validation_loader, test_loader = data_module.dataloader_pipeline()
        
        target_scaler = data_module.target_scalers["target_days_to_payment"]
        continuous_scalers  = data_module.continuous_scalers 

        model   = Model(embedding_dimensions=data_module.embedding_dimensions, num_continuous=len(data_module.continuous_columns))
        trainer = Trainer(model=model, train_loader=train_loader, validation_loader=validation_loader, checkpoint_dir=checkpoint_dir, target_scaler=target_scaler, feature_scaler=continuous_scalers)
        
        best_model = trainer.fit()
        
        logger.section("Model Persistence")
        torch.save(best_model.state_dict(), model_save_path)
        logger.info(f"[Save] Best model saved to: {model_save_path}")
          
        with torch.inference_mode():
            model.eval()
            metrics = trainer.evaluate(test_loader)
            logger.section("Final Evaluation on Test Set")
            for metric_name, metric_value in metrics.items():
                logger.info(f"[Test] {metric_name}: {metric_value:.4f}")
           
    except Exception as e:
        logger.error(f"[Error] Training failed: {e}")
    finally:
        logger.close()


if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Train the rental churn model")
    parser.add_argument("--overfit-single-batch", action="store_true", help="Enable overfitting on a single batch for debugging")
    args = parser.parse_args()
    
    main(overfit_single_batch=args.overfit_single_batch)
