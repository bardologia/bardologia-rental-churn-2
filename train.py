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


def main():

    project_root = current_dir 
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(project_root, config.paths.runs_dir, run_id)
    checkpoint_dir = os.path.join(run_dir, config.paths.checkpoints_dir)
    model_save_path = os.path.join(checkpoint_dir, config.paths.best_model_name)
    
    raw_path = os.path.join(project_root, config.paths.raw_data)
    data_path = os.path.join(project_root, config.paths.train_data)
    
    if not os.path.exists(data_path):
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        pipeline = DataPipeline()
        pipeline.run(raw_path, data_path)
        print(f"Generated new training data: {data_path}")
    
    logger = Logger(name="train", level="INFO", log_dir=run_dir)
    trainer = None

    if config.model.overfit_single_batch:
        config.model.dropout  = config.model.overfit_dropout
        config.model.embedding_dropout = config.model.overfit_dropout
        config.model.use_augmentation = config.model.overfit_use_augmentation
        config.model.weight_decay = config.model.overfit_weight_decay
        config.model.patience = config.model.overfit_patience
        config.model.epochs = config.model.overfit_epochs
        config.model.scheduler_patience = config.model.overfit_patience
        config.model.mixed_precision = config.model.overfit_mixed_precision
        config.model.use_ema = config.model.overfit_use_ema
        config.data.val_size = config.model.overfit_val_size
        config.data.test_size = config.model.overfit_test_size
        config.data.user_sample_num = config.model.overfit_num_users
        config.model.min_seq_len = config.model.overfit_min_lenghth
        logger.warning("Overfit single batch mode: Disabled dropout, augmentation, weight decay, mixed precision, and increased epochs and patience.")

    try:
        data_module = DatasetLoader(data_path)
        
        train_loader, validation_loader, test_loader = data_module.dataloader_pipeline()
        
        target_scaler = data_module.target_scalers[config.columns.target_col_name]
        continuous_scalers  = data_module.continuous_scalers 

        model   = Model(embedding_dimensions=data_module.embedding_dimensions, num_continuous=len(data_module.continuous_columns))    

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            checkpoint_dir=checkpoint_dir,
            target_scaler=target_scaler,
            feature_scaler=continuous_scalers,
            embedding_dimensions=data_module.embedding_dimensions,
            log_dir=run_dir
        )
        
        best_model = trainer.fit()
        
        trainer.logger.section("Model Persistence")
        torch.save(best_model.state_dict(), model_save_path)
        trainer.logger.info(f"[Save] Best model saved to: {model_save_path}")
        
        last_checkpoint_path = os.path.join(checkpoint_dir, config.paths.last_checkpoint_name)
        trainer.logger.info(f"[Save] Last checkpoint path: {last_checkpoint_path}")
          
        with torch.inference_mode():
            model.eval()
            metrics = trainer.evaluate(test_loader)
            trainer.logger.section("Final Evaluation on Test Set")
            for metric_name, metric_value in metrics.items():
                trainer.logger.info(f"[Test] {metric_name}: {metric_value:.4f}")
           
    except Exception as e:
        if trainer is not None:
            trainer.logger.error(f"[Error] Training failed: {e}")
        else:
            logger.error(f"[Error] Training failed before trainer initialization: {e}")
    finally:
        if trainer is not None:
            trainer.logger.close()
        else:
            logger.close()


if __name__ == "__main__":    
    main()
