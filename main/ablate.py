import sys
import os
from venv import logger
import pandas as pd
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import torch
import numpy as np
from datetime import datetime

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(enabled=False)
torch.autograd.profiler.emit_nvtx(enabled=False)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))

from core.data    import DataPipeline
from core.logger  import Logger
from core.ablation import FeatureAblation


def ablate(config, use_cache = True, save_cache = True, save_results = True):
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    run_id       = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    run_dir      = os.path.join(project_root, config.paths.ablation_dir, run_id)
    results_path = os.path.join(run_dir, f"ablation_results_{run_id}.csv")

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    raw_path   = os.path.join(project_root, config.paths.raw_data)
    train_path = os.path.join(project_root, config.paths.train_data)
    
    logger = Logger(name="ablate", level="INFO", log_dir=run_dir)
    
    config.load.user_sample_count = config.ablation.user_sample_count
    config.architecture.hidden_dim = config.ablation.hidden_dim
    config.architecture.num_invoice_encoder_layers = config.ablation.num_invoice_encoder_layers
    config.architecture.num_sequence_encoder_layers = config.ablation.num_sequence_encoder_layers
    config.architecture.num_attention_heads = config.ablation.num_attention_heads
    config.training.epochs = config.ablation.training_epochs
    config.training.patience = config.ablation.patience
    config.scheduler.scheduler_patience = config.ablation.scheduler_patience
    config.overfit.overfit_single_batch = config.ablation.overfit_single_batch

    if config.overfit.overfit_single_batch:
        config.training.dropout  = config.overfit.overfit_dropout
        config.architecture.embedding_dropout = config.overfit.overfit_dropout
        config.architecture.use_augmentation = config.overfit.overfit_use_augmentation
        config.training.weight_decay = config.overfit.overfit_weight_decay
        config.training.patience = config.overfit.overfit_patience
        config.training.epochs = config.overfit.overfit_epochs
        config.scheduler.scheduler_patience = config.overfit.overfit_patience
        config.training.mixed_precision = config.overfit.overfit_mixed_precision
        config.ema.use_ema = config.overfit.overfit_use_ema
        config.split.val_size = config.overfit.overfit_val_size
        config.split.test_size = config.overfit.overfit_test_size
        config.load.user_sample_count = config.overfit.overfit_number_of_users
        config.architecture.min_seq_len = config.overfit.overfit_min_length
        logger.warning("Overfit single batch mode: Disabled dropout, augmentation, weight decay, mixed precision, and increased epochs and patience.")

    if use_cache:
        dataframe = pd.read_parquet(train_path)
    else:
        pipeline  = DataPipeline(config, logger)
        dataframe = pipeline.run(raw_path, train_path)
        if save_cache:
            os.makedirs(os.path.dirname(train_path), exist_ok=True)
            dataframe.to_parquet(train_path, index=False)
            logger.info(f"[Cache] Training data cached at: {train_path}")
  
    study = FeatureAblation(
        dataframe = dataframe,
        config    = config,
        logger    = logger
    )

    results = study.run(metric=config.ablation.metric)

    if save_results:
        rows = []
        for feat, info in results.items():
            row = {
                "feature": feat,
                "feature_type": info.get("feature_type"),
                "delta_metric": info.get("delta_metric")
            }
            val_metrics = info.get("val_metrics") or {}
            for k, v in val_metrics.items():
                row[f"val_{k}"] = v
            rows.append(row)

        results_df = pd.DataFrame(rows)
        results_df.to_csv(results_path, index=False)
        logger.info(f"[Results] Ablation results saved to: {results_path}")
        
    logger.close()
    return results

if __name__ == "__main__":
    from main.config import config

    results = ablate(
        config       = config,
        use_cache    = True,
        save_cache   = True,
        save_results = True
    )

