import os
import sys
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

project_root = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.ablation import FeatureAblation
from main.config import config


def main():
    data_path = os.path.join(project_root, config.paths.train_data)
    output_dir = os.path.join(project_root, config.paths.ablation_dir)
    metric = "rmse"
 
    ab_cfg = config.ablation

    config.load.user_sample_count = ab_cfg.user_sample_count
    config.architecture.hidden_dim = ab_cfg.hidden_dim
    config.architecture.num_invoice_encoder_layers = ab_cfg.num_invoice_encoder_layers
    config.architecture.num_sequence_encoder_layers = ab_cfg.num_sequence_encoder_layers
    config.architecture.num_attention_heads = ab_cfg.num_attention_heads
    config.training.epochs = ab_cfg.training_epochs
    config.training.patience = ab_cfg.patience
    config.scheduler.scheduler_patience = ab_cfg.scheduler_patience
    config.overfit.overfit_single_batch = ab_cfg.overfit_single_batch

    metric = ab_cfg.metric

    study = FeatureAblation(
        data_path=data_path,
        output_dir=output_dir,
        metric=metric,
    )

    df = study.run()
    print(df[["feature", "feature_type", "delta_metric", "num_params", "run_dir"]])


if __name__ == "__main__":
    main()
