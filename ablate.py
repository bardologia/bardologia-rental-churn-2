import os
import sys
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from core.ablation import FeatureAblation
from core.config import config


def main():
    data_path = config.paths.train_data
    output_dir = "runs/ablation"
    metric = "rmse"
 
    config.load.user_sample_count = 300
    config.architecture.hidden_dim = 128
    config.architecture.num_invoice_encoder_layers = 1
    config.architecture.num_sequence_encoder_layers = 1
    config.architecture.num_attention_heads = 4
    config.training.epochs = 10
    config.training.patience = 4
    config.scheduler.scheduler_patience = 2
    config.overfit.overfit_single_batch = False

    study = FeatureAblation(
        data_path=data_path,
        output_dir=output_dir,
        metric=metric,
    )

    df = study.run()
    print(df[["feature", "feature_type", "delta_metric", "num_params", "run_dir"]])


if __name__ == "__main__":
    main()
