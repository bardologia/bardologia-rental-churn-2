import os
import sys

# Ensure project root is on sys.path so `core` package imports work
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from core.ablation import FeatureAblation
from core.config import config


def main():
    data_path = config.paths.train_data
    output_dir = "runs/ablation"
    metric = "rmse"
    user_sample_num = 3000

    config.architecture.hidden_dim = 128
    config.architecture.num_layers = 1
    config.architecture.num_sequence_layers = 1
    config.architecture.n_heads = 4
    config.training.epochs = 10
    config.training.patience = 4
    config.overfit.enabled = False

    study = FeatureAblation(
        data_path=data_path,
        output_dir=output_dir,
        metric=metric,
        user_sample_num=user_sample_num,
    )

    df = study.run()
    print(df[["feature", "feature_type", "delta_metric", "num_params", "run_dir"]].head(25))


if __name__ == "__main__":
    main()
