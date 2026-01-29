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
    user_sample_num = 3000

    config.model.hidden_dim = 128
    config.model.num_layers = 1
    config.model.num_sequence_layers = 1
    config.model.n_heads = 4
    config.model.epochs = 10
    config.model.patience = 4
    

    study = FeatureAblation(
        data_path=data_path,
        output_dir=output_dir,
        metric=metric,
        user_sample_num=user_sample_num,
    )

    df = study.run()
    print(df[["feature", "feature_type", "delta_metric", "num_params", "run_dir"]])


if __name__ == "__main__":
    main()
