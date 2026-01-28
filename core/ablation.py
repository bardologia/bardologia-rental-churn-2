# feature_ablation.py
import os

from tqdm import tqdm
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import copy
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple, Any

import pandas as pd
import torch

from core.config import config
from core.dataset import DatasetLoader
from core.trainer import Trainer
from core.model import Model  


class FeatureAblation:
    def __init__(
        self,
        data_path: str,
        output_dir: str = "runs/ablation",
        metric: str = "rmse",
        user_sample_num: int = 500,
    ):
        self.data_path = data_path
        self.output_dir = output_dir
        self.metric = metric
        self.user_sample_num = int(user_sample_num)

        os.makedirs(self.output_dir, exist_ok=True)

        self._original_categorical_columns = list(config.columns.cat_cols)
        self._original_continuous_columns = list(config.columns.cont_cols)
        self._original_user_sample_num = config.data_sampling

        self.protected = ["target_col_name", "delay_clipped_col", "delay_is_known_col"]

 
    def run(self) -> pd.DataFrame:

        config.data_sampling.user_sample_num = self.user_sample_num

        try:
            config.columns.cat_cols = list(self._original_categorical_columns)
            config.columns.cont_cols = list(self._original_continuous_columns)
            baseline_val_metrics, baseline_test_metrics, baseline_num_params, baseline_run_dir = self._train(tag="baseline")

            baseline_metric = float(baseline_val_metrics[self.metric])
          

            rows = [self._row("__baseline__", "baseline", False, baseline_val_metrics, baseline_test_metrics, baseline_num_params, baseline_run_dir, baseline_metric, None, self.metric)]
            features = self._original_categorical_columns + self._original_continuous_columns
           
            for feature in tqdm(features, desc="Ablating features", ncols=100):
                config.columns.cat_cols  = list(self._original_categorical_columns)
                config.columns.cont_cols = list(self._original_continuous_columns)
                removed, feature_type    = self._remove_feature(feature)

                if not removed:
                    rows.append(self._row(feature, feature_type, False, baseline_val_metrics, baseline_test_metrics, baseline_num_params, baseline_run_dir, baseline_metric, None, self.metric))
                    continue

                val_metrics, test_metrics, num_params, run_dir = self._train(tag=f"minus_{feature}")
              
                delta = float(val_metrics[self.metric]) - baseline_metric
                rows.append(self._row(feature, feature_type, True, val_metrics, test_metrics, num_params, run_dir, baseline_metric, delta, self.metric))

            df = pd.DataFrame(rows).sort_values("delta_metric", ascending=False, na_position="last").reset_index(drop=True)
            out_csv = os.path.join(self.output_dir, f"ablation_{self.metric}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            df.to_csv(out_csv, index=False)
            return df

        finally:
            config.columns.cat_cols = list(self._original_categorical_columns)
            config.columns.cont_cols = list(self._original_continuous_columns)
            config.data_sampling.user_sample_num = self._original_user_sample_num

    def _remove_feature(self, feat: str):
        if feat in self.protected:
            return False, "protected"

        cat = list(config.columns.cat_cols)
        cont = list(config.columns.cont_cols)

        if feat in cat:
            cat.remove(feat)
            config.columns.cat_cols = cat
            return True, "categorical"

        if feat in cont:
            cont.remove(feat)
            config.columns.cont_cols = cont
            return True, "continuous"

        return False, "not_found"

    def _train(self, tag: str):
        run_dir = os.path.join(self.output_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}__{tag}")
        ckpt_dir = os.path.join(run_dir, config.paths.checkpoints_dir)
        os.makedirs(ckpt_dir, exist_ok=True)

        data = DatasetLoader(self.data_path, active=False)
        train_loader, val_loader, test_loader = data.dataloader_pipeline()

        model = Model(
            data.embedding_dimensions,
            len(data.continuous_columns),
        )

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            validation_loader=val_loader,
            checkpoint_dir=ckpt_dir,
            target_scaler=data.target_scalers[config.columns.target_col_name],
            feature_scaler=data.continuous_scalers,
            embedding_dimensions=data.embedding_dimensions,
            log_dir=run_dir,
            active=False,
        )

        trainer.fit()
        
        with torch.inference_mode():
            model.eval()
            val_metrics = trainer.evaluate(val_loader)
            test_metrics = trainer.evaluate(test_loader)
            trainer.logger.close()

        return val_metrics, test_metrics, num_params, run_dir

    @staticmethod
    def _row(
        feature: str,
        feature_type: str,
        removed: bool,
        val_metrics: Optional[Dict[str, float]],
        test_metrics: Optional[Dict[str, float]],
        num_params: Optional[int],
        run_dir: Optional[str],
        baseline_metric: Optional[float],
        delta_metric: Optional[float],
        metric: str,
    ) -> Dict[str, Any]:

        val_metric = float(val_metrics[metric])
       
        return {
            "feature": feature,
            "feature_type": feature_type,
            "removed": removed,
            "num_params": num_params,
            "run_dir": run_dir,
            "baseline_metric": baseline_metric,
            "val_metric": val_metric,
            "delta_metric": delta_metric,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        }

