import os
from tqdm import tqdm
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pandas as pd
import copy

from core.dataset import DatasetLoader
from core.trainer import Trainer
from core.model import Model  
from core.results import Results


class FeatureAblation:
    def __init__(
        self,
        dataframe,
        config = None,
        logger = None
    ):
        self.logger = logger
        
        self.original_dataframe           = dataframe.copy()
        self.original_config              = config
        self.original_categorical_columns = list(config.columns.cat_cols)
        self.original_continuous_columns  = list(config.columns.cont_cols)
    
        self.protected = ["target_days_to_payment", "delay_is_known", "delay_clipped"]
 
    def remove_feature(self, feat: str, dataframe: pd.DataFrame = None, config = None):

        if dataframe is None or config is None:
            return None, "invalid_input", config

        working_dataframe = dataframe.copy()
        working_cfg       = copy.deepcopy(config)

        if feat in self.protected:
            return working_dataframe, "protected", working_cfg

        cat  = list(working_cfg.columns.cat_cols)
        cont = list(working_cfg.columns.cont_cols)

        if feat in cat:
            working_dataframe = working_dataframe.drop(columns=[feat])
            cat.remove(feat)
            working_cfg.columns.cat_cols = cat
            return working_dataframe, "categorical", working_cfg

        if feat in cont:
            working_dataframe = working_dataframe.drop(columns=[feat])
            cont.remove(feat)
            working_cfg.columns.cont_cols = cont
            return working_dataframe, "continuous", working_cfg

        return None, "not_found", config
    
    def train(self, dataset_loader, config):
        train_loader, validation_loader, _ = dataset_loader.run()
        target_scaler       = dataset_loader.target_scalers[config.columns.target_col_name]
        continuous_scalers  = dataset_loader.continuous_scalers 

        model = Model(
            embedding_dimensions = dataset_loader.embedding_dimensions,             
            num_continuous = len(dataset_loader.continuous_columns), 
            target_scaler = target_scaler,
            feature_scaler = continuous_scalers,
            config = config
        )    

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            target_scaler=target_scaler,
            logger = self.logger,
            config=config
        )
        
        trained_model = trainer.fit()

        evaluator = Results(
            model=trained_model,
            device=trainer.device
        )

        val_results   = evaluator.run(validation_loader, make_plots=False)
        val_metrics   = val_results['metrics']
   
        return val_metrics

    def run(self, metric):
        results = {}

        self.original_config.columns.cat_cols         = list(self.original_categorical_columns)
        self.original_config.columns.cont_cols        = list(self.original_continuous_columns)

        dataset_loader = DatasetLoader(dataframe=self.original_dataframe, cfg=self.original_config, logger=self.logger)
        baseline_metrics = self.train(dataset_loader, self.original_config)
        
        features = self.original_categorical_columns + self.original_continuous_columns
        
        for feature in tqdm(features, desc="Ablating features", ncols=100):
            short_dataframe, feature_type, short_config = self.remove_feature(feature, dataframe=self.original_dataframe, config=self.original_config)
            
            if short_dataframe is None:
                if self.logger:
                    self.logger.warning(f"Feature '{feature}' not found in either categorical or continuous columns. Skipping.")
                continue

            short_dataset_loader = DatasetLoader(dataframe=short_dataframe, cfg=short_config, logger=self.logger)
            short_metrics = self.train(short_dataset_loader, short_config)

            delta = baseline_metrics[metric] - short_metrics[metric]

            results[feature] = {
                "feature_type": feature_type,
                "val_metrics": short_metrics,
                "delta_metric": delta
            }

        return results


