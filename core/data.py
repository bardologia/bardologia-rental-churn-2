import pandas as pd
import numpy as np


class Preprocessor:  
    def __init__(self, cfg, logger):
        self.config = cfg
        self.logger = logger
    
    def filter_category(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        category_col = self.config.columns.category_col
        category_filter = self.config.columns.category_filter

        before = len(dataframe)
        dataframe = dataframe[dataframe[category_col].astype(str).str.lower().str.contains(category_filter)]
        after = len(dataframe)
        reduction_pct = (before - after) / before * 100 if before > 0 else 0
        self.logger.info(f"[Category Filter] Filtered by '{category_filter}': {reduction_pct:.1f}%) \n")
        return dataframe

    def drop_columns(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        keep_cols = self.config.columns.keep_cols
        keep    = [col for col in keep_cols if col in dataframe.columns]
        removed = [col for col in dataframe.columns if col not in keep]
        dataframe = dataframe[keep].copy()
        self.logger.info(f"[Column Pruning] Kept {len(keep)} columns, removed {len(removed)} columns \n")
        return dataframe
    
    def process_dates(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        due_date_col = self.config.columns.due_date_col
        date_cols_processed = [col for col in self.config.columns.date_cols if col in dataframe.columns]
        
        for column in date_cols_processed:
            dataframe[column] = pd.to_datetime(dataframe[column], errors='coerce', utc=True)
        
        now_utc = pd.Timestamp.now(tz='UTC')
        dataframe = dataframe[dataframe[due_date_col] <= now_utc].reset_index(drop=True)    
        self.logger.info(f"[Date Processing] Parsed {len(date_cols_processed)} date columns, filtered future dates \n")
        return dataframe

    def run(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        self.logger.section("Preprocessing")
        
        self.logger.subsection("Dropping unnecessary columns")
        dataframe = self.drop_columns(dataframe)

        self.logger.subsection("Filtering by category")
        dataframe = self.filter_category(dataframe)
        
        self.logger.subsection("Processing dates")
        dataframe = self.process_dates(dataframe)   
        return dataframe


class FeatureEngineer:
    def __init__(self, cfg, logger):
        self.config = cfg
        self.logger = logger
    
    def create_temporal_features(self, dataframe: pd.DataFrame) -> pd.DataFrame: 
        due_date_col = self.config.columns.due_date_col
        weekend_start_day = self.config.temporal.weekend_start_day
        days_in_week = self.config.temporal.days_in_week
        months_in_year = self.config.temporal.months_in_year
        
        d = dataframe[due_date_col]
        days_in_month = d.dt.daysinmonth

        dataframe['venc_dayofweek']      = dataframe[due_date_col].dt.dayofweek
        dataframe['venc_day']            = dataframe[due_date_col].dt.day
        dataframe['venc_month']          = dataframe[due_date_col].dt.month
        dataframe['venc_quarter']        = dataframe[due_date_col].dt.quarter
        dataframe['venc_is_weekend']     = (dataframe['venc_dayofweek'] >= weekend_start_day).astype(int)
        dataframe['venc_is_month_start'] = d.dt.is_month_start.astype(int)
        dataframe['venc_is_month_end']   = d.dt.is_month_end.astype(int)
     
        dataframe['venc_dayofweek_sin'] = np.sin(2 * np.pi * dataframe['venc_dayofweek'] / days_in_week)
        dataframe['venc_dayofweek_cos'] = np.cos(2 * np.pi * dataframe['venc_dayofweek'] / days_in_week)
        dataframe['venc_day_sin']       = np.sin(2 * np.pi * dataframe['venc_day'] / days_in_month)
        dataframe['venc_day_cos']       = np.cos(2 * np.pi * dataframe['venc_day'] / days_in_month)
        dataframe['venc_month_sin']     = np.sin(2 * np.pi * dataframe['venc_month'] / months_in_year)
        dataframe['venc_month_cos']     = np.cos(2 * np.pi * dataframe['venc_month'] / months_in_year)
        
        self.logger.info(f"[Temporal Features] Extracted 13 features \n")
        return dataframe

    def create_history_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        user_col = self.config.columns.user_id_col
        delay_col = self.config.columns.delay_col
        sort_cols = self.config.columns.sort_cols

        def process_user_history(user_df):
            user_df = user_df.sort_values(sort_cols).copy()
            user_df['hist_mean_delay'] = user_df[delay_col].expanding().mean().shift(1).fillna(0).clip(lower=0)
            user_df['hist_std_delay']  = user_df[delay_col].expanding().std().shift(1).fillna(0).clip(lower=0)
            user_df['hist_max_delay']  = user_df[delay_col].expanding().max().shift(1).fillna(0).clip(lower=0)
            user_df['last_delay']      = user_df[delay_col].shift(1).fillna(0).clip(lower=0)
            user_df['delay_trend']     = user_df[delay_col].diff().shift(1).fillna(0)
            return user_df

        dataframe = (dataframe
            .sort_values(sort_cols)
            .groupby(user_col, group_keys=False)
            .apply(process_user_history)
            .reset_index(drop=True)
        )
        
        self.logger.info(f"[History Features] Extracted 5 features \n")
        return dataframe

    def create_sequence_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        user_col     = self.config.columns.user_id_col
        contract_col = self.config.columns.contract_id_col
        order_col    = self.config.columns.order_col
        sort_cols    = self.config.columns.sort_cols
        delay_col    = self.config.columns.delay_col
        due_date_col = self.config.columns.due_date_col
        
        dataframe = dataframe.sort_values(sort_cols).reset_index(drop=True)
        
        dataframe['seq_position']            = dataframe.groupby(user_col).cumcount()
        dataframe['seq_total']               = dataframe.groupby(user_col)[user_col].transform(lambda s: s.expanding().count())
        dataframe['seq_position_norm']       = dataframe['seq_position'] / dataframe['seq_total'].clip(lower=1)
        dataframe['is_first_invoice']        = (dataframe['seq_position'] == 0).astype(int)
            
        dataframe['days_since_last_invoice'] = dataframe.groupby(user_col)[due_date_col].diff().dt.days.fillna(0)
    
        for window_size in self.config.target.rolling_window_sizes:
            dataframe[f'rolling_mean_delay_{window_size}'] = dataframe.groupby(user_col)[delay_col].transform(lambda values: values.rolling(window_size, min_periods=1).mean().shift(1)).fillna(0).clip(lower=0)
            dataframe[f'rolling_max_delay_{window_size}'] = dataframe.groupby(user_col)[delay_col].transform(lambda values: values.rolling(window_size, min_periods=1).max().shift(1)).fillna(0).clip(lower=0)

        dataframe['is_improving']            = (dataframe['delay_trend'] < 0).astype(int) if 'delay_trend' in dataframe.columns else 0
        dataframe['parcela_position']        = dataframe.groupby([user_col, contract_col]).cumcount()

        dataframe['parcela_total']           = dataframe.groupby([user_col, contract_col])[order_col].transform(lambda s: s.expanding().count())
        dataframe['parcela_position_norm']   = dataframe['parcela_position'] / dataframe['parcela_total'].clip(lower=1)
        
        first_in_contract = dataframe.groupby([user_col, contract_col]).cumcount().eq(0).astype(int)
        dataframe['num_contracts']          = first_in_contract.groupby(dataframe[user_col]).cumsum()
        dataframe['is_first_contract']      = (dataframe['num_contracts'] == 1).astype(int)
        dataframe['contract_mean_delay']    = dataframe.groupby([user_col, contract_col])[delay_col].transform(lambda values: values.expanding().mean().shift(1)).fillna(0).clip(lower=0)
        
        self.logger.info(f"[Sequence Features] Extracted 12 features \n")
        return dataframe

    def create_value_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        user_col = self.config.columns.user_id_col
        billed_value_col = self.config.columns.billed_value_col
        paid_value_col = self.config.columns.paid_value_col
        sort_cols = self.config.columns.sort_cols
        due_date_col = self.config.columns.due_date_col
        creation_date_col = self.config.columns.creation_date_col

        dataframe['grace_period'] = (dataframe[due_date_col] - dataframe[creation_date_col]).dt.days 
        
        dataframe = dataframe.sort_values(sort_cols).reset_index(drop=True)
        
        dataframe['total_paid']      = (dataframe.groupby(user_col)[paid_value_col].transform(lambda s: s.expanding().sum().shift(1)).fillna(0))
  
        dataframe['total_billed']    = (dataframe.groupby(user_col)[billed_value_col].transform(lambda s: s.expanding().sum()).fillna(0))
        dataframe['hist_mean_value'] = dataframe.groupby(user_col)[billed_value_col].transform(lambda values: values.expanding().mean().shift(1)).clip(lower=0)
        dataframe['value_ratio']     = dataframe[billed_value_col] / dataframe['hist_mean_value']
                
        self.logger.info("[Value Features] Extracted 5 features \n")
        return dataframe

    def create_target(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        delay_col = self.config.columns.delay_col
        target_col = self.config.columns.target_col_name
        dataframe[self.config.columns.delay_clipped_col] = dataframe[delay_col].clip(lower=0)
        dataframe[self.config.columns.delay_is_known_col] = self.config.target.delay_known_value
        dataframe[self.config.columns.target_col_name] = dataframe[self.config.columns.delay_clipped_col]
        self.logger.info(f"[Target Creation] Created target column: {target_col} from {delay_col} \n")
        return dataframe

    def run(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        self.logger.section("Feature Engineering")
        
        self.logger.subsection("Creating temporal features")
        dataframe = self.create_temporal_features(dataframe)
        
        self.logger.subsection("Creating history features")
        dataframe = self.create_history_features(dataframe)
        
        self.logger.subsection("Creating sequence features")
        dataframe = self.create_sequence_features(dataframe)
        
        self.logger.subsection("Creating value features")
        dataframe = self.create_value_features(dataframe)
        
        self.logger.subsection("Creating target features")
        dataframe = self.create_target(dataframe)
        return dataframe


class DataPipeline:
    def __init__(self, cfg, logger):
        self.config = cfg
        self.logger = logger
        
        self.preprocessor     = Preprocessor(cfg, logger)
        self.feature_engineer = FeatureEngineer(cfg, logger)
    
    def _load_data(self, input_path: str) -> pd.DataFrame:
        dataframe = pd.read_parquet(input_path)
        dataframe = dataframe.sample(frac=self.config.load.sample_fraction, random_state=self.config.load.random_state)
        self.logger.info(f"[Data Loading] Loaded {len(dataframe):,} rows from {input_path.split('/')[-1]} (sample_fraction={self.config.load.sample_fraction}) \n")
        return dataframe
    
    def _finalize_data(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        required_cols = set(
            self.config.columns.cat_cols +
            self.config.columns.cont_cols +
            self.config.columns.target_cols +
            self.config.columns.group_cols +
            self.config.columns.sort_cols +
            [self.config.columns.order_col]
        )
        columns = [col for col in dataframe.columns if col in required_cols]

        result_dataframe = dataframe[columns].copy()
        numeric_columns = result_dataframe.select_dtypes(include=[np.number]).columns
        result_dataframe[numeric_columns] = result_dataframe[numeric_columns].replace([np.inf, -np.inf], 0).fillna(0)
        
        self.logger.info(f"[Finalization] Final dataset has {len(result_dataframe):,} rows and {len(result_dataframe.columns)} columns \n")
        return result_dataframe
    
    def run(self, input_path: str, output_path: str) -> pd.DataFrame:
        self.logger.section("Feature Engineering Pipeline")
        
        dataframe = self._load_data(input_path)
        
        self.logger.subsection("Preprocessing")
        dataframe = self.preprocessor.run(dataframe)
        
        self.logger.subsection("Feature Extraction & Target Engineering")
        dataframe = self.feature_engineer.run(dataframe)
        
        self.logger.subsection("Finalization")
        dataframe = self._finalize_data(dataframe)
        
        return dataframe


