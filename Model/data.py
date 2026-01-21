import pandas as pd
import numpy as np
import torch
import logging
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from configs.config import config
from utils.logger import Logger


class Augmentation:
    @staticmethod
    def temporal_cutout(categorical_features: torch.Tensor, continuous_features: torch.Tensor, probability: float = 0.1) -> tuple:
        if torch.rand(1) > probability:
            return categorical_features, continuous_features
            
        sequence_length = categorical_features.shape[0]
        if sequence_length <= 1:
            return categorical_features, continuous_features
            
        num_cutout = max(1, int(sequence_length * config.augmentation.temporal_cutout_ratio))
        cutout_indices = torch.randperm(sequence_length)[:num_cutout]
        
        categorical_augmented = categorical_features.clone()
        continuous_augmented = continuous_features.clone()
        
        categorical_augmented[cutout_indices] = 0
        continuous_augmented[cutout_indices] = 0.0
        
        return categorical_augmented, continuous_augmented
    
    @staticmethod
    def feature_dropout(categorical_features: torch.Tensor, continuous_features: torch.Tensor, probability: float = 0.1) -> tuple:
        if torch.rand(1) > probability:
            return categorical_features, continuous_features
        
        categorical_augmented = categorical_features.clone()
        continuous_augmented = continuous_features.clone()
        
        if categorical_features.shape[1] > 0:
            num_categorical_drop = max(1, int(categorical_features.shape[1] * config.augmentation.feature_dropout_ratio))
            categorical_drop_indices = torch.randperm(categorical_features.shape[1])[:num_categorical_drop]
            categorical_augmented[:, categorical_drop_indices] = 0
        
        if continuous_features.shape[1] > 0:
            num_continuous_drop = max(1, int(continuous_features.shape[1] * config.augmentation.feature_dropout_ratio))
            continuous_drop_indices = torch.randperm(continuous_features.shape[1])[:num_continuous_drop]
            continuous_augmented[:, continuous_drop_indices] = 0.0
        
        return categorical_augmented, continuous_augmented
    
    @staticmethod
    def gaussian_noise(continuous_features: torch.Tensor, probability: float = 0.1, standard_deviation: float = 0.1) -> torch.Tensor:
        if torch.rand(1) > probability:
            return continuous_features
            
        noise = torch.randn_like(continuous_features) * standard_deviation
        return continuous_features + noise
    
    @staticmethod
    def time_warp(categorical_features: torch.Tensor, continuous_features: torch.Tensor, probability: float = 0.05) -> tuple:
        if torch.rand(1) > probability:
            return categorical_features, continuous_features
            
        sequence_length = categorical_features.shape[0]
        if sequence_length <= 2:
            return categorical_features, continuous_features
        
        if torch.rand(1) > 0.5:
            duplicate_index = torch.randint(0, sequence_length, (1,))[0]
            categorical_augmented = torch.cat([categorical_features[:duplicate_index+1], categorical_features[duplicate_index:]], dim=0)
            continuous_augmented = torch.cat([continuous_features[:duplicate_index+1], continuous_features[duplicate_index:]], dim=0)
        else:
            remove_index = torch.randint(0, sequence_length-1, (1,))[0]
            categorical_augmented = torch.cat([categorical_features[:remove_index], categorical_features[remove_index+1:]], dim=0)
            continuous_augmented = torch.cat([continuous_features[:remove_index], continuous_features[remove_index+1:]], dim=0)
        
        return categorical_augmented, continuous_augmented


class SequentialDataset(Dataset):
    def __init__(self, categorical_data, continuous_data, targets, indices, augment=False, augment_probability=0.1):
        self.categorical_data = torch.as_tensor(categorical_data, dtype=torch.long)
        self.continuous_data = torch.as_tensor(continuous_data, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)
        self.indices = indices
        self.augment = augment
        self.augment_probability = augment_probability
        self.augmenter = Augmentation()
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        start_index, end_index, target_index = self.indices[index]
        
        categorical_features = self.categorical_data[start_index:end_index]
        continuous_features = self.continuous_data[start_index:end_index]
        target = self.targets[target_index]
        
        if self.augment and self.training_mode:
            categorical_features, continuous_features = self.augmenter.temporal_cutout(categorical_features, continuous_features, probability=self.augment_probability)
            categorical_features, continuous_features = self.augmenter.feature_dropout(categorical_features, continuous_features, probability=self.augment_probability)
            continuous_features = self.augmenter.gaussian_noise(continuous_features, probability=self.augment_probability, standard_deviation=config.augmentation.gaussian_noise_std)
        
        length = categorical_features.shape[0]
        return categorical_features, continuous_features, target, length
    
    @property
    def training_mode(self):
        return self.augment


def collate_sequences(batch):
    categorical_list, continuous_list, target_list, lengths = zip(*batch)
    
    categorical_padded  = pad_sequence(categorical_list, batch_first=True, padding_value=0)
    continuous_padded = pad_sequence(continuous_list, batch_first=True, padding_value=0.0)
    
    targets = torch.stack(target_list)
    lengths = torch.tensor(lengths, dtype=torch.long)
    
    return categorical_padded, continuous_padded, targets, lengths


class DataModule:
    def __init__(
        self, 
        data_path, 
        batch_size=32,
        num_workers=4, 
        pin_memory=False, 
        logger=None,
        max_sequence_length=50,
        min_sequence_length=2
    ):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.max_sequence_length = max_sequence_length
        self.min_sequence_length = min_sequence_length
        
        if logger is None:
            self.logger = Logger(log_dir="logs/data_module", name="DataModule", enable_tensorboard=False)
        
        self.categorical_columns = config.columns.cat_cols
        self.continuous_columns = config.columns.cont_cols
        self.target_columns = config.columns.target_cols
        
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.embedding_dimensions = []
        
    def prepare_data(self):
        self.logger.section("Data Preparation Pipeline")
        
        dataframe = self._load_and_clean()
        dataframe = self._encode_categorical(dataframe)
        dataframe, unique_users = self._apply_stratified_sampling(dataframe)
        train_users, validation_users, test_users = self._split_users(unique_users)
        train_df, validation_df, test_df = self._split_dataframes(dataframe, train_users, validation_users, test_users)
        self._extract_features(train_df, validation_df, test_df)
        self._compute_class_weights()
        self._create_datasets()
    
    def _load_and_clean(self):
        self.logger.info(f"[Data Loading] Loading sequential data from: {self.data_path}")
        
        dataframe = pd.read_parquet(self.data_path)
        self.logger.info(f"[Data Loading] Loaded {len(dataframe):,} rows, {len(dataframe.columns)} columns")
        
        payment_date_col = config.columns.payment_date_col
        user_col = config.columns.user_id_col
        
        if payment_date_col in dataframe.columns:
            if not pd.api.types.is_datetime64_any_dtype(dataframe[payment_date_col]):
                dataframe[payment_date_col] = pd.to_datetime(dataframe[payment_date_col], errors='coerce')
                
            bad_users = dataframe.loc[dataframe[payment_date_col].isna(), user_col].unique()
            if len(bad_users) > 0:
                self.logger.warning(f"[Data Cleaning] Removing {len(bad_users):,} users with missing payment dates (NaT)")
                dataframe = dataframe[~dataframe[user_col].isin(bad_users)]

        self.categorical_columns = [column for column in self.categorical_columns if column in dataframe.columns]
        self.continuous_columns = [column for column in self.continuous_columns if column in dataframe.columns]
        
        self.logger.subsection("Feature Configuration")
        self.logger.info(f"[Features] Categorical: {len(self.categorical_columns)} features")
        self.logger.info(f"[Features] Continuous: {len(self.continuous_columns)} features")
        self.logger.info(f"[Features] Targets: {len(self.target_columns)} ({', '.join(self.target_columns)})")
        
        dataframe = dataframe.dropna(subset=self.target_columns)
        
        return dataframe
    
    def _encode_categorical(self, dataframe):
        for column in self.categorical_columns:
            dataframe[column] = dataframe[column].astype(str)
            
        self.embedding_dimensions = []
        for column in self.categorical_columns:
            label_encoder = LabelEncoder()
            dataframe[column] = label_encoder.fit_transform(dataframe[column]) + 1
            self.label_encoders[column] = label_encoder
            cardinality = len(label_encoder.classes_)
            self.embedding_dimensions.append(cardinality)
        
        user_col = config.columns.user_id_col
        contract_col = config.columns.contract_id_col
        order_col = config.columns.order_col
        due_date_col = config.columns.due_date_col
        
        sort_columns = [user_col, contract_col, order_col] if order_col in dataframe.columns else [user_col, contract_col, due_date_col]
        dataframe = dataframe.sort_values(sort_columns).reset_index(drop=True)
        
        return dataframe
    
    def _apply_stratified_sampling(self, dataframe):
        user_col = config.columns.user_id_col
        unique_users = dataframe[user_col].unique()
        
        if config.data.sample_frac >= 1.0:
            return dataframe, unique_users
        
        np.random.seed(config.data.random_state)
        num_users = len(unique_users)
        num_sample = int(num_users * config.data.sample_frac)
        
        target_long_col = self.target_columns[2]
        target_medium_col = self.target_columns[1] 
        target_short_col = self.target_columns[0]  
        
        user_targets = dataframe.groupby(user_col)[self.target_columns].max().reset_index()
        users_with_long = user_targets[user_targets[target_long_col] == 1][user_col].values
        
        users_with_medium_only = user_targets[
            (user_targets[target_medium_col] == 1) & (user_targets[target_long_col] == 0)
        ][user_col].values
        
        users_with_short_only = user_targets[
            (user_targets[target_short_col] == 1) & (user_targets[target_medium_col] == 0)
        ][user_col].values
        
        users_no_default = user_targets[
            user_targets[target_short_col] == 0
        ][user_col].values
        
        self.logger.info(f"[Stratified Sampling] Users with target_long (>14d): {len(users_with_long):,}")
        self.logger.info(f"[Stratified Sampling] Users with target_medium only (7-14d): {len(users_with_medium_only):,}")
        self.logger.info(f"[Stratified Sampling] Users with target_short only (3-7d): {len(users_with_short_only):,}")
        self.logger.info(f"[Stratified Sampling] Users with no defaults (<=3d): {len(users_no_default):,}")
        
        selected_users = list(users_with_long) + list(users_with_medium_only)
        remaining_quota = num_sample - len(selected_users)
        
        if remaining_quota > 0:
            majority_users = np.concatenate([users_with_short_only, users_no_default])
            
            if len(majority_users) > remaining_quota:
                sampled_majority = np.random.choice(majority_users, size=remaining_quota, replace=False)
                selected_users.extend(sampled_majority)
            else:
                selected_users.extend(majority_users)
        
        unique_users = np.array(selected_users)
        
        self.logger.info(f"[Subsampling] Stratified sampling: {len(unique_users):,}/{num_users:,} users ({len(unique_users)/num_users:.1%})")
        self.logger.info(f"[Subsampling] Target quota was {num_sample:,}, selected {len(unique_users):,} (prioritizing minority classes)")
        
        dataframe = dataframe[dataframe[user_col].isin(unique_users)]
        
        return dataframe, unique_users
    
    def _split_users(self, unique_users):
        user_col = config.columns.user_id_col
        user_dataframe = pd.DataFrame({user_col: unique_users})
        
        test_size = config.data.test_size
        validation_size = config.data.val_size
        
        group_shuffle_split_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=config.data.random_state)
        train_validation_indices, test_indices = next(group_shuffle_split_test.split(user_dataframe, groups=user_dataframe[user_col]))
        
        train_validation_users = user_dataframe.iloc[train_validation_indices][user_col].values
        test_users = user_dataframe.iloc[test_indices][user_col].values
        
        validation_size_adjusted = validation_size / (1 - test_size)
        group_shuffle_split_validation = GroupShuffleSplit(n_splits=1, test_size=validation_size_adjusted, random_state=config.data.random_state)
        train_indices, validation_indices = next(group_shuffle_split_validation.split(
            pd.DataFrame({user_col: train_validation_users}), 
            groups=train_validation_users
        ))
        
        train_users = set(train_validation_users[train_indices])
        validation_users = set(train_validation_users[validation_indices])
        test_users = set(test_users)
        
        self.logger.subsection("Dataset Splitting")
        total_users = len(train_users) + len(validation_users) + len(test_users)
        self.logger.info(f"[Split] Train: {len(train_users):,} users ({len(train_users)/total_users:.1%})")
        self.logger.info(f"[Split] Validation: {len(validation_users):,} users ({len(validation_users)/total_users:.1%})")
        self.logger.info(f"[Split] Test: {len(test_users):,} users ({len(test_users)/total_users:.1%})")
        
        return train_users, validation_users, test_users
    
    def _split_dataframes(self, dataframe, train_users, validation_users, test_users):
        user_col = config.columns.user_id_col
        delay_col = config.columns.delay_col
        
        train_dataframe = dataframe[dataframe[user_col].isin(train_users)].copy()
        validation_dataframe = dataframe[dataframe[user_col].isin(validation_users)].copy()
        test_dataframe = dataframe[dataframe[user_col].isin(test_users)].copy()
        
        if self.continuous_columns:
            self.continuous_columns = [column for column in self.continuous_columns if column != delay_col]

            train_dataframe[self.continuous_columns] = self.scaler.fit_transform(train_dataframe[self.continuous_columns])
            validation_dataframe[self.continuous_columns] = self.scaler.transform(validation_dataframe[self.continuous_columns])
            test_dataframe[self.continuous_columns] = self.scaler.transform(test_dataframe[self.continuous_columns])
        
        return train_dataframe, validation_dataframe, test_dataframe
    
    def _extract_features(self, train_dataframe, validation_dataframe, test_dataframe):
        group_cols = config.columns.group_cols
        
        self.train_indices = self._create_expanding_indices(train_dataframe, group_cols)
        self.validation_indices = self._create_expanding_indices(validation_dataframe, group_cols)
        self.test_indices = self._create_expanding_indices(test_dataframe, group_cols)

        self.train_categorical = train_dataframe[self.categorical_columns].values
        self.train_continuous = train_dataframe[self.continuous_columns].values
        self.train_targets = train_dataframe[self.target_columns].values

        self.validation_categorical = validation_dataframe[self.categorical_columns].values
        self.validation_continuous = validation_dataframe[self.continuous_columns].values
        self.validation_targets = validation_dataframe[self.target_columns].values

        self.test_categorical = test_dataframe[self.categorical_columns].values
        self.test_continuous = test_dataframe[self.continuous_columns].values
        self.test_targets = test_dataframe[self.target_columns].values
        
        self.logger.subsection("Expanding Window Sequences")
        total_samples = len(self.train_indices) + len(self.validation_indices) + len(self.test_indices)
        self.logger.info(f"[Sequences] Train: {len(self.train_indices):,} samples")
        self.logger.info(f"[Sequences] Validation: {len(self.validation_indices):,} samples")
        self.logger.info(f"[Sequences] Test: {len(self.test_indices):,} samples")
        self.logger.info(f"[Sequences] Total: {total_samples:,} samples (max_len={self.max_sequence_length}, min_len={self.min_sequence_length})")
    
    def _compute_class_weights(self):
        train_targets_values = np.array([self.train_targets[index[2]] for index in self.train_indices])
        
        train_targets_tensor = torch.from_numpy(train_targets_values).float()
        num_positive = torch.sum(train_targets_tensor, dim=0)
        total_samples = len(train_targets_tensor)
        num_negative = total_samples - num_positive
        
        self.positive_weight = num_negative / torch.clamp(num_positive, min=1.0)
        
        self.logger.subsection("Target Imbalance Analysis")
        imbalance_data = []
        for index, column in enumerate(self.target_columns):
            num_pos = int(num_positive[index].item())
            num_neg = int(num_negative[index].item())
            ratio = num_pos / total_samples
            weight = self.positive_weight[index].item()
            imbalance_data.append([column, f"{num_pos:,}", f"{ratio:.2%}", f"{num_neg:,}", f"{weight:.4f}"])
            
        self.logger.metrics_table(
            headers=["Target", "Positive", "Ratio", "Negative", "Recommended Weight"],
            rows=imbalance_data,
            title="Class Distribution per Target"
        )
        
        self.logger.info(f"[Weights] Final pos_weight tensor: {[f'{w:.4f}' for w in self.positive_weight.tolist()]}")
    
    def _create_datasets(self):
        use_augmentation = getattr(config.model, 'use_augmentation', False)
        augmentation_probability = getattr(config.model, 'augment_prob', 0.1)
        
        self.logger.subsection("Data Augmentation Configuration")
        if use_augmentation:
            self.logger.info(f"[Augmentation] Status: ENABLED")
            self.logger.info(f"[Augmentation] Probability: {augmentation_probability:.1%}")
            self.logger.info(f"[Augmentation] Methods: Temporal Cutout, Feature Dropout, Gaussian Noise")
        else:
            self.logger.info(f"[Augmentation] Status: DISABLED")
        
        self.train_dataset = SequentialDataset(
            self.train_categorical, self.train_continuous, self.train_targets, self.train_indices,
            augment=use_augmentation, augment_probability=augmentation_probability
        )
        self.validation_dataset = SequentialDataset(
            self.validation_categorical, self.validation_continuous, self.validation_targets, self.validation_indices,
            augment=False
        )
        self.test_dataset = SequentialDataset(
            self.test_categorical, self.test_continuous, self.test_targets, self.test_indices,
            augment=False
        )
    
    def _create_expanding_indices(self, dataframe, group_column):
        indices = []
        dataframe_reset = dataframe.reset_index(drop=True)
        
        group_offsets = dataframe_reset.groupby(group_column).indices
        
        for user_id, user_indices in group_offsets.items():
            user_indices = np.sort(user_indices)
            num_invoices = len(user_indices)
            if num_invoices < self.min_sequence_length:
                continue
            
            for invoice_index in range(num_invoices - 1):
                target_index = user_indices[invoice_index+1]
                sequence_end = target_index 
                sequence_start = sequence_end - self.max_sequence_length
                if sequence_start < user_indices[0]:
                    sequence_start = user_indices[0]
                           
                indices.append((sequence_start, sequence_end, target_index))
                
        return indices
     
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
            collate_fn=collate_sequences
        )
    
    def validation_dataloader(self):
        return DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
            collate_fn=collate_sequences
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
            collate_fn=collate_sequences
        )
