import pandas as pd
import numpy as np
import torch
import logging
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupShuffleSplit

from core.config import config
from core.logger import Logger


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
    def __init__(self, categorical_data, continuous_data, targets, indices, augment=False, mask_last_cont=None, last_known_idx=None):
        self.categorical_data = torch.as_tensor(categorical_data, dtype=torch.long)
        self.continuous_data = torch.as_tensor(continuous_data, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)
        self.indices = indices
        self.augment = config.model.use_augmentation and augment
        self.augment_probability = config.model.augment_prob
        self.augmenter = Augmentation()
        self.logger = Logger(name="SequentialDataset", level=logging.INFO)

        self.mask_last_cont = mask_last_cont or []  
        self.last_known_idx = last_known_idx 
        
        self.logger.section("Data Augmentation Configuration")
        if self.augment:
            self.logger.info(f"[Augmentation] Status: ENABLED")
            self.logger.info(f"[Augmentation] Probability: {self.augment_probability:.1%}")
            self.logger.info(f"[Augmentation] Methods: Temporal Cutout, Feature Dropout, Gaussian Noise")

    def __len__(self):
        return len(self.indices)
    
    def mask_target(self, continuous_features):
        last_t = continuous_features.shape[0] - 1
        if last_t >= 0:
            for j in self.mask_last_cont:
                continuous_features[last_t, j] = 0.0

            if self.last_known_idx is not None:
                continuous_features[last_t, self.last_known_idx] = 0.0  
        
        return continuous_features

    def __getitem__(self, index):
        start_index, end_index, target_index = self.indices[index]
        
        categorical_features = self.categorical_data[start_index:end_index].clone()
        continuous_features  = self.continuous_data[start_index:end_index].clone()
        target = self.targets[target_index]
        
        self.mask_target(continuous_features)

        if self.augment and self.training_mode:
            categorical_features, continuous_features = self.augmenter.feature_dropout(categorical_features, continuous_features, probability=self.augment_probability)
            continuous_features = self.augmenter.gaussian_noise(continuous_features, probability=self.augment_probability, standard_deviation=config.augmentation.gaussian_noise_std)
        
        self.mask_target(continuous_features)

        length = categorical_features.shape[0]
        return categorical_features, continuous_features, target, length
    
    @property
    def training_mode(self):
        return self.augment


class DatasetLoader:
    def __init__(
        self, 
        data_path,    
    ):
        self.data_path = data_path
        self.logger = Logger(name="DatasetLoader", level=logging.INFO)

        self.categorical_columns = config.columns.cat_cols
        self.continuous_columns = config.columns.cont_cols
        self.target_columns = config.columns.target_cols
        self.random_state = config.data.random_state

        self.label_encoders = {}
        self.continuous_scalers = {}
        self.target_scalers = {} 
        self.embedding_dimensions = []


    def dataloader_pipeline(self):
        self.logger.section("Dataloader Pipeline")
        
        self.logger.subsection("Data Loading")
        dataframe = self._load_data()
        
        self.logger.subsection("Data Cleaning")
        dataframe = self._clean_data(dataframe)
        
        self.logger.subsection("Categorical Encoding")
        dataframe = self._encode_categorical(dataframe)
       
        self.logger.subsection("Sampling targets")
        dataframe, unique_users = self._apply_sampling(dataframe)
        
        self.logger.subsection("User Subsampling")
        train_users, validation_users, test_users = self._split_users(unique_users)
        
        self.logger.subsection("Dataframe Splitting")
        train_df, val_df, test_df = self._split_dataframes(dataframe, train_users, validation_users, test_users)
        
        self.logger.subsection("Feature Normalization")
        train_df, val_df, test_df = self._normalize_continuous_features(train_df, val_df, test_df)
        
        self.logger.subsection("Target Normalization")
        train_df, val_df, test_df = self._normalize_targets(train_df, val_df, test_df)
        
        self.logger.subsection("Index Creation")
        indices, (train_df, val_df, test_df) = self._create_indices(train_df, val_df, test_df)
        
        self.logger.subsection("Array Creation")
        continuous_array, categorical_array, targets_array  = self._create_arrays(train_df, val_df, test_df)
        
        self.logger.subsection("Dataset Creation")
        train_dataset, val_dataset, test_dataset = self._create_datasets(
            indices=indices,
            continuous=continuous_array,
            categorical=categorical_array,
            targets=targets_array,
        )

        self.logger.subsection("Dataloader Creation")
        train_dataloader, val_dataloader, test_dataloader = self.create_dataloaders(train_dataset, val_dataset, test_dataset)

        self.logger.section("Dataloader Pipeline Complete")
        return train_dataloader, val_dataloader, test_dataloader


    def _load_data(self):
        self.logger.info(f"[Data Loading] Loading sequential data from: {self.data_path}")
        dataframe = pd.read_parquet(self.data_path)
        self.logger.info(f"[Data Loading] Loaded {len(dataframe):,} rows, {len(dataframe.columns)} columns")
        dataframe = dataframe.sample(frac=config.data.load_sample_frac, random_state=self.random_state)
        self.logger.info(f"[Data Loading] Sampled {config.data.load_sample_frac:.1%} of data: {len(dataframe):,} rows\n")
        return dataframe


    def _clean_data(self, dataframe):
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
        self.continuous_columns  = [column for column in self.continuous_columns if column in dataframe.columns]
        
        self.logger.subsection("Feature Configuration")
        self.logger.info(f"[Clean] Categorical: {len(self.categorical_columns)} features")
        self.logger.info(f"[Clean] Continuous: {len(self.continuous_columns)} features")
        self.logger.info(f"[Clean] Targets: {len(self.target_columns)} ({', '.join(self.target_columns)}) \n")
        
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
            self.logger.info(f"[Categorical Encoding] Encoded column '{column}' with cardinality {cardinality} \n")
        
        self.logger.info(f"[Categorical Encoding] Encoded {len(self.categorical_columns)} categorical features \n")
        return dataframe
    

    def _apply_sampling(self, dataframe):
        user_col = config.columns.user_id_col
        target_col = config.columns.target_cols[0]  
        unique_users = dataframe[user_col].unique()
        
        np.random.seed(config.data.random_state)
        num_users = len(unique_users)
        num_sample = int(num_users)

        user_max_target = dataframe.groupby(user_col)[target_col].max()
        users_gt = user_max_target[user_max_target > 10].index.values
        users_le = user_max_target[user_max_target <= 10].index.values

        remaining_quota = num_sample - len(users_gt)
        
        if len(users_gt) >= num_sample:
            selected_users = np.random.choice(users_gt, size=num_sample, replace=False)
        elif remaining_quota > 0:
            sampled_le = np.random.choice(users_le, size=min(remaining_quota, len(users_le)), replace=False)
            selected_users = np.concatenate([users_gt, sampled_le])
        else:
            selected_users = users_gt

        self.logger.info(f"[Subsampling] All users with target > 10 days: {len(users_gt):,}")
        self.logger.info(f"[Subsampling] Users randomly selected with target <= 10 days: {len(selected_users) - len(users_gt):,}")
        self.logger.info(f"[Subsampling] Total selected: {len(selected_users):,}/{num_users:,} users ({len(selected_users)/num_users:.1%}) \n")

        dataframe = dataframe[dataframe[user_col].isin(selected_users)]
        return dataframe, selected_users
    

    def _split_users(self, unique_users):
        user_col       = config.columns.user_id_col
        user_dataframe = pd.DataFrame({user_col: unique_users})
        
        test_size       = config.data.test_size
        validation_size = config.data.val_size
        
        group_shuffle_split_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state = config.data.random_state)
        train_validation_indices, test_indices = next(group_shuffle_split_test.split(user_dataframe, groups=user_dataframe[user_col]))
        
        train_validation_users = user_dataframe.iloc[train_validation_indices][user_col].values
        test_users = user_dataframe.iloc[test_indices][user_col].values
        
        validation_size_adjusted = validation_size / (1 - test_size)
        group_shuffle_split_validation = GroupShuffleSplit(n_splits=1, test_size=validation_size_adjusted, random_state = config.data.random_state)
        train_indices, validation_indices = next(group_shuffle_split_validation.split(pd.DataFrame({user_col: train_validation_users}), groups=train_validation_users))
        
        train_users      = set(train_validation_users[train_indices])
        validation_users = set(train_validation_users[validation_indices])
        test_users       = set(test_users)
        
        total_users = len(train_users) + len(validation_users) + len(test_users)
        self.logger.info(f"[Split] Train: {len(train_users):,} users ({len(train_users)/total_users:.1%})")
        self.logger.info(f"[Split] Validation: {len(validation_users):,} users ({len(validation_users)/total_users:.1%})")
        self.logger.info(f"[Split] Test: {len(test_users):,} users ({len(test_users)/total_users:.1%}) \n")
        
        return train_users, validation_users, test_users
    

    def _split_dataframes(self, dataframe, train_users, validation_users, test_users):
        user_col  = config.columns.user_id_col
     
        train_dataframe      = dataframe[dataframe[user_col].isin(train_users)].copy()
        validation_dataframe = dataframe[dataframe[user_col].isin(validation_users)].copy()
        test_dataframe       = dataframe[dataframe[user_col].isin(test_users)].copy()

        self.logger.info(f"[Dataframe Split] Training set: {train_dataframe.shape}")
        self.logger.info(f"[Dataframe Split] Validation set: {validation_dataframe.shape}")
        self.logger.info(f"[Dataframe Split] Test set: {test_dataframe.shape} \n")
        
        return train_dataframe, validation_dataframe, test_dataframe
    

    def _normalize_continuous_features(self, train_dataframe, validation_dataframe, test_dataframe):
        no_scale = {'delay_is_known'}
        for col in self.continuous_columns:
            if col in no_scale:
                train_dataframe[col]      = train_dataframe[col].astype(float)
                validation_dataframe[col] = validation_dataframe[col].astype(float)
                test_dataframe[col]       = test_dataframe[col].astype(float)
                continue
            
            self.logger.info(f"Feature before normalization - {col}: mean={train_dataframe[col].mean():.4f}, std={train_dataframe[col].std():.4f}")
            scaler = StandardScaler()
            train_values = train_dataframe[[col]].values
            scaler.fit(train_values)
            train_dataframe[col]         = scaler.transform(train_values)
            validation_dataframe[col]    = scaler.transform(validation_dataframe[[col]].values)
            test_dataframe[col]          = scaler.transform(test_dataframe[[col]].values)
            self.continuous_scalers[col] = scaler
            self.logger.info(f"Feature after normalization - {col}: mean={train_dataframe[col].mean():.4f}, std={train_dataframe[col].std():.4f} \n")
    
        self.logger.info(f"[Continuous Normalization] Normalized {len(self.continuous_columns)} continuous features \n")
        return train_dataframe, validation_dataframe, test_dataframe


    def _normalize_targets(self, train_dataframe, validation_dataframe, test_dataframe):
        for target_col in self.target_columns:
            self.logger.info(f"[Target Normalization] Target {target_col} mean before normalization: {train_dataframe[target_col].mean():.4f}, std: {train_dataframe[target_col].std():.4f}")
            scaler = StandardScaler()
            train_targets = np.log1p(np.maximum(train_dataframe[[target_col]].values, 0))
            scaler.fit(train_targets)
            train_dataframe[target_col] = scaler.transform(train_targets)
            validation_dataframe[target_col] = scaler.transform(np.log1p(np.maximum(validation_dataframe[[target_col]].values, 0)))
            test_dataframe[target_col] = scaler.transform(np.log1p(np.maximum(test_dataframe[[target_col]].values, 0)))
            self.target_scalers[target_col] = scaler
            self.logger.info(f"[Target Normalization] Target {target_col} mean after normalization: {train_dataframe[target_col].mean():.4f}, std: {train_dataframe[target_col].std():.4f}")

        self.logger.info(f"[Target Normalization] Normalized {len(self.target_columns)} target features \n")
        return train_dataframe, validation_dataframe, test_dataframe
        

    def _create_expanding_indices(self, dataframe, group_column):
        indices = []
        dataframe_reset = dataframe.reset_index(drop=True)
        group_offsets = dataframe_reset.groupby(group_column).indices

        min_start = config.model.min_seq_len - 1

        for _, group_indices in group_offsets.items():
            group_indices = np.sort(group_indices)
            num_invoices = len(group_indices)

            if num_invoices < config.model.min_seq_len:
                continue
            
            for invoice_index in range(min_start, num_invoices):
                target_index = group_indices[invoice_index]
                sequence_end = target_index + 1
                sequence_start = max(sequence_end - config.model.max_seq_len, group_indices[0])
                indices.append((sequence_start, sequence_end, target_index))

        return indices


    def _create_indices(self, train_dataframe, validation_dataframe, test_dataframe):
        group_cols = config.columns.group_cols
        sort_cols  = config.columns.sort_cols
    
        train_dataframe      = train_dataframe.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
        validation_dataframe = validation_dataframe.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
        test_dataframe       = test_dataframe.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    
        train_indices      = self._create_expanding_indices(train_dataframe, group_cols)
        validation_indices = self._create_expanding_indices(validation_dataframe, group_cols)
        test_indices       = self._create_expanding_indices(test_dataframe, group_cols)
    
        self.logger.info(f"[Indices] Created {len(train_indices):,} training sequences")
        self.logger.info(f"[Indices] Created {len(validation_indices):,} validation sequences")    
        self.logger.info(f"[Indices] Created {len(test_indices):,} test sequences \n")

        sequence_lengths_train = [end - start for (start, end, _) in train_indices]
        sequence_lengths_validation = [end - start for (start, end, _) in validation_indices]
        sequence_lengths_test = [end - start for (start, end, _) in test_indices]

        avg_length_train, avg_length_val, avg_length_test = np.mean(sequence_lengths_train), np.mean(sequence_lengths_validation), np.mean(sequence_lengths_test)
        std_length_train, std_length_val, std_length_test = np.std(sequence_lengths_train), np.std(sequence_lengths_validation), np.std(sequence_lengths_test)
        min_length_train, min_length_val, min_length_test = np.min(sequence_lengths_train), np.min(sequence_lengths_validation), np.min(sequence_lengths_test)
        max_length_train, max_length_val, max_length_test = np.max(sequence_lengths_train), np.max(sequence_lengths_validation), np.max(sequence_lengths_test)
        
        self.logger.info(f"[Indices] Training sequence lengths: mean={avg_length_train:.2f}, std={std_length_train:.2f} timesteps")
        self.logger.info(f"[Indices] Training sequence lengths: min={min_length_train} timesteps, max={max_length_train} timesteps \n")
        
        self.logger.info(f"[Indices] Validation sequence lengths: mean={avg_length_val:.2f}, std={std_length_val:.2f} timesteps")
        self.logger.info(f"[Indices] Validation sequence lengths: min={min_length_val} timesteps, max={max_length_val} timesteps \n")

        self.logger.info(f"[Indices] Test sequence lengths: mean={avg_length_test:.2f}, std={std_length_test:.2f} timesteps")
        self.logger.info(f"[Indices] Test sequence lengths: min={min_length_test} timesteps, max={max_length_test} timesteps \n")
        
        return (train_indices, validation_indices, test_indices), (train_dataframe, validation_dataframe, test_dataframe)


    def _create_arrays(self, train_dataframe, validation_dataframe, test_dataframe):
        train_continuous       = train_dataframe[self.continuous_columns].values
        validation_continuous  = validation_dataframe[self.continuous_columns].values
        test_continuous        = test_dataframe[self.continuous_columns].values

        self.logger.info(f"[Arrays] Extracted training features: continuous={train_continuous.shape}")
        self.logger.info(f"[Arrays] Extracted validation features: continuous={validation_continuous.shape}")
        self.logger.info(f"[Arrays] Extracted test features: continuous={test_continuous.shape} \n")
    
        continuous_array = train_continuous, validation_continuous, test_continuous

        train_categorical      = train_dataframe[self.categorical_columns].values
        validation_categorical = validation_dataframe[self.categorical_columns].values
        test_categorical       = test_dataframe[self.categorical_columns].values
     
        self.logger.info(f"[Arrays] Extracted training features: categorical={train_categorical.shape}")
        self.logger.info(f"[Arrays] Extracted validation features: categorical={validation_categorical.shape}")
        self.logger.info(f"[Arrays] Extracted test features: categorical={test_categorical.shape} \n")
    
        categorical_array = train_categorical, validation_categorical, test_categorical

        train_targets      = train_dataframe[self.target_columns].values
        validation_targets = validation_dataframe[self.target_columns].values
        test_targets       = test_dataframe[self.target_columns].values

        self.logger.info(f"[Arrays] Extracted training targets: {train_targets.shape}")
        self.logger.info(f"[Arrays] Extracted validation targets: {validation_targets.shape}")
        self.logger.info(f"[Arrays] Extracted test targets: {test_targets.shape} \n")

        targets_array = train_targets, validation_targets, test_targets

        return continuous_array, categorical_array, targets_array


    def _create_datasets(self, indices, continuous, categorical, targets):
        mask_idxs = []
        known_idx = None
       
        mask_idxs.append(self.continuous_columns.index('delay_clipped'))
        known_idx = self.continuous_columns.index('delay_is_known')

        train_indices,     validation_indices,     test_indices     = indices
        train_targets,     validation_targets,     test_targets     = targets
        train_continuous,  validation_continuous,  test_continuous  = continuous
        train_categorical, validation_categorical, test_categorical = categorical
        
        train_dataset      = SequentialDataset(train_categorical, train_continuous, train_targets, train_indices, augment=True,  mask_last_cont=mask_idxs, last_known_idx=known_idx)
        validation_dataset = SequentialDataset(validation_categorical, validation_continuous, validation_targets, validation_indices, augment=False,  mask_last_cont=mask_idxs, last_known_idx=known_idx)        
        test_dataset       = SequentialDataset(test_categorical, test_continuous, test_targets, test_indices, augment=False,  mask_last_cont=mask_idxs, last_known_idx=known_idx)
    
        self.logger.info(f"[Datasets] Created training dataset with {len(train_dataset):,} samples")
        self.logger.info(f"[Datasets] Created validation dataset with {len(validation_dataset):,} samples")
        self.logger.info(f"[Datasets] Created test dataset with {len(test_dataset):,} samples \n")
        
        return train_dataset, validation_dataset, test_dataset


    @staticmethod
    def collate_sequences(batch):
        categorical_list, continuous_list, target_list, lengths = zip(*batch)
        
        categorical_padded  = pad_sequence(categorical_list, batch_first=True, padding_value=0)
        continuous_padded   = pad_sequence(continuous_list, batch_first=True, padding_value=0.0)
        
        targets = torch.stack(target_list)
        lengths = torch.tensor(lengths, dtype=torch.long)
    
        return categorical_padded, continuous_padded, targets, lengths


    def create_dataloaders(self, train_dataset, validation_dataset, test_dataset):

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.model.batch_size,
            shuffle=True,
            num_workers=config.model.num_workers,
            pin_memory=config.model.pin_memory,
            persistent_workers=config.model.num_workers > 0,
            collate_fn=self.collate_sequences
        )

        validation_dataloader = DataLoader(
            validation_dataset,
            batch_size=config.model.batch_size,
            shuffle=False,
            num_workers=config.model.num_workers,
            pin_memory=config.model.pin_memory,
            persistent_workers=config.model.num_workers > 0,
            collate_fn=self.collate_sequences
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.model.batch_size,
            shuffle=False,
            num_workers=config.model.num_workers,
            pin_memory=config.model.pin_memory,
            persistent_workers=config.model.num_workers > 0,
            collate_fn=self.collate_sequences
        )

        self.logger.info(f"[Dataloaders] Created training dataloader with {len(train_dataloader):,} batches")
        self.logger.info(f"[Dataloaders] Created validation dataloader with {len(validation_dataloader):,} batches")
        self.logger.info(f"[Dataloaders] Created test dataloader with {len(test_dataloader):,} batches \n")
        return train_dataloader, validation_dataloader, test_dataloader
    
 