"""
Test 01: Configuration Validation
Tests that all configuration parameters are properly defined and have valid values.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import config, Config, PathsDetails, DataParams, ModelParams


class TestConfigStructure:
    """Test configuration structure and defaults."""
    
    def test_config_instance_exists(self):
        """Verify config singleton is properly instantiated."""
        assert config is not None
        assert isinstance(config, Config)
    
    def test_paths_config(self):
        """Verify paths configuration."""
        assert hasattr(config, 'paths')
        assert isinstance(config.paths, PathsDetails)
        assert config.paths.raw_data.endswith('.parquet')
        assert config.paths.train_data.endswith('.parquet')
        assert config.paths.model_save.endswith('.pth')
    
    def test_data_params(self):
        """Verify data parameters are valid."""
        assert hasattr(config, 'data')
        assert isinstance(config.data, DataParams)
        
        # Test size constraints
        assert 0 < config.data.test_size < 1
        assert 0 < config.data.val_size < 1
        assert config.data.test_size + config.data.val_size < 1
        
        # Delay thresholds should be ordered
        assert config.data.delay_threshold_1 < config.data.delay_threshold_2
        assert config.data.delay_threshold_2 < config.data.delay_threshold_3
        
        # Minimum sequence length
        assert config.data.min_sequence_length >= 1
    
    def test_model_params(self):
        """Verify model parameters are valid."""
        assert hasattr(config, 'model')
        assert isinstance(config.model, ModelParams)
        
        # Batch size
        assert config.model.batch_size > 0
        
        # Hidden dimension should be divisible by heads
        assert config.model.hidden_dim % config.model.n_heads == 0
        
        # Learning rate
        assert 0 < config.model.lr < 1
        assert config.model.min_lr < config.model.lr
        
        # Dropout range
        assert 0 <= config.model.dropout < 1
        
        # Epochs and patience
        assert config.model.epochs > 0
        assert config.model.patience > 0
        assert config.model.patience < config.model.epochs
    
    def test_columns_config(self):
        """Verify columns configuration."""
        assert hasattr(config, 'columns')
        
        # Target columns should exist
        assert len(config.columns.target_cols) == 3
        assert 'target_short' in config.columns.target_cols
        assert 'target_medium' in config.columns.target_cols
        assert 'target_long' in config.columns.target_cols
        
        # Categorical and continuous columns should be defined
        assert len(config.columns.cat_cols) > 0
        assert len(config.columns.cont_cols) > 0
    
    def test_loss_params(self):
        """Verify loss parameters."""
        assert hasattr(config, 'loss')
        
        # Asymmetric loss params
        assert config.loss.asymmetric_gamma_negative >= 0
        assert config.loss.asymmetric_gamma_positive >= 0
        assert 0 <= config.loss.asymmetric_clip < 1
        
        # Label smoothing
        assert 0 <= config.loss.label_smoothing < 0.5
        
        # Temperature
        assert config.loss.temperature_init > 0
    
    def test_temporal_params(self):
        """Verify temporal feature parameters."""
        assert hasattr(config, 'temporal')
        
        assert config.temporal.days_in_week == 7
        assert config.temporal.days_in_month == 31
        assert config.temporal.months_in_year == 12
        assert 1 <= config.temporal.month_start_threshold <= config.temporal.days_in_month
        assert 1 <= config.temporal.month_end_threshold <= config.temporal.days_in_month
    
    def test_augmentation_params(self):
        """Verify augmentation parameters."""
        assert hasattr(config, 'augmentation')
        
        assert 0 <= config.augmentation.temporal_cutout_ratio < 1
        assert 0 <= config.augmentation.feature_dropout_ratio < 1
        assert config.augmentation.gaussian_noise_std > 0


class TestConfigConsistency:
    """Test configuration consistency across components."""
    
    def test_worker_settings_consistency(self):
        """Verify worker settings are consistent."""
        if config.model.num_workers > 0:
            assert config.model.pin_memory == True
            assert config.model.persistent_workers == True
        else:
            assert config.model.prefetch_factor == 0
    
    def test_head_dropout_multipliers(self):
        """Verify head dropout multipliers are reasonable."""
        base_dropout = config.model.dropout
        medium_dropout = base_dropout * config.model.head_dropout_multiplier_medium
        long_dropout = base_dropout * config.model.head_dropout_multiplier_long
        
        # Should be increasing but still < 1
        assert medium_dropout > base_dropout or base_dropout == 0
        assert long_dropout >= medium_dropout or base_dropout == 0
        assert long_dropout < 1
