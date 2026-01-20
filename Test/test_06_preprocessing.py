"""
Test 06: Data Preprocessing Pipeline
Tests the Preprocessor and FeatureEngineer classes from core.py.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model.core import Preprocessor, FeatureEngineer
from Configs.config import config


class TestPreprocessor:
    """Test Preprocessor class."""
    
    @pytest.fixture
    def preprocessor(self, mock_logger):
        """Create a Preprocessor instance."""
        return Preprocessor(logger=mock_logger)
    
    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing."""
        n = 50
        base_date = datetime.now() - timedelta(days=180)
        
        df = pd.DataFrame({
            'categoria': ['aluguel'] * 30 + ['venda'] * 20,
            'vencimentoData': pd.to_datetime([
                base_date + timedelta(days=i) for i in range(n)
            ], utc=True),
            'pagamentoData': pd.to_datetime([
                base_date + timedelta(days=i + np.random.randint(0, 5)) for i in range(n)
            ]),
            'usuarioId': np.random.randint(1, 10, n),
            'valor': np.random.uniform(100, 1000, n),
            'emissaoNotaFiscalData': ['2024-01-01'] * n,  # Column to drop
        })
        return df
    
    def test_filter_category(self, preprocessor, sample_df):
        """Test category filtering."""
        filtered = preprocessor.filter_category(sample_df.copy())
        
        # Should only keep 'aluguel' rows
        assert len(filtered) < len(sample_df)
        assert all(filtered['categoria'].str.contains('aluguel'))
    
    def test_drop_columns(self, preprocessor, sample_df):
        """Test column dropping."""
        result = preprocessor.drop_columns(sample_df.copy())
        
        # emissaoNotaFiscalData should be dropped
        assert 'emissaoNotaFiscalData' not in result.columns
    
    def test_process_dates(self, preprocessor, sample_df):
        """Test date processing."""
        # Add some future dates
        future_date = datetime.now() + timedelta(days=30)
        sample_df.loc[0, 'vencimentoData'] = pd.Timestamp(future_date, tz='UTC')
        
        result = preprocessor.process_dates(sample_df.copy())
        
        # Future dates should be filtered out
        assert len(result) <= len(sample_df)
        
        # Should be sorted by due date
        dates = result['vencimentoData'].values
        assert all(dates[i] <= dates[i + 1] for i in range(len(dates) - 1))
    
    def test_run_pipeline(self, preprocessor, sample_df):
        """Test full preprocessing pipeline."""
        result = preprocessor.run(sample_df.copy())
        
        assert len(result) > 0
        assert 'emissaoNotaFiscalData' not in result.columns


class TestFeatureEngineer:
    """Test FeatureEngineer class."""
    
    @pytest.fixture
    def engineer(self, mock_logger):
        """Create a FeatureEngineer instance."""
        return FeatureEngineer(logger=mock_logger)
    
    @pytest.fixture
    def preprocessed_df(self):
        """Create a preprocessed DataFrame for feature engineering."""
        n = 100
        n_users = 5
        n_contracts = 10
        base_date = datetime.now() - timedelta(days=365)
        
        df = pd.DataFrame({
            'usuarioId': np.repeat(range(1, n_users + 1), n // n_users),
            'locacaoId': np.tile(range(1, n_contracts + 1), n // n_contracts),
            'ordem_parcela': np.tile(range(1, 11), n // 10),
            'vencimentoData': pd.to_datetime([
                base_date + timedelta(days=i * 3) for i in range(n)
            ], utc=True),
            'Dias_atraso': np.random.choice([0, 1, 2, 3, 5, 7, 10, 15], n),
            'valor_caucao_brl': np.random.uniform(100, 1000, n),
        })
        
        return df.sort_values(['usuarioId', 'locacaoId', 'ordem_parcela']).reset_index(drop=True)
    
    def test_create_temporal_features(self, engineer, preprocessed_df):
        """Test temporal feature creation."""
        result = engineer.create_temporal_features(preprocessed_df.copy())
        
        # Check new columns exist
        temporal_cols = [
            'venc_dayofweek', 'venc_day', 'venc_month', 'venc_quarter',
            'venc_is_weekend', 'venc_is_month_start', 'venc_is_month_end',
            'venc_dayofweek_sin', 'venc_dayofweek_cos',
            'venc_day_sin', 'venc_day_cos',
            'venc_month_sin', 'venc_month_cos'
        ]
        
        for col in temporal_cols:
            assert col in result.columns, f"Missing column: {col}"
    
    def test_temporal_features_ranges(self, engineer, preprocessed_df):
        """Test temporal features have valid ranges."""
        result = engineer.create_temporal_features(preprocessed_df.copy())
        
        # Day of week: 0-6
        assert result['venc_dayofweek'].between(0, 6).all()
        
        # Day: 1-31
        assert result['venc_day'].between(1, 31).all()
        
        # Month: 1-12
        assert result['venc_month'].between(1, 12).all()
        
        # Quarter: 1-4
        assert result['venc_quarter'].between(1, 4).all()
        
        # Binary features
        assert result['venc_is_weekend'].isin([0, 1]).all()
        
        # Sin/cos features: -1 to 1
        assert result['venc_dayofweek_sin'].between(-1, 1).all()
        assert result['venc_dayofweek_cos'].between(-1, 1).all()
    
    def test_create_history_features(self, engineer, preprocessed_df):
        """Test history feature creation."""
        result = engineer.create_history_features(preprocessed_df.copy())
        
        history_cols = [
            'hist_mean_delay', 'hist_std_delay', 'hist_max_delay', 'hist_default_rate'
        ]
        
        for col in history_cols:
            assert col in result.columns, f"Missing column: {col}"
    
    def test_history_features_no_future_leak(self, engineer, preprocessed_df):
        """Test that history features don't leak future information."""
        result = engineer.create_history_features(preprocessed_df.copy())
        
        # First row for each user should have 0 (or NaN filled) history
        for user_id in result['usuarioId'].unique():
            user_data = result[result['usuarioId'] == user_id].iloc[0]
            assert user_data['hist_mean_delay'] == 0 or pd.isna(user_data['hist_mean_delay'])


class TestTargetCreation:
    """Test target variable creation."""
    
    def test_target_thresholds(self):
        """Test that delay thresholds create valid targets."""
        delays = pd.Series([0, 1, 2, 3, 5, 7, 10, 14, 15, 30])
        
        target_short = (delays >= config.data.delay_threshold_1).astype(int)
        target_medium = (delays >= config.data.delay_threshold_2).astype(int)
        target_long = (delays >= config.data.delay_threshold_3).astype(int)
        
        # Check ordering: short >= medium >= long
        assert (target_short >= target_medium).all()
        assert (target_medium >= target_long).all()
    
    def test_target_values_binary(self):
        """Test targets are binary."""
        delays = pd.Series(np.random.randint(0, 30, 100))
        
        target_short = (delays >= config.data.delay_threshold_1).astype(int)
        
        assert set(target_short.unique()).issubset({0, 1})
