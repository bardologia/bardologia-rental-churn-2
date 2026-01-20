"""
Pytest configuration and shared fixtures for all tests.
"""

import pytest
import torch
import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime, timedelta

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from Configs.config import config


@pytest.fixture
def device():
    """Return the best available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def batch_size():
    """Standard batch size for testing."""
    return 8


@pytest.fixture
def sequence_length():
    """Standard sequence length for testing."""
    return 10


@pytest.fixture
def hidden_dim():
    """Standard hidden dimension for testing."""
    return 64


@pytest.fixture
def num_heads():
    """Number of attention heads."""
    return 4


@pytest.fixture
def num_categorical():
    """Number of categorical features."""
    return 5


@pytest.fixture
def num_continuous():
    """Number of continuous features."""
    return 8


@pytest.fixture
def embedding_dims(num_categorical):
    """Embedding cardinalities for categorical features."""
    return [10, 20, 15, 8, 12]


@pytest.fixture
def sample_categorical_batch(batch_size, sequence_length, num_categorical):
    """Generate sample categorical tensor."""
    return torch.randint(1, 10, (batch_size, sequence_length, num_categorical))


@pytest.fixture
def sample_continuous_batch(batch_size, sequence_length, num_continuous):
    """Generate sample continuous tensor."""
    return torch.randn(batch_size, sequence_length, num_continuous)


@pytest.fixture
def sample_lengths(batch_size, sequence_length):
    """Generate sample sequence lengths."""
    return torch.randint(3, sequence_length + 1, (batch_size,))


@pytest.fixture
def sample_targets(batch_size):
    """Generate sample targets (3 outputs for short, medium, long)."""
    return torch.randint(0, 2, (batch_size, 3)).float()


@pytest.fixture
def sample_dataframe():
    """Generate a sample DataFrame that mimics the raw data structure."""
    np.random.seed(42)
    n_samples = 100
    n_users = 10
    n_contracts = 20
    
    base_date = datetime.now() - timedelta(days=365)
    
    data = {
        'usuarioId': np.random.choice(range(1, n_users + 1), n_samples),
        'locacaoId': np.random.choice(range(1, n_contracts + 1), n_samples),
        'ordem_parcela': np.random.randint(1, 12, n_samples),
        'vencimentoData': [base_date + timedelta(days=np.random.randint(0, 300)) for _ in range(n_samples)],
        'pagamentoData': [base_date + timedelta(days=np.random.randint(0, 350)) for _ in range(n_samples)],
        'Dias_atraso': np.random.choice([0, 1, 2, 3, 5, 7, 10, 15, 30], n_samples),
        'quantidadeDiarias': np.random.randint(1, 30, n_samples),
        'valor_caucao_brl': np.random.uniform(100, 5000, n_samples),
        'recorrencia_pagamento': np.random.choice(['mensal', 'semanal', 'quinzenal'], n_samples),
        'sexo': np.random.choice(['M', 'F'], n_samples),
        'faixa_idade_resumida': np.random.choice(['18-25', '26-35', '36-45', '46+'], n_samples),
        'veiculo_modelo': np.random.choice(['modelo_a', 'modelo_b', 'modelo_c'], n_samples),
        'pacoteNome': np.random.choice(['basico', 'intermediario', 'premium'], n_samples),
        'formaPagamento': np.random.choice(['cartao', 'boleto', 'pix'], n_samples),
        'lugar': np.random.choice(['cidade_a', 'cidade_b', 'cidade_c'], n_samples),
        'regiao': np.random.choice(['norte', 'sul', 'sudeste', 'nordeste'], n_samples),
        'produto_categoria': np.random.choice(['categoria_1', 'categoria_2'], n_samples),
        'categoria': np.random.choice(['aluguel', 'aluguel_mensal', 'venda'], n_samples),
    }
    
    df = pd.DataFrame(data)
    df['vencimentoData'] = pd.to_datetime(df['vencimentoData'], utc=True)
    df['pagamentoData'] = pd.to_datetime(df['pagamentoData'], utc=True)
    
    return df


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    import logging
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)
    return logger
