# Test Suite for Payment Default Prediction Model

This directory contains comprehensive tests to validate the full code flow of the payment default prediction model.

## Test Structure

The tests are organized step-by-step to validate each component:

| File | Description |
|------|-------------|
| `test_01_config.py` | Configuration validation - ensures all parameters are valid |
| `test_02_network_layers.py` | Individual neural network layers and modules |
| `test_03_model.py` | Full model integration and forward pass |
| `test_04_data_augmentation.py` | Data augmentation strategies |
| `test_05_dataset.py` | Dataset and DataLoader functionality |
| `test_06_preprocessing.py` | Data preprocessing pipeline |
| `test_07_trainer_components.py` | EMA, loss functions, and trainer utilities |
| `test_08_integration.py` | End-to-end training pipeline |
| `test_09_metrics.py` | Metrics and evaluation |

## Running Tests

### Run All Tests
```bash
cd "project - 2"
python -m pytest Test/ -v
```

### Run Specific Test File
```bash
python -m pytest Test/test_01_config.py -v
```

### Run Specific Test Class
```bash
python -m pytest Test/test_02_network_layers.py::TestDropPath -v
```

### Run Specific Test
```bash
python -m pytest Test/test_02_network_layers.py::TestDropPath::test_forward_training -v
```

### Run with Coverage
```bash
python -m pytest Test/ -v --cov=Model --cov=Configs --cov-report=html
```

### Run Tests with Detailed Output
```bash
python -m pytest Test/ -v --tb=long
```

### Using the Test Runner Script
```bash
python Test/run_tests.py
```

## Test Fixtures

Common fixtures are defined in `conftest.py`:

- `device`: Best available device (CUDA or CPU)
- `batch_size`: Standard batch size (8)
- `sequence_length`: Standard sequence length (10)
- `hidden_dim`: Hidden dimension (64)
- `embedding_dims`: Categorical embedding cardinalities
- `sample_categorical_batch`: Sample categorical tensor
- `sample_continuous_batch`: Sample continuous tensor
- `sample_lengths`: Sample sequence lengths
- `sample_targets`: Sample target tensor
- `sample_dataframe`: Sample pandas DataFrame
- `mock_logger`: Mock logger for testing

## Requirements

```
pytest>=7.0.0
pytest-cov>=4.0.0
torch>=2.0.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
```

## Test Coverage Goals

- **Unit Tests**: Individual components work correctly in isolation
- **Integration Tests**: Components work together properly
- **Edge Cases**: Handling of edge cases (empty inputs, extreme values)
- **Device Compatibility**: Tests run on both CPU and CUDA
- **Numerical Stability**: Loss functions and gradients remain stable

## Adding New Tests

1. Create a new test file following the naming convention `test_XX_description.py`
2. Use fixtures from `conftest.py` for common setup
3. Group related tests in classes
4. Use descriptive test method names starting with `test_`
5. Add docstrings explaining what each test validates
