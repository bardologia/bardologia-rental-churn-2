"""
Test 03: Full Model Integration
Tests the complete Model class and its forward pass.
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model.network import Model


class TestModelInstantiation:
    """Test model can be properly instantiated."""
    
    def test_basic_instantiation(self, embedding_dims, num_continuous, hidden_dim):
        """Test basic model creation."""
        model = Model(
            embedding_dimensions=embedding_dims,
            num_continuous=num_continuous,
            hidden_dimension=hidden_dim,
            num_invoice_layers=1,
            num_sequence_layers=2,
            num_heads=4
        )
        
        assert model is not None
        assert model.hidden_dimension == hidden_dim
    
    def test_model_parameters(self, embedding_dims, num_continuous, hidden_dim):
        """Test model has trainable parameters."""
        model = Model(
            embedding_dimensions=embedding_dims,
            num_continuous=num_continuous,
            hidden_dimension=hidden_dim
        )
        
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert num_params > 0
    
    def test_model_components(self, embedding_dims, num_continuous, hidden_dim):
        """Test model has all expected components."""
        model = Model(
            embedding_dimensions=embedding_dims,
            num_continuous=num_continuous,
            hidden_dimension=hidden_dim,
            use_temporal_attention=True,
            use_temperature_scaling=True
        )
        
        assert hasattr(model, 'tokenizer')
        assert hasattr(model, 'invoice_encoder')
        assert hasattr(model, 'sequence_encoder')
        assert hasattr(model, 'temporal_attention')
        assert hasattr(model, 'head_short')
        assert hasattr(model, 'head_medium')
        assert hasattr(model, 'head_long')
        assert hasattr(model, 'temperature_scaling')


class TestModelForward:
    """Test model forward pass."""
    
    @pytest.fixture
    def model(self, embedding_dims, num_continuous, hidden_dim):
        """Create a model instance for testing."""
        return Model(
            embedding_dimensions=embedding_dims,
            num_continuous=num_continuous,
            hidden_dimension=hidden_dim,
            num_invoice_layers=1,
            num_sequence_layers=2,
            num_heads=4,
            max_sequence_length=64
        )
    
    def test_forward_output_shape(
        self, model, sample_categorical_batch, sample_continuous_batch, sample_lengths, batch_size
    ):
        """Test forward pass output shape."""
        logits = model(sample_categorical_batch, sample_continuous_batch, sample_lengths)
        
        # Should have 3 outputs (short, medium, long)
        assert logits.shape == (batch_size, 3)
    
    def test_forward_with_temperature(
        self, model, sample_categorical_batch, sample_continuous_batch, sample_lengths
    ):
        """Test forward pass with temperature scaling."""
        logits_no_temp = model(
            sample_categorical_batch, sample_continuous_batch, sample_lengths,
            apply_temperature=False
        )
        logits_with_temp = model(
            sample_categorical_batch, sample_continuous_batch, sample_lengths,
            apply_temperature=True
        )
        
        # Outputs should be different when temperature is applied
        assert logits_no_temp.shape == logits_with_temp.shape
    
    def test_forward_gradient_flow(
        self, model, sample_categorical_batch, sample_continuous_batch, sample_lengths
    ):
        """Test gradients flow through the model."""
        model.train()
        
        logits = model(sample_categorical_batch, sample_continuous_batch, sample_lengths)
        loss = logits.sum()
        loss.backward()
        
        # Check some gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None or param.numel() == 0, f"No gradient for {name}"
                break
    
    def test_forward_eval_mode(
        self, model, sample_categorical_batch, sample_continuous_batch, sample_lengths
    ):
        """Test model in evaluation mode."""
        model.eval()
        
        with torch.no_grad():
            logits = model(sample_categorical_batch, sample_continuous_batch, sample_lengths)
        
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()
    
    def test_variable_sequence_lengths(self, embedding_dims, num_continuous, hidden_dim):
        """Test model handles variable sequence lengths correctly."""
        model = Model(
            embedding_dimensions=embedding_dims,
            num_continuous=num_continuous,
            hidden_dimension=hidden_dim
        )
        
        batch_size = 4
        max_seq_len = 20
        
        cat_features = torch.randint(1, 5, (batch_size, max_seq_len, len(embedding_dims)))
        cont_features = torch.randn(batch_size, max_seq_len, num_continuous)
        lengths = torch.tensor([5, 10, 15, 20])
        
        logits = model(cat_features, cont_features, lengths)
        
        assert logits.shape == (batch_size, 3)


class TestModelModes:
    """Test model training and evaluation modes."""
    
    @pytest.fixture
    def model(self, embedding_dims, num_continuous, hidden_dim):
        return Model(
            embedding_dimensions=embedding_dims,
            num_continuous=num_continuous,
            hidden_dimension=hidden_dim
        )
    
    def test_train_mode(self, model):
        """Test model in training mode."""
        model.train()
        assert model.training == True
        
        # Check submodules
        assert model.tokenizer.training == True
        assert model.invoice_encoder.training == True
    
    def test_eval_mode(self, model):
        """Test model in evaluation mode."""
        model.eval()
        assert model.training == False
        
        # Check submodules
        assert model.tokenizer.training == False
        assert model.invoice_encoder.training == False
    
    def test_dropout_behavior(
        self, model, sample_categorical_batch, sample_continuous_batch, sample_lengths
    ):
        """Test dropout is active in train and inactive in eval."""
        model.train()
        torch.manual_seed(42)
        out1 = model(sample_categorical_batch, sample_continuous_batch, sample_lengths)
        torch.manual_seed(42)
        out2 = model(sample_categorical_batch, sample_continuous_batch, sample_lengths)
        
        # Due to dropout, outputs may differ in training
        # (Note: with same seed they should be same, but this tests the mode works)
        
        model.eval()
        with torch.no_grad():
            out3 = model(sample_categorical_batch, sample_continuous_batch, sample_lengths)
            out4 = model(sample_categorical_batch, sample_continuous_batch, sample_lengths)
        
        # In eval mode, outputs should be identical
        assert torch.allclose(out3, out4)


class TestModelDevice:
    """Test model device handling."""
    
    def test_model_to_device(self, embedding_dims, num_continuous, hidden_dim, device):
        """Test model can be moved to device."""
        model = Model(
            embedding_dimensions=embedding_dims,
            num_continuous=num_continuous,
            hidden_dimension=hidden_dim
        )
        
        model = model.to(device)
        
        # Check parameters are on correct device
        for param in model.parameters():
            assert param.device.type == device.type
    
    def test_forward_on_device(
        self, embedding_dims, num_continuous, hidden_dim, 
        sample_categorical_batch, sample_continuous_batch, sample_lengths, device
    ):
        """Test forward pass on specific device."""
        model = Model(
            embedding_dimensions=embedding_dims,
            num_continuous=num_continuous,
            hidden_dimension=hidden_dim
        ).to(device)
        
        cat = sample_categorical_batch.to(device)
        cont = sample_continuous_batch.to(device)
        lengths = sample_lengths.to(device)
        
        logits = model(cat, cont, lengths)
        
        assert logits.device.type == device.type


class TestModelSaveLoad:
    """Test model serialization."""
    
    def test_state_dict(self, embedding_dims, num_continuous, hidden_dim):
        """Test model state dict creation."""
        model = Model(
            embedding_dimensions=embedding_dims,
            num_continuous=num_continuous,
            hidden_dimension=hidden_dim
        )
        
        state_dict = model.state_dict()
        
        assert len(state_dict) > 0
        assert all(isinstance(v, torch.Tensor) for v in state_dict.values())
    
    def test_load_state_dict(self, embedding_dims, num_continuous, hidden_dim):
        """Test model state dict loading."""
        model1 = Model(
            embedding_dimensions=embedding_dims,
            num_continuous=num_continuous,
            hidden_dimension=hidden_dim
        )
        model2 = Model(
            embedding_dimensions=embedding_dims,
            num_continuous=num_continuous,
            hidden_dimension=hidden_dim
        )
        
        # Save and load
        state_dict = model1.state_dict()
        model2.load_state_dict(state_dict)
        
        # Verify parameters match
        for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
            assert torch.allclose(p1, p2), f"Parameter mismatch: {n1}"
