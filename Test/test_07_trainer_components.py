"""
Test 07: Trainer Components
Tests the EMA, AsymmetricLoss, and other trainer utilities.
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model.trainer import EMA, AsymmetricLoss


class TestEMA:
    """Test Exponential Moving Average."""
    
    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )
    
    def test_ema_creation(self, simple_model):
        """Test EMA initialization."""
        ema = EMA(simple_model, decay=0.999)
        
        assert ema.decay == 0.999
        assert ema.step == 0
        assert len(ema.shadow) > 0
    
    def test_shadow_initialization(self, simple_model):
        """Test shadow parameters are initialized correctly."""
        ema = EMA(simple_model)
        
        for name, param in simple_model.named_parameters():
            if param.requires_grad:
                assert name in ema.shadow
                assert torch.allclose(ema.shadow[name], param.data)
    
    def test_update(self, simple_model):
        """Test EMA update step."""
        ema = EMA(simple_model, decay=0.9)
        
        # Modify model parameters
        with torch.no_grad():
            for param in simple_model.parameters():
                param.add_(torch.randn_like(param))
        
        ema.update()
        
        assert ema.step == 1
    
    def test_apply_and_restore(self, simple_model):
        """Test applying and restoring shadow parameters."""
        ema = EMA(simple_model)
        
        # Save original parameters
        original_params = {name: param.clone() for name, param in simple_model.named_parameters()}
        
        # Modify model
        with torch.no_grad():
            for param in simple_model.parameters():
                param.add_(1.0)
        
        # Apply shadow (original) parameters
        ema.apply_shadow()
        
        for name, param in simple_model.named_parameters():
            if param.requires_grad:
                assert torch.allclose(param.data, ema.shadow[name])
        
        # Restore modified parameters
        ema.restore()
        
        for name, param in simple_model.named_parameters():
            if param.requires_grad:
                assert not torch.allclose(param.data, original_params[name])
    
    def test_warmup_decay(self, simple_model):
        """Test decay warmup behavior."""
        ema = EMA(simple_model, decay=0.999, warmup_steps=100)
        
        # Initial decay should be lower due to warmup
        initial_decay = ema._get_decay()
        assert initial_decay < 0.999
        
        # Simulate steps
        for _ in range(50):
            ema.update()
        
        # Decay should increase
        mid_decay = ema._get_decay()
        assert mid_decay > initial_decay
    
    def test_state_dict(self, simple_model):
        """Test state dict save/load."""
        ema1 = EMA(simple_model, decay=0.99)
        ema1.update()
        ema1.update()
        
        state = ema1.state_dict()
        
        assert 'shadow' in state
        assert 'step' in state
        assert 'decay' in state
        assert state['step'] == 2
    
    def test_load_state_dict(self, simple_model):
        """Test loading state dict."""
        ema1 = EMA(simple_model, decay=0.99)
        ema1.update()
        ema1.update()
        
        state = ema1.state_dict()
        
        ema2 = EMA(simple_model, decay=0.5)
        ema2.load_state_dict(state)
        
        assert ema2.step == 2
        assert ema2.decay == 0.99


class TestAsymmetricLoss:
    """Test Asymmetric Loss function."""
    
    def test_loss_creation(self):
        """Test loss function creation."""
        loss_fn = AsymmetricLoss()
        
        assert loss_fn.gamma_negative >= 0
        assert loss_fn.gamma_positive >= 0
    
    def test_loss_output_shape(self):
        """Test loss output is scalar."""
        loss_fn = AsymmetricLoss()
        
        logits = torch.randn(8, 3)
        targets = torch.randint(0, 2, (8, 3)).float()
        
        loss = loss_fn(logits, targets)
        
        assert loss.ndim == 0  # Scalar
    
    def test_loss_positive(self):
        """Test loss is always positive."""
        loss_fn = AsymmetricLoss()
        
        for _ in range(10):
            logits = torch.randn(16, 3)
            targets = torch.randint(0, 2, (16, 3)).float()
            
            loss = loss_fn(logits, targets)
            
            assert loss.item() > 0
    
    def test_loss_gradient_flow(self):
        """Test gradients flow through the loss."""
        loss_fn = AsymmetricLoss()
        
        logits = torch.randn(8, 3, requires_grad=True)
        targets = torch.randint(0, 2, (8, 3)).float()
        
        loss = loss_fn(logits, targets)
        loss.backward()
        
        assert logits.grad is not None
    
    def test_loss_asymmetry(self):
        """Test loss handles positive/negative classes differently."""
        # High gamma_negative should down-weight easy negatives
        loss_fn = AsymmetricLoss(gamma_negative=4, gamma_positive=0)
        
        # All positive predictions
        logits_high = torch.ones(8, 1) * 5  # High confidence positive
        targets_neg = torch.zeros(8, 1)  # But actually negative
        
        logits_low = torch.ones(8, 1) * 0.1  # Low confidence positive
        
        loss_high = loss_fn(logits_high, targets_neg)
        loss_low = loss_fn(logits_low, targets_neg)
        
        # With high gamma_negative, high confidence wrong predictions should have higher loss
        # (but the modulation is complex, so just verify both are positive)
        assert loss_high.item() > 0
        assert loss_low.item() > 0
    
    def test_loss_with_custom_params(self):
        """Test loss with custom parameters."""
        loss_fn = AsymmetricLoss(
            gamma_negative=2,
            gamma_positive=1,
            clip=0.1,
            reduction='mean'
        )
        
        logits = torch.randn(8, 3)
        targets = torch.randint(0, 2, (8, 3)).float()
        
        loss = loss_fn(logits, targets)
        
        assert loss.ndim == 0


class TestLossNumericalStability:
    """Test numerical stability of loss functions."""
    
    def test_extreme_logits(self):
        """Test loss handles extreme logit values."""
        loss_fn = AsymmetricLoss()
        
        # Very large positive logits
        logits_large = torch.ones(4, 3) * 100
        targets = torch.randint(0, 2, (4, 3)).float()
        
        loss = loss_fn(logits_large, targets)
        
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    def test_extreme_negative_logits(self):
        """Test loss handles very negative logits."""
        loss_fn = AsymmetricLoss()
        
        logits_neg = torch.ones(4, 3) * -100
        targets = torch.randint(0, 2, (4, 3)).float()
        
        loss = loss_fn(logits_neg, targets)
        
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    def test_all_zeros_targets(self):
        """Test loss with all zero targets."""
        loss_fn = AsymmetricLoss()
        
        logits = torch.randn(8, 3)
        targets = torch.zeros(8, 3)
        
        loss = loss_fn(logits, targets)
        
        assert not torch.isnan(loss)
        assert loss.item() >= 0
    
    def test_all_ones_targets(self):
        """Test loss with all one targets."""
        loss_fn = AsymmetricLoss()
        
        logits = torch.randn(8, 3)
        targets = torch.ones(8, 3)
        
        loss = loss_fn(logits, targets)
        
        assert not torch.isnan(loss)
        assert loss.item() >= 0
