"""
Test 02: Network Layer Components
Tests individual neural network layers and modules in isolation.
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model.network import (
    DropPath,
    PeriodicEmbedding,
    GatedResidualNetwork,
    SwiGLU,
    RotaryPositionalEmbedding,
    GRNPredictionHead,
    TemperatureScaling,
    FeatureTokenizer,
    InvoiceEncoder,
    TransformerBlock,
    SequenceEncoder,
    HistoryAttention
)


class TestDropPath:
    """Test DropPath regularization layer."""
    
    def test_forward_training(self):
        """Test forward pass in training mode."""
        layer = DropPath(drop_prob=0.1)
        layer.train()
        x = torch.randn(4, 10, 64)
        out = layer(x)
        
        assert out.shape == x.shape
    
    def test_forward_eval(self):
        """Test forward pass in eval mode (should be identity)."""
        layer = DropPath(drop_prob=0.5)
        layer.eval()
        x = torch.randn(4, 10, 64)
        out = layer(x)
        
        assert torch.allclose(out, x)
    
    def test_zero_drop_prob(self):
        """Test with zero drop probability."""
        layer = DropPath(drop_prob=0.0)
        layer.train()
        x = torch.randn(4, 10, 64)
        out = layer(x)
        
        assert torch.allclose(out, x)


class TestPeriodicEmbedding:
    """Test periodic embedding for continuous features."""
    
    def test_output_shape(self):
        """Test output tensor shape."""
        num_features = 5
        embedding_dim = 32
        batch_size = 8
        
        layer = PeriodicEmbedding(num_features, embedding_dim)
        x = torch.randn(batch_size, num_features)
        out = layer(x)
        
        assert out.shape == (batch_size, num_features, embedding_dim)
    
    def test_gradient_flow(self):
        """Test gradients flow through the layer."""
        layer = PeriodicEmbedding(4, 16)
        x = torch.randn(2, 4, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert layer.frequencies.grad is not None


class TestGatedResidualNetwork:
    """Test Gated Residual Network module."""
    
    def test_same_dimension(self):
        """Test GRN with same input/output dimension."""
        layer = GatedResidualNetwork(64, 128, 64)
        x = torch.randn(8, 64)
        out = layer(x)
        
        assert out.shape == (8, 64)
    
    def test_different_dimension(self):
        """Test GRN with different input/output dimension."""
        layer = GatedResidualNetwork(64, 128, 32)
        x = torch.randn(8, 64)
        out = layer(x)
        
        assert out.shape == (8, 32)
    
    def test_with_context(self):
        """Test GRN with context input."""
        layer = GatedResidualNetwork(64, 128, 64, context_dimension=32)
        x = torch.randn(8, 64)
        context = torch.randn(8, 32)
        out = layer(x, context)
        
        assert out.shape == (8, 64)
    
    def test_gradient_flow(self):
        """Test gradients flow through GRN."""
        layer = GatedResidualNetwork(32, 64, 32)
        x = torch.randn(4, 32, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None


class TestSwiGLU:
    """Test SwiGLU activation module."""
    
    def test_output_shape(self):
        """Test output tensor shape."""
        layer = SwiGLU(64, 256, 64)
        x = torch.randn(8, 10, 64)
        out = layer(x)
        
        assert out.shape == (8, 10, 64)
    
    def test_different_dimensions(self):
        """Test with different input/output dimensions."""
        layer = SwiGLU(32, 128, 64)
        x = torch.randn(4, 5, 32)
        out = layer(x)
        
        assert out.shape == (4, 5, 64)


class TestRotaryPositionalEmbedding:
    """Test Rotary Positional Embedding (RoPE)."""
    
    def test_output_shapes(self):
        """Test query and key output shapes."""
        dim = 32
        max_seq = 128
        layer = RotaryPositionalEmbedding(dim, max_seq)
        
        batch_size = 4
        num_heads = 4
        seq_len = 16
        
        query = torch.randn(batch_size, num_heads, seq_len, dim)
        key = torch.randn(batch_size, num_heads, seq_len, dim)
        
        q_out, k_out = layer(query, key, seq_len)
        
        assert q_out.shape == query.shape
        assert k_out.shape == key.shape
    
    def test_cache_building(self):
        """Test that cache is built correctly."""
        layer = RotaryPositionalEmbedding(32, 64)
        
        assert hasattr(layer, 'cos_cached')
        assert hasattr(layer, 'sin_cached')
        assert layer.cos_cached.shape[-2] == 64


class TestGRNPredictionHead:
    """Test GRN-based prediction head."""
    
    def test_single_output(self):
        """Test prediction head with single output."""
        head = GRNPredictionHead(64, 32, num_outputs=1)
        x = torch.randn(8, 64)
        out = head(x)
        
        assert out.shape == (8, 1)
    
    def test_multi_output(self):
        """Test prediction head with multiple outputs."""
        head = GRNPredictionHead(128, 64, num_outputs=3)
        x = torch.randn(8, 128)
        out = head(x)
        
        assert out.shape == (8, 3)


class TestTemperatureScaling:
    """Test temperature scaling for calibration."""
    
    def test_forward(self):
        """Test forward scaling."""
        layer = TemperatureScaling(num_outputs=3)
        logits = torch.randn(8, 3)
        scaled = layer(logits)
        
        assert scaled.shape == logits.shape
    
    def test_temperature_parameter(self):
        """Test temperature is a learnable parameter."""
        layer = TemperatureScaling(num_outputs=3)
        
        assert layer.temperature.shape == (3,)
        assert layer.temperature.requires_grad == True


class TestFeatureTokenizer:
    """Test feature tokenization module."""
    
    def test_output_shape(self, embedding_dims, num_continuous, hidden_dim, batch_size, sequence_length):
        """Test tokenizer output shape."""
        tokenizer = FeatureTokenizer(
            cardinalities=embedding_dims,
            num_continuous=num_continuous,
            token_dimension=hidden_dim
        )
        
        cat_features = torch.randint(1, 5, (batch_size, sequence_length, len(embedding_dims)))
        cont_features = torch.randn(batch_size, sequence_length, num_continuous)
        
        tokens = tokenizer(cat_features, cont_features)
        
        expected_num_tokens = len(embedding_dims) + num_continuous
        assert tokens.shape == (batch_size, sequence_length, expected_num_tokens, hidden_dim)
    
    def test_categorical_only(self):
        """Test tokenizer with only categorical features."""
        embedding_dims = [5, 10, 8]
        tokenizer = FeatureTokenizer(
            cardinalities=embedding_dims,
            num_continuous=0,
            token_dimension=32
        )
        
        cat_features = torch.randint(1, 5, (4, 8, 3))
        cont_features = torch.randn(4, 8, 0)
        
        tokens = tokenizer(cat_features, cont_features)
        
        assert tokens.shape == (4, 8, 3, 32)


class TestInvoiceEncoder:
    """Test invoice-level transformer encoder."""
    
    def test_output_shape(self, hidden_dim, batch_size, sequence_length):
        """Test encoder output shape."""
        num_features = 10
        encoder = InvoiceEncoder(hidden_dim, num_heads=4, num_layers=2)
        
        tokens = torch.randn(batch_size, sequence_length, num_features, hidden_dim)
        out = encoder(tokens)
        
        assert out.shape == (batch_size, sequence_length, hidden_dim)
    
    def test_gradient_flow(self, hidden_dim):
        """Test gradients flow through encoder."""
        encoder = InvoiceEncoder(hidden_dim, num_heads=4, num_layers=1)
        
        tokens = torch.randn(2, 5, 8, hidden_dim, requires_grad=True)
        out = encoder(tokens)
        loss = out.sum()
        loss.backward()
        
        assert tokens.grad is not None


class TestTransformerBlock:
    """Test transformer block."""
    
    def test_forward(self, hidden_dim, batch_size, sequence_length):
        """Test forward pass."""
        block = TransformerBlock(hidden_dim, num_heads=4)
        x = torch.randn(batch_size, sequence_length, hidden_dim)
        out = block(x)
        
        assert out.shape == x.shape
    
    def test_with_rope(self, hidden_dim):
        """Test with rotary positional embedding."""
        head_dim = hidden_dim // 4
        rope = RotaryPositionalEmbedding(head_dim, max_sequence_length=64)
        block = TransformerBlock(hidden_dim, num_heads=4, rotary_positional_embedding=rope)
        
        x = torch.randn(4, 16, hidden_dim)
        out = block(x)
        
        assert out.shape == x.shape
    
    def test_causal_masking(self, hidden_dim):
        """Test causal attention mode."""
        block = TransformerBlock(hidden_dim, num_heads=4, is_causal=True)
        x = torch.randn(4, 16, hidden_dim)
        out = block(x)
        
        assert out.shape == x.shape


class TestSequenceEncoder:
    """Test sequence-level transformer encoder."""
    
    def test_output_shapes(self, hidden_dim, batch_size, sequence_length):
        """Test encoder outputs context and hidden states."""
        encoder = SequenceEncoder(hidden_dim, num_heads=4, num_layers=2)
        
        x = torch.randn(batch_size, sequence_length, hidden_dim)
        lengths = torch.randint(3, sequence_length + 1, (batch_size,))
        
        context, hidden = encoder(x, lengths)
        
        assert context.shape == (batch_size, hidden_dim)
        assert hidden.shape == (batch_size, sequence_length, hidden_dim)
    
    def test_without_lengths(self, hidden_dim):
        """Test encoder without explicit lengths."""
        encoder = SequenceEncoder(hidden_dim, num_heads=4, num_layers=1)
        
        x = torch.randn(4, 10, hidden_dim)
        context, hidden = encoder(x)
        
        assert context.shape == (4, hidden_dim)


class TestHistoryAttention:
    """Test history attention module."""
    
    def test_output_shape(self, hidden_dim, batch_size, sequence_length):
        """Test attention output shape."""
        attn = HistoryAttention(hidden_dim, num_heads=4)
        
        current = torch.randn(batch_size, hidden_dim)
        history = torch.randn(batch_size, sequence_length, hidden_dim)
        
        attended, weights = attn(current, history)
        
        assert attended.shape == (batch_size, hidden_dim)
        assert weights.shape[0] == batch_size
    
    def test_with_mask(self, hidden_dim):
        """Test attention with padding mask."""
        attn = HistoryAttention(hidden_dim, num_heads=4)
        
        current = torch.randn(4, hidden_dim)
        history = torch.randn(4, 8, hidden_dim)
        mask = torch.zeros(4, 8, dtype=torch.bool)
        mask[:, 5:] = True  # Mask last 3 positions
        
        attended, weights = attn(current, history, mask)
        
        assert attended.shape == (4, hidden_dim)
