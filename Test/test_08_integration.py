"""
Test 08: End-to-End Integration Tests
Tests the complete training pipeline flow.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model.network import Model
from Model.trainer import AsymmetricLoss, EMA
from Model.data import SequentialDataset, collate_sequences
from torch.utils.data import DataLoader


class TestTrainingStep:
    """Test a single training step."""
    
    @pytest.fixture
    def training_setup(self, embedding_dims, num_continuous, hidden_dim, device):
        """Set up model, optimizer, and loss for training."""
        model = Model(
            embedding_dimensions=embedding_dims,
            num_continuous=num_continuous,
            hidden_dimension=hidden_dim,
            num_invoice_layers=1,
            num_sequence_layers=1,
            num_heads=4
        ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = AsymmetricLoss()
        
        return model, optimizer, criterion
    
    def test_single_training_step(
        self, training_setup, sample_categorical_batch, sample_continuous_batch, 
        sample_lengths, sample_targets, device
    ):
        """Test a single forward-backward pass."""
        model, optimizer, criterion = training_setup
        
        # Move data to device
        cat = sample_categorical_batch.to(device)
        cont = sample_continuous_batch.to(device)
        lengths = sample_lengths.to(device)
        targets = sample_targets.to(device)
        
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(cat, cont, lengths)
        loss = criterion(logits, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        assert not torch.isnan(loss)
        assert loss.item() > 0
    
    def test_gradient_clipping(
        self, training_setup, sample_categorical_batch, sample_continuous_batch,
        sample_lengths, sample_targets, device
    ):
        """Test training with gradient clipping."""
        model, optimizer, criterion = training_setup
        
        cat = sample_categorical_batch.to(device)
        cont = sample_continuous_batch.to(device)
        lengths = sample_lengths.to(device)
        targets = sample_targets.to(device)
        
        model.train()
        optimizer.zero_grad()
        
        logits = model(cat, cont, lengths)
        loss = criterion(logits, targets)
        loss.backward()
        
        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Check gradients are clipped
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        
        assert total_norm <= 1.5  # Allow some tolerance


class TestDataLoaderTraining:
    """Test training with DataLoader."""
    
    @pytest.fixture
    def dataloader(self, batch_size):
        """Create a DataLoader with synthetic data."""
        n_samples = 100
        n_cat = 5
        n_cont = 8
        
        categorical = np.random.randint(0, 10, (n_samples, n_cat))
        continuous = np.random.randn(n_samples, n_cont).astype(np.float32)
        targets = np.random.randint(0, 2, (n_samples, 3)).astype(np.float32)
        
        indices = [(i, min(i + 5, n_samples - 1), min(i + 4, n_samples - 1)) 
                   for i in range(0, n_samples - 10)]
        
        dataset = SequentialDataset(categorical, continuous, targets, indices)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_sequences
        )
    
    def test_epoch_training(self, dataloader, hidden_dim, device):
        """Test training for one epoch."""
        model = Model(
            embedding_dimensions=[10] * 5,
            num_continuous=8,
            hidden_dimension=hidden_dim,
            num_invoice_layers=1,
            num_sequence_layers=1
        ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = AsymmetricLoss()
        
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for cat, cont, targets, lengths in dataloader:
            cat = cat.to(device)
            cont = cont.to(device)
            targets = targets.to(device)
            lengths = lengths.to(device)
            
            optimizer.zero_grad()
            logits = model(cat, cont, lengths)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        assert avg_loss > 0
        assert not np.isnan(avg_loss)


class TestEMAIntegration:
    """Test EMA integration with training."""
    
    def test_ema_during_training(
        self, embedding_dims, num_continuous, hidden_dim,
        sample_categorical_batch, sample_continuous_batch, sample_lengths, sample_targets, device
    ):
        """Test EMA updates during training."""
        model = Model(
            embedding_dimensions=embedding_dims,
            num_continuous=num_continuous,
            hidden_dimension=hidden_dim
        ).to(device)
        
        ema = EMA(model, decay=0.99)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = AsymmetricLoss()
        
        cat = sample_categorical_batch.to(device)
        cont = sample_continuous_batch.to(device)
        lengths = sample_lengths.to(device)
        targets = sample_targets.to(device)
        
        # Training steps with EMA
        for _ in range(5):
            model.train()
            optimizer.zero_grad()
            
            logits = model(cat, cont, lengths)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
            ema.update()
        
        assert ema.step == 5
        
        # Apply EMA for evaluation
        ema.apply_shadow()
        model.eval()
        
        with torch.no_grad():
            ema_logits = model(cat, cont, lengths)
        
        assert not torch.isnan(ema_logits).any()
        
        # Restore
        ema.restore()


class TestMixedPrecision:
    """Test mixed precision training (if CUDA available)."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_amp_training(
        self, embedding_dims, num_continuous, hidden_dim,
        sample_categorical_batch, sample_continuous_batch, sample_lengths, sample_targets
    ):
        """Test training with automatic mixed precision."""
        device = torch.device("cuda")
        
        model = Model(
            embedding_dimensions=embedding_dims,
            num_continuous=num_continuous,
            hidden_dimension=hidden_dim
        ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = AsymmetricLoss()
        scaler = torch.amp.GradScaler()
        
        cat = sample_categorical_batch.to(device)
        cont = sample_continuous_batch.to(device)
        lengths = sample_lengths.to(device)
        targets = sample_targets.to(device)
        
        model.train()
        optimizer.zero_grad()
        
        with torch.amp.autocast(device_type='cuda'):
            logits = model(cat, cont, lengths)
            loss = criterion(logits, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        assert not torch.isnan(loss)


class TestModelValidation:
    """Test validation/evaluation flow."""
    
    @pytest.fixture
    def model_and_data(self, embedding_dims, num_continuous, hidden_dim, device):
        """Create model and validation data."""
        model = Model(
            embedding_dimensions=embedding_dims,
            num_continuous=num_continuous,
            hidden_dimension=hidden_dim
        ).to(device)
        
        # Create validation data
        n_samples = 50
        categorical = np.random.randint(0, 10, (n_samples, len(embedding_dims)))
        continuous = np.random.randn(n_samples, num_continuous).astype(np.float32)
        targets = np.random.randint(0, 2, (n_samples, 3)).astype(np.float32)
        
        indices = [(i, min(i + 5, n_samples - 1), min(i + 4, n_samples - 1)) 
                   for i in range(0, n_samples - 10)]
        
        dataset = SequentialDataset(categorical, continuous, targets, indices)
        loader = DataLoader(dataset, batch_size=8, collate_fn=collate_sequences)
        
        return model, loader
    
    def test_validation_loop(self, model_and_data, device):
        """Test validation evaluation loop."""
        model, loader = model_and_data
        criterion = AsymmetricLoss()
        
        model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for cat, cont, targets, lengths in loader:
                cat = cat.to(device)
                cont = cont.to(device)
                targets = targets.to(device)
                lengths = lengths.to(device)
                
                logits = model(cat, cont, lengths)
                loss = criterion(logits, targets)
                
                total_loss += loss.item()
                all_predictions.append(torch.sigmoid(logits).cpu())
                all_targets.append(targets.cpu())
        
        predictions = torch.cat(all_predictions)
        targets = torch.cat(all_targets)
        
        assert predictions.shape == targets.shape
        assert (predictions >= 0).all() and (predictions <= 1).all()
    
    def test_prediction_output_range(self, model_and_data, device):
        """Test that sigmoid predictions are in [0, 1]."""
        model, loader = model_and_data
        
        model.eval()
        
        with torch.no_grad():
            cat, cont, targets, lengths = next(iter(loader))
            cat = cat.to(device)
            cont = cont.to(device)
            lengths = lengths.to(device)
            
            logits = model(cat, cont, lengths)
            probs = torch.sigmoid(logits)
        
        assert (probs >= 0).all()
        assert (probs <= 1).all()


class TestCheckpointing:
    """Test model checkpointing."""
    
    def test_save_and_load_checkpoint(
        self, embedding_dims, num_continuous, hidden_dim,
        sample_categorical_batch, sample_continuous_batch, sample_lengths, tmp_path
    ):
        """Test saving and loading model checkpoint."""
        model1 = Model(
            embedding_dimensions=embedding_dims,
            num_continuous=num_continuous,
            hidden_dimension=hidden_dim
        )
        
        # Get prediction before saving
        model1.eval()
        with torch.no_grad():
            pred1 = model1(sample_categorical_batch, sample_continuous_batch, sample_lengths)
        
        # Save checkpoint
        checkpoint_path = tmp_path / "checkpoint.pth"
        torch.save({
            'model_state_dict': model1.state_dict(),
            'embedding_dimensions': embedding_dims,
            'num_continuous': num_continuous,
            'hidden_dimension': hidden_dim
        }, checkpoint_path)
        
        # Load in new model
        model2 = Model(
            embedding_dimensions=embedding_dims,
            num_continuous=num_continuous,
            hidden_dimension=hidden_dim
        )
        
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model2.load_state_dict(checkpoint['model_state_dict'])
        
        # Verify predictions match
        model2.eval()
        with torch.no_grad():
            pred2 = model2(sample_categorical_batch, sample_continuous_batch, sample_lengths)
        
        assert torch.allclose(pred1, pred2)
