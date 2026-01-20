"""
Test 05: Dataset and DataLoader
Tests the SequentialDataset and DataModule classes.
"""

import pytest
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model.data import SequentialDataset, collate_sequences


class TestSequentialDataset:
    """Test SequentialDataset class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        n_samples = 100
        n_cat_features = 5
        n_cont_features = 8
        n_targets = 3
        
        categorical = np.random.randint(0, 10, (n_samples, n_cat_features))
        continuous = np.random.randn(n_samples, n_cont_features).astype(np.float32)
        targets = np.random.randint(0, 2, (n_samples, n_targets)).astype(np.float32)
        
        # Create indices: (start, end, target_idx)
        indices = []
        for i in range(0, n_samples - 10, 5):
            end = min(i + np.random.randint(3, 10), n_samples)
            indices.append((i, end, end - 1))
        
        return categorical, continuous, targets, indices
    
    def test_dataset_creation(self, sample_data):
        """Test dataset can be created."""
        cat, cont, targets, indices = sample_data
        
        dataset = SequentialDataset(cat, cont, targets, indices)
        
        assert len(dataset) == len(indices)
    
    def test_getitem(self, sample_data):
        """Test __getitem__ returns correct shapes."""
        cat, cont, targets, indices = sample_data
        
        dataset = SequentialDataset(cat, cont, targets, indices)
        
        cat_seq, cont_seq, target, length = dataset[0]
        
        assert isinstance(cat_seq, torch.Tensor)
        assert isinstance(cont_seq, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        assert isinstance(length, int)
        
        # Check dimensions
        assert cat_seq.ndim == 2
        assert cont_seq.ndim == 2
        assert cat_seq.shape[0] == length
        assert cont_seq.shape[0] == length
    
    def test_getitem_dtypes(self, sample_data):
        """Test data types of returned items."""
        cat, cont, targets, indices = sample_data
        
        dataset = SequentialDataset(cat, cont, targets, indices)
        
        cat_seq, cont_seq, target, length = dataset[0]
        
        assert cat_seq.dtype == torch.long
        assert cont_seq.dtype == torch.float32
        assert target.dtype == torch.float32
    
    def test_augmentation_flag(self, sample_data):
        """Test augmentation flag affects behavior."""
        cat, cont, targets, indices = sample_data
        
        dataset_no_aug = SequentialDataset(cat, cont, targets, indices, augment=False)
        dataset_aug = SequentialDataset(cat, cont, targets, indices, augment=True)
        
        assert dataset_no_aug.augment == False
        assert dataset_aug.augment == True
    
    def test_training_mode_property(self, sample_data):
        """Test training_mode property."""
        cat, cont, targets, indices = sample_data
        
        dataset = SequentialDataset(cat, cont, targets, indices, augment=True)
        
        assert dataset.training_mode == True
        
        dataset.augment = False
        assert dataset.training_mode == False


class TestCollateSequences:
    """Test collate function for variable-length sequences."""
    
    def test_collate_basic(self):
        """Test basic collation."""
        batch = [
            (torch.randint(0, 10, (5, 3)), torch.randn(5, 4), torch.tensor([1.0, 0.0, 1.0]), 5),
            (torch.randint(0, 10, (7, 3)), torch.randn(7, 4), torch.tensor([0.0, 1.0, 0.0]), 7),
            (torch.randint(0, 10, (3, 3)), torch.randn(3, 4), torch.tensor([1.0, 1.0, 0.0]), 3),
        ]
        
        cat_padded, cont_padded, targets, lengths = collate_sequences(batch)
        
        # Check shapes
        assert cat_padded.shape == (3, 7, 3)  # batch, max_len, features
        assert cont_padded.shape == (3, 7, 4)
        assert targets.shape == (3, 3)
        assert lengths.shape == (3,)
    
    def test_collate_padding_values(self):
        """Test padding uses correct values."""
        batch = [
            (torch.ones(3, 2, dtype=torch.long) * 5, torch.ones(3, 2) * 2.0, torch.tensor([1.0]), 3),
            (torch.ones(5, 2, dtype=torch.long) * 5, torch.ones(5, 2) * 2.0, torch.tensor([0.0]), 5),
        ]
        
        cat_padded, cont_padded, targets, lengths = collate_sequences(batch)
        
        # Check padding values (categorical should be 0, continuous should be 0.0)
        # First sequence should have padding in last 2 positions
        assert cat_padded[0, 3, 0].item() == 0
        assert cont_padded[0, 3, 0].item() == 0.0
    
    def test_collate_lengths_tensor(self):
        """Test lengths are returned as tensor."""
        batch = [
            (torch.randint(0, 10, (4, 3)), torch.randn(4, 4), torch.tensor([1.0]), 4),
            (torch.randint(0, 10, (6, 3)), torch.randn(6, 4), torch.tensor([0.0]), 6),
        ]
        
        _, _, _, lengths = collate_sequences(batch)
        
        assert isinstance(lengths, torch.Tensor)
        assert lengths.dtype == torch.long
        assert lengths.tolist() == [4, 6]


class TestDataLoaderIntegration:
    """Test integration with PyTorch DataLoader."""
    
    @pytest.fixture
    def dataset(self):
        """Create a dataset for testing."""
        n_samples = 50
        categorical = np.random.randint(0, 10, (n_samples, 5))
        continuous = np.random.randn(n_samples, 8).astype(np.float32)
        targets = np.random.randint(0, 2, (n_samples, 3)).astype(np.float32)
        
        indices = [(i, min(i + 5, n_samples - 1), min(i + 4, n_samples - 1)) 
                   for i in range(0, n_samples - 6)]
        
        return SequentialDataset(categorical, continuous, targets, indices)
    
    def test_dataloader_creation(self, dataset):
        """Test DataLoader can be created with the dataset."""
        from torch.utils.data import DataLoader
        
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=collate_sequences
        )
        
        assert len(loader) > 0
    
    def test_dataloader_iteration(self, dataset):
        """Test iterating through DataLoader."""
        from torch.utils.data import DataLoader
        
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=collate_sequences
        )
        
        for batch_idx, (cat, cont, targets, lengths) in enumerate(loader):
            assert cat.ndim == 3
            assert cont.ndim == 3
            assert targets.ndim == 2
            assert lengths.ndim == 1
            
            if batch_idx > 2:
                break
    
    def test_dataloader_batch_sizes(self, dataset):
        """Test DataLoader handles different batch sizes."""
        from torch.utils.data import DataLoader
        
        for batch_size in [1, 4, 8]:
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=collate_sequences
            )
            
            cat, cont, targets, lengths = next(iter(loader))
            assert cat.shape[0] <= batch_size
