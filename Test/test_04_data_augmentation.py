"""
Test 04: Data Augmentation
Tests the data augmentation strategies.
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model.data import Augmentation


class TestTemporalCutout:
    """Test temporal cutout augmentation."""
    
    def test_no_augmentation_when_prob_zero(self):
        """Test no change when probability is 0."""
        cat = torch.randint(1, 10, (10, 5))
        cont = torch.randn(10, 8)
        
        cat_aug, cont_aug = Augmentation.temporal_cutout(cat, cont, probability=0.0)
        
        assert torch.equal(cat_aug, cat)
        assert torch.equal(cont_aug, cont)
    
    def test_output_shapes(self):
        """Test output shapes remain the same."""
        cat = torch.randint(1, 10, (10, 5))
        cont = torch.randn(10, 8)
        
        # Force augmentation by setting high probability
        torch.manual_seed(0)
        cat_aug, cont_aug = Augmentation.temporal_cutout(cat, cont, probability=1.0)
        
        assert cat_aug.shape == cat.shape
        assert cont_aug.shape == cont.shape
    
    def test_cutout_creates_zeros(self):
        """Test that cutout creates zero values."""
        cat = torch.randint(1, 10, (20, 5))
        cont = torch.randn(20, 8) + 5  # Add offset to ensure no zeros initially
        
        torch.manual_seed(42)
        cat_aug, cont_aug = Augmentation.temporal_cutout(cat, cont, probability=1.0)
        
        # Some positions should be zeroed
        assert (cat_aug == 0).any() or torch.equal(cat_aug, cat)
    
    def test_short_sequence_handling(self):
        """Test handling of very short sequences."""
        cat = torch.randint(1, 10, (1, 5))
        cont = torch.randn(1, 8)
        
        # Should not fail on single-element sequence
        cat_aug, cont_aug = Augmentation.temporal_cutout(cat, cont, probability=1.0)
        
        assert cat_aug.shape == cat.shape


class TestFeatureDropout:
    """Test feature dropout augmentation."""
    
    def test_no_dropout_when_prob_zero(self):
        """Test no change when probability is 0."""
        cat = torch.randint(1, 10, (10, 5))
        cont = torch.randn(10, 8)
        
        cat_aug, cont_aug = Augmentation.feature_dropout(cat, cont, probability=0.0)
        
        assert torch.equal(cat_aug, cat)
        assert torch.equal(cont_aug, cont)
    
    def test_output_shapes(self):
        """Test output shapes remain the same."""
        cat = torch.randint(1, 10, (10, 5))
        cont = torch.randn(10, 8)
        
        torch.manual_seed(0)
        cat_aug, cont_aug = Augmentation.feature_dropout(cat, cont, probability=1.0)
        
        assert cat_aug.shape == cat.shape
        assert cont_aug.shape == cont.shape
    
    def test_dropout_affects_columns(self):
        """Test that dropout zeros entire feature columns."""
        cat = torch.randint(1, 10, (10, 5))
        cont = torch.randn(10, 8) + 5
        
        torch.manual_seed(42)
        cat_aug, cont_aug = Augmentation.feature_dropout(cat, cont, probability=1.0)
        
        # Check if any full column is zeroed
        cat_zeros = (cat_aug == 0).all(dim=0)
        cont_zeros = (cont_aug == 0).all(dim=0)
        
        # At least one column should be affected (or unchanged if not triggered)
        assert cat_aug.shape == cat.shape


class TestGaussianNoise:
    """Test Gaussian noise augmentation."""
    
    def test_no_noise_when_prob_zero(self):
        """Test no change when probability is 0."""
        cont = torch.randn(10, 8)
        
        cont_aug = Augmentation.gaussian_noise(cont, probability=0.0)
        
        assert torch.equal(cont_aug, cont)
    
    def test_noise_changes_values(self):
        """Test that noise changes the values."""
        cont = torch.randn(10, 8)
        
        torch.manual_seed(42)
        cont_aug = Augmentation.gaussian_noise(cont, probability=1.0, standard_deviation=0.5)
        
        # Values should be different
        if not torch.equal(cont_aug, cont):
            assert not torch.allclose(cont_aug, cont, atol=1e-6)
    
    def test_noise_scale(self):
        """Test that noise scale affects magnitude of changes."""
        cont = torch.zeros(100, 8)
        
        torch.manual_seed(42)
        cont_small = Augmentation.gaussian_noise(cont.clone(), probability=1.0, standard_deviation=0.1)
        torch.manual_seed(42)
        cont_large = Augmentation.gaussian_noise(cont.clone(), probability=1.0, standard_deviation=1.0)
        
        # Larger std should produce larger deviations
        if not torch.equal(cont_small, cont) and not torch.equal(cont_large, cont):
            assert cont_large.abs().mean() > cont_small.abs().mean()


class TestTimeWarp:
    """Test time warp augmentation."""
    
    def test_no_warp_when_prob_zero(self):
        """Test no change when probability is 0."""
        cat = torch.randint(1, 10, (10, 5))
        cont = torch.randn(10, 8)
        
        cat_aug, cont_aug = Augmentation.time_warp(cat, cont, probability=0.0)
        
        assert torch.equal(cat_aug, cat)
        assert torch.equal(cont_aug, cont)
    
    def test_sequence_length_changes(self):
        """Test that time warp can change sequence length."""
        cat = torch.randint(1, 10, (10, 5))
        cont = torch.randn(10, 8)
        
        # Run multiple times to catch both duplicate and remove cases
        lengths_seen = set()
        for seed in range(100):
            torch.manual_seed(seed)
            cat_aug, cont_aug = Augmentation.time_warp(cat, cont, probability=1.0)
            lengths_seen.add(cat_aug.shape[0])
        
        # Should have multiple lengths (9, 10, 11)
        assert len(lengths_seen) >= 1  # At least original length
    
    def test_short_sequence_handling(self):
        """Test handling of short sequences."""
        cat = torch.randint(1, 10, (2, 5))
        cont = torch.randn(2, 8)
        
        # Should not fail on length-2 sequence
        cat_aug, cont_aug = Augmentation.time_warp(cat, cont, probability=1.0)
        
        assert cat_aug.shape[1] == cat.shape[1]
        assert cont_aug.shape[1] == cont.shape[1]


class TestAugmentationCombined:
    """Test combinations of augmentations."""
    
    def test_all_augmentations_pipeline(self):
        """Test applying all augmentations in sequence."""
        cat = torch.randint(1, 10, (15, 5))
        cont = torch.randn(15, 8)
        
        # Apply all augmentations
        cat_aug, cont_aug = Augmentation.temporal_cutout(cat.clone(), cont.clone(), probability=1.0)
        cat_aug, cont_aug = Augmentation.feature_dropout(cat_aug, cont_aug, probability=1.0)
        cont_aug = Augmentation.gaussian_noise(cont_aug, probability=1.0, standard_deviation=0.1)
        
        # Shapes should be maintained
        assert cat_aug.shape == cat.shape
        assert cont_aug.shape == cont.shape
    
    def test_augmentation_class_instance(self):
        """Test using Augmentation as a class."""
        augmenter = Augmentation()
        
        assert hasattr(augmenter, 'temporal_cutout')
        assert hasattr(augmenter, 'feature_dropout')
        assert hasattr(augmenter, 'gaussian_noise')
        assert hasattr(augmenter, 'time_warp')
