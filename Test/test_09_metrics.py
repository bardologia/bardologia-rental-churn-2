"""
Test 09: Metrics and Evaluation
Tests metric calculations used for model evaluation.
"""

import pytest
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, brier_score_loss
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestMetricCalculations:
    """Test metric calculations match sklearn."""
    
    @pytest.fixture
    def predictions_and_targets(self):
        """Generate sample predictions and targets."""
        np.random.seed(42)
        n_samples = 100
        
        # Targets (binary)
        targets = np.random.randint(0, 2, n_samples).astype(float)
        
        # Predictions (probabilities)
        # Make predictions somewhat correlated with targets
        predictions = targets * 0.6 + np.random.uniform(0, 0.4, n_samples)
        predictions = np.clip(predictions, 0.01, 0.99)
        
        return predictions, targets
    
    def test_roc_auc_calculation(self, predictions_and_targets):
        """Test ROC-AUC calculation."""
        predictions, targets = predictions_and_targets
        
        auc = roc_auc_score(targets, predictions)
        
        assert 0 <= auc <= 1
        # With correlated predictions, AUC should be above 0.5
        assert auc > 0.5
    
    def test_average_precision(self, predictions_and_targets):
        """Test Average Precision calculation."""
        predictions, targets = predictions_and_targets
        
        ap = average_precision_score(targets, predictions)
        
        assert 0 <= ap <= 1
    
    def test_brier_score(self, predictions_and_targets):
        """Test Brier score calculation."""
        predictions, targets = predictions_and_targets
        
        brier = brier_score_loss(targets, predictions)
        
        # Brier score is MSE of probabilities, should be in [0, 1]
        assert 0 <= brier <= 1
    
    def test_f1_score(self, predictions_and_targets):
        """Test F1 score calculation."""
        predictions, targets = predictions_and_targets
        
        # Convert probabilities to binary predictions
        binary_predictions = (predictions > 0.5).astype(int)
        
        f1 = f1_score(targets, binary_predictions)
        
        assert 0 <= f1 <= 1


class TestMultiOutputMetrics:
    """Test metrics for multi-output predictions (short, medium, long)."""
    
    @pytest.fixture
    def multi_output_data(self):
        """Generate multi-output predictions and targets."""
        np.random.seed(42)
        n_samples = 100
        n_outputs = 3
        
        targets = np.random.randint(0, 2, (n_samples, n_outputs)).astype(float)
        predictions = np.random.uniform(0.1, 0.9, (n_samples, n_outputs))
        
        return predictions, targets
    
    def test_per_output_auc(self, multi_output_data):
        """Test AUC calculation for each output."""
        predictions, targets = multi_output_data
        
        aucs = []
        for i in range(predictions.shape[1]):
            try:
                auc = roc_auc_score(targets[:, i], predictions[:, i])
                aucs.append(auc)
            except ValueError:
                # Skip if only one class in targets
                pass
        
        assert len(aucs) > 0
        for auc in aucs:
            assert 0 <= auc <= 1
    
    def test_average_auc(self, multi_output_data):
        """Test average AUC across outputs."""
        predictions, targets = multi_output_data
        
        aucs = []
        for i in range(predictions.shape[1]):
            try:
                auc = roc_auc_score(targets[:, i], predictions[:, i])
                aucs.append(auc)
            except ValueError:
                pass
        
        if aucs:
            avg_auc = np.mean(aucs)
            assert 0 <= avg_auc <= 1


class TestThresholdOptimization:
    """Test optimal threshold finding."""
    
    def test_find_optimal_threshold(self):
        """Test finding optimal threshold for F1 score."""
        np.random.seed(42)
        n_samples = 200
        
        targets = np.random.randint(0, 2, n_samples).astype(float)
        # Predictions somewhat correlated with targets
        predictions = targets * 0.5 + np.random.uniform(0, 0.5, n_samples)
        predictions = np.clip(predictions, 0.01, 0.99)
        
        best_threshold = 0.5
        best_f1 = 0.0
        
        for threshold in np.arange(0.1, 0.9, 0.05):
            binary_preds = (predictions > threshold).astype(int)
            current_f1 = f1_score(targets, binary_preds)
            
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_threshold = threshold
        
        assert 0 < best_threshold < 1
        assert best_f1 > 0


class TestPredictionStatistics:
    """Test prediction statistics and distributions."""
    
    def test_prediction_distribution(self):
        """Test that predictions have reasonable distribution."""
        torch.manual_seed(42)
        
        # Simulated logits
        logits = torch.randn(100, 3)
        predictions = torch.sigmoid(logits)
        
        # Check predictions are in valid range
        assert (predictions >= 0).all()
        assert (predictions <= 1).all()
        
        # Check reasonable spread
        assert predictions.std() > 0.1
    
    def test_class_balance_handling(self):
        """Test handling of imbalanced classes."""
        np.random.seed(42)
        
        # Highly imbalanced targets (95% negative)
        n_samples = 1000
        targets = np.zeros(n_samples)
        targets[:50] = 1  # Only 5% positive
        
        # Random predictions
        predictions = np.random.uniform(0, 1, n_samples)
        
        # AUC should still be calculable
        try:
            auc = roc_auc_score(targets, predictions)
            assert 0 <= auc <= 1
        except ValueError:
            pytest.fail("Should handle imbalanced classes")
    
    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        targets = np.array([0, 0, 0, 1, 1, 1], dtype=float)
        predictions = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9])
        
        auc = roc_auc_score(targets, predictions)
        assert auc == 1.0
        
        binary_preds = (predictions > 0.5).astype(int)
        f1 = f1_score(targets, binary_preds)
        assert f1 == 1.0


class TestCalibration:
    """Test prediction calibration metrics."""
    
    def test_calibration_error(self):
        """Test expected calibration error concept."""
        np.random.seed(42)
        n_samples = 1000
        
        targets = np.random.randint(0, 2, n_samples).astype(float)
        
        # Well-calibrated predictions: predicted probability matches actual frequency
        # For simplicity, use targets with noise as predictions
        predictions = np.clip(targets + np.random.normal(0, 0.2, n_samples), 0.01, 0.99)
        
        # Bin predictions and check calibration
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        
        calibration_errors = []
        for i in range(n_bins):
            mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i + 1])
            if mask.sum() > 0:
                avg_pred = predictions[mask].mean()
                avg_target = targets[mask].mean()
                calibration_errors.append(abs(avg_pred - avg_target))
        
        # Average calibration error
        if calibration_errors:
            ece = np.mean(calibration_errors)
            assert ece >= 0
            assert ece <= 1
