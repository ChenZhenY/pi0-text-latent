import torch
import torch.nn as nn
import numpy as np
import unittest


def compute_accuracy(predictions: torch.Tensor, 
                    targets: torch.Tensor) -> float:
    """Compute classification accuracy using cosine similarity with proper handling of duplicate targets."""
    with torch.no_grad():
        # Normalize predictions and targets
        predictions_norm = nn.functional.normalize(predictions, p=2, dim=1)
        targets_norm = nn.functional.normalize(targets, p=2, dim=1)
        
        # Compute cosine similarities between all pairs
        similarities = torch.mm(predictions_norm, targets_norm.T)
        
        # For each prediction, find all targets that have the maximum similarity
        # This handles the case where multiple targets might have the same T5 embedding
        max_similarities = torch.max(similarities, dim=1, keepdim=True)[0]
        max_mask = (similarities >= max_similarities - 1e-6)  # Allow for small numerical differences
        
        # For each prediction, check if the true target is among the best matches
        correct_predictions = 0
        for i in range(predictions.shape[0]):
            # Get the true target for this prediction
            true_target = targets[i]
            
            # Find all targets that match the true target (handles duplicates)
            target_matches = torch.all(targets == true_target, dim=1)
            
            # Check if any of the best matches for this prediction include the true target
            best_matches_for_prediction = max_mask[i]
            
            # Only consider it correct if the max similarity is above a threshold
            # This prevents the case where all similarities are 0 (orthogonal vectors)
            max_sim = max_similarities[i]
            if max_sim > 0.1 and torch.any(best_matches_for_prediction & target_matches):
                correct_predictions += 1
        
        accuracy = correct_predictions / predictions.shape[0]
        return accuracy


class TestComputeAccuracy(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def test_basic_case_no_duplicates(self):
        """Test basic case with no duplicate targets."""
        # Create simple test case: 3 predictions, 3 unique targets
        predictions = torch.tensor([
            [1.0, 0.0, 0.0],  # Should match target 0
            [0.0, 1.0, 0.0],  # Should match target 1
            [0.0, 0.0, 1.0]   # Should match target 2
        ], dtype=torch.float32)
        
        targets = torch.tensor([
            [1.0, 0.0, 0.0],  # Target 0
            [0.0, 1.0, 0.0],  # Target 1
            [0.0, 0.0, 1.0]   # Target 2
        ], dtype=torch.float32)
        
        accuracy = compute_accuracy(predictions, targets)
        self.assertEqual(accuracy, 1.0, "Perfect predictions should give 100% accuracy")
    
    def test_duplicate_targets_same_embedding(self):
        """Test case where multiple targets have identical T5 embeddings."""
        # Create test case: 4 predictions, but targets 0 and 1 have identical embeddings
        predictions = torch.tensor([
            [1.0, 0.0, 0.0],  # Should match target 0 or 1 (both are correct)
            [1.0, 0.0, 0.0],  # Should match target 0 or 1 (both are correct)
            [0.0, 0.0, 1.0],  # Should match target 2
            [0.0, 1.0, 0.0]   # Should match target 3
        ], dtype=torch.float32)
        
        targets = torch.tensor([
            [1.0, 0.0, 0.0],  # Target 0 (identical to target 1)
            [1.0, 0.0, 0.0],  # Target 1 (identical to target 0)
            [0.0, 0.0, 1.0],  # Target 2
            [0.0, 1.0, 0.0]   # Target 3
        ], dtype=torch.float32)
        
        accuracy = compute_accuracy(predictions, targets)
        self.assertEqual(accuracy, 1.0, "All predictions should be correct when targets have identical embeddings")
    
    def test_duplicate_targets_with_imperfect_predictions(self):
        """Test case with duplicate targets and some imperfect predictions."""
        # Create test case: 5 predictions, targets 0 and 1 have identical embeddings
        predictions = torch.tensor([
            [0.9, 0.1, 0.0],  # Should match target 0 or 1 (both are correct)
            [0.8, 0.2, 0.0],  # Should match target 0 or 1 (both are correct)
            [0.1, 0.9, 0.0],  # Should match target 0 or 1 (both are correct)
            [0.0, 0.0, 1.0],  # Should match target 3 or 4 (both are correct)
            [0.0, 0.1, 0.9]   # Should match target 3 or 4 (both are correct)
        ], dtype=torch.float32)
        
        targets = torch.tensor([
            [1.0, 0.0, 0.0],  # Target 0 (identical to target 1)
            [1.0, 0.0, 0.0],  # Target 1 (identical to target 0)
            [1.0, 0.0, 0.0],  # Target 2 (identical to targets 0 and 1)
            [0.0, 0.0, 1.0],  # Target 3
            [0.0, 0.0, 1.0]   # Target 4 (identical to target 3)
        ], dtype=torch.float32)
        
        accuracy = compute_accuracy(predictions, targets)
        expected_accuracy = 1.0  # All predictions should be correct
        self.assertAlmostEqual(accuracy, expected_accuracy, places=5, 
                              msg=f"Expected {expected_accuracy}, got {accuracy}")
    
    def test_multiple_duplicate_groups(self):
        """Test case with multiple groups of duplicate targets."""
        # Create test case with multiple duplicate groups
        # Each prediction should match its corresponding target group
        predictions = torch.tensor([
            [1.0, 0.0, 0.0],  # Should match targets 0,1,2 (Group 1)
            [0.0, 1.0, 0.0],  # Should match targets 3,4 (Group 2)
            [0.0, 0.0, 1.0],  # Should match target 5 (Unique)
            [1.0, 0.0, 0.0],  # Should match targets 0,1,2 (Group 1)
            [0.0, 1.0, 0.0],  # Should match targets 3,4 (Group 2)
        ], dtype=torch.float32)
        
        targets = torch.tensor([
            [1.0, 0.0, 0.0],  # Target 0 (Group 1) - matches prediction 0
            [0.0, 1.0, 0.0],  # Target 1 (Group 2) - matches prediction 1
            [0.0, 0.0, 1.0],  # Target 2 (Unique) - matches prediction 2
            [1.0, 0.0, 0.0],  # Target 3 (Group 1) - matches prediction 3
            [0.0, 1.0, 0.0],  # Target 4 (Group 2) - matches prediction 4
        ], dtype=torch.float32)
        
        accuracy = compute_accuracy(predictions, targets)
        self.assertEqual(accuracy, 1.0, "All predictions should be correct with multiple duplicate groups")
    
    def test_edge_case_single_prediction(self):
        """Test edge case with single prediction and target."""
        predictions = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
        targets = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
        
        accuracy = compute_accuracy(predictions, targets)
        self.assertEqual(accuracy, 1.0, "Single correct prediction should give 100% accuracy")
    
    def test_edge_case_all_wrong_predictions(self):
        """Test edge case where all predictions are wrong."""
        predictions = torch.tensor([
            [0.0, 1.0, 0.0],  # Wrong prediction - should not match any target
            [0.0, 0.0, 1.0],  # Wrong prediction - should not match any target
        ], dtype=torch.float32)
        
        targets = torch.tensor([
            [1.0, 0.0, 0.0],  # Target 0
            [1.0, 0.0, 0.0],  # Target 1 (identical to target 0)
        ], dtype=torch.float32)
        
        accuracy = compute_accuracy(predictions, targets)
        self.assertEqual(accuracy, 0.0, "All wrong predictions should give 0% accuracy")
    
    def test_numerical_precision(self):
        """Test that the function handles small numerical differences correctly."""
        # Create predictions that are very close but not exactly equal
        predictions = torch.tensor([
            [1.0, 0.0, 0.0],
            [1.0 + 1e-7, 0.0, 0.0],  # Very close to target
        ], dtype=torch.float32)
        
        targets = torch.tensor([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],  # Identical to target 0
        ], dtype=torch.float32)
        
        accuracy = compute_accuracy(predictions, targets)
        self.assertEqual(accuracy, 1.0, "Small numerical differences should not affect accuracy")
    
    def test_realistic_t5_embeddings(self):
        """Test with more realistic T5-like embeddings."""
        # Create random T5-like embeddings (768-dimensional)
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create 3 unique embeddings
        unique_embeddings = torch.randn(3, 768)
        
        # Create targets with duplicates
        targets = torch.cat([
            unique_embeddings[0].unsqueeze(0).repeat(2, 1),  # Duplicate of embedding 0
            unique_embeddings[1].unsqueeze(0).repeat(3, 1),  # Triplicate of embedding 1
            unique_embeddings[2].unsqueeze(0).repeat(1, 1),  # Single copy of embedding 2
        ], dim=0)
        
        # Create predictions that should match the targets
        predictions = torch.cat([
            unique_embeddings[0].unsqueeze(0).repeat(2, 1),  # Should match targets 0,1
            unique_embeddings[1].unsqueeze(0).repeat(3, 1),  # Should match targets 2,3,4
            unique_embeddings[2].unsqueeze(0).repeat(1, 1),  # Should match target 5
        ], dim=0)
        
        accuracy = compute_accuracy(predictions, targets)
        self.assertEqual(accuracy, 1.0, "Realistic T5 embeddings with duplicates should work correctly")
    
    def test_compare_with_old_implementation(self):
        """Test that the new implementation gives different (correct) results than the old one."""
        # Create a case where the old implementation would fail
        predictions = torch.tensor([
            [0.9, 0.1, 0.0],  # Should match target 0 or 1
            [0.8, 0.2, 0.0],  # Should match target 0 or 1
        ], dtype=torch.float32)
        
        targets = torch.tensor([
            [1.0, 0.0, 0.0],  # Target 0
            [1.0, 0.0, 0.0],  # Target 1 (identical to target 0)
        ], dtype=torch.float32)
        
        # New implementation
        new_accuracy = compute_accuracy(predictions, targets)
        
        # Old implementation (simulated)
        def old_compute_accuracy(predictions, targets):
            with torch.no_grad():
                predictions_norm = nn.functional.normalize(predictions, p=2, dim=1)
                targets_norm = nn.functional.normalize(targets, p=2, dim=1)
                similarities = torch.mm(predictions_norm, targets_norm.T)
                predicted_indices = torch.argmax(similarities, dim=1)
                unique_targets, true_indices = torch.unique(targets, dim=0, return_inverse=True)
                accuracy = torch.mean((predicted_indices == true_indices).float()).item()
                return accuracy
        
        old_accuracy = old_compute_accuracy(predictions, targets)
        
        # The new implementation should give better accuracy in this case
        self.assertGreaterEqual(new_accuracy, old_accuracy, 
                               "New implementation should not be worse than old implementation")
        
        # In this specific case, new should be 100% while old might be less
        self.assertEqual(new_accuracy, 1.0, "New implementation should give 100% accuracy for this case")


if __name__ == '__main__':
    unittest.main()
