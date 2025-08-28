#!/usr/bin/env python3
"""
Test script to verify accuracy computation logic.
"""

import torch
import torch.nn.functional as F

def compute_accuracy(predictions: torch.Tensor, 
                    targets: torch.Tensor) -> float:
    """Compute classification accuracy using cosine similarity."""
    with torch.no_grad():
        # Normalize predictions and targets
        predictions_norm = F.normalize(predictions, p=2, dim=1)
        targets_norm = F.normalize(targets, p=2, dim=1)
        
        # Compute cosine similarities between all pairs
        similarities = torch.mm(predictions_norm, targets_norm.T)
        
        # For each prediction, find the target with maximum similarity
        max_indices = torch.argmax(similarities, dim=1)
        
        # Check if the predicted target matches the true target
        correct_predictions = 0
        for i in range(predictions.shape[0]):
            predicted_target_idx = max_indices[i]
            true_target = targets[i]
            predicted_target = targets[predicted_target_idx]
            
            # Check if predicted target matches true target (allowing for small numerical differences)
            if torch.allclose(true_target, predicted_target, atol=1e-6):
                correct_predictions += 1
        
        accuracy = correct_predictions / predictions.shape[0]
        return accuracy

def test_accuracy():
    """Test accuracy computation with different scenarios."""
    
    # Test 1: Perfect predictions
    print("Test 1: Perfect predictions")
    targets = torch.randn(10, 8)
    predictions = targets.clone()  # Perfect predictions
    acc = compute_accuracy(predictions, targets)
    print(f"Accuracy: {acc:.4f} (expected: 1.0)")
    
    # Test 2: Random predictions
    print("\nTest 2: Random predictions")
    predictions = torch.randn(10, 8)
    acc = compute_accuracy(predictions, targets)
    print(f"Accuracy: {acc:.4f}")
    
    # Test 3: Multiple samples per task (simulating visual expert)
    print("\nTest 3: Multiple samples per task")
    # Create 3 tasks, each with 3 samples
    task_embeddings = torch.randn(3, 8)
    targets = torch.cat([task_embeddings[0].unsqueeze(0).repeat(3, 1),
                        task_embeddings[1].unsqueeze(0).repeat(3, 1),
                        task_embeddings[2].unsqueeze(0).repeat(3, 1)], dim=0)
    
    # Perfect predictions
    predictions = targets.clone()
    acc = compute_accuracy(predictions, targets)
    print(f"Perfect predictions accuracy: {acc:.4f} (expected: 1.0)")
    
    # Slightly noisy predictions
    predictions = targets + 0.1 * torch.randn_like(targets)
    acc = compute_accuracy(predictions, targets)
    print(f"Noisy predictions accuracy: {acc:.4f}")
    
    # Test 4: Per-task accuracy vs overall accuracy
    print("\nTest 4: Per-task vs overall accuracy")
    # Create predictions that are good within each task but confused between tasks
    predictions = torch.cat([
        task_embeddings[0].unsqueeze(0).repeat(3, 1) + 0.05 * torch.randn(3, 8),  # Task 0: good
        task_embeddings[1].unsqueeze(0).repeat(3, 1) + 0.05 * torch.randn(3, 8),  # Task 1: good
        task_embeddings[0].unsqueeze(0).repeat(3, 1) + 0.05 * torch.randn(3, 8),  # Task 2: confused with task 0
    ], dim=0)
    
    # Overall accuracy
    overall_acc = compute_accuracy(predictions, targets)
    print(f"Overall accuracy: {overall_acc:.4f}")
    
    # Per-task accuracy
    for task_idx in range(3):
        task_mask = torch.arange(9) // 3 == task_idx
        task_predictions = predictions[task_mask]
        task_targets = targets[task_mask]
        task_acc = compute_accuracy(task_predictions, task_targets)
        print(f"Task {task_idx} accuracy: {task_acc:.4f}")

if __name__ == "__main__":
    test_accuracy() 