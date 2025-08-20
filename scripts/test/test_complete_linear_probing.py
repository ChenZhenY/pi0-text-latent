#!/usr/bin/env python3
"""
Test script for complete linear probing implementation.
This script creates mock data to test the full training pipeline.
"""

import logging
import pathlib
import pickle
import tempfile
import numpy as np
import torch
from linear_probing_inference import (
    DataConfig, TrainingConfig, InferenceLatentDataset, 
    create_t5_labels, LinearProbe, CosineSimilarityLoss,
    train, analyze_results, save_results
)


def create_mock_data():
    """Create mock inference latent data for testing."""
    mock_data = {
        "task_name": "pick_up_the_cream_cheese_and_place_it_in_the_basket",
        "task_id": 0,
        "task_description": "pick up the cream cheese and place it in the basket",
        "episodes": {
            "episode_0": {
                "rollout_steps": {
                    "step_10": {
                        "timestep": 10,
                        "action": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                        "vlm_layer_output": {
                            "hidden_states": np.random.randn(18, 1, 816, 2048).astype(np.float32),
                            "post_attn_embedding": np.random.randn(18, 1, 816, 2048).astype(np.float32),
                            "pre_attn_norm_scales": np.random.randn(18, 1, 816, 2048).astype(np.float32),
                            "pre_mlp_norm_scales": np.random.randn(18, 1, 816, 2048).astype(np.float32)
                        },
                        "action_expert_hidden_state_t0.9": {
                            "hidden_states": np.random.randn(18, 1, 200, 2048).astype(np.float32),
                            "post_attn_embedding": np.random.randn(18, 1, 200, 2048).astype(np.float32),
                            "pre_attn_norm_scales": np.random.randn(18, 1, 200, 2048).astype(np.float32),
                            "pre_mlp_norm_scales": np.random.randn(18, 1, 200, 2048).astype(np.float32)
                        },
                        "observation": {
                            "agentview_image": np.random.randn(256, 256, 3).astype(np.uint8),
                            "wrist_image": np.random.randn(256, 256, 3).astype(np.uint8),
                            "robot_state": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
                        }
                    }
                },
                "metadata": {
                    "success": True,
                    "total_steps": 45
                }
            },
            "episode_1": {
                "rollout_steps": {
                    "step_10": {
                        "timestep": 10,
                        "action": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                        "vlm_layer_output": {
                            "hidden_states": np.random.randn(18, 1, 816, 2048).astype(np.float32),
                            "post_attn_embedding": np.random.randn(18, 1, 816, 2048).astype(np.float32),
                            "pre_attn_norm_scales": np.random.randn(18, 1, 816, 2048).astype(np.float32),
                            "pre_mlp_norm_scales": np.random.randn(18, 1, 816, 2048).astype(np.float32)
                        },
                        "action_expert_hidden_state_t0.9": {
                            "hidden_states": np.random.randn(18, 1, 200, 2048).astype(np.float32),
                            "post_attn_embedding": np.random.randn(18, 1, 200, 2048).astype(np.float32),
                            "pre_attn_norm_scales": np.random.randn(18, 1, 200, 2048).astype(np.float32),
                            "pre_mlp_norm_scales": np.random.randn(18, 1, 200, 2048).astype(np.float32)
                        },
                        "observation": {
                            "agentview_image": np.random.randn(256, 256, 3).astype(np.uint8),
                            "wrist_image": np.random.randn(256, 256, 3).astype(np.uint8),
                            "robot_state": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                        }
                    }
                },
                "metadata": {
                    "success": False,
                    "total_steps": 30
                }
            }
        }
    }
    
    # Create second task with different description
    mock_data_2 = {
        "task_name": "open_the_top_drawer_and_put_the_bowl_inside",
        "task_id": 1,
        "task_description": "open the top drawer and put the bowl inside",
        "episodes": {
            "episode_0": {
                "rollout_steps": {
                    "step_10": {
                        "timestep": 10,
                        "action": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                        "vlm_layer_output": {
                            "hidden_states": np.random.randn(18, 1, 816, 2048).astype(np.float32),
                            "post_attn_embedding": np.random.randn(18, 1, 816, 2048).astype(np.float32),
                            "pre_attn_norm_scales": np.random.randn(18, 1, 816, 2048).astype(np.float32),
                            "pre_mlp_norm_scales": np.random.randn(18, 1, 816, 2048).astype(np.float32)
                        },
                        "action_expert_hidden_state_t0.9": {
                            "hidden_states": np.random.randn(18, 1, 200, 2048).astype(np.float32),
                            "post_attn_embedding": np.random.randn(18, 1, 200, 2048).astype(np.float32),
                            "pre_attn_norm_scales": np.random.randn(18, 1, 200, 2048).astype(np.float32),
                            "pre_mlp_norm_scales": np.random.randn(18, 1, 200, 2048).astype(np.float32)
                        },
                        "observation": {
                            "agentview_image": np.random.randn(256, 256, 3).astype(np.uint8),
                            "wrist_image": np.random.randn(256, 256, 3).astype(np.uint8),
                            "robot_state": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                        }
                    }
                },
                "metadata": {
                    "success": True,
                    "total_steps": 40
                }
            }
        }
    }
    
    return mock_data, mock_data_2


def test_complete_pipeline():
    """Test the complete linear probing pipeline."""
    print("\n=== Testing Complete Pipeline ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock data files
        mock_data, mock_data_2 = create_mock_data()
        
        data_path = pathlib.Path(temp_dir) / "data" / "inference_latents"
        data_path.mkdir(parents=True, exist_ok=True)
        
        # Save mock data
        with open(data_path / "task_0_pick_up_the_cream_cheese_and_place_it_in_the_basket.pkl", 'wb') as f:
            pickle.dump(mock_data, f)
        with open(data_path / "task_1_open_the_top_drawer_and_put_the_bowl_inside.pkl", 'wb') as f:
            pickle.dump(mock_data_2, f)
        
        # Test configuration
        data_config = DataConfig(
            data_path=str(data_path),
            task_range=(0, 2),
            episode_range=(0, 2),
            rollout_step=10,
            expert="vlm",
            layer=5
        )
        
        training_config = TrainingConfig(
            learning_rate=0.001,
            num_epochs=5,  # Small number for testing
            batch_size=2,
            seed=42
        )
        
        # Load dataset
        dataset = InferenceLatentDataset(data_config)
        print(f"✓ Dataset loaded: {len(dataset)} samples, {len(dataset.task_descriptions)} tasks")
        
        # Create T5 labels
        t5_labels = create_t5_labels(dataset.task_descriptions)
        print(f"✓ T5 labels created: {t5_labels.shape}")
        
        # Set device
        device = torch.device("cpu")  # Use CPU for testing
        print(f"✓ Using device: {device}")
        
        # Train model
        print("Training model...")
        model, training_metrics = train(dataset, t5_labels, training_config, device)
        print(f"✓ Training completed. Best loss: {training_metrics['best_loss']:.4f}")
        
        # Analyze results
        print("Analyzing results...")
        results = analyze_results(model, dataset, t5_labels, data_config, device)
        print(f"✓ Analysis completed. Accuracy: {results['accuracy']:.4f}")
        
        # Save results
        output_dir = pathlib.Path(temp_dir) / "results"
        save_results(results, data_config, str(output_dir))
        print(f"✓ Results saved to {output_dir}")
        
        # Check saved files
        saved_files = list(output_dir.glob("*"))
        print(f"✓ Saved {len(saved_files)} files")
        
        return True


def test_model_components():
    """Test individual model components."""
    print("\n=== Testing Model Components ===")
    
    # Test LinearProbe
    model = LinearProbe(input_dim=2048, output_dim=512)
    x = torch.randn(4, 2048)
    output = model(x)
    print(f"✓ LinearProbe: input {x.shape} -> output {output.shape}")
    
    # Test CosineSimilarityLoss
    criterion = CosineSimilarityLoss()
    predictions = torch.randn(4, 512)
    targets = torch.randn(4, 512)
    loss = criterion(predictions, targets)
    print(f"✓ CosineSimilarityLoss: {loss.item():.4f}")
    
    return True


def main():
    """Run all tests."""
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Complete Linear Probing Implementation")
    print("=" * 60)
    
    tests = [
        test_model_components,
        test_complete_pipeline
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Complete implementation is working correctly.")
    else:
        print("❌ Some tests failed. Please check the implementation.")


if __name__ == "__main__":
    main()
