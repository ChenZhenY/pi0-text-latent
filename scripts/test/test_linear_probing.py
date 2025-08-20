#!/usr/bin/env python3
"""
Test script for linear probing Phase 1 and 2 implementation.
This script creates mock data to test the data loading and T5 label generation.
"""

import logging
import pathlib
import pickle
import tempfile
import numpy as np
from linear_probing_inference import DataConfig, InferenceLatentDataset, create_t5_labels, sanity_check_data_loading


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
                        "action_expert_hidden_state_t0.8": {
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
                    },
                    "step_20": {
                        "timestep": 20,
                        "action": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
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
                            "robot_state": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
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


def test_vlm_expert():
    """Test VLM expert data loading."""
    print("\n=== Testing VLM Expert ===")
    
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
        
        # Test VLM expert
        config = DataConfig(
            data_path=str(data_path),
            task_range=(0, 2),
            episode_range=(0, 2),
            rollout_step=10,
            expert="vlm",
            layer=5
        )
        
        dataset = InferenceLatentDataset(config)
        print(f"✓ VLM Dataset loaded: {len(dataset)} samples, {len(dataset.task_descriptions)} tasks")
        
        # Test T5 labels
        t5_labels = create_t5_labels(dataset.task_descriptions)
        print(f"✓ T5 labels created: {t5_labels.shape}")
        
        # Test sample
        features, label = dataset[0]
        print(f"✓ Sample features: {features.shape}, label: {label}")
        
        return True


def test_action_expert():
    """Test Action expert data loading."""
    print("\n=== Testing Action Expert ===")
    
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
        
        # Test Action expert
        config = DataConfig(
            data_path=str(data_path),
            task_range=(0, 2),
            episode_range=(0, 2),
            rollout_step=10,
            expert="action",
            layer=5,
            action_timestep=0.9
        )
        
        dataset = InferenceLatentDataset(config)
        print(f"✓ Action Dataset loaded: {len(dataset)} samples, {len(dataset.task_descriptions)} tasks")
        
        # Test sample
        features, label = dataset[0]
        print(f"✓ Sample features: {features.shape}, label: {label}")
        
        return True


def test_text_only_expert():
    """Test Text-only expert data loading."""
    print("\n=== Testing Text-Only Expert ===")
    
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
        
        # Test Text-only expert
        config = DataConfig(
            data_path=str(data_path),
            task_range=(0, 2),
            episode_range=(0, 2),
            rollout_step=10,
            expert="text_only",
            layer=5
        )
        
        dataset = InferenceLatentDataset(config)
        print(f"✓ Text-Only Dataset loaded: {len(dataset)} samples, {len(dataset.task_descriptions)} tasks")
        
        # Test sample
        features, label = dataset[0]
        print(f"✓ Sample features: {features.shape}, label: {label}")
        
        return True


def test_sanity_checks():
    """Test sanity check function."""
    print("\n=== Testing Sanity Checks ===")
    
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
        
        # Test sanity checks
        config = DataConfig(
            data_path=str(data_path),
            task_range=(0, 2),
            episode_range=(0, 2),
            rollout_step=10,
            expert="vlm",
            layer=5
        )
        
        success = sanity_check_data_loading(config)
        if success:
            print("✓ Sanity checks passed!")
        else:
            print("✗ Sanity checks failed!")
        
        return success


def main():
    """Run all tests."""
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Linear Probing Phase 1 and 2 Implementation")
    print("=" * 60)
    
    tests = [
        test_vlm_expert,
        test_action_expert,
        test_text_only_expert,
        test_sanity_checks
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Phase 1 and 2 implementation is working correctly.")
    else:
        print("❌ Some tests failed. Please check the implementation.")


if __name__ == "__main__":
    main()
