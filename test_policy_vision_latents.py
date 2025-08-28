#!/usr/bin/env python3
"""Test script to verify policy vision latents handling works correctly."""

import jax
import jax.numpy as jnp
import numpy as np
from src.openpi.models.pi0_linear_probing import Pi0Config, Pi0
from src.openpi.policies.policy import Policy

def test_policy_vision_latents():
    """Test that policy correctly handles vision latents."""
    
    # Create a simple config
    config = Pi0Config()
    
    # Create model
    rng = jax.random.key(42)
    model = config.create(rng)
    
    # Create policy
    policy = Policy(model, rng=rng)
    
    # Create fake observation
    batch_size = 1
    obs_spec, action_spec = config.inputs_spec(batch_size=batch_size)
    
    # Create fake observation data
    fake_obs = {
        'images': {
            'base_0_rgb': np.ones((batch_size, 224, 224, 3), dtype=np.float32),
            'left_wrist_0_rgb': np.ones((batch_size, 224, 224, 3), dtype=np.float32),
            'right_wrist_0_rgb': np.ones((batch_size, 224, 224, 3), dtype=np.float32),
        },
        'image_masks': {
            'base_0_rgb': np.ones((batch_size,), dtype=np.bool_),
            'left_wrist_0_rgb': np.ones((batch_size,), dtype=np.bool_),
            'right_wrist_0_rgb': np.ones((batch_size,), dtype=np.bool_),
        },
        'state': np.ones((batch_size, config.action_dim), dtype=np.float32),
        'tokenized_prompt': np.ones((batch_size, config.max_token_len), dtype=np.int32),
        'tokenized_prompt_mask': np.ones((batch_size, config.max_token_len), dtype=np.bool_),
        'prompt': 'test prompt',
        'done': False,
        'mask_prompt_method': None,
        'layer_to_intervene': None,
        'collect_latents': True,
    }
    
    # Test policy inference with vision latents collection
    print("Testing policy inference with vision latents...")
    outputs = policy.infer(fake_obs)
    
    print(f"Output keys: {list(outputs.keys())}")
    print(f"Actions shape: {outputs['actions'].shape}")
    
    # Check if vision latents are present
    if 'vision_latents' in outputs:
        print(f"Vision latents keys: {list(outputs['vision_latents'].keys())}")
        
        # Check structure of vision latents
        for image_name, latents in outputs['vision_latents'].items():
            print(f"\nVision latents for {image_name}:")
            for key, value in latents.items():
                if hasattr(value, 'shape'):
                    print(f"  {key}: {value.shape}")
                else:
                    print(f"  {key}: {type(value)}")
    else:
        print("No vision latents found in outputs")
    
    # Test policy inference without vision latents collection
    print("\nTesting policy inference without vision latents...")
    fake_obs['collect_latents'] = False
    outputs_no_latents = policy.infer(fake_obs)
    
    print(f"Output keys (no latents): {list(outputs_no_latents.keys())}")
    print(f"Actions shape: {outputs_no_latents['actions'].shape}")
    
    if 'vision_latents' in outputs_no_latents:
        print("Warning: Vision latents found when collect_latents=False")
    else:
        print("Correctly no vision latents when collect_latents=False")
    
    print("\n✅ Policy vision latents handling test completed!")

if __name__ == "__main__":
    test_policy_vision_latents()




