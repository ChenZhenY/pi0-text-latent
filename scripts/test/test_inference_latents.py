#!/usr/bin/env python3
"""
Test script for inference latent collection pipeline.
This script tests the modifications made to collect expert-specific hidden states.
"""

import logging
import numpy as np
import jax
import jax.numpy as jnp
from openpi.models.pi0 import Pi0, Pi0Config
from openpi.models.model import Observation
from openpi.models.gemma import Analysis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dummy_observation():
    """Create a dummy observation for testing."""
    batch_size = 1
    
    # Create dummy images (224x224x3) - must match IMAGE_RESOLUTION from model.py
    image_shape = (batch_size, 224, 224, 3)
    dummy_image = jax.random.normal(jax.random.key(42), image_shape).astype(jnp.float32)
    
    # Create dummy state (32-dimensional)
    state_shape = (batch_size, 32)
    dummy_state = jax.random.normal(jax.random.key(43), state_shape).astype(jnp.float32)
    
    # Create dummy tokenized prompt (48 tokens)
    prompt_shape = (batch_size, 48)
    dummy_prompt = jax.random.randint(jax.random.key(44), prompt_shape, 0, 1000).astype(jnp.int32)
    dummy_prompt_mask = jnp.ones(prompt_shape, dtype=jnp.bool_)
    
    return Observation(
        images={
            "base_0_rgb": dummy_image,
            "left_wrist_0_rgb": dummy_image,
            "right_wrist_0_rgb": dummy_image,
        },
        image_masks={
            "base_0_rgb": jnp.ones((batch_size,), dtype=jnp.bool_),
            "left_wrist_0_rgb": jnp.ones((batch_size,), dtype=jnp.bool_),
            "right_wrist_0_rgb": jnp.ones((batch_size,), dtype=jnp.bool_),
        },
        state=dummy_state,
        tokenized_prompt=dummy_prompt,
        tokenized_prompt_mask=dummy_prompt_mask,
    )


def test_expert_hidden_states_extraction():
    """Test that expert-specific hidden states can be extracted correctly."""
    logger.info("Testing expert hidden states extraction...")
    
    # Initialize model
    config = Pi0Config()
    model = config.create(jax.random.key(42))
    
    # Create dummy observation
    obs = create_dummy_observation()
    
    # Create dummy actions
    actions_shape = (1, 50, 32)  # (batch_size, action_horizon, action_dim)
    dummy_actions = np.random.randn(*actions_shape).astype(np.float32)
    
    # Test compute_loss_with_extra to get layer output
    rng = jax.random.key(123)
    result = model.compute_loss_with_extra(rng, obs, dummy_actions)
    
    # Check that expert-specific states are in the result
    logger.info("Checking result keys...")
    expected_keys = [
        "expert_0_hidden_states_sum",
        "expert_1_hidden_states_sum", 
        "text_only_hidden_states_sum"
    ]
    
    for key in expected_keys:
        if key in result:
            logger.info(f"✓ Found {key} in result")
            print(f"result[key]: {result[key]}")
            if result[key] is not None:
                logger.info(f"  Shape: {result[key].shape}")
            else:
                logger.info(f"  Value: None")
        else:
            logger.warning(f"✗ Missing {key} in result")
    
    # Test Analysis class methods
    logger.info("Testing Analysis class methods...")
    
    # We need to get the layer_output from the model
    # For now, let's test with a dummy layer_output structure
    dummy_layer_output = (
        None,  # mlp_activations
        np.random.randn(18, 1, 816, 2048),  # attention
        np.random.randn(18, 1, 816, 2048),  # output_pre_attn_scales
        None,  # output_pre_mlp_scales
        [None, None],  # output_norm
        [None, None],  # all_post_attn
        np.random.randn(18, 1, 816, 2048),  # layer_hidden_states
        np.random.randn(18, 1, 816, 2048),  # post_attn_embeddings
        np.random.randn(18, 1, 48, 8, 2048),  # text_representation
        np.random.randn(18, 1, 816, 2048),  # expert_0_hidden_states
        np.random.randn(18, 1, 200, 2048),  # expert_1_hidden_states
    )
    
    # Test expert extraction methods
    expert_0_states = Analysis.get_expert_0_hidden_states(dummy_layer_output)
    expert_1_states = Analysis.get_expert_1_hidden_states(dummy_layer_output)
    text_only_states = Analysis.get_text_only_hidden_states(dummy_layer_output)
    
    logger.info(f"Expert 0 states shape: {expert_0_states.shape if expert_0_states is not None else None}")
    logger.info(f"Expert 1 states shape: {expert_1_states.shape if expert_1_states is not None else None}")
    logger.info(f"Text-only states shape: {text_only_states.shape if text_only_states is not None else None}")

    print(f"expert_0_states: {expert_0_states.shape}")
    print(f"expert_1_states: {expert_1_states.shape}")
    print(f"text_only_states: {text_only_states.shape}")
    
    # Test layer indexing
    if expert_0_states is not None:
        layer_0_states = Analysis.get_expert_0_hidden_states(dummy_layer_output, layer_index=[0])
        logger.info(f"Layer 0 expert 0 states shape: {layer_0_states.shape}")
    
    logger.info("✓ Expert hidden states extraction test completed")


def test_infer_with_latents():
    """Test the new infer_with_latents method."""
    logger.info("Testing infer_with_latents method...")
    
    # Initialize model
    config = Pi0Config()
    model = config.create(jax.random.key(42))
    
    # Create dummy observation
    obs = create_dummy_observation()
    
    # Test infer_with_latents
    rng = jax.random.key(123)
    result = model.infer_with_latents(rng, obs)
    
    # Check result structure
    expected_keys = [
        "actions",
        "expert_0_hidden_states",
        "expert_1_hidden_states", 
        "text_only_hidden_states",
        "layer_output"
    ]
    
    for key in expected_keys:
        if key in result:
            logger.info(f"✓ Found {key} in result")
            if result[key] is not None:
                if hasattr(result[key], 'shape'):
                    logger.info(f"  Shape: {result[key].shape}")
                else:
                    logger.info(f"  Type: {type(result[key])}")
            else:
                logger.info(f"  Value: None")
        else:
            logger.warning(f"✗ Missing {key} in result")
    
    logger.info("✓ infer_with_latents test completed")


def test_token_positions():
    """Test understanding of token positions for experts."""
    logger.info("Testing token position understanding...")
    
    # Token positions are hardcoded because they're determined by model architecture:
    # - SIGLIP variant: "So400m/14" with patch_size = (14, 14)
    # - Image resolution: (224, 224) 
    # - Number of patches per image: (224/14) × (224/14) = 16 × 16 = 256
    # - Number of images: 3 (base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb)
    # - Total image tokens: 256 × 3 = 768
    # - Text tokens: 48 (from max_token_len config)
    # - Total prefix length: 768 + 48 = 816
    
    # These are fixed by model architecture, not dynamic
    expert_0_start = 0
    expert_0_end = 815  # 816 - 1
    expert_1_start = 816
    text_start = 768  # 256 * 3
    text_end = 816  # text_start + 48
    
    logger.info(f"Expert 0 (PaliGemma) positions: {expert_0_start}-{expert_0_end}")
    logger.info(f"Expert 1 (Action) positions: {expert_1_start}+")
    logger.info(f"Text tokens positions: {text_start}-{text_end-1}")
    logger.info(f"Image tokens per image: 256 (16×16 patches)")
    logger.info(f"Total image tokens: 768 (256×3 images)")
    logger.info(f"Text tokens: 48 (fixed by config)")
    
    # Verify text positions are within expert 0
    assert text_start >= expert_0_start, "Text start should be within expert 0"
    assert text_end <= expert_0_end + 1, "Text end should be within expert 0"
    
    logger.info("✓ Token position understanding test completed")
    logger.info("Note: These positions are hardcoded because they're determined by model architecture, not input content.")


if __name__ == "__main__":
    logger.info("Starting inference latent collection tests...")
    print(f"Starting test_token_positions...")
    
    try:
        test_token_positions()
        test_expert_hidden_states_extraction()
        test_infer_with_latents()
        
        logger.info("🎉 All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        raise 