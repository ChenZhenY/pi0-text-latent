# Vision Latents Collection in Pi0 Model

This document explains how to collect vision encoder latents during action sampling in the Pi0 model.

## Overview

The Pi0 model has been enhanced to collect and return vision encoder latents during action sampling. This allows you to access intermediate representations from the siglip vision encoder for analysis, visualization, or further processing.

## Changes Made

### 1. Modified `embed_prefix` method

The `embed_prefix` method now returns vision latents along with the existing tokens:

```python
def embed_prefix(self, obs: _model.Observation) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"], dict]:
    # ... existing code ...
    return tokens, input_mask, ar_mask, vision_latents
```

### 2. Modified `sample_actions` method

The `sample_actions` method now returns vision latents:

```python
def sample_actions(self, rng, observation, **kwargs) -> tuple[_model.Actions, dict]:
    # ... existing code ...
    return actions, vision_latents
```

### 3. Modified `sample_actions_with_latents` method

The `sample_actions_with_latents` method now returns vision latents along with layer outputs:

```python
def sample_actions_with_latents(self, rng, observation, **kwargs) -> tuple[_model.Actions, dict, list]:
    # ... existing code ...
    return actions, vision_latents, layer_outputs
```

## Vision Latents Structure

The `vision_latents` dictionary contains vision encoder outputs for each image. The structure is:

```python
vision_latents = {
    "base_0_rgb": {
        "encoded": jnp.array,      # Final encoded features [B, H*W, D]
        "pre_logits": jnp.array,   # Pooled features [B, D]
        "pre_logits_2d": jnp.array, # 2D features [B, H, W, D]
        "stem": jnp.array,         # Initial patch embeddings
        "with_posemb": jnp.array,  # Features with positional embeddings
        "encoder": dict,           # All transformer layer outputs
    },
    "left_wrist_0_rgb": { ... },
    "right_wrist_0_rgb": { ... },
}
```

## Usage Examples

### Basic Usage

```python
from src.openpi.models.pi0_linear_probing import Pi0Config, Pi0

# Create model
config = Pi0Config()
model = config.create(rng)

# Sample actions and get vision latents
actions, vision_latents = model.sample_actions(rng, observation, num_steps=10)

# Access vision latents for specific images
base_latents = vision_latents["base_0_rgb"]
encoded_features = base_latents["encoded"]  # [B, H*W, D]
pooled_features = base_latents["pre_logits"]  # [B, D]
```

### Advanced Usage with All Latents

```python
# Get both vision latents and layer outputs
actions, vision_latents, layer_outputs = model.sample_actions_with_latents(
    rng, observation, num_steps=10
)

# Access different types of latents
for image_name, latents in vision_latents.items():
    print(f"Image: {image_name}")
    print(f"  Encoded shape: {latents['encoded'].shape}")
    print(f"  Pooled shape: {latents['pre_logits'].shape}")
    print(f"  2D features shape: {latents['pre_logits_2d'].shape}")
```

### Direct Access to embed_prefix

```python
# Get vision latents directly from embed_prefix
tokens, input_mask, ar_mask, vision_latents = model.embed_prefix(observation)

# Process vision latents before action sampling
for image_name, latents in vision_latents.items():
    # Do something with the latents
    features = latents["encoded"]
    # ... your processing code ...
```

## Key Features

1. **Non-intrusive**: The changes don't affect the core functionality of the model
2. **Comprehensive**: All intermediate representations from the vision encoder are available
3. **Structured**: Latents are organized by image name for easy access
4. **Efficient**: Vision latents are computed once during the prefix embedding phase
5. **Flexible**: Can be used with both basic and advanced sampling methods

## Notes

- Vision latents are computed during the prefix embedding phase and cached
- The latents represent the final state after all transformer layers
- All latents are JAX arrays and can be used with JAX transformations
- The structure is consistent across different batch sizes and configurations




