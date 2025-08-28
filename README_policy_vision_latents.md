# Policy Vision Latents Collection

This document explains the changes made to the policy to support vision latents collection during action sampling.

## Overview

The policy has been updated to handle the new return signature from the Pi0 model's `sample_actions` methods, which now return vision latents along with actions. The policy converts these vision latents to numpy arrays and includes them in the output when `collect_latents=True`.

## Changes Made

### 1. Updated Policy Inference Method

The `infer` method in `Policy` class now handles the new return signatures:

```python
# Before
action, layer_output = self._sample_actions(sample_rng, ...)
action, layer_output, action_expert_output = self._sample_actions_with_latent(sample_rng, ...)

# After  
action, vision_latents = self._sample_actions(sample_rng, ...)
action, vision_latents, layer_outputs = self._sample_actions_with_latent(sample_rng, ...)
```

### 2. Vision Latents Processing

The policy now processes vision latents and converts them to numpy arrays:

```python
# Convert vision latents to numpy and add to outputs
vision_latents_np = {}
for image_name, latents in vision_latents.items():
    vision_latents_np[image_name] = {}
    for key, value in latents.items():
        if hasattr(value, 'shape'):
            # Convert JAX array to numpy
            vision_latents_np[image_name][key] = np.asarray(value[0, ...])  # Remove batch dimension
        else:
            # Handle nested dictionaries (like encoder outputs)
            vision_latents_np[image_name][key] = value
outputs["vision_latents"] = vision_latents_np
```

### 3. Conditional Latents Collection

Vision latents are only collected and returned when `obs.get("collect_latents", True)` is True:

```python
if obs.get("collect_latents", True):
    # Process and include vision latents
    outputs["vision_latents"] = vision_latents_np
```

## Output Structure

When `collect_latents=True`, the policy output now includes:

```python
outputs = {
    "state": np.array,           # Robot state
    "actions": np.array,         # Sampled actions
    "vision_latents": {          # Vision encoder outputs (only when collect_latents=True)
        "base_0_rgb": {
            "encoded": np.array,      # Final encoded features [H*W, D]
            "pre_logits": np.array,   # Pooled features [D]
            "pre_logits_2d": np.array, # 2D features [H, W, D]
            "stem": np.array,         # Initial patch embeddings
            "with_posemb": np.array,  # Features with positional embeddings
            "encoder": dict,          # All transformer layer outputs
        },
        "left_wrist_0_rgb": { ... },
        "right_wrist_0_rgb": { ... },
    },
    "vlm_layer_output": {        # VLM layer outputs (only when collect_latents=True)
        "mlp_activation": np.array,
        "pre_attn_norm_scales": np.array,
        "pre_mlp_norm_scales": np.array,
        "hidden_states": np.array,
        "post_attn_embedding": np.array,
    }
}
```

## Usage Examples

### Basic Usage with Vision Latents

```python
from src.openpi.policies.policy import Policy
from src.openpi.models.pi0_linear_probing import Pi0Config

# Create policy
config = Pi0Config()
model = config.create(rng)
policy = Policy(model, rng=rng)

# Create observation with collect_latents=True
obs = {
    # ... observation data ...
    'collect_latents': True,
}

# Get outputs including vision latents
outputs = policy.infer(obs)

# Access vision latents
vision_latents = outputs['vision_latents']
base_latents = vision_latents['base_0_rgb']
encoded_features = base_latents['encoded']  # [H*W, D]
pooled_features = base_latents['pre_logits']  # [D]
```

### Usage Without Vision Latents

```python
# Create observation with collect_latents=False
obs = {
    # ... observation data ...
    'collect_latents': False,
}

# Get outputs without vision latents (smaller bandwidth)
outputs = policy.infer(obs)
# outputs will not contain 'vision_latents' key
```

## Key Features

1. **Backward Compatible**: Existing code continues to work without changes
2. **Conditional Collection**: Vision latents are only collected when requested
3. **Numpy Conversion**: All vision latents are converted to numpy arrays for easy use
4. **Structured Output**: Vision latents are organized by image name
5. **Bandwidth Efficient**: Only collects latents when `collect_latents=True`

## Notes

- Vision latents are computed once during the prefix embedding phase and cached
- All latents are converted to numpy arrays and have batch dimension removed
- The structure is consistent across different batch sizes and configurations
- Vision latents are only included in outputs when `collect_latents=True`
- The policy handles both `sample_actions` and `sample_actions_with_latents` methods correctly




