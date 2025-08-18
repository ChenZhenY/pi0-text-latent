# Inference Latent Collection Pipeline

This document describes the implementation of the first 5 steps of the expert separation TODO list, which creates a data collection pipeline during inference time to analyze how language information flows and degrades during task execution.

## Overview

The pipeline enables collection of expert-specific hidden states during inference time, providing insights into:
- **Rollout Steps**: How language information changes across timesteps during episode execution
- **Expert Separation**: How PaliGemma (Expert 0) and Action Expert (Expert 1) process information differently
- **Text-Only Analysis**: Focused analysis of text token representations (positions 768-815)

## Architecture Changes

### Step 1: Modified Module Class (`src/openpi/models/gemma.py`)

**Changes Made:**
- Added tracking for `expert_0_hidden_states` (PaliGemma expert, positions 0-815)
- Added tracking for `expert_1_hidden_states` (Action expert, positions 816+)
- Modified `attn_output` tuple to include expert-specific states

**Key Code:**
```python
# Added to layer loop
expert_0_hidden_states.append(embedded[0])  # PaliGemma
expert_1_hidden_states.append(embedded[1])  # Action expert

# Modified return tuple
attn_output = (..., expert_0_hidden_states, expert_1_hidden_states)
```

### Step 2: Enhanced Analysis Class (`src/openpi/models/gemma.py`)

**New Methods Added:**
- `get_expert_0_hidden_states()`: Extract PaliGemma expert states
- `get_expert_1_hidden_states()`: Extract Action expert states  
- `get_text_only_hidden_states()`: Extract text-only states (positions 768-815)

**Usage:**
```python
from openpi.models.gemma import Analysis

expert_0_states = Analysis.get_expert_0_hidden_states(layer_output)
expert_1_states = Analysis.get_expert_1_hidden_states(layer_output)
text_only_states = Analysis.get_text_only_hidden_states(layer_output)
```

### Step 3: Updated Pi0 Model (`src/openpi/models/pi0.py`)

**Changes Made:**
- Modified `compute_loss_with_extra()` to include expert-specific states in results
- Added `infer_with_latents()` method for inference with latent collection

**New Result Fields:**
```python
result = {
    "expert_0_hidden_states_sum": expert_0_hidden_states.sum(axis=1),
    "expert_1_hidden_states_sum": expert_1_hidden_states.sum(axis=1),
    "text_only_hidden_states_sum": text_only_hidden_states.sum(axis=1),
    # ... existing fields
}
```

### Step 4: Enhanced Policy Interface (`src/openpi/policies/policy.py`)

**Changes Made:**
- Modified `infer()` method to handle `collect_latents` flag
- Returns expert-specific hidden states when requested

**Usage:**
```python
# Set flag to collect latents
element = {
    "collect_latents": True,
    # ... other observation data
}

# Response includes latent states
response = client.infer(element)
expert_0_states = response.get("expert_0_hidden_states")
expert_1_states = response.get("expert_1_hidden_states")
text_only_states = response.get("text_only_hidden_states")
```

### Step 5: Inference-Time Data Collection (`scripts/inference_latent_collection.py`)

**Key Features:**
- Collects latent states during real episode execution
- Tracks rollout steps and episode context
- Memory-efficient sampling with configurable stride
- Incremental saving to avoid memory buildup

**Configuration:**
```python
args = InferenceLatentCollectionArgs(
    task_suite_name="libero_object",
    tasks_to_collect=(10, 15),  # Range of tasks
    num_trials_per_task=3,
    timestep_stride=5,  # Collect every 5th timestep
    max_steps=200,
)
```

## Data Structure

### Collected Data Format

```python
task_data = {
    "task_name": "pick_up_the_cream_cheese_and_place_it_in_the_basket",
    "task_id": 10,
    "task_description": "pick up the cream cheese and place it in the basket",
    "episodes": {
        "episode_0": {
            "rollout_steps": {
                "step_10": {
                    "timestep": 10,
                    "action": [...],
                    "expert_0_hidden_states": np.array(...),  # Shape: (18, 1, 816, 2048)
                    "expert_1_hidden_states": np.array(...),  # Shape: (18, 1, 200, 2048)
                    "text_only_hidden_states": np.array(...), # Shape: (18, 1, 48, 2048)
                    "observation": {...}
                },
                # ... more steps
            },
            "metadata": {
                "success": True,
                "total_steps": 45,
                "episode_idx": 0
            }
        }
        # ... more episodes
    }
}
```

### Token Position Mapping

- **Expert 0 (PaliGemma)**: Positions 0-815
  - Images: Positions 0-767 (256×3 tokens)
  - Text: Positions 768-815 (48 tokens)
- **Expert 1 (Action Expert)**: Positions 816+
  - State + Action tokens

## Usage Examples

### 1. Collect Inference Data

```bash
# Run the collection script
python scripts/inference_latent_collection.py
```

### 2. Analyze Collected Data

```python
from scripts.expert_data_utils import (
    load_inference_task_data,
    get_language_loss_trajectory,
    analyze_expert_separation
)

# Load task data
task_data = load_inference_task_data(task_id=10)

# Get language loss trajectory
trajectory = get_language_loss_trajectory(
    task_data, 
    episode_idx=0, 
    expert_type="text_only",
    layer_idx=5
)

# Analyze expert separation
separation = analyze_expert_separation(
    task_data, 
    episode_idx=0, 
    rollout_step=20, 
    layer_idx=0
)
```

### 3. Test the Pipeline

```bash
# Run tests to verify functionality
python scripts/test_inference_latents.py
```

## Memory Efficiency

The pipeline implements several memory-saving strategies:

1. **Stratified Sampling**: Collect every Nth timestep instead of all timesteps
2. **Text-Only Focus**: Extract only text token representations (48 vs 816 tokens)
3. **Incremental Processing**: Save data per task to avoid memory buildup
4. **Configurable Scope**: Limit number of tasks, episodes, and layers

## Key Advantages Over Training-Time Collection

### Training Time (Original `text_latent.py`)
- ❌ No rollout context
- ❌ No episode progression tracking
- ❌ Batch processing across episodes
- ❌ No temporal analysis

### Inference Time (New Pipeline)
- ✅ Full episode context
- ✅ Rollout step tracking
- ✅ Real-time collection
- ✅ Temporal analysis capabilities
- ✅ Success/failure correlation

## Analysis Capabilities

### 1. Language Loss Trajectory
```python
# Track how language information degrades over time
trajectory = get_language_loss_trajectory(task_data, 0, "text_only")
# Analyze correlation with task success
```

### 2. Expert Separation Analysis
```python
# Measure how experts process information differently
separation = analyze_expert_separation(task_data, 0, 20, layer_idx=0)
# cosine_similarity, l2_distance, etc.
```

### 3. Layer-Wise Analysis
```python
# Focus on specific layers
layer_5_states = get_hidden_states_for_dimensions(
    task_data, 0, 20, layer_idx=5, expert_type="expert_0"
)
```

## Next Steps

The pipeline is ready for the remaining TODO steps:

1. **Step 6**: Inference step sampling (AR process and flow matching)
2. **Step 7**: Advanced analysis utilities
3. **Step 8**: Language loss quantification and visualization

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Memory Issues**: Reduce `timestep_stride` or number of tasks
3. **Model Server**: Ensure the model server is running with latent collection support

### Debug Mode

Enable debug logging to troubleshoot issues:
```python
logging.basicConfig(level=logging.DEBUG)
```

## Performance Considerations

- **Collection Speed**: ~2-3x slower than regular inference due to latent extraction
- **Storage**: ~100MB per task (with 3 episodes, stride=5)
- **Memory**: Peak usage ~2GB during collection

The pipeline provides a solid foundation for analyzing expert separation and language information flow during real task execution. 