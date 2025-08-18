# Language Information Loss Analysis Guide

## Overview

This guide outlines a **memory-efficient approach** to investigate where language information is lost in your Vision-Language-Action (VLA) model across three key dimensions:

1. **Rollout Steps**: Different timesteps during episode execution
2. **Inference Steps**: Different layers within the model during a single inference
3. **Model Layers**: Different transformer layers in the architecture

## Key Insights from Current Code

### What the Original Script Does
```python
# Lines 118-121: Accumulates across ALL episodes and timesteps
hidden_states_sum = jnp.zeros((18, prefix_len, 2048), dtype=jnp.float32)
# ...
hidden_states_sum += result["hidden_states_sum"]  # Averages everything together
```

### What We Need Instead
- **Per-timestep** hidden states
- **Per-layer** analysis
- **Per-episode** tracking
- **Text-specific** regions only

## Memory-Efficient Strategy

### 1. Hierarchical Sampling

Instead of collecting everything, use **stratified sampling**:

```python
sampling_config = {
    "tasks": range(10, 15),           # 5 tasks instead of 30
    "episodes_per_task": 2,           # 2 episodes instead of 20
    "timestep_stride": 10,            # Every 10th timestep instead of all
    "layers": [0, 3, 6, 9, 12, 15, 17],  # 7 key layers instead of all 18
}
```

**Memory Reduction**: ~90% less data to store

### 2. Focus on Text Regions

The model processes:
- **Images**: Positions 0-767 (256×3 tokens)
- **Text**: Positions 768-815 (48 tokens)

```python
text_start = 256 * 3  # 768
text_end = text_start + 48  # 816
text_hidden_states = hidden_states[:, text_start:text_end, :]  # (layers, 48, 2048)
```

**Memory Reduction**: ~94% less data (48 vs 816 tokens)

### 3. Incremental Processing

```python
# Process one task at a time
for task_id in sampling_config["tasks"]:
    task_data = collect_language_loss_data(...)
    save_incrementally(task_data)  # Avoid memory buildup
```

## Data Collection Strategy

### Phase 1: Baseline Collection (5-10 tasks)
```python
# Start small to validate approach
sampling_config = {
    "tasks": range(10, 15),
    "episodes_per_task": 2,
    "timestep_stride": 10,
    "layers": [0, 3, 6, 9, 12, 15, 17],
}
```

### Phase 2: Targeted Analysis
Based on Phase 1 results, focus on:
- **Critical layers** where information loss occurs
- **Specific timesteps** where loss accelerates
- **Problematic tasks** that show high loss

### Phase 3: Deep Dive
For identified critical regions:
- **Higher resolution** timestep sampling
- **All layers** in critical ranges
- **More episodes** for statistical significance

## Key Metrics to Track

### 1. Text Reconstruction Accuracy
```python
def calculate_text_accuracy(original, reconstructed):
    original_tokens = original.split()
    reconstructed_tokens = reconstructed.split()
    return len(set(original_tokens) & set(reconstructed_tokens)) / len(original_tokens)
```

### 2. Hidden State Variance
```python
def calculate_state_variance(hidden_states):
    return jnp.var(hidden_states, axis=0).mean()  # Across token dimensions
```

### 3. Layer-wise Information Flow
```python
def calculate_layer_information_loss(layer_states, next_layer_states):
    # Measure how much information is preserved between layers
    return cosine_similarity(layer_states, next_layer_states)
```

## Expected Findings

### 1. Layer-wise Loss Pattern
```
Layer 0-3:   High text preservation (95-100%)
Layer 4-9:   Gradual information mixing (80-95%)
Layer 10-15: Significant language loss (50-80%)
Layer 16-17: Minimal language info (20-50%)
```

### 2. Timestep-wise Loss Pattern
```
Timestep 0-20:   Stable language representation
Timestep 21-60:  Gradual degradation
Timestep 61-120: Significant language loss
```

### 3. Task-specific Patterns
- **Simple tasks**: Better language preservation
- **Complex tasks**: Faster language loss
- **Long sequences**: More degradation

## Implementation Steps

### Step 1: Run Baseline Collection
```bash
python scripts/text_latent_analysis.py
```

### Step 2: Analyze Results
```python
# The script automatically generates:
# - Layer-wise accuracy plots
# - Timestep-wise degradation curves
# - Critical layer identification
```

### Step 3: Targeted Follow-up
Based on results, run focused analysis on:
- Critical layers (e.g., layers 10-15)
- Problematic timesteps (e.g., 40-80)
- Specific task types

## Memory Usage Estimates

### Original Approach
```
30 tasks × 20 episodes × 120 timesteps × 18 layers × 816 tokens × 2048 dims × 4 bytes
= ~4.3 GB per task = ~130 GB total
```

### Optimized Approach
```
5 tasks × 2 episodes × 12 timesteps × 7 layers × 48 tokens × 2048 dims × 4 bytes
= ~0.4 MB per task = ~2 MB total
```

**Memory Reduction**: ~99.998% less memory usage

## Key Code Modifications Needed

### 1. Modify Model Output
```python
# In compute_loss_with_extra, return per-layer hidden states
result = {
    "hidden_states_per_layer": hidden_states,  # (18, seq_len, 2048)
    "text_hidden_states": text_hidden_states,  # (18, 48, 2048)
    "attention_weights": attention_weights,    # (18, num_heads, seq_len, seq_len)
}
```

### 2. Add Text-specific Analysis
```python
def analyze_text_region(hidden_states, text_start=768, text_end=816):
    return hidden_states[:, text_start:text_end, :]
```

### 3. Implement Incremental Saving
```python
def save_incrementally(data, task_id):
    with open(f"task_{task_id}.pkl", 'wb') as f:
        pickle.dump(data, f)
```

## Expected Outcomes

### 1. Identify Critical Layers
- Which layers cause the most language information loss
- Where the model transitions from language to action representation

### 2. Understand Temporal Dynamics
- How language information degrades over time
- Whether loss is gradual or sudden

### 3. Task-specific Insights
- Which tasks preserve language better
- Whether task complexity affects language retention

### 4. Architectural Recommendations
- Which layers could be modified to preserve language
- Whether additional language-specific components are needed

## Next Steps

1. **Run the baseline analysis** with the provided script
2. **Analyze the results** to identify critical regions
3. **Design targeted experiments** based on findings
4. **Implement architectural modifications** to preserve language information
5. **Validate improvements** with the same analysis framework

This approach will give you a comprehensive understanding of where and how language information is lost in your VLA model while using minimal computational resources. 