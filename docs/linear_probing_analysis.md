# Linear Probing Analysis for Inference Latent Data

## Table of Contents

1. [Overview](#overview)
2. [Context and Data Structure](#context-and-data-structure)
3. [Implementation Guide](#implementation-guide)
4. [Usage Instructions](#usage-instructions)
5. [Technical Details](#technical-details)
6. [Troubleshooting](#troubleshooting)
7. [Extending the Script](#extending-the-script)

---

## Overview

This document provides a comprehensive guide for performing linear probing analysis on inference latent data to understand how well different experts (VLM vs Action) preserve language information across rollout steps and inference steps.

### Purpose

The linear probing script analyzes the hidden states from different experts in the Pi0 model to determine how well they encode task descriptions. It trains a linear probe to predict T5 embeddings of task descriptions from the hidden states.

### Key Features

- **Expert Separation**: Analyze VLM vs Action expert representations
- **Layer-wise Analysis**: Probe specific layers (0-17) of the model
- **Rollout Step Tracking**: Analyze specific inference timesteps
- **T5 Label Generation**: Use T5 embeddings as ground truth labels
- **Comprehensive Metrics**: Accuracy and cosine similarity evaluation

---

## Context and Data Structure

### Data Source

- **Input**: Inference latent data from `scripts/inference_latent_collection.py`
- **Format**: Pickle files in `data/inference_latents/task_{id}_{name}.pkl`
- **Structure**: Nested dictionary with episodes and rollout steps

### Data Structure Understanding

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
                    "action_expert_hidden_state_t*": {
                        "hidden_states": np.array(...),  # Shape: (18, 1, seq_len, 2048)
                        "post_attn_embedding": np.array(...),
                        "pre_attn_norm_scales": np.array(...),
                        "pre_mlp_norm_scales": np.array(...)
                    },
                    "vlm_layer_output": {
                        "hidden_states": np.array(...),  # Shape: (18, 1, 816, 2048)
                        "post_attn_embedding": np.array(...),
                        "pre_attn_norm_scales": np.array(...),
                        "pre_mlp_norm_scales": np.array(...)
                    },
                    "observation": {...}
                }
            },
            "metadata": {
                "success": True,
                "total_steps": 45
            }
        }
    }
}
```

### Expert Token Positions

Based on the model architecture in `gemma.py`:

- **Expert 0 (VLM/PaliGemma)**: Positions 0-815
  - Image tokens: 0-767 (3 images × 256 patches)
  - Text tokens: 768-815 (48 tokens)
- **Expert 1 (Action Expert)**: Positions 816+
  - Action tokens: 816+ (variable length)

---

## Implementation Guide

### Phase 1: Script Setup and Argument Parsing

#### Command Line Arguments

**Required Arguments:**
- `--rollout_step`: Specific rollout step to analyze (int)
- `--expert`: Expert type - "vlm", "action", or "text_only" (str)
- `--layer`: Layer ID to analyze (0-17) (int)
- `--data_path`: Path to inference latent data (str)
- `--task_range`: Range of tasks to analyze (start, end) (int, int)
- `--episode_range`: Range of episodes to analyze (start, end) (int, int)

**Optional Arguments:**
- `--output_dir`: Output directory for results (str, default: "results/linear_probing")
- `--learning_rate`: Learning rate for training (float, default: 0.001)
- `--num_epochs`: Number of training epochs (int, default: 100)
- `--batch_size`: Batch size for training (int, default: 32)
- `--seed`: Random seed (int, default: 42)

### Phase 2: Data Loading and Processing

#### Data Loader Implementation

```python
@dataclasses.dataclass
class DataConfig:
    """Configuration for data loading."""
    data_path: str
    task_range: Tuple[int, int]
    episode_range: Tuple[int, int]
    rollout_step: int
    expert: str  # "vlm", "action", "text_only"
    layer: int
    feature_type: str = "hidden_states"

class InferenceLatentDataLoader:
    """DataLoader for inference latent data with expert separation."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self._load_data()
        
    def get_features_and_labels(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return features and T5 labels."""
        
    def get_task_descriptions(self) -> List[str]:
        """Return list of task descriptions."""
```

#### Feature Extraction Strategy

**Expert Mapping:**
- **VLM Expert**: Use `vlm_layer_output["hidden_states"]` (positions 0-815)
- **Action Expert**: Use `action_expert_hidden_states["hidden_states"]` (positions 816+)
- **Text Only**: Extract positions 768-815 from VLM expert

**Data Loading Strategy:**
- **Lazy Loading**: Don't load entire pickle files at once
- **Selective Loading**: Only load required rollout steps and layers
- **Memory Management**: Use generators for large datasets
- **Error Handling**: Gracefully handle missing data

### Phase 3: T5 Label Generation

#### Implementation

```python
def create_t5_labels(task_descriptions: List[str], 
                    model_name: str = "t5-small") -> np.ndarray:
    """
    Use T5 to encode task descriptions into embeddings as labels.
    
    Args:
        task_descriptions: List of task description strings
        model_name: T5 model variant to use
    
    Returns:
        T5 embeddings array of shape (num_tasks, embedding_dim)
    """
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5EncoderModel.from_pretrained(model_name)
    
    embeddings = []
    with torch.no_grad():
        for description in task_descriptions:
            inputs = tokenizer(description, return_tensors="pt", 
                             max_length=512, truncation=True, padding=True)
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            embeddings.append(embedding)
    
    return np.stack(embeddings, axis=0)
```

### Phase 4: Linear Network and Training

#### Linear Probe Architecture

```python
class LinearProbe(nn.Module):
    """Linear probe for inference latent analysis."""
    
    input_dim: int  # Hidden state dimension (2048)
    output_dim: int  # T5 embedding dimension
    
    @nn.compact
    def __call__(self, x):
        return nn.Dense(features=self.output_dim)(x)
```

#### Training Configuration

```python
@dataclasses.dataclass
class TrainingConfig:
    """Configuration for training."""
    learning_rate: float = 0.001
    num_epochs: int = 100
    batch_size: int = 32
    seed: int = 42
```

#### Training Loop

**Loss Function:** Cosine similarity loss for embedding regression
**Optimization:** Adam optimizer with configurable learning rate
**Training Strategy:** Batch training with early stopping

```python
@jax.jit
def train_step(state: train_state.TrainState, 
               batch: Tuple[np.ndarray, np.ndarray]) -> Tuple[train_state.TrainState, Dict]:
    """Single training step with cosine similarity loss."""
    features, targets = batch
    
    def loss_fn(params):
        predictions = state.apply_fn({'params': params}, features)
        
        # Normalize predictions and targets
        predictions_norm = predictions / (jnp.linalg.norm(predictions, axis=-1, keepdims=True) + 1e-8)
        targets_norm = targets / (jnp.linalg.norm(targets, axis=-1, keepdims=True) + 1e-8)
        
        # Compute cosine similarity loss
        cosine_sim = jnp.sum(predictions_norm * targets_norm, axis=-1)
        loss = 1.0 - jnp.mean(cosine_sim)
        
        return loss, cosine_sim
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, cosine_sim), grads = grad_fn(state.params)
    
    state = state.apply_gradients(grads=grads)
    
    return state, {'loss': loss, 'cosine_similarity': jnp.mean(cosine_sim)}
```

### Phase 5: Evaluation and Analysis

#### Metrics Computation

**Accuracy:** Classification accuracy using cosine similarity
**Cosine Similarity:** Direct similarity between predictions and targets

```python
def compute_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute classification accuracy using cosine similarity."""
    predictions_norm = predictions / (np.linalg.norm(predictions, axis=-1, keepdims=True) + 1e-8)
    targets_norm = targets / (np.linalg.norm(targets, axis=-1, keepdims=True) + 1e-8)
    
    similarities = np.dot(predictions_norm, targets_norm.T)
    predicted_indices = np.argmax(similarities, axis=1)
    true_indices = np.arange(len(targets))
    
    return np.mean(predicted_indices == true_indices)

def compute_cosine_similarity(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute average cosine similarity between predictions and targets."""
    predictions_norm = predictions / (np.linalg.norm(predictions, axis=-1, keepdims=True) + 1e-8)
    targets_norm = targets / (np.linalg.norm(targets, axis=-1, keepdims=True) + 1e-8)
    
    similarities = np.sum(predictions_norm * targets_norm, axis=-1)
    return np.mean(similarities)
```

---

## Usage Instructions

### Installation

1. **Install Dependencies:**
```bash
pip install -r scripts/requirements_linear_probing.txt
```

2. **Ensure Data Availability:**
   - Collect inference latent data using `scripts/inference_latent_collection.py`
   - Verify data structure in `data/inference_latents/`

### Basic Usage

```bash
python scripts/linear_probing_inference.py \
    --rollout_step 10 \
    --expert vlm \
    --layer 5 \
    --data_path data/inference_latents \
    --task_range 0 10 \
    --episode_range 0 5
```

### Advanced Usage

```bash
python scripts/linear_probing_inference.py \
    --rollout_step 20 \
    --expert action \
    --layer 10 \
    --data_path data/inference_latents \
    --task_range 5 15 \
    --episode_range 2 8 \
    --learning_rate 0.001 \
    --num_epochs 100 \
    --batch_size 32 \
    --output_dir results/linear_probing
```

### Expert Types

- **vlm**: VLM/PaliGemma expert (positions 0-815)
- **action**: Action expert (positions 816+)
- **text_only**: Text-only tokens from VLM expert (positions 768-815)

### Expected Output

```
results/linear_probing/
├── results_vlm_layer5_step10.pkl
├── summary_vlm_layer5_step10.txt
└── ...
```

**Example Summary Output:**
```
Linear Probing Results
=====================

Expert: vlm
Layer: 5
Rollout Step: 10
Feature Type: hidden_states

Accuracy: 0.8500
Cosine Similarity: 0.7234
Number of Samples: 45
Number of Tasks: 10

Task Descriptions:
  0: pick up the cream cheese and place it in the basket
  1: open the top drawer and put the bowl inside
  ...
```

---

## Technical Details

### Dependencies

**Core ML Libraries:**
- `jax>=0.4.0`, `jaxlib>=0.4.0`
- `flax>=0.7.0`, `optax>=0.1.0`
- `numpy>=1.21.0`

**Transformers:**
- `transformers>=4.20.0`
- `torch>=1.12.0`

**Optional:**
- `tqdm>=4.64.0` (progress bars)
- `matplotlib>=3.5.0` (plotting)
- `seaborn>=0.11.0` (visualizations)

### Performance Considerations

**Memory Management:**
- Use selective loading to avoid loading entire pickle files
- Implement batch processing for large datasets
- Consider using generators for memory-intensive operations

**Training Optimization:**
- JAX JIT compilation for faster training
- Configurable batch sizes for different hardware
- Early stopping to prevent overfitting

**Data Processing:**
- Lazy loading of task data
- Efficient feature extraction
- Proper error handling for missing data

---

## Troubleshooting

### Common Issues

1. **No Data Files Found**
   - **Cause**: Incorrect data path or missing pickle files
   - **Solution**: Verify data path and ensure files exist
   ```bash
   ls data/inference_latents/
   ```

2. **Memory Issues**
   - **Cause**: Large batch size or too many tasks/episodes
   - **Solution**: Reduce batch size or data range
   ```bash
   --batch_size 16 --task_range 0 5 --episode_range 0 3
   ```

3. **CUDA Out of Memory**
   - **Cause**: GPU memory insufficient for batch size
   - **Solution**: Use CPU or reduce batch size
   ```bash
   export CUDA_VISIBLE_DEVICES=""  # Use CPU
   ```

4. **Missing Rollout Steps**
   - **Cause**: Specified rollout step doesn't exist in data
   - **Solution**: Check available steps in data files
   ```python
   # Check available steps
   print(list(task_data["episodes"]["episode_0"]["rollout_steps"].keys()))
   ```

### Debug Mode

Enable verbose logging for debugging:

```python
logging.basicConfig(level=logging.DEBUG)
```

### Performance Monitoring

Monitor training progress:
- Loss and cosine similarity metrics
- Memory usage during training
- Training time per epoch

---

## Extending the Script

### Adding New Expert Types

1. **Update Argument Parser:**
```python
parser.add_argument("--expert", choices=["vlm", "action", "text_only", "new_expert"])
```

2. **Implement Feature Extraction:**
```python
elif self.config.expert == "new_expert":
    # Implement extraction logic
    expert_data = step_data["new_expert_data"]
```

3. **Update Documentation:**
   - Add expert type to documentation
   - Update usage examples

### Adding New Metrics

1. **Implement Metric Function:**
```python
def compute_new_metric(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute new evaluation metric."""
    # Implementation
    return metric_value
```

2. **Add to Analysis:**
```python
def analyze_results(...):
    # ... existing code ...
    new_metric = compute_new_metric(predictions, labels)
    results['new_metric'] = new_metric
```

3. **Update Output:**
   - Include in summary file
   - Add to logging output

### Custom Loss Functions

Modify the training step to use different loss functions:

```python
def custom_loss_fn(predictions, targets):
    """Custom loss function implementation."""
    # Implement custom loss
    return loss

@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        predictions = state.apply_fn({'params': params}, features)
        return custom_loss_fn(predictions, targets)
    # ... rest of implementation
```

### Batch Processing Improvements

1. **Multi-GPU Training:**
   - Implement data parallelism
   - Use JAX's `pmap` for multi-device training

2. **Distributed Training:**
   - Add support for multiple machines
   - Implement checkpointing and recovery

3. **Advanced Data Loading:**
   - Implement prefetching
   - Add data augmentation options

### Visualization Enhancements

1. **Training Curves:**
   - Plot loss and accuracy over time
   - Add confidence intervals

2. **Confusion Matrices:**
   - Visualize task classification performance
   - Add per-task analysis plots

3. **Embedding Visualizations:**
   - t-SNE or UMAP plots of embeddings
   - Similarity heatmaps

---

## Related Documentation

- **Inference Latent Collection**: `scripts/inference_latent_collection.py`
- **Model Architecture**: `src/openpi/models/gemma.py`
- **Policy Implementation**: `src/openpi/policies/policy.py`
- **Original Linear Probing**: `scripts/linear_probing.py`

## Citation

If you use this linear probing analysis in your research, please cite:

1. The Pi0 project and related papers
2. The inference latent collection methodology
3. The T5 model for label generation
4. The JAX/Flax framework for implementation

---

*This document provides a comprehensive guide for linear probing analysis of inference latent data. For questions or contributions, please refer to the project repository.*
