# Linear Probing TODO: Expert Separation Analysis (PyTorch Version)

## Overview

This document outlines the complete TODO list for creating a linear probing script that analyzes inference latent data with expert separation capabilities using PyTorch. The script will analyze how well different experts (VLM vs Action) preserve language information across rollout steps and inference steps.

## Context and Data Structure

### Data Source
- **Input**: Inference latent data from `scripts/inference_latent_collection.py`
- **Format**: Pickle files in `data/inference_latents/task_{id}_{name}.pkl`
- **Structure**: Nested dictionary with episodes and rollout steps

### Updated Data Structure Understanding
Based on `pi0.py` implementation, action expert states are stored for every rollout timestep:

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
                    "action_expert_hidden_state_t0.9": {
                        "hidden_states": np.array(...),  # Shape: (18, 1, seq_len, 2048)
                        "post_attn_embedding": np.array(...),
                        "pre_attn_norm_scales": np.array(...),
                        "pre_mlp_norm_scales": np.array(...)
                    },
                    "action_expert_hidden_state_t0.8": {
                        "hidden_states": np.array(...),
                        "post_attn_embedding": np.array(...),
                        "pre_attn_norm_scales": np.array(...),
                        "pre_mlp_norm_scales": np.array(...)
                    },
                    # ... more timesteps (t0.7, t0.6, ..., t0.1)
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

### Expert Token Positions (from gemma.py)
- **Expert 0 (VLM/PaliGemma)**: Positions 0-815
  - Image tokens: 0-767 (3 images × 256 patches)
  - Text tokens: 768-815 (48 tokens)
- **Expert 1 (Action Expert)**: Positions 816+
  - Action tokens: 816+ (variable length)

## TODO List

### Phase 1: Script Setup and Argument Parsing

#### TODO 1.1: Create Main Script Structure
**File**: `scripts/linear_probing_inference.py`
**Requirements**:
```python
#!/usr/bin/env python3
"""
Linear Probing Analysis for Inference Latent Data with Expert Separation (PyTorch)

This script performs linear probing analysis on inference latent data to understand
how well different experts (VLM vs Action) preserve language information across
rollout steps and inference steps.

Usage:
    python scripts/linear_probing_inference.py \
        --rollout_step 10 \
        --expert vlm \
        --layer 5 \
        --data_path data/inference_latents \
        --task_range 0 10 \
        --episode_range 0 5
"""

import argparse
import dataclasses
import logging
import pathlib
import pickle
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5EncoderModel
```

#### TODO 1.2: Define Command Line Arguments
**Function**: `parse_args()`
**Arguments**:
- `--rollout_step`: Specific rollout step to analyze (int)
- `--expert`: Expert type - "vlm", "action", or "text_only" (str)
- `--layer`: Layer ID to analyze (0-17) (int)
- `--data_path`: Path to inference latent data (str)
- `--task_range`: Range of tasks to analyze (start, end) (int, int)
- `--episode_range`: Range of episodes to analyze (start, end) (int, int)
- `--output_dir`: Output directory for results (str)
- `--learning_rate`: Learning rate for training (float)
- `--num_epochs`: Number of training epochs (int)
- `--batch_size`: Batch size for training (int)
- `--seed`: Random seed (int)
- `--action_timestep`: For action expert, which diffusion timestep to use (float, e.g., 0.9, 0.8, etc.)

### Phase 2: Data Loading and Processing

#### TODO 2.1: Implement Data Loader
**Class**: `InferenceLatentDataset`
**Requirements**:
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
    feature_type: str = "hidden_states"  # "hidden_states", "post_attn_embedding", etc.
    action_timestep: float = 0.9  # For action expert, which diffusion timestep

class InferenceLatentDataset(Dataset):
    """PyTorch Dataset for inference latent data with expert separation."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.features = []
        self.labels = []
        self.task_descriptions = []
        self._load_data()
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
        
    def _load_data(self):
        """Load and preprocess inference latent data."""
        # TODO: Implement data loading logic
        
    def get_task_descriptions(self) -> List[str]:
        """Return list of task descriptions."""
        return self.task_descriptions
        
    def get_data_info(self) -> Dict:
        """Return information about loaded data."""
        # TODO: Return data statistics
```

#### TODO 2.2: Implement Feature Extraction
**Function**: `extract_expert_features()`
**Requirements**:
```python
def extract_expert_features(step_data: Dict, config: DataConfig) -> Optional[torch.Tensor]:
    """
    Extract features from step data based on expert configuration.
    
    Args:
        step_data: Step data dictionary from rollout
        config: Data configuration specifying expert, layer, etc.
    
    Returns:
        Features tensor of shape (hidden_dim,) or None if not available
    """
    # TODO: Implement feature extraction logic
    # - Map expert types to correct data keys
    # - For action expert, use specific timestep key (e.g., "action_expert_hidden_state_t0.9")
    # - Extract specific layer and rollout step
    # - Handle shape transformations
    # - Convert numpy arrays to torch tensors
```

#### TODO 2.3: Create T5 Label Generation
**Function**: `create_t5_labels()`
**Requirements**:
```python
def create_t5_labels(task_descriptions: List[str], 
                    model_name: str = "t5-small") -> torch.Tensor:
    """
    Use T5 to encode task descriptions into embeddings as labels.
    
    Args:
        task_descriptions: List of task description strings
        model_name: T5 model variant to use
    
    Returns:
        T5 embeddings tensor of shape (num_tasks, embedding_dim)
    """
    # TODO: Implement T5 encoding
    # - Load T5 model and tokenizer
    # - Encode each task description
    # - Handle padding and truncation
    # - Return mean-pooled embeddings as torch tensors
```

### Phase 3: Linear Network and Training

#### TODO 3.1: Create Linear Probe Network
**Class**: `LinearProbe`
**Requirements**:
```python
class LinearProbe(nn.Module):
    """Linear probe for inference latent analysis."""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.linear(x)
```

#### TODO 3.2: Implement Training Loop
**Functions**: `train_epoch()`, `train()`
**Requirements**:
```python
def train_epoch(model: nn.Module, 
                dataloader: DataLoader,
                optimizer: optim.Optimizer,
                criterion: nn.Module,
                device: torch.device) -> Dict[str, float]:
    """Train for one epoch."""
    # TODO: Implement epoch training
    # - Set model to training mode
    # - Iterate through batches
    # - Forward pass, loss computation, backward pass
    # - Return metrics

def train(features: torch.Tensor, 
          labels: torch.Tensor,
          config: TrainingConfig) -> Tuple[nn.Module, Dict]:
    """Main training function."""
    # TODO: Implement complete training loop
    # - Initialize model, optimizer, criterion
    # - Create DataLoader
    # - Run training epochs
    # - Track and return metrics
```

### Phase 4: Evaluation and Analysis

#### TODO 4.1: Implement Evaluation Metrics
**Functions**: `compute_accuracy()`, `compute_cosine_similarity()`
**Requirements**:
```python
def compute_accuracy(predictions: torch.Tensor, 
                    targets: torch.Tensor) -> float:
    """Compute classification accuracy using cosine similarity."""
    # TODO: Implement accuracy computation
    # - Compute cosine similarities
    # - Find nearest neighbor predictions
    # - Calculate accuracy

def compute_cosine_similarity(predictions: torch.Tensor, 
                            targets: torch.Tensor) -> float:
    """Compute average cosine similarity between predictions and targets."""
    # TODO: Implement cosine similarity computation
```

#### TODO 4.2: Create Analysis and Reporting
**Functions**: `analyze_results()`, `save_results()`
**Requirements**:
```python
def analyze_results(model: nn.Module,
                   features: torch.Tensor,
                   labels: torch.Tensor,
                   task_descriptions: List[str],
                   config: DataConfig) -> Dict:
    """Analyze and report results."""
    # TODO: Implement comprehensive analysis
    # - Get predictions from model
    # - Compute final accuracy and similarity
    # - Analyze per-task performance
    # - Generate detailed report

def save_results(results: Dict, config: DataConfig, 
                output_dir: str):
    """Save results to disk."""
    # TODO: Implement result saving
    # - Save metrics and plots
    # - Save model weights
    # - Create summary report
```

### Phase 5: Main Execution

#### TODO 5.1: Implement Main Function
**Function**: `main()`
**Requirements**:
```python
def main():
    """Main execution function."""
    # TODO: Implement main execution flow
    # - Parse arguments
    # - Set random seeds
    # - Set device (CPU/GPU)
    # - Load data
    # - Train linear probe
    # - Evaluate results
    # - Save outputs
```

## Implementation Notes

### Data Loading Strategy
- **Lazy Loading**: Don't load entire pickle files at once
- **Selective Loading**: Only load required rollout steps and layers
- **Memory Management**: Use PyTorch DataLoader for efficient batching
- **Error Handling**: Gracefully handle missing data

### Expert Mapping
- **VLM Expert**: Use `vlm_layer_output["hidden_states"]` (positions 0-815)
- **Action Expert**: Use `action_expert_hidden_state_t{timestep}["hidden_states"]` (positions 816+)
- **Text Only**: Extract positions 768-815 from VLM expert

### Training Considerations
- **Loss Function**: Use cosine similarity loss for embedding regression
- **Normalization**: Normalize features and labels before training
- **Device Management**: Handle CPU/GPU device placement
- **Early Stopping**: Implement early stopping to prevent overfitting

### Output Format
- **Metrics**: Accuracy, cosine similarity, per-task performance
- **Visualizations**: Confusion matrix, similarity heatmap
- **Model**: Saved linear probe weights
- **Report**: Detailed analysis summary

## Usage Examples

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

### Advanced Usage with Action Expert
```bash
python scripts/linear_probing_inference.py \
    --rollout_step 20 \
    --expert action \
    --layer 10 \
    --action_timestep 0.9 \
    --data_path data/inference_latents \
    --task_range 5 15 \
    --episode_range 2 8 \
    --learning_rate 0.001 \
    --num_epochs 100 \
    --batch_size 32 \
    --output_dir results/linear_probing
```

## Expected Outputs

1. **Training Logs**: Progress updates during training
2. **Final Metrics**: Overall accuracy and cosine similarity
3. **Per-Task Analysis**: Individual task performance breakdown
4. **Visualizations**: Confusion matrix and similarity plots
5. **Model Weights**: Saved linear probe parameters
6. **Summary Report**: Comprehensive analysis document

## Checklist

### Phase 1: Setup and Arguments
- [ ] Create main script structure with PyTorch imports
- [ ] Define command line arguments including action_timestep
- [ ] Add argument validation and help messages
- [ ] Set up logging configuration

### Phase 2: Data Loading
- [ ] Implement DataConfig dataclass
- [ ] Create InferenceLatentDataset class
- [ ] Implement feature extraction for VLM expert
- [ ] Implement feature extraction for Action expert (with timestep selection)
- [ ] Implement feature extraction for text_only
- [ ] Create T5 label generation function
- [ ] Add data validation and error handling
- [ ] Test data loading with sample files

### Phase 3: Model and Training
- [ ] Implement LinearProbe class
- [ ] Create training configuration
- [ ] Implement train_epoch function
- [ ] Implement main training loop
- [ ] Add cosine similarity loss function
- [ ] Implement early stopping
- [ ] Add training progress tracking
- [ ] Test training on small dataset

### Phase 4: Evaluation
- [ ] Implement compute_accuracy function
- [ ] Implement compute_cosine_similarity function
- [ ] Create analyze_results function
- [ ] Implement save_results function
- [ ] Add visualization functions (optional)
- [ ] Test evaluation metrics

### Phase 5: Integration
- [ ] Implement main function
- [ ] Add device management (CPU/GPU)
- [ ] Set up random seed handling
- [ ] Add comprehensive error handling
- [ ] Test end-to-end pipeline
- [ ] Add documentation and usage examples

### Phase 6: Optimization and Testing
- [ ] Optimize data loading performance
- [ ] Add memory usage monitoring
- [ ] Test with different expert types
- [ ] Test with different layers and timesteps
- [ ] Validate results against expected outputs
- [ ] Add unit tests for critical functions
- [ ] Performance benchmarking

### Phase 7: Documentation and Deployment
- [ ] Update README with PyTorch version
- [ ] Add usage examples and troubleshooting
- [ ] Create requirements file for PyTorch dependencies
- [ ] Add code comments and docstrings
- [ ] Create example output files
- [ ] Final testing and validation

