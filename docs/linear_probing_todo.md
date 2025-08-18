# Linear Probing TODO: Expert Separation Analysis

## Overview

This document outlines the complete TODO list for rewriting `scripts/linear_probing.py` to perform linear probing analysis on inference latent data with expert separation capabilities. The new script will analyze how well different experts (VLM vs Action) preserve language information across rollout steps and inference steps.

## Current State Analysis

### Existing `linear_probing.py` Limitations:
- ❌ Uses training-time data collection (no rollout context)
- ❌ No expert separation (analyzes combined hidden states)
- ❌ No inference step analysis (only diffusion steps)
- ❌ No rollout step tracking
- ❌ Limited command line arguments
- ❌ Basic T5 label generation

### New Requirements:
- ✅ Use inference-time latent data (from `scripts/inference_latent_collection.py`)
- ✅ Support expert separation (VLM vs Action expert)
- ✅ Analyze specific rollout steps and inference steps
- ✅ Layer-wise analysis (0-17 layers)
- ✅ Comprehensive metrics and reporting
- ✅ Batch processing capabilities

## Data Structure Understanding

### Inference Latent Data Format:
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

### Expert Token Positions:
- **Expert 0 (VLM/PaliGemma)**: Positions 0-815
  - Images: Positions 0-767 (256×3 tokens)
  - Text: Positions 768-815 (48 tokens)
- **Expert 1 (Action Expert)**: Positions 816+
  - State + Action tokens

## Phase-by-Phase TODO List

## **Phase 1: Setup and Infrastructure**

### **TODO 1.1: Update Command Line Arguments**
**File**: `scripts/linear_probing.py`
**Current**: Basic flags for model path, T5 model, etc.
**New Requirements**:
```python
# Add these new flags to FLAGS
flags.DEFINE_integer("rollout_step", 0, "Which rollout step to analyze (e.g., 0, 5, 10, 15, 20)")
flags.DEFINE_integer("inference_step", 0, "Which inference step to analyze (0-9 for 10 diffusion steps)")
flags.DEFINE_integer("layer", 0, "Which layer to analyze (0-17 for 18 layers)")
flags.DEFINE_enum("expert", "vlm", ["vlm", "action", "text_only"], "Which expert to analyze")
flags.DEFINE_string("data_path", "exp_data/inference_latents", "Path to inference latents directory")
flags.DEFINE_list("task_range", ["0", "10"], "Range of tasks to analyze (start, end)")
flags.DEFINE_list("episode_range", ["0", "5"], "Range of episodes per task (start, end)")
flags.DEFINE_string("output_dir", "results/linear_probing", "Output directory for results")
flags.DEFINE_boolean("batch_mode", False, "Run analysis for multiple configurations")
```

### **TODO 1.2: Create Data Loading Infrastructure**
**File**: `scripts/linear_probing.py`
**New Functions Needed**:
```python
def load_inference_latent_data(data_path: str, task_range: tuple, episode_range: tuple) -> Dict:
    """Load inference latent data from pickle files."""
    
def extract_features_from_data(data: Dict, rollout_step: int, inference_step: int, 
                              layer: int, expert: str) -> np.ndarray:
    """Extract features based on configuration."""
    
def create_task_descriptions(data: Dict) -> List[str]:
    """Extract task descriptions from loaded data."""
```

**Implementation Details**:
- Handle nested data structure: `task_data["episodes"]["episode_0"]["rollout_steps"]["step_10"]`
- Load data from `exp_data/inference_latents/task_{id}_{name}.pkl` files
- Handle missing data gracefully (some episodes may not have all rollout steps)
- Aggregate features across episodes and tasks

---

## **Phase 2: Feature Extraction and Data Processing**

### **TODO 2.1: Implement Expert-Specific Feature Extraction**
**File**: `scripts/linear_probing.py`
**New Function**:
```python
def extract_expert_features(data: Dict, rollout_step: int, inference_step: int, 
                           layer: int, expert: str) -> np.ndarray:
    """
    Extract features based on expert type:
    - "vlm": Use expert_0_hidden_states (PaliGemma, positions 0-815)
    - "action": Use expert_1_hidden_states (Action expert, positions 816+)
    - "text_only": Use text_only_hidden_states (positions 768-815)
    
    Returns: (num_samples, hidden_dim) array
    """
```

**Implementation Details**:
- Map expert types to correct hidden state keys:
  - `"vlm"` → `"expert_0_hidden_states"`
  - `"action"` → `"expert_1_hidden_states"`
  - `"text_only"` → `"text_only_hidden_states"`
- Extract specific layer and inference step from nested structure
- Handle shape transformations: `(18, 1, seq_len, 2048)` → `(num_samples, 2048)`
- Aggregate across episodes and tasks

### **TODO 2.2: Create T5 Label Generation**
**File**: `scripts/linear_probing.py`
**New Function**:
```python
def create_t5_labels(task_descriptions: List[str], t5_model_name: str = "t5-small") -> np.ndarray:
    """
    Use T5-small to encode task descriptions into embeddings as labels.
    
    Args:
        task_descriptions: List of task description strings
        t5_model_name: T5 model variant to use
        
    Returns: (num_tasks, embedding_dim) array
    """
```

**Implementation Details**:
- Load T5-small model and tokenizer using transformers library
- Encode each task description with proper tokenization
- Handle padding and truncation for variable-length descriptions
- Return mean-pooled embeddings as labels
- Cache T5 embeddings to avoid recomputation

### **TODO 2.3: Create DataLoader Class**
**File**: `scripts/linear_probing.py`
**New Class**:
```python
class InferenceLatentDataLoader:
    """DataLoader for inference latent data with expert separation."""
    
    def __init__(self, data_path: str, task_range: tuple, episode_range: tuple,
                 rollout_step: int, inference_step: int, layer: int, expert: str):
        self.data_path = data_path
        self.task_range = task_range
        self.episode_range = episode_range
        self.rollout_step = rollout_step
        self.inference_step = inference_step
        self.layer = layer
        self.expert = expert
        self._load_data()
        
    def _load_data(self):
        """Load and preprocess inference latent data."""
        
    def get_features_and_labels(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return features and T5 labels."""
        
    def get_task_descriptions(self) -> List[str]:
        """Return list of task descriptions."""
        
    def get_data_info(self) -> Dict:
        """Return information about loaded data."""
```

---

## **Phase 3: Linear Network and Training**

### **TODO 3.1: Update Linear Probe Architecture**
**File**: `scripts/linear_probing.py`
**Current**: Simple linear probe with fixed input size
**New Requirements**:
```python
class LinearProbe(nn.Module):
    """Linear probe for inference latent analysis."""
    input_dim: int  # Hidden state dimension (2048)
    output_dim: int  # T5 embedding dimension
    
    @nn.compact
    def __call__(self, x):
        return nn.Dense(features=self.output_dim)(x)
```

### **TODO 3.2: Implement Training Loop**
**File**: `scripts/linear_probing.py`
**New Functions**:
```python
def create_train_state(rng, learning_rate, input_dim, output_dim):
    """Create initial training state for linear probe."""
    
def train_step(state, batch):
    """Single training step with cosine similarity loss."""
    
def eval_step(state, batch, all_targets):
    """Evaluation step with multiple metrics."""
    
def train_probe(features, labels, learning_rate, num_epochs, batch_size, 
                validation_split=0.2):
    """Complete training pipeline with validation."""
```

**Training Details**:
- Use cosine similarity loss for comparing embeddings
- Implement proper train/validation split
- Add early stopping based on validation loss
- Track training metrics (loss, accuracy, cosine similarity)
- Use JAX JIT compilation for efficiency

### **TODO 3.3: Add Comprehensive Metrics**
**File**: `scripts/linear_probing.py`
**New Functions**:
```python
def compute_metrics(predictions, targets) -> Dict:
    """
    Compute multiple metrics:
    - Cosine similarity
    - L2 distance
    - Classification accuracy (nearest neighbor)
    - Top-k accuracy
    - Mean squared error
    """
    
def compute_cosine_similarity(predictions, targets) -> float:
    """Compute average cosine similarity between predictions and targets."""
    
def compute_nearest_neighbor_accuracy(predictions, targets) -> float:
    """Compute classification accuracy using nearest neighbor."""
```

---

## **Phase 4: Analysis and Reporting**

### **TODO 4.1: Create Analysis Pipeline**
**File**: `scripts/linear_probing.py`
**New Function**:
```python
def analyze_expert_language_loss(data_path: str, rollout_step: int, inference_step: int,
                                layer: int, expert: str, task_range: tuple, 
                                episode_range: tuple, **kwargs) -> Dict:
    """
    Complete analysis pipeline:
    1. Load data
    2. Extract features
    3. Create labels
    4. Train probe
    5. Evaluate and report metrics
    
    Returns: Dictionary with all results and metrics
    """
```

### **TODO 4.2: Implement Reporting System**
**File**: `scripts/linear_probing.py`
**New Functions**:
```python
def print_analysis_results(results: Dict, rollout_step: int, inference_step: int,
                          layer: int, expert: str):
    """Print detailed analysis results."""
    
def save_results_to_file(results: Dict, output_path: str):
    """Save results to JSON file."""
    
def create_summary_table(all_results: List[Dict]) -> str:
    """Create summary table comparing different configurations."""
    
def generate_analysis_report(results: Dict, config: Dict) -> str:
    """Generate comprehensive analysis report."""
```

**Reporting Details**:
- Print detailed metrics for each configuration
- Save results to JSON/CSV files with timestamps
- Create summary tables comparing different experts/layers/steps
- Generate markdown reports for easy reading
- Include configuration parameters in results

---

## **Phase 5: Integration and Testing**

### **TODO 5.1: Update Main Function**
**File**: `scripts/linear_probing.py`
**Current**: Simple main function with basic data collection
**New Requirements**:
```python
def main(_):
    """Main function for linear probing analysis."""
    # Parse new command line arguments
    config = parse_config_from_flags()
    
    # Validate configuration
    validate_config(config)
    
    # Run analysis
    if FLAGS.batch_mode:
        results = run_batch_analysis(config)
    else:
        results = analyze_expert_language_loss(**config)
    
    # Report results
    print_analysis_results(results, **config)
    save_results_to_file(results, FLAGS.output_dir)
```

### **TODO 5.2: Add Error Handling**
**File**: `scripts/linear_probing.py`
**New Requirements**:
```python
def validate_config(config: Dict) -> bool:
    """Validate configuration parameters."""
    
def handle_missing_data(data_path: str, task_range: tuple) -> List[int]:
    """Handle missing data files gracefully."""
    
def check_memory_requirements(features: np.ndarray, labels: np.ndarray) -> bool:
    """Check if system has enough memory for analysis."""
```

**Error Handling Details**:
- Handle missing data files gracefully
- Validate command line arguments
- Add progress bars for data loading
- Handle memory issues with large datasets
- Provide informative error messages

### **TODO 5.3: Add Configuration Validation**
**File**: `scripts/linear_probing.py`
**New Function**:
```python
def validate_config(rollout_step: int, inference_step: int, layer: int, 
                   expert: str, data_path: str) -> bool:
    """
    Validate that the requested configuration is valid:
    - Check if data files exist
    - Validate step ranges
    - Check expert type
    - Validate layer range
    """
```

---

## **Phase 6: Advanced Features**

### **TODO 6.1: Add Batch Processing**
**File**: `scripts/linear_probing.py`
**New Function**:
```python
def run_batch_analysis(config: Dict) -> Dict:
    """
    Run analysis for multiple configurations:
    - Different rollout steps
    - Different inference steps  
    - Different layers
    - Different experts
    
    Returns: Dictionary with results for all configurations
    """
```

**Batch Processing Details**:
- Support multiple rollout steps: `[0, 5, 10, 15, 20]`
- Support multiple inference steps: `[0, 1, 2, 3, 4, 5]`
- Support multiple layers: `[0, 5, 10, 15]`
- Support all expert types: `["vlm", "action", "text_only"]`
- Parallel processing for efficiency

### **TODO 6.2: Add Cross-Validation**
**File**: `scripts/linear_probing.py`
**New Function**:
```python
def cross_validate_probe(features, labels, n_folds=5, **kwargs) -> Dict:
    """
    Perform k-fold cross-validation to get more robust results.
    
    Returns: Dictionary with cross-validation metrics
    """
```

### **TODO 6.3: Add Visualization**
**File**: `scripts/linear_probing.py`
**New Functions**:
```python
def plot_metrics_comparison(results: Dict, output_path: str):
    """Create comparison plots for different configurations."""
    
def plot_training_curves(training_history: Dict, output_path: str):
    """Plot training and validation curves."""
```

---

## **Implementation Order**

### **Priority 1: Core Functionality**
1. **Phase 1**: Setup infrastructure and command line arguments
2. **Phase 2**: Implement data loading and feature extraction
3. **Phase 3**: Update linear probe and training loop
4. **Phase 4**: Add basic analysis and reporting

### **Priority 2: Integration and Testing**
5. **Phase 5**: Integrate and test with real data
6. **Phase 6**: Add advanced features (batch processing, cross-validation)

### **Priority 3: Optimization and Enhancement**
7. Add visualization capabilities
8. Optimize memory usage for large datasets
9. Add parallel processing for batch mode

---

## **Key Files to Modify**

1. **`scripts/linear_probing.py`** - Main script (complete rewrite)
2. **`scripts/expert_data_utils.py`** - May need updates for data loading
3. **`src/openpi/models/gemma.py`** - Already has expert separation methods

## **Expected Data Flow**

```
Command Line Args → Data Loading → Feature Extraction → T5 Labeling → 
Linear Probe Training → Metrics Computation → Results Reporting
```

## **Usage Examples**

### **Single Configuration Analysis**
```bash
python scripts/linear_probing.py \
    --data_path exp_data/inference_latents \
    --rollout_step 10 \
    --inference_step 5 \
    --layer 10 \
    --expert vlm \
    --task_range 0 10 \
    --episode_range 0 5
```

### **Batch Mode Analysis**
```bash
python scripts/linear_probing.py \
    --data_path exp_data/inference_latents \
    --batch_mode \
    --task_range 0 10 \
    --episode_range 0 5 \
    --output_dir results/batch_analysis
```

## **Expected Output**

### **Console Output**
```
=== Linear Probing Analysis Results ===
Configuration:
  - Rollout Step: 10
  - Inference Step: 5
  - Layer: 10
  - Expert: vlm
  - Tasks: 0-10
  - Episodes: 0-5

Data Loading:
  - Loaded 50 samples from 10 tasks
  - Feature shape: (50, 2048)
  - Label shape: (10, 768)

Training Results:
  - Final Loss: 0.1234
  - Cosine Similarity: 0.8567
  - Nearest Neighbor Accuracy: 0.8000
  - Top-3 Accuracy: 0.9000

Analysis Complete!
Results saved to: results/linear_probing/analysis_2024-01-15_14-30-25.json
```

### **JSON Output**
```json
{
  "config": {
    "rollout_step": 10,
    "inference_step": 5,
    "layer": 10,
    "expert": "vlm",
    "task_range": [0, 10],
    "episode_range": [0, 5]
  },
  "data_info": {
    "num_samples": 50,
    "num_tasks": 10,
    "feature_shape": [50, 2048],
    "label_shape": [10, 768]
  },
  "metrics": {
    "final_loss": 0.1234,
    "cosine_similarity": 0.8567,
    "nearest_neighbor_accuracy": 0.8000,
    "top_3_accuracy": 0.9000,
    "l2_distance": 0.2345
  },
  "training_history": {
    "train_loss": [...],
    "val_loss": [...],
    "epochs": [...]
  }
}
```

## **Success Criteria**

1. **Functionality**: Script successfully analyzes inference latent data with expert separation
2. **Accuracy**: Linear probe achieves reasonable performance (>0.7 cosine similarity)
3. **Usability**: Clear command line interface with helpful error messages
4. **Efficiency**: Handles large datasets without memory issues
5. **Reproducibility**: Results are saved and can be reproduced
6. **Extensibility**: Easy to add new metrics or analysis types

This TODO list provides a comprehensive roadmap for rewriting the linear probing script to work with the new inference latent data structure and support the expert separation analysis requirements. 