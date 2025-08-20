# Linear Probing Analysis for Inference Latent Data

This directory contains scripts for performing linear probing analysis on inference latent data to understand how well different experts (VLM vs Action) preserve language information across rollout steps and inference steps.

## Overview

The linear probing script analyzes the hidden states from different experts in the Pi0 model to determine how well they encode task descriptions. It trains a linear probe to predict T5 embeddings of task descriptions from the hidden states.

## Files

- `linear_probing_inference.py`: Main script for linear probing analysis
- `linear_probing_todo.md`: Detailed TODO list and implementation guide
- `requirements_linear_probing.txt`: Python dependencies
- `README_linear_probing.md`: This file

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements_linear_probing.txt
```

2. Ensure you have inference latent data collected using `scripts/inference_latent_collection.py`

## Usage

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

## Arguments

### Required Arguments

- `--rollout_step`: Specific rollout step to analyze (int)
- `--expert`: Expert type to analyze - "vlm", "action", or "text_only" (str)
- `--layer`: Layer ID to analyze (0-17) (int)
- `--data_path`: Path to inference latent data (str)
- `--task_range`: Range of tasks to analyze (start, end) (int, int)
- `--episode_range`: Range of episodes to analyze (start, end) (int, int)

### Optional Arguments

- `--learning_rate`: Learning rate for training (float, default: 0.001)
- `--num_epochs`: Number of training epochs (int, default: 100)
- `--batch_size`: Batch size for training (int, default: 32)
- `--seed`: Random seed (int, default: 42)
- `--output_dir`: Output directory for results (str, default: "results/linear_probing")

## Expert Types

- **vlm**: VLM/PaliGemma expert (positions 0-815)
- **action**: Action expert (positions 816+)
- **text_only**: Text-only tokens from VLM expert (positions 768-815)

## Data Structure

The script expects inference latent data in the following format:

```
data/inference_latents/
├── task_0_pick_up_the_cream_cheese_and_place_it_in_the_basket.pkl
├── task_1_open_the_top_drawer_and_put_the_bowl_inside.pkl
└── ...
```

Each pickle file contains:
- Task metadata (name, description, ID)
- Episode data with rollout steps
- Hidden states for both VLM and Action experts
- Layer outputs and attention information

## Output

The script generates:

1. **Training Logs**: Progress updates during training
2. **Results File**: Pickle file with detailed results
3. **Summary File**: Text file with key metrics
4. **Model Weights**: Saved linear probe parameters

### Example Output Structure

```
results/linear_probing/
├── results_vlm_layer5_step10.pkl
├── summary_vlm_layer5_step10.txt
└── ...
```

## Metrics

The script computes:

- **Accuracy**: Classification accuracy using cosine similarity
- **Cosine Similarity**: Average cosine similarity between predictions and targets
- **Per-task Performance**: Individual task performance breakdown

## Example Results

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

## Troubleshooting

### Common Issues

1. **No data files found**: Ensure the data path is correct and contains pickle files
2. **Memory issues**: Reduce batch size or number of tasks/episodes
3. **CUDA out of memory**: Use CPU or reduce batch size
4. **Missing rollout steps**: Check that the specified rollout step exists in the data

### Debug Mode

Add verbose logging by modifying the logging level in the script:

```python
logging.basicConfig(level=logging.DEBUG)
```

## Extending the Script

### Adding New Expert Types

1. Add the expert type to the choices in `parse_args()`
2. Implement feature extraction logic in `_extract_features()`
3. Update the expert mapping documentation

### Adding New Metrics

1. Implement the metric function
2. Add it to the `analyze_results()` function
3. Include it in the output summary

### Custom Loss Functions

Modify the `train_step()` function to use different loss functions:

```python
def custom_loss_fn(predictions, targets):
    # Implement custom loss
    return loss
```

## Related Scripts

- `scripts/inference_latent_collection.py`: Collects inference latent data
- `scripts/linear_probing.py`: Original linear probing script (training-time data)

## Citation

If you use this script in your research, please cite the relevant papers and acknowledge the Pi0 project.

