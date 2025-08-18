import dataclasses
import os.path
import pickle
import numpy as np
import h5py

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import tqdm
from toolz.tests.test_dicttoolz import defaultdict

import openpi.models.model as _model
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.sharding as sharding
import openpi.transforms as _transforms
from openpi import EXP_DATA_PATH
from openpi.models.pi0 import Pi0
from openpi.shared import nnx_utils
from openpi.shared.download import maybe_download
from openpi.training.data_loader import TorchDataLoader, TransformedDataset


def normalize(vectors):
    return vectors / jnp.linalg.norm(vectors, axis=1, keepdims=True)


@dataclasses.dataclass
class Checkpoint:
    config: str
    dir: str


@dataclasses.dataclass
class Args:
    default_prompt: str | None = None
    policy = None


def create_dataloader(train_config, data_config, ckpt_dir, task_range, batch_size, episode_to_use_per_task, debug):
    norm_stats = _checkpoints.load_norm_stats(ckpt_dir / "assets", data_config.asset_id)
    mesh = sharding.make_mesh(train_config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id, local_files_only=False)
    episode_per_task = defaultdict(list)
    for index, episode in enumerate(dataset_meta.episodes):
        task_desc = episode["tasks"][0]
        if task_range[0] <= dataset_meta.task_to_task_index[task_desc] < task_range[1]:
            episode_per_task[task_desc].append(index)

    assert len(episode_per_task) == task_range[1] - task_range[0], "episode picking error"
    episode_to_use = []
    for key in episode_per_task:
        episode_to_use += episode_per_task[key][:episode_to_use_per_task] \
            if episode_to_use_per_task else episode_per_task[key]

    delta_time = {key: [t / dataset_meta.fps for t in range(train_config.model.action_horizon)]
                  for key in data_config.action_sequence_keys}
    dataset = lerobot_dataset.LeRobotDataset(data_config.repo_id, delta_timestamps=delta_time,
                                             local_files_only=False,
                                             episodes=episode_to_use)
    assert data_config.norm_stats is None, "we will overwrite the norm stats with the training one"
    dataset = TransformedDataset(dataset, [
        _transforms.PromptFromLeRobotTask(dataset_meta.tasks),
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs])
    local_batch_size = batch_size
    data_loader = TorchDataLoader(dataset,
                                  local_batch_size=local_batch_size,
                                  sharding=data_sharding,
                                  shuffle=False,
                                  num_workers=0 if debug else 8,
                                  drop_last=False,
                                  seed=train_config.seed)

    num_iters = len(data_loader.torch_loader)
    data_loader.set_num_batches(num_iters)
    return data_loader, dataset_meta


def decode(obs, dataset):
    """Decode tokenized prompt back to text."""
    if not isinstance(obs, list):
        obs = obs.tokenized_prompt[0].tolist()
    return dataset._transform.transforms[-1].tokenizer._tokenizer.decode(obs)


def collect_language_loss_data(model, data_loader, dataset, sampling_config):
    """
    Collect language information loss data across three dimensions:
    1. Rollout steps (different timesteps in episode)
    2. Inference steps (different layers in model)
    3. Model layers (different transformer layers)
    """
    
    # Configuration
    text_start = 256 * 3  # 768
    text_end = text_start + 48  # 816
    selected_layers = sampling_config["layers"]
    timestep_stride = sampling_config.get("timestep_stride", 6)  # Sample every 6th timestep
    
    # Initialize storage
    all_data = {
        "task_id": [],
        "episode_id": [],
        "timestep": [],
        "layer": [],
        "text_hidden_states": [],
        "original_prompt": [],
        "reconstructed_prompt": [],
        "language_loss_metrics": []
    }
    
    compute_loss = nnx_utils.module_jit(model.compute_loss_with_extra)
    
    for batch_idx, batch in enumerate(tqdm.tqdm(data_loader, desc="Processing batches")):
        # Skip timesteps based on stride
        if batch_idx % timestep_stride != 0:
            continue
            
        actions = batch.pop("actions")
        obs = _model.Observation.from_dict(batch)
        
        # Get model outputs
        result = compute_loss(jax.random.key(0), obs, actions)
        
        # Extract text-specific hidden states for selected layers
        hidden_states = result["hidden_states_sum"]  # (18, prefix_len, 2048)
        text_hidden_states = hidden_states[selected_layers, text_start:text_end, :]  # (num_layers, 48, 2048)
        
        # Get original and reconstructed prompts
        original_prompt = decode(obs, dataset)
        
        # Reconstruct prompt from hidden states
        embedder = model.PaliGemma.llm.embedder["input_embedding"]
        norm_embedding_matrix = normalize(embedder)
        
        reconstructed_prompts = []
        for layer_idx, layer_states in enumerate(text_hidden_states):
            norm_embedded_tokens = normalize(layer_states)
            similarity = jnp.dot(norm_embedded_tokens, norm_embedding_matrix.T)
            token_indices = jnp.argmax(similarity, axis=1)
            reconstructed_prompt = decode(token_indices.tolist(), dataset)
            reconstructed_prompts.append(reconstructed_prompt)
        
        # Calculate language loss metrics
        language_loss_metrics = calculate_language_loss_metrics(
            original_prompt, reconstructed_prompts, text_hidden_states
        )
        
        # Store data
        for layer_idx, layer_id in enumerate(selected_layers):
            all_data["task_id"].append(batch_idx)
            all_data["episode_id"].append(batch_idx)
            all_data["timestep"].append(batch_idx)
            all_data["layer"].append(layer_id)
            all_data["text_hidden_states"].append(text_hidden_states[layer_idx])
            all_data["original_prompt"].append(original_prompt)
            all_data["reconstructed_prompt"].append(reconstructed_prompts[layer_idx])
            all_data["language_loss_metrics"].append(language_loss_metrics[layer_idx])
    
    return all_data


def calculate_language_loss_metrics(original_prompt, reconstructed_prompts, text_hidden_states):
    """Calculate various metrics to quantify language information loss."""
    metrics = []
    
    for layer_idx, reconstructed_prompt in enumerate(reconstructed_prompts):
        # 1. Text similarity (using simple string comparison for now)
        exact_match = original_prompt == reconstructed_prompt
        
        # 2. Token-level accuracy
        original_tokens = original_prompt.split()
        reconstructed_tokens = reconstructed_prompt.split()
        token_accuracy = len(set(original_tokens) & set(reconstructed_tokens)) / max(len(original_tokens), 1)
        
        # 3. Hidden state variance (measure of information preservation)
        layer_states = text_hidden_states[layer_idx]  # (48, 2048)
        state_variance = jnp.var(layer_states, axis=0).mean()  # Average variance across dimensions
        
        metrics.append({
            "exact_match": exact_match,
            "token_accuracy": token_accuracy,
            "state_variance": float(state_variance),
            "layer_idx": layer_idx
        })
    
    return metrics


if __name__ == '__main__':
    # Configuration for memory-efficient data collection
    sampling_config = {
        "tasks": range(10, 15),  # Start with 5 tasks
        "episodes_per_task": 5,  # Reduced from 20
        "timestep_stride": 10,  # Sample every 10th timestep
        "layers": [0, 3, 6, 9, 12, 15, 17],  # Key layers
    }
    
    # Load model
    prefix_len = 816
    args = Args()
    args.policy = Checkpoint(config="pi0_libero", dir="s3://openpi-assets/checkpoints/pi0_libero")
    train_config = _config.get_config(args.policy.config)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    ckpt_dir = maybe_download(args.policy.dir)
    model: Pi0 = train_config.model.load(_model.restore_params(ckpt_dir / "params", dtype=jnp.bfloat16))
    
    # Collect data for each task
    all_task_data = {}
    
    for task_id in sampling_config["tasks"]:
        print(f"Processing task {task_id}")
        task_range = (task_id, task_id + 1)
        
        data_loader, dataset_meta = create_dataloader(
            train_config, data_config, ckpt_dir, task_range, 1,
            sampling_config["episodes_per_task"], True
        )
        dataset = data_loader.torch_loader.dataset
        
        # Collect data for this task
        task_data = collect_language_loss_data(model, data_loader, dataset, sampling_config)
        all_task_data[task_id] = task_data
        
        # Save incrementally to avoid memory issues
        save_path = f"{EXP_DATA_PATH}/language_loss_analysis/task_{task_id}.pkl"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Convert to numpy arrays and save
        with open(save_path, 'wb') as f:
            pickle.dump(task_data, f)
    
    # Analyze and visualize results
    analyze_language_loss(all_task_data, sampling_config)


def analyze_language_loss(all_task_data, sampling_config):
    """Analyze the collected data to identify where language information is lost."""
    
    # Aggregate metrics across tasks
    layer_metrics = defaultdict(list)
    
    for task_id, task_data in all_task_data.items():
        for layer_idx, layer_id in enumerate(sampling_config["layers"]):
            layer_mask = [i for i, layer in enumerate(task_data["layer"]) if layer == layer_id]
            
            for idx in layer_mask:
                metrics = task_data["language_loss_metrics"][idx]
                layer_metrics[layer_id].append({
                    "token_accuracy": metrics["token_accuracy"],
                    "state_variance": metrics["state_variance"],
                    "timestep": task_data["timestep"][idx]
                })
    
    # Calculate average metrics per layer
    layer_averages = {}
    for layer_id, metrics_list in layer_metrics.items():
        avg_token_accuracy = np.mean([m["token_accuracy"] for m in metrics_list])
        avg_state_variance = np.mean([m["state_variance"] for m in metrics_list])
        layer_averages[layer_id] = {
            "avg_token_accuracy": avg_token_accuracy,
            "avg_state_variance": avg_state_variance,
            "num_samples": len(metrics_list)
        }
    
    # Print results
    print("\n=== Language Information Loss Analysis ===")
    print("Layer\tToken Accuracy\tState Variance\tSamples")
    print("-" * 50)
    for layer_id in sorted(layer_averages.keys()):
        metrics = layer_averages[layer_id]
        print(f"{layer_id}\t{metrics['avg_token_accuracy']:.3f}\t\t{metrics['avg_state_variance']:.3f}\t\t{metrics['num_samples']}")
    
    # Identify critical layers where information is lost
    token_accuracies = [layer_averages[layer_id]["avg_token_accuracy"] for layer_id in sorted(layer_averages.keys())]
    critical_layers = []
    
    for i, layer_id in enumerate(sorted(layer_averages.keys())):
        if i > 0 and token_accuracies[i] < token_accuracies[i-1] * 0.8:  # 20% drop
            critical_layers.append(layer_id)
    
    print(f"\nCritical layers where language information is lost: {critical_layers}")
    
    return layer_averages, critical_layers 