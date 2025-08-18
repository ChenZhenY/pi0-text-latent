import os.path
import os.path as osp
import pickle
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import json

EXP_DATA_PATH = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "exp_data", "inference_latents")


def load_inference_task_data(task_id: int, task_name: Optional[str] = None, 
                           data_path: str = EXP_DATA_PATH) -> Dict[str, Any]:
    """Load inference task data from pickle file."""
    if task_name is None:
        # Try to find the file by task_id
        for filename in os.listdir(data_path):
            if filename.startswith(f"task_{task_id}_") and filename.endswith(".pkl"):
                task_name = filename.replace(f"task_{task_id}_", "").replace(".pkl", "")
                break
        if task_name is None:
            raise FileNotFoundError(f"No data file found for task_id {task_id}")
    
    filepath = osp.join(data_path, f"task_{task_id}_{task_name}.pkl")
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def get_hidden_states_for_dimensions(task_data: Dict[str, Any], 
                                   episode_idx: int, 
                                   rollout_step: int, 
                                   layer_idx: Optional[int] = None,
                                   expert_type: str = "expert_0") -> np.ndarray:
    """Get hidden states for specific dimensions from inference data.
    
    Args:
        task_data: Task data dictionary
        episode_idx: Episode index
        rollout_step: Rollout step (timestep)
        layer_idx: Layer index (None for all layers)
        expert_type: Type of expert ("expert_0", "expert_1", "text_only")
    
    Returns:
        Hidden states array
    """
    episode_data = task_data["episodes"][f"episode_{episode_idx}"]
    step_data = episode_data["rollout_steps"][f"step_{rollout_step}"]
    
    if expert_type not in step_data:
        raise KeyError(f"Expert type '{expert_type}' not found in step data")
    
    hidden_states = step_data[expert_type]
    
    if layer_idx is not None:
        return hidden_states[layer_idx]
    else:
        return hidden_states


def get_language_loss_trajectory(task_data: Dict[str, Any], 
                                episode_idx: int,
                                expert_type: str = "text_only",
                                layer_idx: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Get language loss trajectory across rollout steps for a specific episode.
    
    Args:
        task_data: Task data dictionary
        episode_idx: Episode index
        expert_type: Type of expert to analyze
        layer_idx: Layer index (None for all layers)
    
    Returns:
        Dictionary with timesteps and corresponding hidden states
    """
    episode_data = task_data["episodes"][f"episode_{episode_idx}"]
    rollout_steps = episode_data["rollout_steps"]
    
    timesteps = []
    hidden_states_list = []
    
    for step_key, step_data in rollout_steps.items():
        if expert_type in step_data:
            timestep = step_data["timestep"]
            hidden_states = step_data[expert_type]
            
            if layer_idx is not None:
                hidden_states = hidden_states[layer_idx]
            
            timesteps.append(timestep)
            hidden_states_list.append(hidden_states)
    
    return {
        "timesteps": np.array(timesteps),
        "hidden_states": np.array(hidden_states_list)
    }


def analyze_expert_separation(task_data: Dict[str, Any], 
                            episode_idx: int,
                            rollout_step: int,
                            layer_idx: int = 0) -> Dict[str, float]:
    """Analyze separation between experts at a specific timestep.
    
    Args:
        task_data: Task data dictionary
        episode_idx: Episode index
        rollout_step: Rollout step
        layer_idx: Layer index to analyze
    
    Returns:
        Dictionary with separation metrics
    """
    # Get hidden states for both experts
    expert_0_states = get_hidden_states_for_dimensions(
        task_data, episode_idx, rollout_step, layer_idx, "expert_0"
    )
    expert_1_states = get_hidden_states_for_dimensions(
        task_data, episode_idx, rollout_step, layer_idx, "expert_1"
    )
    
    # Calculate cosine similarity between experts
    expert_0_flat = expert_0_states.flatten()
    expert_1_flat = expert_1_states.flatten()
    
    cosine_sim = np.dot(expert_0_flat, expert_1_flat) / (
        np.linalg.norm(expert_0_flat) * np.linalg.norm(expert_1_flat)
    )
    
    # Calculate L2 distance
    l2_distance = np.linalg.norm(expert_0_flat - expert_1_flat)
    
    return {
        "cosine_similarity": float(cosine_sim),
        "l2_distance": float(l2_distance),
        "expert_0_norm": float(np.linalg.norm(expert_0_flat)),
        "expert_1_norm": float(np.linalg.norm(expert_1_flat))
    }


def get_task_summary_stats(data_path: str = EXP_DATA_PATH) -> Dict[str, Any]:
    """Get summary statistics for all collected tasks.
    
    Args:
        data_path: Path to data directory
    
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        "total_tasks": 0,
        "total_episodes": 0,
        "total_successes": 0,
        "tasks": {}
    }
    
    for filename in os.listdir(data_path):
        if filename.startswith("task_") and filename.endswith(".pkl"):
            filepath = osp.join(data_path, filename)
            with open(filepath, 'rb') as f:
                task_data = pickle.load(f)
            
            task_id = task_data["task_id"]
            task_name = task_data["task_name"]
            
            task_episodes = len(task_data["episodes"])
            task_successes = sum(
                1 for episode in task_data["episodes"].values()
                if episode["metadata"]["success"]
            )
            
            summary["total_tasks"] += 1
            summary["total_episodes"] += task_episodes
            summary["total_successes"] += task_successes
            
            summary["tasks"][task_id] = {
                "name": task_name,
                "episodes": task_episodes,
                "successes": task_successes,
                "success_rate": task_successes / task_episodes if task_episodes > 0 else 0
            }
    
    if summary["total_episodes"] > 0:
        summary["overall_success_rate"] = summary["total_successes"] / summary["total_episodes"]
    else:
        summary["overall_success_rate"] = 0
    
    return summary


def save_analysis_results(analysis_results: Dict[str, Any], 
                         output_path: str,
                         filename: str = "analysis_results.json") -> None:
    """Save analysis results to JSON file.
    
    Args:
        analysis_results: Analysis results dictionary
        output_path: Output directory path
        filename: Output filename
    """
    os.makedirs(output_path, exist_ok=True)
    filepath = osp.join(output_path, filename)
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_numpy(analysis_results)
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)


if __name__ == "__main__":
    # Example usage
    print("Loading task summary...")
    summary = get_task_summary_stats()
    print(f"Total tasks: {summary['total_tasks']}")
    print(f"Total episodes: {summary['total_episodes']}")
    print(f"Overall success rate: {summary['overall_success_rate']:.3f}")
    
    # Example: Load specific task data
    if summary["total_tasks"] > 0:
        first_task_id = list(summary["tasks"].keys())[0]
        print(f"\nLoading data for task {first_task_id}...")
        task_data = load_inference_task_data(first_task_id)
        print(f"Task: {task_data['task_name']}")
        print(f"Episodes: {len(task_data['episodes'])}")
        
        # Example: Get language loss trajectory
        if len(task_data["episodes"]) > 0:
            trajectory = get_language_loss_trajectory(task_data, 0, "text_only")
            print(f"Language loss trajectory timesteps: {len(trajectory['timesteps'])}") 