import collections
import dataclasses
import json
import logging
import os.path
import os.path as osp
import pathlib
import pickle
import random
from datetime import datetime
import numpy as np
import tqdm
import imageio
import cv2
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

# dirty hack, this file can't import openpi under this dir... and Analysis in gemma.py
class Analysis:
    @staticmethod
    def get_mlp_activation(attn_output, layer_index=None):
        layer_index = layer_index or [i for i in range(len(attn_output[0]))]
        return attn_output[0][jnp.asarray(layer_index)] if attn_output[0] is not None else None

    @staticmethod
    def get_attention(attn_output, layer_index=None):
        layer_index = layer_index or [i for i in range(len(attn_output[1]))]
        return attn_output[1][jnp.asarray(layer_index)]

    @staticmethod
    def get_pre_attn_norm_scales(attn_output, layer_index=None):
        layer_index = layer_index or [i for i in range(len(attn_output[2]))]
        return attn_output[2][jnp.asarray(layer_index)]

    @staticmethod
    def get_pre_mlp_norm_scales(attn_output, layer_index=None):
        layer_index = layer_index or [i for i in range(len(attn_output[3]))]
        return attn_output[3][jnp.asarray(layer_index)] if attn_output[3] is not None else None

    @staticmethod
    def get_final_norm_scales(attn_output):
        return attn_output[4]

    @staticmethod
    def get_post_attn_value(attn_output, layer_index=None):
        layer_index = layer_index or [i for i in range(len(attn_output[5][0]))]
        return [x[jnp.asarray(layer_index)] if x is not None else None for x in attn_output[5]]

    @staticmethod
    def get_neuron_memory(nnx_model, module_index, layer_index, neuron_index=None):
        layer = getattr(nnx_model, f"layers_{layer_index}")
        module_name = "mlp" if module_index == 0 else "mlp_1"
        memory = layer[module_name]["linear"]
        if neuron_index is not None:
            return memory[neuron_index]
        else:
            return memory

    @staticmethod
    def get_hidden_states(layer_output, layer_index=None):
        layer_index = layer_index or [i for i in range(len(layer_output[6]))]
        return layer_output[6][jnp.asarray(layer_index)]

    @staticmethod
    def get_post_attn_embedding(layer_output, layer_index=None):
        layer_index = layer_index or [i for i in range(len(layer_output[7]))]
        return layer_output[7][jnp.asarray(layer_index)]

    @staticmethod
    def get_text_representation(layer_output, layer_index=None):
        layer_index = layer_index or [i for i in range(len(layer_output[8]))]
        return layer_output[8][jnp.asarray(layer_index)]

    @staticmethod
    def get_text_only_hidden_states(layer_output, layer_index=None):
        """Extract text-only hidden states from expert 0 (positions 768-815)."""
        expert_0_states = Analysis.get_expert_0_hidden_states(layer_output, layer_index)
        if expert_0_states is None:
            return None
        # Text tokens are at positions 768-815 (48 tokens)
        # These positions are hardcoded because they're determined by model architecture:
        # - SIGLIP patch size: (14, 14) with 224x224 images = 256 patches per image
        # - 3 images × 256 patches = 768 image tokens
        # - Text tokens start at position 768
        text_start = 256 * 3  # 768
        text_end = text_start + 48  # 816
        return expert_0_states[:, :, text_start:text_end, :]

EXP_DATA_PATH = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "exp_data", "inference_latents")

@dataclasses.dataclass
class InferenceLatentCollectionArgs:
    """Arguments for inference-time latent collection."""
    
    # Model server parameters
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    
    # LIBERO environment parameters
    task_suite_name: str = "libero_object"  # libero_spatial, libero_object, libero_goal
    num_steps_wait: int = 10
    num_trials_per_task: int = 5
    max_steps: int = 280
    
    # Data collection parameters
    save_latents: bool = True
    save_observations: bool = True
    save_actions: bool = True
    
    # Video saving parameters
    save_video: bool = True
    video_out_path: str = "data/inference_latents/videos"
    
    # Sampling parameters
    tasks_to_collect: tuple = (10, 15)  # Range of tasks to collect
    timestep_stride: int = 5  # Collect every Nth timestep to save memory
    replan_steps: int = 5
    
    # Utils
    seed: int = 42
    output_path: str = "data/inference_latents"


def collect_inference_latents(args: InferenceLatentCollectionArgs) -> None:
    """Collect latent states during inference time with rollout step tracking."""
    
    # Set random seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    
    # Create output directory
    output_dir = pathlib.Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create video output directory if saving videos
    if args.save_video:
        video_dir = pathlib.Path(args.video_out_path)
        video_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize model client
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    
    # Start collection
    all_task_data = {}
    total_episodes, total_successes = 0, 0
    
    for task_id in tqdm.tqdm(range(*args.tasks_to_collect)):
        if task_id >= num_tasks_in_suite:
            print(f"Task {task_id} is out of range. Only {num_tasks_in_suite} tasks in the suite.")
            break
            
        # Get task
        task = task_suite.get_task(task_id)
        task_description = task.language
        task_name = task_description.replace(" ", "_")
        
        logging.info(f"Collecting data for task {task_id}: {task_description}")
        
        # Create task-specific video directory if saving videos
        if args.save_video:
            task_video_dir = video_dir / f"{task_name}"
            task_video_dir.mkdir(parents=True, exist_ok=True)
        
        task_data = {
            "task_name": task_name,
            "task_id": task_id,
            "task_description": task_description,
            "episodes": {}
        }
        
        task_episodes, task_successes = 0, 0
        
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task), desc=f"Task {task_id}"):
            print(f"Episode {episode_idx} of {args.num_trials_per_task}")
            action_plan = collections.deque()

            episode_data = {
                "rollout_steps": {},
                "metadata": {
                    "success": False,
                    "total_steps": 0,
                    "episode_idx": episode_idx
                }
            }
            episode_data["metadata"]["total_steps"] = 0
            
            # Initialize environment
            env, _ = _get_libero_env(task, resolution=256, seed=args.seed + episode_idx)
            obs = env.reset()
            
            # Initialize video recording lists
            replay_images = []
            replay_wrist_images = []
            
            t = 0
            while t < args.max_steps + args.num_steps_wait:
                # Wait for objects to stabilize
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step([0.0] * 6 + [-1.0])
                    t += 1
                    continue
                
                # Collect data at specified stride
                # Preprocess images
                original_img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                original_wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(original_img, args.resize_size, args.resize_size)
                )
                wrist_img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(original_wrist_img, args.resize_size, args.resize_size)
                )
                
                # Save images for video recording
                if args.save_video:
                    replay_images.append(original_img)
                    replay_wrist_images.append(original_wrist_img)

                if not action_plan:
                
                    # Prepare observation dict
                    element = {
                        "done": t == args.num_steps_wait,  # reset_sever equivalent
                        "observation/image": img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": np.concatenate([
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        ]),
                        "prompt": task_description,
                        "collect_latents": True,  # Signal to collect latents
                        ###### args from text latent collection
                        "mask_prompt_method": None,
                        "use_TEI_and_TLI": False,
                        "hidden_states_mapping_file": None,
                        "task_hidden_states_mapping": None,
                        "layer_to_intervene": None,
                    }
                    
                    # Query model to get action and latents
                    try:
                        received = client.infer(element)

                        #### Get action ####
                        action_chunk = received["actions"]
                        reset_sever = False
                        assert (
                                len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])
                        
                        #### Save the latent states ####
                        if t % args.timestep_stride == 0:
                            # Store rollout step data
                            rollout_data = {
                                "timestep": t,
                                "action": action_chunk[0].tolist() if len(action_chunk) > 0 else None,
                            }
                            
                            # Store latent states if available
                            if "action_expert_hidden_states" in received:
                                for key, value in received.items():
                                    if "action_expert_hidden_states" in key:
                                        rollout_data[key] = value
                            
                            ### Store necessary information for VLM layer output
                            if "vlm_layer_output" in received:
                                # from openpi.models.gemma import Analysis
                                rollout_data["vlm_layer_output"] = received["vlm_layer_output"]

                            # Store observation if requested
                            if args.save_observations:
                                rollout_data["observation"] = {
                                    "agentview_image": original_img,
                                    "wrist_image": original_wrist_img,
                                    "robot_state": obs["robot0_eef_pos"].tolist(),
                                }
                            
                            episode_data["rollout_steps"][f"step_{t}"] = rollout_data
                            episode_data["metadata"]["total_steps"] += 1
                        
                    except Exception as e:
                        logging.error(f"Error during inference at step {t}: {e}")
                        break

                # Execute action in environment
                action = action_plan.popleft()
                obs, reward, done, info = env.step(action.tolist())
                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1

                # print(f"t: {t}, inference_steps: {inference_steps}")
            
            task_data["episodes"][f"episode_{episode_idx}"] = episode_data
            task_episodes += 1
            total_episodes += 1
            
            # Save video if requested
            if args.save_video and len(replay_images) > 0:
                suffix = "success" if episode_data["metadata"]["success"] else "failure"
                try:
                    imageio.mimwrite(
                        task_video_dir / f"{suffix}_{episode_idx}.mp4",
                        [np.asarray(x) for x in replay_images],
                        fps=30,  # the same as openvla
                    )
                    imageio.mimwrite(
                        task_video_dir / f"{suffix}_wrist_{episode_idx}.mp4",
                        [np.asarray(x) for x in replay_wrist_images],
                        fps=30,  # the same as openvla
                    )
                except Exception as e:
                    logging.error(f"Error saving video for episode {episode_idx}: {e}")
            
            # Log progress
            logging.info(f"Episode {episode_idx + 1}/{args.num_trials_per_task} completed. Success: {episode_data['metadata']['success']}")
        
        # Save task data incrementally
        task_save_path = output_dir / f"task_{task_id}_{task_name}.pkl"
        with open(task_save_path, 'wb') as f:
            pickle.dump(task_data, f)
        
        all_task_data[task_id] = task_data
        
        # Log task results
        task_success_rate = task_successes / task_episodes
        logging.info(f"Task {task_id} success rate: {task_success_rate:.3f} ({task_successes}/{task_episodes})")
        logging.info(f"Overall success rate: {total_successes/total_episodes:.3f} ({total_successes}/{total_episodes})")
    
    # Save overall results
    results = {
        "total_success_rate": total_successes / total_episodes,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "args": dataclasses.asdict(args),
        "task_data": all_task_data
    }
    
    results_path = output_dir / f"collection_results_{datetime.now().strftime('%m-%d-%H-%M')}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4, default=str)
    
    logging.info(f"Collection completed. Results saved to {results_path}")
    logging.info(f"Final success rate: {total_successes/total_episodes:.3f}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    """Convert quaternion to axis-angle representation."""
    import math
    
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Define max steps for different task suites
    TASK_SUITE_MAX_STEPS = {
        "libero_object": 300,
        "libero_goal": 300,
        "libero_spatial": 300,
        "libero_object_ood": 280,
        "libero_goal_ood": 300,
        "libero_spatial_ood": 300,
        "libero_10": 520,
        "libero_90": 400
    }
    
    # Example usage
    suite_name = "libero_object"
    args = InferenceLatentCollectionArgs(
        task_suite_name=suite_name,
        tasks_to_collect=(0, 9),  # Collect tasks 1-4
        num_trials_per_task=1,
        timestep_stride=10,  # Collect every 2nd inference timestep
        max_steps=TASK_SUITE_MAX_STEPS[suite_name],
        save_video=True,  # Enable video saving
        video_out_path="data/inference_latents/videos",  # Video output path
    )
    
    collect_inference_latents(args) 