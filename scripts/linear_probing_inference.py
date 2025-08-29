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

Usage for visual encoder:
python scripts/linear_probing_inference.py      --expert visual      --data_path data/inference_latents/0826_visual_latent     --task_range 0 8     --episode_range 0 1     --num_epochs 200     --batch_size 32        
"""

import argparse
import dataclasses
import logging
import pathlib
import pickle
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5EncoderModel
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


@dataclasses.dataclass
class DataConfig:
    """Configuration for data loading."""
    data_path: str
    task_range: Tuple[int, int] = (0, 9)
    episode_range: Tuple[int, int] = (0, 1)
    rollout_step: int = 10
    expert: str = "vlm"  # "vlm", "action", "text_only"
    layer: int = -1
    feature_type: str = "hidden_states"  # "hidden_states", "post_attn_embedding", etc.
    action_timestep: float = 0.9  # For action expert, which diffusion timestep
    seed: int = 42  # Random seed for reproducible splits


@dataclasses.dataclass
class TrainingConfig:
    """Configuration for training."""
    learning_rate: float = 0.001
    weight_decay: float = 0.01 # L2 weight decay penalty
    num_epochs: int = 100
    batch_size: int = 32
    seed: int = 42
    train_split: float = 0.7  # Fraction of data for training
    eval_split: float = 0.15  # Fraction of data for evaluation
    test_split: float = 0.15  # Fraction of data for testing


class LinearProbe(nn.Module):
    """Linear probe for inference latent analysis."""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.linear(x)


class CosineSimilarityLoss(nn.Module):
    """Cosine similarity loss for embedding regression."""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, predictions, targets):
        """
        Compute cosine similarity loss.
        
        Args:
            predictions: (batch_size, embedding_dim)
            targets: (batch_size, embedding_dim)
            
        Returns:
            loss: scalar tensor
        """
        # Normalize predictions and targets
        predictions_norm = nn.functional.normalize(predictions, p=2, dim=1)
        targets_norm = nn.functional.normalize(targets, p=2, dim=1)
        
        # Compute cosine similarity
        cosine_sim = torch.sum(predictions_norm * targets_norm, dim=1)
        
        # Loss is 1 - cosine_similarity (we want to maximize cosine similarity)
        loss = 1.0 - torch.mean(cosine_sim)
        
        return loss


class InferenceLatentDataset(Dataset):
    """PyTorch Dataset for inference latent data with expert separation."""
    
    def __init__(self, config: DataConfig, split: str = "all"):
        """
        Initialize dataset.
        
        Args:
            config: Data configuration
            split: Dataset split ("all", "train", "eval", "test")
        """
        self.config = config
        self.split = split
        self.features = []
        self.labels = []
        self.timestamps = []
        self.task_descriptions = []
        self.task_to_label_idx = {}  # Map task description to label index
        self._load_data()
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
    def get_timestamp(self, idx):
        """Get timestamp for a specific index."""
        if idx < len(self.timestamps):
            return self.timestamps[idx]
        return None
        
    def _load_data(self):
        """Load and preprocess inference latent data."""
        data_path = pathlib.Path(self.config.data_path)
        task_start, task_end = self.config.task_range
        episode_start, episode_end = self.config.episode_range
        
        logging.info(f"Loading data from {data_path}")
        logging.info(f"Task range: {task_start}-{task_end}, Episode range: {episode_start}-{episode_end}")
        logging.info(f"Rollout step: {self.config.rollout_step}, Expert: {self.config.expert}")
        if self.config.expert != "visual":
            logging.info(f"Layer: {self.config.layer}")
        else:
            logging.info(f"Visual feature type: pre_logits")
        
        for task_id in range(task_start, task_end):
            # Find task file
            task_files = list(data_path.glob(f"task_{task_id}_*.pkl"))
            if not task_files:
                logging.warning(f"No data file found for task {task_id}")
                continue
                
            # task_file = task_files[0]
            for task_file in task_files:
                logging.info(f"Loading task data from {task_file}")
                
                try:
                    with open(task_file, 'rb') as f:
                        task_data = pickle.load(f)
                except Exception as e:
                    logging.error(f"Error loading {task_file}: {e}")
                    continue
                
                task_description = task_data["task_description"]
                
                # Add task description to mapping if not already present
                if task_description not in self.task_to_label_idx:
                    self.task_to_label_idx[task_description] = len(self.task_descriptions)
                    self.task_descriptions.append(task_description)
                
                label_idx = self.task_to_label_idx[task_description]
                
                for episode_idx in range(episode_start, episode_end):
                    episode_key = f"episode_{episode_idx}"
                    if episode_key not in task_data["episodes"]:
                        logging.debug(f"Episode {episode_key} not found in task {task_id}")
                        continue
                        
                    episode_data = task_data["episodes"][episode_key]
                    step_key_list = list(episode_data["rollout_steps"].keys())

                    # Loop through all rollout steps by default
                    if self.config.rollout_step is None:
                        logging.info(f"Rollout Step key: {step_key_list}")
                        for step_key in step_key_list:
                            if step_key not in episode_data["rollout_steps"]:
                                logging.debug(f"Step {step_key} not found in episode {episode_key}")
                                continue

                            step_data = episode_data["rollout_steps"][step_key]

                            # Overwrite label_id (task description) if it is different in the step data
                            if "prompt" in step_data:
                                if step_data["prompt"] not in self.task_to_label_idx:
                                    self.task_to_label_idx[step_data["prompt"]] = len(self.task_descriptions)
                                    self.task_descriptions.append(step_data["prompt"])
                                
                                label_idx = self.task_to_label_idx[step_data["prompt"]]
                                logging.info(f"Overwriting label_idx from {task_description} to {step_data['prompt']}")

                            # Extract features based on expert type
                            features = self._extract_features(step_data)
                            if features is not None:
                                if isinstance(features, list):
                                    # For visual expert, each camera view becomes a separate datapoint
                                    self.features.extend(features)
                                    self.labels.extend([label_idx] * len(features))
                                else:
                                    self.features.append(features)
                                    self.labels.append(label_idx)
                            if "timestep" in step_data:
                                self.timestamps.extend([step_data["timestep"]] * len(features))
                    else:
                        step_key = step_key_list[self.config.rollout_step]
                        logging.info(f"Rollout Step key: {step_key}")
                        if step_key not in episode_data["rollout_steps"]:
                            logging.debug(f"Step {step_key} not found in episode {episode_key}")
                            continue

                        step_data = episode_data["rollout_steps"][step_key]

                        # Extract features based on expert type
                        features = self._extract_features(step_data)
                        if features is not None:
                            if isinstance(features, list):
                                # For visual expert, each camera view becomes a separate datapoint
                                self.features.extend(features)
                                self.labels.extend([label_idx] * len(features))
                            else:
                                self.features.append(features)
                                self.labels.append(label_idx)
                        if "timestep" in step_data:
                            self.timestamps.extend([step_data["timestep"]] * len(features))
        
        logging.info(f"Loaded {len(self.features)} samples with {len(self.task_descriptions)} unique tasks")

        
        if self.config.expert == "visual":
            # Count samples per task for visual expert
            task_counts = {}
            for label in self.labels:
                task_desc = self.task_descriptions[label]
                task_counts[task_desc] = task_counts.get(task_desc, 0) + 1
            logging.info(f"Visual expert samples per task:")
            for task_desc, count in task_counts.items():
                logging.info(f"  {task_desc}: {count} samples")
        
        if len(self.features) == 0:
            raise ValueError("No valid features found. Check data path and configuration.")
        
        # Note: Splits are now handled in create_dataset_splits function
        # to avoid loading data multiple times
        
    def _extract_features(self, step_data: Dict) -> Optional[torch.Tensor]:
        # Extract and return features from step data according to the current expert type and configuration.
        try:
            if self.config.expert == "vlm":
                if "vlm_layer_output" not in step_data:
                    logging.debug("vlm_layer_output not found in step data")
                    return None
                expert_data = step_data["vlm_layer_output"]

                if self.config.feature_type not in expert_data:
                    logging.debug(f"{self.config.feature_type} not found in expert data")
                    return None

                hidden_states = expert_data[self.config.feature_type]
                # Convert numpy array to torch tensor
                if isinstance(hidden_states, np.ndarray):
                    hidden_states = torch.from_numpy(hidden_states).float()
                # Extract specific layer
                if len(hidden_states.shape) == 4:  # (layers, batch, seq_len, hidden_dim)
                    layer_features = hidden_states[self.config.layer, 0]  # Remove batch dimension, which is 1 in test time
                else:
                    layer_features = hidden_states[self.config.layer]
                # Mean pool over sequence dimension, shape (hidden_dim)
                features = torch.mean(layer_features, dim=0)
                return features

            elif self.config.expert == "action":
                # For action expert, collect all action_expert keys as a list of features
                action_expert_keys = [k for k in step_data.keys() if "action_expert" in k]
                if not action_expert_keys:
                    logging.debug("No action_expert keys found in step data")
                    return None
                features_list = []
                for key in action_expert_keys:
                    expert_data = step_data[key]
                    if self.config.feature_type not in expert_data:
                        logging.debug(f"{self.config.feature_type} not found in expert data for {key}")
                        continue
                    hidden_states = expert_data[self.config.feature_type]
                    if isinstance(hidden_states, np.ndarray):
                        hidden_states = torch.from_numpy(hidden_states).float()
                    if len(hidden_states.shape) == 4:
                        layer_features = hidden_states[self.config.layer, 0]
                    else:
                        layer_features = hidden_states[self.config.layer]
                    features = torch.mean(layer_features, dim=0)
                    features_list.append(features)
                if not features_list:
                    return None
                return features_list

            elif self.config.expert == "text_only":
                if "vlm_layer_output" not in step_data:
                    logging.debug("vlm_layer_output not found in step data")
                    return None
                expert_data = step_data["vlm_layer_output"]

                if self.config.feature_type not in expert_data:
                    logging.debug(f"{self.config.feature_type} not found in expert data")
                    return None

                hidden_states = expert_data[self.config.feature_type]
                if isinstance(hidden_states, np.ndarray):
                    hidden_states = torch.from_numpy(hidden_states).float()
                if len(hidden_states.shape) == 4:
                    layer_features = hidden_states[self.config.layer, 0]
                else:
                    layer_features = hidden_states[self.config.layer]
                # For text_only, extract positions 768-815
                text_start = 256 * 3  # 768
                text_end = text_start + 48  # 816
                layer_features = layer_features[text_start:text_end]
                features = torch.mean(layer_features, dim=0)
                return features

            elif self.config.expert == "visual":
                if "vision_latents" not in step_data:
                    logging.debug("vision_latents not found in step data")
                    return None
                
                vision_data = step_data["vision_latents"]
                features_list = []
                
                # Extract pre_logits from each RGB camera view
                for camera_key in ["base_0_rgb", "left_wrist_0_rgb"]: # "right_wrist_0_rgb"]:
                    if camera_key not in vision_data:
                        logging.debug(f"{camera_key} not found in vision_latents")
                        continue
                    
                    camera_data = vision_data[camera_key]
                    if "pre_logits" not in camera_data:
                        logging.debug(f"pre_logits not found in {camera_key}")
                        continue
                    
                    pre_logits = camera_data["pre_logits"]
                    if isinstance(pre_logits, np.ndarray):
                        pre_logits = torch.from_numpy(pre_logits).float()
                    
                    # pre_logits shape is (256, 1152), mean pool over spatial dimension
                    features = torch.mean(pre_logits, dim=0)  # Shape: (1152,)
                    features_list.append(features)
                
                if not features_list:
                    logging.debug("No valid visual features found")
                    return None
                
                # Return list of features, one for each camera view
                return features_list

            else:
                raise ValueError(f"Unknown expert type: {self.config.expert}")

        except Exception as e:
            logging.error(f"Error extracting features: {e}")
            return None
            
        except Exception as e:
            logging.error(f"Error extracting features: {e}")
            return None
    
    def get_task_descriptions(self) -> List[str]:
        """Return list of task descriptions."""
        return self.task_descriptions
        
    def get_data_info(self) -> Dict:
        """Return information about loaded data."""
        info = {
            "num_samples": len(self.features),
            "num_tasks": len(self.task_descriptions),
            "task_descriptions": self.task_descriptions,
            "expert_type": self.config.expert,
            "layer": self.config.layer,
            "rollout_step": self.config.rollout_step,
            "feature_type": self.config.feature_type,
            "action_timestep": self.config.action_timestep if self.config.expert == "action" else None,
            "feature_dim": self.features[0].shape[0] if self.features else None
        }
        
        # Add visual-specific information
        if self.config.expert == "visual":
            info["camera_views"] = ["base_0_rgb", "left_wrist_0_rgb"] # , "right_wrist_0_rgb"]
            info["visual_feature_type"] = "pre_logits"
        
        return info


def create_t5_labels(task_descriptions: List[str], 
                    model_name: str = "t5-small",
                    low_dim_projection: bool = True,
                    target_dim: int = 64) -> torch.Tensor:
    """
    Use T5 to encode task descriptions into embeddings as labels.
    
    Args:
        task_descriptions: List of task description strings
        model_name: T5 model variant to use
        low_dim_projection: Whether to apply low-dimensional projection
        target_dim: Target dimension for low-dimensional projection
    
    Returns:
        T5 embeddings tensor of shape (num_tasks, embedding_dim) or (num_tasks, target_dim)
    """
    logging.info(f"Creating T5 labels using {model_name}")
    
    # Load T5 model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5EncoderModel.from_pretrained(model_name)
    
    # Set model to evaluation mode
    model.eval()
    
    embeddings = []
    with torch.no_grad():
        for i, description in enumerate(task_descriptions):
            logging.debug(f"Encoding task {i}: {description}")
            
            # Tokenize and encode
            inputs = tokenizer(description, return_tensors="pt", 
                             max_length=512, truncation=True, padding=True)
            outputs = model(**inputs)
            
            # Mean pool over sequence dimension
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
            embeddings.append(embedding)
    
    embeddings_tensor = torch.stack(embeddings, dim=0)
    original_shape = embeddings_tensor.shape
    logging.info(f"Created T5 embeddings with original shape: {original_shape}")
    
    if low_dim_projection and embeddings_tensor.shape[1] > target_dim:
        # Center the embeddings
        embeddings_centered = embeddings_tensor - embeddings_tensor.mean(dim=0, keepdim=True)
        
        # Compute SVD: embeddings_centered = U * S * V^T
        U, S, Vt = torch.linalg.svd(embeddings_centered, full_matrices=False)
        
        # Select top-k singular values and vectors
        k = min(target_dim, len(S))
        U_k = U[:, :k]
        S_k = S[:k]
        Vt_k = Vt[:k, :]
        
        # Compute orthonormal basis B (shape: [512, k])
        B = Vt_k.T  # Transpose to get shape [512, k]
        
        # Compute new coordinates C (shape: [num_labels, k])
        # C = embeddings_centered * B
        C = torch.matmul(embeddings_centered, B)
        
        # Verify the reconstruction: span = C * B^T
        span_reconstructed = torch.matmul(C, B.T)
        
        # Compute reconstruction error
        reconstruction_error = torch.norm(embeddings_centered - span_reconstructed, p='fro')
        original_norm = torch.norm(embeddings_centered, p='fro')
        relative_error = reconstruction_error / original_norm
        
        logging.info(f"Low-dimensional projection applied:")
        logging.info(f"  Original shape: {original_shape}")
        logging.info(f"  Target dimension: {target_dim}")
        logging.info(f"  Orthonormal basis B shape: {B.shape}")
        logging.info(f"  New coordinates C shape: {C.shape}")
        logging.info(f"  Reconstruction error: {reconstruction_error:.6f}")
        logging.info(f"  Relative error: {relative_error:.6f}")
        logging.info(f"  Preserved variance ratio: {1.0 - relative_error:.6f}")
        
        # Return the low-dimensional coordinates
        return C
    else:
        logging.info(f"No low-dimensional projection applied, returning original shape: {original_shape}")
        return embeddings_tensor


def create_dataset_splits(config: DataConfig, training_config: TrainingConfig) -> Tuple[InferenceLatentDataset, InferenceLatentDataset, InferenceLatentDataset]:
    """Create train, eval, and test dataset splits."""
    logging.info("Creating dataset splits...")
    
    # Load data only once
    full_dataset = InferenceLatentDataset(config, split="all")
    
    # Set random seed for reproducible splits
    random.seed(config.seed)
    
    # Create indices for shuffling
    indices = list(range(len(full_dataset.features)))
    random.shuffle(indices)
    
    # Calculate split boundaries based on training config
    total_samples = len(indices)
    train_end = int(total_samples * training_config.train_split)
    eval_end = train_end + int(total_samples * training_config.eval_split)
    
    # Create train split
    train_indices = indices[:train_end]
    train_dataset = InferenceLatentDataset.__new__(InferenceLatentDataset)
    train_dataset.config = config
    train_dataset.split = "train"
    train_dataset.features = [full_dataset.features[i] for i in train_indices]
    train_dataset.labels = [full_dataset.labels[i] for i in train_indices]
    train_dataset.task_descriptions = full_dataset.task_descriptions
    train_dataset.task_to_label_idx = full_dataset.task_to_label_idx
    
    # Create eval split
    eval_indices = indices[train_end:eval_end]
    eval_dataset = InferenceLatentDataset.__new__(InferenceLatentDataset)
    eval_dataset.config = config
    eval_dataset.split = "eval"
    eval_dataset.features = [full_dataset.features[i] for i in eval_indices]
    eval_dataset.labels = [full_dataset.labels[i] for i in eval_indices]
    eval_dataset.task_descriptions = full_dataset.task_descriptions
    eval_dataset.task_to_label_idx = full_dataset.task_to_label_idx
    
    # Create test split
    test_indices = indices[eval_end:]
    test_dataset = InferenceLatentDataset.__new__(InferenceLatentDataset)
    test_dataset.config = config
    test_dataset.split = "test"
    test_dataset.features = [full_dataset.features[i] for i in test_indices]
    test_dataset.labels = [full_dataset.labels[i] for i in test_indices]
    test_dataset.task_descriptions = full_dataset.task_descriptions
    test_dataset.task_to_label_idx = full_dataset.task_to_label_idx
    
    logging.info(f"Dataset splits created:")
    logging.info(f"  Train: {len(train_dataset)} samples")
    logging.info(f"  Eval: {len(eval_dataset)} samples")
    logging.info(f"  Test: {len(test_dataset)} samples")
    
    return train_dataset, eval_dataset, test_dataset, full_dataset


def evaluate_model(model: nn.Module, 
                  dataloader: DataLoader,
                  criterion: nn.Module,
                  device: torch.device,
                  t5_labels: torch.Tensor) -> Dict[str, float]:
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0.0
    total_cosine_sim = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, (features, label_indices) in enumerate(dataloader):
            # Move data to device
            features = features.to(device)
            label_indices = label_indices.to(device)
            
            # Get T5 labels for this batch
            batch_labels = t5_labels[label_indices.cpu()].to(device)
            
            # Forward pass
            predictions = model(features)
            
            # Compute loss
            loss = criterion(predictions, batch_labels)
            
            # Compute cosine similarity
            predictions_norm = nn.functional.normalize(predictions, p=2, dim=1)
            targets_norm = nn.functional.normalize(batch_labels, p=2, dim=1)
            cosine_sim = torch.mean(torch.sum(predictions_norm * targets_norm, dim=1))
            
            total_loss += loss.item()
            total_cosine_sim += cosine_sim.item()
            num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'cosine_similarity': total_cosine_sim / num_batches
    }


def train_epoch(model: nn.Module, 
                dataloader: DataLoader,
                optimizer: optim.Optimizer,
                criterion: nn.Module,
                device: torch.device,
                t5_labels: torch.Tensor) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_cosine_sim = 0.0
    num_batches = 0
    
    for batch_idx, (features, label_indices) in enumerate(dataloader):
        # Move data to device
        features = features.to(device)
        label_indices = label_indices.to(device)
        
        # Get T5 labels for this batch
        batch_labels = t5_labels[label_indices.cpu()].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(features)
        
        # Compute loss
        loss = criterion(predictions, batch_labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Compute cosine similarity for monitoring
        with torch.no_grad():
            predictions_norm = nn.functional.normalize(predictions, p=2, dim=1)
            targets_norm = nn.functional.normalize(batch_labels, p=2, dim=1)
            cosine_sim = torch.mean(torch.sum(predictions_norm * targets_norm, dim=1))
        
        total_loss += loss.item()
        total_cosine_sim += cosine_sim.item()
        num_batches += 1
        
        if batch_idx % 10 == 0:
            logging.debug(f"Batch {batch_idx}: Loss={loss.item():.4f}, Cosine Sim={cosine_sim.item():.4f}")
    
    return {
        'loss': total_loss / num_batches,
        'cosine_similarity': total_cosine_sim / num_batches
    }


def train(train_dataset: InferenceLatentDataset,
          eval_dataset: InferenceLatentDataset,
          t5_labels: torch.Tensor,
          config: TrainingConfig,
          device: torch.device) -> Tuple[nn.Module, Dict]:
    """Main training function with evaluation."""
    logging.info(f"Starting training on device: {device}")
    
    # Create DataLoaders with shuffling for training
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Initialize model
    feature_dim = train_dataset.features[0].shape[0]
    t5_dim = t5_labels.shape[1]
    model = LinearProbe(input_dim=feature_dim, output_dim=t5_dim).to(device)
    
    # Initialize optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = CosineSimilarityLoss()
    
    # Training loop
    best_eval_loss = float('inf')
    training_history = []
    eval_history = []
    
    logging.info(f"Training for {config.num_epochs} epochs")
    logging.info(f"Model: input_dim={feature_dim}, output_dim={t5_dim}")
    logging.info(f"Train dataset: {len(train_dataset)} samples")
    logging.info(f"Eval dataset: {len(eval_dataset)} samples")
    logging.info(f"Tasks: {len(train_dataset.task_descriptions)}")
    
    for epoch in range(config.num_epochs):
        # Training
        train_metrics = train_epoch(model, train_dataloader, optimizer, criterion, device, t5_labels)
        
        # Evaluation
        eval_metrics = evaluate_model(model, eval_dataloader, criterion, device, t5_labels)
        
        training_history.append(train_metrics)
        eval_history.append(eval_metrics)
        
        # Log every epoch
        logging.info(f"Epoch {epoch:3d}: "
                    f"Train Loss={train_metrics['loss']:.4f}, "
                    f"Train CosSim={train_metrics['cosine_similarity']:.4f} | "
                    f"Eval Loss={eval_metrics['loss']:.4f}, "
                    f"Eval CosSim={eval_metrics['cosine_similarity']:.4f}")
        
        # Early stopping based on eval loss
        if eval_metrics['loss'] < best_eval_loss:
            best_eval_loss = eval_metrics['loss']
            best_epoch = epoch
    
    logging.info(f"Training completed. Best eval loss: {best_eval_loss:.4f} at epoch {best_epoch}")
    
    return model, {
        'training_history': training_history, 
        'eval_history': eval_history,
        'best_eval_loss': best_eval_loss,
        'best_epoch': best_epoch
    }


def compute_accuracy(predictions: torch.Tensor, 
                    targets: torch.Tensor) -> float:
    """Compute classification accuracy using cosine similarity."""
    with torch.no_grad():
        # Normalize predictions and targets
        predictions_norm = nn.functional.normalize(predictions, p=2, dim=1)
        targets_norm = nn.functional.normalize(targets, p=2, dim=1)
        
        # Compute cosine similarities between all pairs
        similarities = torch.mm(predictions_norm, targets_norm.T)
        
        # For each prediction, find the target with maximum similarity
        max_indices = torch.argmax(similarities, dim=1)
        
        # Check if the predicted target matches the true target
        correct_predictions = 0
        for i in range(predictions.shape[0]):
            predicted_target_idx = max_indices[i]
            true_target = targets[i]
            predicted_target = targets[predicted_target_idx]
            
            # Check if predicted target matches true target (allowing for small numerical differences)
            if torch.allclose(true_target, predicted_target, atol=1e-6):
                correct_predictions += 1
        
        accuracy = correct_predictions / predictions.shape[0]
        return accuracy


def compute_cosine_similarity(predictions: torch.Tensor, 
                            targets: torch.Tensor) -> float:
    """Compute average cosine similarity between predictions and targets."""
    with torch.no_grad():
        # Normalize predictions and targets
        predictions_norm = nn.functional.normalize(predictions, p=2, dim=1)
        targets_norm = nn.functional.normalize(targets, p=2, dim=1)
        
        # Compute cosine similarities
        similarities = torch.sum(predictions_norm * targets_norm, dim=1)
        return torch.mean(similarities).item()


def analyze_results(model: nn.Module,
                   test_dataset: InferenceLatentDataset,
                   t5_labels: torch.Tensor,
                   config: DataConfig,
                   device: torch.device) -> Dict:
    """Analyze and report results on test set."""
    logging.info("Analyzing results on test set...")
    
    model.eval()
    
    # Create test dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Collect all predictions and targets
    all_predictions = []
    all_targets = []
    all_labels = []
    
    with torch.no_grad():
        for features, label_indices in test_dataloader:
            features = features.to(device)
            label_indices = label_indices.to(device)
            
            # Get predictions
            predictions = model(features)
            
            # Get T5 targets
            batch_labels = t5_labels[label_indices.cpu()].to(device)
            
            all_predictions.append(predictions.cpu())
            all_targets.append(batch_labels.cpu())
            all_labels.append(label_indices.cpu())
    
    # Concatenate all batches
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    # Compute metrics
    accuracy = compute_accuracy(predictions, targets)
    cosine_sim = compute_cosine_similarity(predictions, targets)
    
    # Per-task analysis
    task_accuracies = {}
    task_sample_counts = {}
    for task_idx, task_desc in enumerate(test_dataset.task_descriptions):
        task_mask = (labels == task_idx)
        if task_mask.sum() > 0:
            task_predictions = predictions[task_mask]
            task_targets = targets[task_mask]
            task_accuracy = compute_accuracy(task_predictions, task_targets)
            task_accuracies[task_desc] = task_accuracy
            task_sample_counts[task_desc] = task_mask.sum().item()

    results = {
        'accuracy': accuracy,
        'cosine_similarity': cosine_sim,
        'config': dataclasses.asdict(config),
        'num_samples': len(test_dataset),
        'num_tasks': len(test_dataset.task_descriptions),
        'task_descriptions': test_dataset.task_descriptions,
        'task_accuracies': task_accuracies,
        'predictions': predictions,
        'targets': targets,
        'feature_dim': predictions.shape[1],
        't5_dim': targets.shape[1]
    }

    # Add detailed analysis to results
    results['task_sample_counts'] = task_sample_counts
    
    logging.info(f"Test Results:")
    logging.info(f"  Accuracy: {accuracy:.4f}")
    logging.info(f"  Cosine Similarity: {cosine_sim:.4f}")
    logging.info(f"  Number of samples: {len(test_dataset)}")
    
    return results


def analyze_results_timesteps(model: nn.Module,
                             full_dataset: InferenceLatentDataset,
                             t5_labels: torch.Tensor,
                             config: DataConfig,
                             device: torch.device,
                             plot_style: str = "both") -> Dict:
    """
    Analyze results on full dataset with timestamp-based analysis.
    
    This function:
    1. Calculates predicted features for the full dataset
    2. Computes distance between predicted and ground truth features
    3. Groups by timestamps and calculates mean distance per timestamp
    4. Creates plots of mean distance vs timestamp
    
    Args:
        plot_style: "both" for mean plots with error bars, "scatter" for individual data points colored by prompt
    """
    logging.info("Analyzing results on full dataset with timestep analysis...")
    
    model.eval()
    
    # Create dataloader for full dataset
    full_dataloader = DataLoader(full_dataset, batch_size=32, shuffle=False)
    
    # Collect all predictions, targets, labels, and timestamps
    all_predictions = []
    all_targets = []
    all_labels = []
    all_timestamps = []
    
    with torch.no_grad():
        for features, label_indices in full_dataloader:
            features = features.to(device)
            label_indices = label_indices.to(device)
            
            # Get predictions
            predictions = model(features)
            
            # Get T5 targets
            batch_labels = t5_labels[label_indices.cpu()].to(device)
            
            all_predictions.append(predictions.cpu())
            all_targets.append(batch_labels.cpu())
            all_labels.append(label_indices.cpu())
    
    # Concatenate all batches
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    # Get timestamps from dataset
    timestamps = full_dataset.timestamps
    
    # Handle timestamp alignment with predictions
    if len(timestamps) != len(predictions):
        raise ValueError(f"Number of timestamps ({len(timestamps)}) doesn't match number of predictions ({len(predictions)})")
        # logging.warning(f"Number of timestamps ({len(timestamps)}) doesn't match number of predictions ({len(predictions)})")
        
        # if len(timestamps) == 0:
        #     # No timestamps available, create dummy timestamps
        #     timestamps = list(range(len(predictions)))
        #     logging.info("No timestamps found in dataset, using dummy timestamps")
        # elif len(timestamps) < len(predictions):
        #     # This can happen with visual expert where multiple camera views create multiple features per timestamp
        #     # We need to expand timestamps to match the number of predictions
        #     expanded_timestamps = []
            
        #     # Calculate how many predictions per timestamp
        #     predictions_per_timestamp = len(predictions) // len(timestamps)
        #     remainder = len(predictions) % len(timestamps)
            
        #     logging.info(f"Expanding timestamps: {len(timestamps)} timestamps -> {len(predictions)} predictions")
        #     logging.info(f"Predictions per timestamp: {predictions_per_timestamp} (remainder: {remainder})")
            
        #     for i, timestamp in enumerate(timestamps):
        #         # Calculate how many predictions this timestamp should have
        #         if i < remainder:
        #             num_predictions = predictions_per_timestamp + 1
        #         else:
        #             num_predictions = predictions_per_timestamp
        
        #         # Add this timestamp the appropriate number of times
        #         expanded_timestamps.extend([timestamp] * num_predictions)
            
        #     timestamps = expanded_timestamps
        #     logging.info(f"Expanded timestamps to match {len(predictions)} predictions")
            
        #     # Verify the expansion worked correctly
        #     if len(timestamps) != len(predictions):
        #         logging.error(f"Timestamp expansion failed: {len(timestamps)} timestamps vs {len(predictions)} predictions")
        #         # Fallback: repeat the last timestamp for remaining predictions
        #         while len(timestamps) < len(predictions):
        #             timestamps.append(timestamps[-1])
        #         logging.info(f"Applied fallback expansion: {len(timestamps)} timestamps")
        # else:
        #     # Truncate timestamps
        #     timestamps = timestamps[:len(predictions)]
        #     logging.info("Truncated timestamps to match number of predictions")
    
    # Verify final alignment
    if len(timestamps) != len(predictions):
        logging.error(f"Final timestamp alignment failed: {len(timestamps)} timestamps vs {len(predictions)} predictions")
        raise ValueError("Timestamp alignment failed")
    
    logging.info(f"Final alignment: {len(timestamps)} timestamps, {len(predictions)} predictions")
    logging.info(f"Timestamp sample: {timestamps[:10]}...")  # Show first 10 timestamps
    
    # Calculate distances between predicted and ground truth features
    distances = torch.norm(predictions - targets, p=2, dim=1)  # L2 distance
    distances_np = distances.numpy()
    
    # Calculate cosine similarities between predicted and ground truth features
    predictions_norm = nn.functional.normalize(predictions, p=2, dim=1)
    targets_norm = nn.functional.normalize(targets, p=2, dim=1)
    cosine_similarities = torch.sum(predictions_norm * targets_norm, dim=1)
    cosine_similarities_np = cosine_similarities.numpy()
    
    # Calculate accuracy for each sample using the same method as analyze_results
    # This compares predictions against the unique set of T5 embeddings
    accuracies = []
    
    # Get unique T5 embeddings (one per task)
    unique_t5_embeddings = t5_labels  # Shape: (num_tasks, embedding_dim)
    
    for i in range(len(predictions)):
        pred_embedding = predictions[i]
        true_label_idx = labels[i].item()
        
        # Compare prediction against all unique T5 embeddings
        similarities = torch.mm(pred_embedding.unsqueeze(0), unique_t5_embeddings.T)
        predicted_label_idx = torch.argmax(similarities).item()
        
        # Check if predicted label matches true label
        is_correct = (predicted_label_idx == true_label_idx)
        accuracies.append(1.0 if is_correct else 0.0)
    
    accuracies_np = np.array(accuracies)
    
    # Log accuracy distribution for debugging
    unique_accuracies, counts = np.unique(accuracies_np, return_counts=True)
    logging.info(f"Accuracy distribution: {dict(zip(unique_accuracies, counts))}")
    logging.info(f"Total samples: {len(accuracies_np)}")
    logging.info(f"Correct predictions: {np.sum(accuracies_np)}")
    logging.info(f"Overall accuracy: {np.mean(accuracies_np):.4f}")
    
    # Group distances, cosine similarities, and accuracies by timestamps
    timestamp_to_distances = defaultdict(list)
    timestamp_to_cosine_sims = defaultdict(list)
    timestamp_to_accuracies = defaultdict(list)
    
    # Verify that we're grouping correctly
    total_grouped = 0
    for i, timestamp in enumerate(timestamps):
        timestamp_to_distances[timestamp].append(distances_np[i])
        timestamp_to_cosine_sims[timestamp].append(cosine_similarities_np[i])
        timestamp_to_accuracies[timestamp].append(accuracies_np[i])
        total_grouped += 1
    
    # Verify grouping integrity
    if total_grouped != len(predictions):
        logging.error(f"Grouping integrity check failed: {total_grouped} grouped vs {len(predictions)} predictions")
        raise ValueError("Grouping integrity check failed")
    
    logging.info(f"Grouped metrics by {len(timestamp_to_distances)} unique timestamps")
    logging.info(f"Timestamp range: {min(timestamps)} to {max(timestamps)}")
    logging.info(f"Total samples grouped: {total_grouped}")
    
    # Log some sample groupings for verification
    sample_timestamps = sorted(timestamp_to_distances.keys())[:5]  # First 5 timestamps
    for ts in sample_timestamps:
        logging.info(f"Timestamp {ts}: {len(timestamp_to_distances[ts])} samples")
    
    # Generate filename for plots
    output_path = pathlib.Path("results/linear_probing")
    output_path.mkdir(parents=True, exist_ok=True)
    
    import datetime
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if config.expert == "visual":
        plot_filename = f"timestep_analysis_{config.expert}_prelogits_step{config.rollout_step}_{time_str}.png"
    else:
        plot_filename = f"timestep_analysis_{config.expert}_layer{config.layer}_step{config.rollout_step}_{time_str}.png"
        if config.expert == "action":
            plot_filename = plot_filename.replace(f"_step{config.rollout_step}", f"_step{config.rollout_step}_t{config.action_timestep}")
    
    # Initialize plot paths
    plot_path = None
    scatter_plot_path = None
    bar_plot_path = None
    
    # Calculate mean and variance for each timestamp
    timestamp_stats = {}
    for timestamp, dist_list in timestamp_to_distances.items():
        dist_array = np.array(dist_list)
        cosine_array = np.array(timestamp_to_cosine_sims[timestamp])
        accuracy_array = np.array(timestamp_to_accuracies[timestamp])
        timestamp_stats[timestamp] = {
            'mean': np.mean(dist_array),
            'var': np.var(dist_array),
            'std': np.std(dist_array),
            'count': len(dist_array),
            'cosine_mean': np.mean(cosine_array),
            'cosine_var': np.var(cosine_array),
            'cosine_std': np.std(cosine_array),
            'accuracy_mean': np.mean(accuracy_array),
            'accuracy_std': np.std(accuracy_array)
        }
    
    # Sort timestamps for plotting
    sorted_timestamps = sorted(timestamp_stats.keys())
    
    # Verify that all lists have the same length
    mean_distances = [timestamp_stats[ts]['mean'] for ts in sorted_timestamps]
    std_distances = [timestamp_stats[ts]['std'] for ts in sorted_timestamps]
    counts = [timestamp_stats[ts]['count'] for ts in sorted_timestamps]
    mean_cosine_sims = [timestamp_stats[ts]['cosine_mean'] for ts in sorted_timestamps]
    std_cosine_sims = [timestamp_stats[ts]['cosine_std'] for ts in sorted_timestamps]
    mean_accuracies = [timestamp_stats[ts]['accuracy_mean'] for ts in sorted_timestamps]
    std_accuracies = [timestamp_stats[ts]['accuracy_std'] for ts in sorted_timestamps]
    
    # Verify plotting data integrity
    list_lengths = [len(mean_distances), len(std_distances), len(counts), 
                   len(mean_cosine_sims), len(std_cosine_sims), 
                   len(mean_accuracies), len(std_accuracies)]
    if len(set(list_lengths)) != 1:
        logging.error(f"Plotting data length mismatch: {list_lengths}")
        raise ValueError("Plotting data length mismatch")
    
    logging.info(f"Plotting data verified: {len(sorted_timestamps)} timestamps, all lists have length {list_lengths[0]}")
    logging.info(f"Sorted timestamps sample: {sorted_timestamps[:5]}...")  # Show first 5 sorted timestamps
    
    # Create the plot with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 16))
    
    # Plot 1: Mean distance with error bars
    ax1.errorbar(sorted_timestamps, mean_distances, yerr=std_distances, 
                fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=6, color='red', label='L2 Distance')
    
    ax1.set_xlabel('Timestamp', fontsize=14)
    ax1.set_ylabel('Mean Distance (L2)', fontsize=14)
    ax1.set_title(f'Mean Distance vs Timestamp - {config.expert} Expert', fontsize=16)
    ax1.grid(True, alpha=0.3)
    
    # Add sample count annotations for distance plot
    for i, (ts, count) in enumerate(zip(sorted_timestamps, counts)):
        ax1.annotate(f'n={count}', (ts, mean_distances[i]), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=8, alpha=0.7)
    
    # Plot 2: Mean cosine similarity with error bars
    ax2.errorbar(sorted_timestamps, mean_cosine_sims, yerr=std_cosine_sims, 
                fmt='s-', capsize=5, capthick=2, linewidth=2, markersize=6, color='blue', label='Cosine Similarity')
    
    ax2.set_xlabel('Timestamp', fontsize=14)
    ax2.set_ylabel('Mean Cosine Similarity', fontsize=14)
    ax2.set_title(f'Mean Cosine Similarity vs Timestamp - {config.expert} Expert', fontsize=16)
    ax2.grid(True, alpha=0.3)
    
    # Add sample count annotations for cosine similarity plot
    for i, (ts, count) in enumerate(zip(sorted_timestamps, counts)):
        ax2.annotate(f'n={count}', (ts, mean_cosine_sims[i]), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=8, alpha=0.7)
    
    # Plot 3: Mean accuracy with error bars
    ax3.errorbar(sorted_timestamps, mean_accuracies, yerr=std_accuracies, 
                fmt='^-', capsize=5, capthick=2, linewidth=2, markersize=6, color='green', label='Accuracy')
    
    ax3.set_xlabel('Timestamp', fontsize=14)
    ax3.set_ylabel('Mean Accuracy', fontsize=14)
    ax3.set_title(f'Mean Accuracy vs Timestamp - {config.expert} Expert', fontsize=16)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)  # Accuracy is between 0 and 1
    
    # Add sample count annotations for accuracy plot
    for i, (ts, count) in enumerate(zip(sorted_timestamps, counts)):
        ax3.annotate(f'n={count}', (ts, mean_accuracies[i]), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    
    # Create scatter plot if requested
    if plot_style in ["scatter", "both"]:
        # Create scatter plot with different colors for different prompts
        fig_scatter, (ax_scatter1, ax_scatter2, ax_scatter3) = plt.subplots(3, 1, figsize=(14, 18))
        
        # Get unique prompts and create color mapping
        unique_prompts = list(set(full_dataset.task_descriptions))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_prompts)))
        prompt_to_color = {prompt: colors[i] for i, prompt in enumerate(unique_prompts)}
        
        # Create scatter plots for each prompt
        for prompt_idx, prompt in enumerate(unique_prompts):
            # Find indices for this prompt
            prompt_indices = [i for i, label in enumerate(labels.numpy()) if full_dataset.task_descriptions[label] == prompt]
            
            if prompt_indices:
                prompt_timestamps = [timestamps[i] for i in prompt_indices]
                prompt_distances = [distances_np[i] for i in prompt_indices]
                prompt_cosine_sims = [cosine_similarities_np[i] for i in prompt_indices]
                prompt_accuracies = [accuracies_np[i] for i in prompt_indices]
                
                # Plot distance scatter
                ax_scatter1.scatter(prompt_timestamps, prompt_distances, 
                                  c=[prompt_to_color[prompt]], label=prompt[:50] + "..." if len(prompt) > 50 else prompt,
                                  alpha=0.7, s=30)
                
                # Plot cosine similarity scatter
                ax_scatter2.scatter(prompt_timestamps, prompt_cosine_sims, 
                                  c=[prompt_to_color[prompt]], label=prompt[:50] + "..." if len(prompt) > 50 else prompt,
                                  alpha=0.7, s=30)
                
                # Plot accuracy scatter (keep individual points for other metrics)
                ax_scatter3.scatter(prompt_timestamps, prompt_accuracies, 
                                  c=[prompt_to_color[prompt]], label=prompt[:50] + "..." if len(prompt) > 50 else prompt,
                                  alpha=0.7, s=30)
        
        # Configure distance scatter plot
        ax_scatter1.set_xlabel('Timestamp', fontsize=14)
        ax_scatter1.set_ylabel('L2 Distance', fontsize=14)
        ax_scatter1.set_title(f'Individual L2 Distances vs Timestamp - {config.expert} Expert', fontsize=16)
        ax_scatter1.grid(True, alpha=0.3)
        ax_scatter1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # Configure cosine similarity scatter plot
        ax_scatter2.set_xlabel('Timestamp', fontsize=14)
        ax_scatter2.set_ylabel('Cosine Similarity', fontsize=14)
        ax_scatter2.set_title(f'Individual Cosine Similarities vs Timestamp - {config.expert} Expert', fontsize=16)
        ax_scatter2.grid(True, alpha=0.3)
        ax_scatter2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # Configure accuracy scatter plot
        ax_scatter3.set_xlabel('Timestamp', fontsize=14)
        ax_scatter3.set_ylabel('Accuracy', fontsize=14)
        ax_scatter3.set_title(f'Individual Accuracies vs Timestamp - {config.expert} Expert', fontsize=16)
        ax_scatter3.grid(True, alpha=0.3)
        ax_scatter3.set_ylim(0, 1)  # Accuracy is between 0 and 1
        ax_scatter3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        plt.tight_layout()
        
        # Save scatter plot
        scatter_plot_filename = plot_filename.replace('.png', '_scatter.png')
        scatter_plot_path = output_path / scatter_plot_filename
        plt.savefig(scatter_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Scatter plot saved to: {scatter_plot_path}")
        
        # Create bar plot for accuracy success rate
        fig_bar, ax_bar = plt.subplots(1, 1, figsize=(14, 8))
        
        # Calculate success rate (percentage of correct predictions) for each timestamp
        success_rates = []
        for ts in sorted_timestamps:
            accuracy_values = timestamp_to_accuracies[ts]
            success_rate = np.mean(accuracy_values) * 100  # Convert to percentage
            success_rates.append(success_rate)
        
        # Create bar plot
        bars = ax_bar.bar(sorted_timestamps, success_rates, 
                         color='skyblue', alpha=0.7, edgecolor='navy', linewidth=1)
        
        # Add value labels on top of bars
        for i, (ts, rate) in enumerate(zip(sorted_timestamps, success_rates)):
            ax_bar.text(ts, rate + 1, f'{rate:.1f}%', 
                       ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Add sample count annotations
        for i, (ts, count) in enumerate(zip(sorted_timestamps, counts)):
            ax_bar.text(ts, success_rates[i] - 5, f'n={count}', 
                       ha='center', va='top', fontsize=8, alpha=0.7)
        
        ax_bar.set_xlabel('Timestamp', fontsize=14)
        ax_bar.set_ylabel('Success Rate (%)', fontsize=14)
        ax_bar.set_title(f'Prediction Success Rate vs Timestamp - {config.expert} Expert', fontsize=16)
        ax_bar.grid(True, alpha=0.3, axis='y')
        ax_bar.set_ylim(0, 105)  # Give some space for labels
        
        # Add a horizontal line at 50% for reference
        ax_bar.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% Reference')
        ax_bar.legend()
        
        plt.tight_layout()
        
        # Save bar plot
        bar_plot_filename = plot_filename.replace('.png', '_accuracy_bar.png')
        bar_plot_path = output_path / bar_plot_filename
        plt.savefig(bar_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Accuracy bar plot saved to: {bar_plot_path}")
    
    # Create bar plot for accuracy success rate (always create this)
    if bar_plot_path is None:  # Only create if not already created in scatter section
        fig_bar, ax_bar = plt.subplots(1, 1, figsize=(14, 8))
        
        # Calculate success rate (percentage of correct predictions) for each timestamp
        success_rates = []
        for ts in sorted_timestamps:
            accuracy_values = timestamp_to_accuracies[ts]
            success_rate = np.mean(accuracy_values) * 100  # Convert to percentage
            success_rates.append(success_rate)
        
        # Create bar plot
        bars = ax_bar.bar(sorted_timestamps, success_rates, 
                         color='skyblue', alpha=0.7, edgecolor='navy', linewidth=1)
        
        # Add value labels on top of bars
        for i, (ts, rate) in enumerate(zip(sorted_timestamps, success_rates)):
            ax_bar.text(ts, rate + 1, f'{rate:.1f}%', 
                       ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Add sample count annotations
        for i, (ts, count) in enumerate(zip(sorted_timestamps, counts)):
            ax_bar.text(ts, success_rates[i] - 5, f'n={count}', 
                       ha='center', va='top', fontsize=8, alpha=0.7)
        
        ax_bar.set_xlabel('Timestamp', fontsize=14)
        ax_bar.set_ylabel('Success Rate (%)', fontsize=14)
        ax_bar.set_title(f'Prediction Success Rate vs Timestamp - {config.expert} Expert', fontsize=16)
        ax_bar.grid(True, alpha=0.3, axis='y')
        ax_bar.set_ylim(0, 105)  # Give some space for labels
        
        # Add a horizontal line at 50% for reference
        ax_bar.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% Reference')
        ax_bar.legend()
        
        plt.tight_layout()
        
        # Save bar plot
        bar_plot_filename = plot_filename.replace('.png', '_accuracy_bar.png')
        bar_plot_path = output_path / bar_plot_filename
        plt.savefig(bar_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Accuracy bar plot saved to: {bar_plot_path}")
    
    # Save the main plot (mean plots with error bars)
    if plot_style in ["mean", "both"]:
        plot_path = output_path / plot_filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Mean plot saved to: {plot_path}")
    
    if plot_path:
        logging.info(f"Timestep analysis mean plot saved to: {plot_path}")
    
    # Print summary statistics
    overall_mean = np.mean(distances_np)
    overall_var = np.var(distances_np)
    overall_std = np.std(distances_np)
    
    overall_cosine_mean = np.mean(cosine_similarities_np)
    overall_cosine_var = np.var(cosine_similarities_np)
    overall_cosine_std = np.std(cosine_similarities_np)
    
    overall_accuracy_mean = np.mean(accuracies_np)
    overall_accuracy_std = np.std(accuracies_np)
    
    logging.info(f"Overall distance statistics:")
    logging.info(f"  Mean: {overall_mean:.4f}")
    logging.info(f"  Variance: {overall_var:.4f}")
    logging.info(f"  Standard deviation: {overall_std:.4f}")
    logging.info(f"Overall cosine similarity statistics:")
    logging.info(f"  Mean: {overall_cosine_mean:.4f}")
    logging.info(f"  Variance: {overall_cosine_var:.4f}")
    logging.info(f"  Standard deviation: {overall_cosine_std:.4f}")
    logging.info(f"Overall accuracy statistics:")
    logging.info(f"  Mean: {overall_accuracy_mean:.4f}")
    logging.info(f"  Standard deviation: {overall_accuracy_std:.4f}")
    logging.info(f"  Number of timestamps: {len(sorted_timestamps)}")
    
    # Return results
    results = {
        'overall_mean_distance': overall_mean,
        'overall_var_distance': overall_var,
        'overall_std_distance': overall_std,
        'overall_mean_cosine_similarity': overall_cosine_mean,
        'overall_var_cosine_similarity': overall_cosine_var,
        'overall_std_cosine_similarity': overall_cosine_std,
        'overall_mean_accuracy': overall_accuracy_mean,
        'overall_std_accuracy': overall_accuracy_std,
        'timestamp_stats': timestamp_stats,
        'sorted_timestamps': sorted_timestamps,
        'mean_distances': mean_distances,
        'std_distances': std_distances,
        'mean_cosine_similarities': mean_cosine_sims,
        'std_cosine_similarities': std_cosine_sims,
        'mean_accuracies': mean_accuracies,
        'std_accuracies': std_accuracies,
        'sample_counts': counts,
        'all_distances': distances_np,
        'all_cosine_similarities': cosine_similarities_np,
        'all_accuracies': accuracies_np,
        'all_timestamps': timestamps,
        'plot_path': str(plot_path) if plot_path else None,
        'scatter_plot_path': str(scatter_plot_path) if scatter_plot_path else None,
        'bar_plot_path': str(bar_plot_path) if bar_plot_path else None
    }
    
    return results


def save_results(results: Dict, config: DataConfig, 
                output_dir: str):
    """Save results to disk."""
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create filename based on configuration
    import datetime
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if config.expert == "visual":
        filename_base = f"results_{config.expert}_prelogits_step{config.rollout_step}"
    else:
        filename_base = f"results_{config.expert}_layer{config.layer}_step{config.rollout_step}"
        if config.expert == "action":
            filename_base += f"_t{config.action_timestep}"
    filename_base += f"_{time_str}"
    
    # Save results
    results_file = output_path / f"{filename_base}.pkl"
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    # Save summary
    summary_file = output_path / f"summary_{filename_base}.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Linear Probing Results\n")
        f.write(f"=====================\n\n")
        f.write(f"Expert: {config.expert}\n")
        if config.expert == "visual":
            f.write(f"Visual Feature Type: pre_logits\n")
        else:
            f.write(f"Layer: {config.layer}\n")
        f.write(f"Rollout Step: {config.rollout_step}\n")
        f.write(f"Feature Type: {config.feature_type}\n")
        if config.action_timestep:
            f.write(f"Action Timestep: {config.action_timestep}\n")
        f.write(f"\n")
        f.write(f"Test Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Test Cosine Similarity: {results['cosine_similarity']:.4f}\n")
        f.write(f"Number of Test Samples: {results['num_samples']}\n")
        f.write(f"Number of Tasks: {results['num_tasks']}\n")
        f.write(f"Feature Dimension: {results['feature_dim']}\n")
        f.write(f"T5 Embedding Dimension: {results['t5_dim']}\n\n")
        
        # Add training metrics if available
        if 'training_metrics' in results:
            metrics = results['training_metrics']
            f.write(f"Training Metrics:\n")
            f.write(f"  Best Eval Loss: {metrics.get('best_eval_loss', 'N/A'):.4f}\n")
            f.write(f"  Best Epoch: {metrics.get('best_epoch', 'N/A')}\n")
            f.write(f"\n")
        
        f.write(f"Per-Task Accuracies:\n")
        for task_desc, acc in results['task_accuracies'].items():
            sample_count = results.get('task_sample_counts', {}).get(task_desc, 'N/A')
            f.write(f"  {task_desc}: {acc:.4f} ({sample_count} samples)\n")
        
        # Add timestep analysis results if available
        if 'timestep_analysis' in results:
            timestep_results = results['timestep_analysis']
            f.write(f"\nTimestep Analysis Results:\n")
            f.write(f"  Overall Mean Distance: {timestep_results['overall_mean_distance']:.4f}\n")
            f.write(f"  Overall Distance Variance: {timestep_results['overall_var_distance']:.4f}\n")
            f.write(f"  Overall Distance Std: {timestep_results['overall_std_distance']:.4f}\n")
            f.write(f"  Overall Mean Cosine Similarity: {timestep_results['overall_mean_cosine_similarity']:.4f}\n")
            f.write(f"  Overall Cosine Similarity Variance: {timestep_results['overall_var_cosine_similarity']:.4f}\n")
            f.write(f"  Overall Cosine Similarity Std: {timestep_results['overall_std_cosine_similarity']:.4f}\n")
            f.write(f"  Overall Mean Accuracy: {timestep_results['overall_mean_accuracy']:.4f}\n")
            f.write(f"  Overall Accuracy Std: {timestep_results['overall_std_accuracy']:.4f}\n")
            f.write(f"  Number of Timestamps: {len(timestep_results['sorted_timestamps'])}\n")
            if timestep_results.get('plot_path'):
                f.write(f"  Mean Plot Saved: {timestep_results['plot_path']}\n")
            if timestep_results.get('scatter_plot_path'):
                f.write(f"  Scatter Plot Saved: {timestep_results['scatter_plot_path']}\n")
            if timestep_results.get('bar_plot_path'):
                f.write(f"  Accuracy Bar Plot Saved: {timestep_results['bar_plot_path']}\n")
            
            f.write(f"\nPer-Timestamp Statistics:\n")
            for ts in timestep_results['sorted_timestamps']:
                stats = timestep_results['timestamp_stats'][ts]
                f.write(f"  Timestamp {ts}: dist_mean={stats['mean']:.4f}, dist_std={stats['std']:.4f}, cos_mean={stats['cosine_mean']:.4f}, cos_std={stats['cosine_std']:.4f}, acc_mean={stats['accuracy_mean']:.4f}, acc_std={stats['accuracy_std']:.4f}, count={stats['count']}\n")
        
        f.write(f"\nTask Descriptions:\n")
        for i, desc in enumerate(results['task_descriptions']):
            f.write(f"  {i}: {desc}\n")
    
    # Save model weights
    model_file = output_path / f"model_{filename_base}.pth"
    torch.save(results.get('model_state_dict', {}), model_file)
    
    logging.info(f"Results saved to {output_path}")
    logging.info(f"  - Results: {results_file}")
    logging.info(f"  - Summary: {summary_file}")
    logging.info(f"  - Model: {model_file}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Linear probing analysis for inference latent data")
    
    # Data arguments
    parser.add_argument("--rollout_step", type=int, default=None,
                       help="Specific rollout step to analyze, this is used to index into the rollout_steps dictionary, not necessarily the same as the step number in the episode")
    parser.add_argument("--expert", type=str, required=True, 
                       choices=["vlm", "action", "text_only", "visual"],
                       help="Expert type to analyze")
    parser.add_argument("--layer", type=int, required=False, default=-1,
                       help="Layer ID to analyze (0-17), not needed for visual expert")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to inference latent data")
    parser.add_argument("--task_range", type=int, nargs=2, required=True,
                       help="Range of tasks to analyze (start, end)")
    parser.add_argument("--episode_range", type=int, nargs=2, required=True,
                       help="Range of episodes to analyze (start, end)")
    parser.add_argument("--action_timestep", type=int, default=9,
                       help="For action expert, which diffusion timestep to use. Range [0-9] This is the index of the action_expert_hidden_state_t dictionary, not necessarily the same as the diffusion timestep")
    
    # Training arguments
    parser.add_argument("--learning_rate", type=float, default=0.001,
                       help="Learning rate for training")
    parser.add_argument("--num_epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for training")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="results/linear_probing",
                       help="Output directory for results")
    
    # Debug arguments
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    parser.add_argument("--sanity_check", action="store_true",
                       help="Run sanity checks and exit")
    
    # T5 arguments
    parser.add_argument("--t5_target_dim", type=int, default=64,
                       help="Target dimension for T5 embedding projection (default: 64)")
    
    # Plot arguments
    parser.add_argument("--plot_style", type=str, default="both", 
                       choices=["both", "scatter", "mean"],
                       help="Plot style: 'both' for mean plots with error bars and scatter plots, 'scatter' for scatter plots only, 'mean' for mean plots only")
    
    return parser.parse_args()


def sanity_check_data_loading(config: DataConfig):
    """Run sanity checks on data loading."""
    logging.info("Running sanity checks...")
    
    try:
        # Test dataset creation
        dataset = InferenceLatentDataset(config, split="all")
        logging.info("✓ Dataset created successfully")
        
        # Check dataset properties
        logging.info(f"✓ Dataset length: {len(dataset)}")
        logging.info(f"✓ Number of unique tasks: {len(dataset.task_descriptions)}")
        
        if len(dataset) == 0:
            logging.error("✗ Dataset is empty!")
            return False
        
        # Test getting a sample
        sample_features, sample_label = dataset[0]
        logging.info(f"✓ Sample features shape: {sample_features.shape}")
        logging.info(f"✓ Sample label: {sample_label}")
        
        # Test T5 label creation
        t5_labels = create_t5_labels(dataset.task_descriptions)
        logging.info(f"✓ T5 labels shape: {t5_labels.shape}")
        
        # Check feature and label dimensions
        feature_dim = sample_features.shape[0]
        t5_dim = t5_labels.shape[1]
        logging.info(f"✓ Feature dimension: {feature_dim}")
        logging.info(f"✓ T5 embedding dimension: {t5_dim}")
        
        # Test data info
        data_info = dataset.get_data_info()
        logging.info("✓ Data info retrieved successfully")
        logging.info(f"  - Expert: {data_info['expert_type']}")
        if data_info['expert_type'] == "visual":
            logging.info(f"  - Visual feature type: {data_info.get('visual_feature_type', 'N/A')}")
            logging.info(f"  - Camera views: {data_info.get('camera_views', [])}")
        else:
            logging.info(f"  - Layer: {data_info['layer']}")
        logging.info(f"  - Rollout step: {data_info['rollout_step']}")
        logging.info(f"  - Feature type: {data_info['feature_type']}")
        if data_info['action_timestep']:
            logging.info(f"  - Action timestep: {data_info['action_timestep']}")
        
        # Test dataset splits
        logging.info("✓ Testing dataset splits...")
        train_dataset, eval_dataset, test_dataset, full_dataset = create_dataset_splits(config, TrainingConfig())
        
        logging.info(f"✓ Train split: {len(train_dataset)} samples")
        logging.info(f"✓ Eval split: {len(eval_dataset)} samples")
        logging.info(f"✓ Test split: {len(test_dataset)} samples")
        
        # Test DataLoader creation
        dataloader = DataLoader(dataset, batch_size=min(4, len(dataset)), shuffle=True)
        logging.info("✓ DataLoader created successfully")
        
        # Test batch iteration
        for batch_idx, (batch_features, batch_labels) in enumerate(dataloader):
            logging.info(f"✓ Batch {batch_idx}: features {batch_features.shape}, labels {batch_labels.shape}")
            if batch_idx >= 2:  # Only test first few batches
                break
        
        logging.info("✓ All sanity checks passed!")
        return True
        
    except Exception as e:
        logging.error(f"✗ Sanity check failed: {e}")
        return False


def main():
    """Main execution function."""
    args = parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Validate arguments
    if args.expert != "visual" and args.layer == -1:
        logging.error("Layer argument is required for non-visual experts")
        return
    
    # Create configurations
    data_config = DataConfig(
        data_path=args.data_path,
        task_range=tuple(args.task_range),
        episode_range=tuple(args.episode_range),
        rollout_step=args.rollout_step,
        expert=args.expert,
        layer=args.layer if args.expert != "visual" else -1,
        action_timestep=args.action_timestep,
        seed=args.seed
    )
    
    training_config = TrainingConfig(
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        seed=args.seed
    )
    
    # Run sanity checks if requested
    if args.sanity_check:
        success = sanity_check_data_loading(data_config)
        if success:
            logging.info("Sanity checks completed successfully. Ready for training.")
        else:
            logging.error("Sanity checks failed. Please check your data and configuration.")
        return
    
    # Load data and create splits
    logging.info("Loading inference latent data and creating splits...")
    train_dataset, eval_dataset, test_dataset, full_dataset = create_dataset_splits(data_config, training_config)
    
    # Create T5 labels
    logging.info("Creating T5 labels...")
    t5_labels = create_t5_labels(train_dataset.task_descriptions, target_dim=args.t5_target_dim)
    
    # Print data summary
    data_info = train_dataset.get_data_info()
    logging.info(f"Data Summary:")
    logging.info(f"  - Train samples: {len(train_dataset)}")
    logging.info(f"  - Eval samples: {len(eval_dataset)}")
    logging.info(f"  - Test samples: {len(test_dataset)}")
    logging.info(f"  - Tasks: {data_info['num_tasks']}")
    logging.info(f"  - Feature dimension: {data_info['feature_dim']}")
    logging.info(f"  - T5 embedding dimension: {t5_labels.shape[1]}")
    if data_info['expert_type'] == "visual":
        logging.info(f"  - Camera views: {data_info.get('camera_views', [])}")
        logging.info(f"  - Visual feature type: {data_info.get('visual_feature_type', 'N/A')}")
    
    # Train linear probe
    logging.info("Training linear probe...")
    model, training_metrics = train(train_dataset, eval_dataset, t5_labels, training_config, device)
    
    # Analyze results on test set
    logging.info("Analyzing results on test set...")
    results = analyze_results(model, test_dataset, t5_labels, data_config, device)

    # Analyze results on full set
    logging.info("Analyzing results on full set...")
    results_full = analyze_results_timesteps(model, full_dataset, t5_labels, data_config, device, args.plot_style)
    
    # Add model state dict and training metrics to results for saving
    results['model_state_dict'] = model.state_dict()
    results['training_metrics'] = training_metrics
    
    # Add timestep analysis results
    results['timestep_analysis'] = results_full
    
    # Save results
    logging.info("Saving results...")
    save_results(results, data_config, args.output_dir)
    
    # Print final metrics
    logging.info(f"Final Accuracy: {results['accuracy']:.4f}")
    logging.info(f"Final Cosine Similarity: {results['cosine_similarity']:.4f}")
    logging.info(f"Overall Mean Distance: {results_full['overall_mean_distance']:.4f}")
    logging.info(f"Overall Distance Variance: {results_full['overall_var_distance']:.4f}")
    logging.info(f"Overall Mean Cosine Similarity: {results_full['overall_mean_cosine_similarity']:.4f}")
    logging.info(f"Overall Cosine Similarity Variance: {results_full['overall_var_cosine_similarity']:.4f}")
    logging.info(f"Overall Mean Accuracy: {results_full['overall_mean_accuracy']:.4f}")
    logging.info(f"Overall Accuracy Std: {results_full['overall_std_accuracy']:.4f}")
    
    logging.info("Linear probing analysis completed successfully!")


if __name__ == "__main__":
    main()

