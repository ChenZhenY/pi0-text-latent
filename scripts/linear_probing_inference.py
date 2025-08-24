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
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5EncoderModel


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
        self.task_descriptions = []
        self.task_to_label_idx = {}  # Map task description to label index
        self._load_data()
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
        
    def _load_data(self):
        """Load and preprocess inference latent data."""
        data_path = pathlib.Path(self.config.data_path)
        task_start, task_end = self.config.task_range
        episode_start, episode_end = self.config.episode_range
        
        logging.info(f"Loading data from {data_path}")
        logging.info(f"Task range: {task_start}-{task_end}, Episode range: {episode_start}-{episode_end}")
        logging.info(f"Rollout step: {self.config.rollout_step}, Expert: {self.config.expert}, Layer: {self.config.layer}")
        
        for task_id in range(task_start, task_end):
            # Find task file
            task_files = list(data_path.glob(f"task_{task_id}_*.pkl"))
            if not task_files:
                logging.warning(f"No data file found for task {task_id}")
                continue
                
            task_file = task_files[0]
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

                        # Extract features based on expert type
                        features = self._extract_features(step_data)
                        if features is not None:
                            if isinstance(features, list):
                                self.features.extend(features)
                                self.labels.extend([label_idx] * len(features))
                            else:
                                self.features.append(features)
                                self.labels.append(label_idx)
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
                        self.features.append(features)
                        self.labels.append(label_idx)
        
        logging.info(f"Loaded {len(self.features)} samples with {len(self.task_descriptions)} unique tasks")
        
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
        return {
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
    
    return train_dataset, eval_dataset, test_dataset


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
    """Compute classification accuracy using cosine similarity with proper handling of duplicate targets."""
    with torch.no_grad():
        # Normalize predictions and targets
        predictions_norm = nn.functional.normalize(predictions, p=2, dim=1)
        targets_norm = nn.functional.normalize(targets, p=2, dim=1)
        
        # Compute cosine similarities between all pairs
        similarities = torch.mm(predictions_norm, targets_norm.T)
        
        # For each prediction, find all targets that have the maximum similarity
        # This handles the case where multiple targets might have the same T5 embedding
        max_similarities = torch.max(similarities, dim=1, keepdim=True)[0]
        max_mask = (similarities >= max_similarities - 1e-6)  # Allow for small numerical differences
        
        # For each prediction, check if the true target is among the best matches
        correct_predictions = 0
        for i in range(predictions.shape[0]):
            # Get the true target for this prediction
            true_target = targets[i]
            
            # Find all targets that match the true target (handles duplicates)
            target_matches = torch.all(targets == true_target, dim=1)
            
            # Check if any of the best matches for this prediction include the true target
            best_matches_for_prediction = max_mask[i]
            if torch.any(best_matches_for_prediction & target_matches):
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
    for task_idx, task_desc in enumerate(test_dataset.task_descriptions):
        task_mask = (labels == task_idx)
        if task_mask.sum() > 0:
            task_predictions = predictions[task_mask]
            task_targets = targets[task_mask]
            task_accuracy = compute_accuracy(task_predictions, task_targets)
            task_accuracies[task_desc] = task_accuracy
    
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
    
    logging.info(f"Test Results:")
    logging.info(f"  Accuracy: {accuracy:.4f}")
    logging.info(f"  Cosine Similarity: {cosine_sim:.4f}")
    logging.info(f"  Number of samples: {len(test_dataset)}")
    
    return results


def save_results(results: Dict, config: DataConfig, 
                output_dir: str):
    """Save results to disk."""
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create filename based on configuration
    import datetime
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
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
            f.write(f"  {task_desc}: {acc:.4f}\n")
        
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
                       choices=["vlm", "action", "text_only"],
                       help="Expert type to analyze")
    parser.add_argument("--layer", type=int, required=True,
                       help="Layer ID to analyze (0-17)")
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
        logging.info(f"  - Layer: {data_info['layer']}")
        logging.info(f"  - Rollout step: {data_info['rollout_step']}")
        logging.info(f"  - Feature type: {data_info['feature_type']}")
        if data_info['action_timestep']:
            logging.info(f"  - Action timestep: {data_info['action_timestep']}")
        
        # Test dataset splits
        logging.info("✓ Testing dataset splits...")
        train_dataset, eval_dataset, test_dataset = create_dataset_splits(config, TrainingConfig())
        
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
    
    # Create configurations
    data_config = DataConfig(
        data_path=args.data_path,
        task_range=tuple(args.task_range),
        episode_range=tuple(args.episode_range),
        rollout_step=args.rollout_step,
        expert=args.expert,
        layer=args.layer,
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
    train_dataset, eval_dataset, test_dataset = create_dataset_splits(data_config, training_config)
    
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
    
    # Train linear probe
    logging.info("Training linear probe...")
    model, training_metrics = train(train_dataset, eval_dataset, t5_labels, training_config, device)
    
    # Analyze results on test set
    logging.info("Analyzing results on test set...")
    results = analyze_results(model, test_dataset, t5_labels, data_config, device)
    
    # Add model state dict and training metrics to results for saving
    results['model_state_dict'] = model.state_dict()
    results['training_metrics'] = training_metrics
    
    # Save results
    logging.info("Saving results...")
    save_results(results, data_config, args.output_dir)
    
    # Print final metrics
    logging.info(f"Final Accuracy: {results['accuracy']:.4f}")
    logging.info(f"Final Cosine Similarity: {results['cosine_similarity']:.4f}")
    
    logging.info("Linear probing analysis completed successfully!")


if __name__ == "__main__":
    main()

