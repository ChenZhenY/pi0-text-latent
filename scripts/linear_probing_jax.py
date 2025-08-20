"""
A script to perform linear probing on pi0 models to investigate the relevance of latent features to language input.
"""
import dataclasses
import logging
from typing import Any, Dict, List, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from absl import app, flags
from flax.training import train_state
import optax
from transformers import T5EncoderModel, T5Tokenizer

from openpi.models import model as model_lib
from openpi.models import pi0_linear_probing
from openpi.models.gemma import Analysis
from openpi.models.tokenizer import PaligemmaTokenizer


FLAGS = flags.FLAGS

# Flags for model loading, data collection, training, etc.
flags.DEFINE_string("model_path", None, "Path to the model checkpoint.")
flags.DEFINE_string("t5_model_name", "t5-small", "Name of the T5 model to use for text embeddings.")
flags.DEFINE_integer("num_tasks", 10, "Number of tasks to use for data collection.")
flags.DEFINE_integer("num_steps", 10, "Number of diffusion steps for sampling.")
flags.DEFINE_integer("batch_size", 8, "Batch size for training the linear probe.")
flags.DEFINE_float("learning_rate", 1e-3, "Learning rate for the linear probe.")
flags.DEFINE_integer("num_epochs", 10, "Number of epochs for training the linear probe.")


class LinearProbe(nn.Module):
    """A simple linear probe model."""
    num_features: int

    @nn.compact
    def __call__(self, x):
        return nn.Dense(features=self.num_features)(x)

def get_t5_embeddings(prompts: List[str], tokenizer, model) -> np.ndarray:
    """Gets T5 embeddings for a list of prompts."""
    inputs = tokenizer(prompts, return_tensors="jax", padding=True, truncation=True)
    embeddings = model(**inputs).last_hidden_state.mean(axis=1)
    return embeddings

# def collect_data(pi0_model: pi0_linear_probing.Pi0, pi0_config: pi0_linear_probing.Pi0Config, t5_tokenizer: T5Tokenizer, t5_model: T5EncoderModel, tasks: List[str], num_steps: int):
#     """Collects latent features and text embeddings."""
#     paligemma_tokenizer = PaligemmaTokenizer(max_len=pi0_config.max_token_len)

#     collected_data = []
#     for task in tasks:
#         # Get T5 embedding
#         t5_embedding = get_t5_embeddings([task], t5_tokenizer, t5_model)

#         # Create a fake observation
#         obs = pi0_config.fake_obs()
#         tokens, masks = paligemma_tokenizer.tokenize(task)
#         obs = obs.replace(tokenized_prompt=tokens[None, :], tokenized_prompt_mask=masks[None, :])

#         # Sample actions and get latents
#         rng = jax.random.key(0)
#         actions, latents = pi0_model.sample_actions_with_latents(rng, obs, num_steps=num_steps)

#         # Extract hidden states from latents
#         hidden_states = Analysis.get_hidden_states(latents)
#         collected_data.append((hidden_states, t5_embedding))

#     return collected_data


def create_train_state(rng, learning_rate, num_features):
    """Creates initial TrainState."""
    probe = LinearProbe(num_features=num_features)
    params = probe.init(rng, jnp.ones([1, num_features]))["params"]
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=probe.apply, params=params, tx=tx)


@jax.jit
def train_step(state, batch):
    """Train for a single step."""
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, batch["data"])
        loss = optax.losses.mean_squared_error(predictions=logits, targets=batch["target"])
        return loss.mean()

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


@jax.jit
def eval_step(state, batch, all_targets):
    """Evaluate for a single step."""
    logits = state.apply_fn({"params": state.params}, batch["data"])
    # Cosine similarity
    similarity = jnp.dot(logits, all_targets.T) / (jnp.linalg.norm(logits, axis=-1, keepdims=True) * jnp.linalg.norm(all_targets, axis=-1, keepdims=True).T)
    predictions = jnp.argmax(similarity, axis=-1)
    # The target is a one-hot vector, so we can get the index by argmax
    true_labels = jnp.argmax(jnp.all(batch["target"][:, None, :] == all_targets[None, :, :], axis=-1), axis=-1)
    correct = jnp.equal(predictions, true_labels)
    return correct.mean()

def train_probe(data, learning_rate, num_epochs, batch_size, num_features, all_targets):
    """Trains the linear probe."""
    rng = jax.random.key(0)
    state = create_train_state(rng, learning_rate, num_features)

    for epoch in range(num_epochs):
        epoch_loss = 0
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i+batch_size]
            batch = {"data": jnp.array([item[0] for item in batch_data]), "target": jnp.array([item[1] for item in batch_data])}
            state, loss = train_step(state, batch)
            epoch_loss += loss
        epoch_loss /= (len(data) // batch_size)

    # Evaluate the probe
    batch = {"data": jnp.array([item[0] for item in data]), "target": jnp.array([item[1] for item in data])}
    accuracy = eval_step(state, batch, all_targets)
    return accuracy

def analyze_results(results: Dict[Tuple[int, int], float]):
    """Analyzes and prints the results."""
    print("\n--- Linear Probing Results ---")
    print("Layer | Diffusion Step | Accuracy")
    print("---------------------------------")
    sorted_results = sorted(results.items())
    for (layer, step), acc in sorted_results:
        print(f"{layer:5d} | {step:14d} | {acc:.4f}")

def main(_):
    # Load pi0 model
    pi0_config = pi0_linear_probing.Pi0Config()
    if FLAGS.model_path is None:
        print("Please provide a path to the model checkpoint using --model_path")
        return
    params = model_lib.restore_params(FLAGS.model_path)
    pi0_model = pi0_config.load(params)

    # Initialize T5 model and tokenizer
    t5_tokenizer = T5Tokenizer.from_pretrained(FLAGS.t5_model_name)
    t5_model = T5EncoderModel.from_pretrained(FLAGS.t5_model_name)

    # Define a set of tasks (prompts)
    tasks = [f"task {i}" for i in range(FLAGS.num_tasks)]

    # Collect data
    collected_data = collect_data(pi0_model, pi0_config, t5_tokenizer, t5_model, tasks, FLAGS.num_steps)
    print(f"Successfully collected data for {len(collected_data)} tasks.")

    all_hidden_states = jnp.array([item[0] for item in collected_data])
    all_text_embeddings = jnp.squeeze(jnp.array([item[1] for item in collected_data]), axis=1)

    num_layers = all_hidden_states.shape[2]
    num_diffusion_steps = all_hidden_states.shape[1]
    num_features = all_hidden_states.shape[-1]

    results = {}
    for layer in range(num_layers):
        for step in range(num_diffusion_steps):
            print(f"Training probe for layer {layer}, diffusion step {step}")
            # Extract data for this layer and step
            # Taking the mean over tokens for simplicity
            data_for_probe = [(all_hidden_states[i, step, layer].mean(axis=0), all_text_embeddings[i]) for i in range(len(tasks))]
            
            accuracy = train_probe(data_for_probe, FLAGS.learning_rate, FLAGS.num_epochs, FLAGS.batch_size, num_features, all_text_embeddings)
            results[(layer, step)] = accuracy

    analyze_results(results)


if __name__ == "__main__":
    app.run(main)
