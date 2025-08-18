# Expert Separation TODO: Collecting Separate Hidden States

## Current Architecture Analysis

### How Experts Currently Work:
1. **Two Experts**: PaliGemma (Expert 0) and Action Expert (Expert 1)
2. **Input Routing**: 
   - Expert 0: Processes image + text tokens (positions 0-815)
   - Expert 1: Processes state + action tokens (positions 816+)
3. **Current Output**: Combined hidden states from both experts merged together

### Current Data Flow:
```
embed_prefix() → [image_tokens, text_tokens] → Expert 0
embed_suffix() → [state_tokens, action_tokens] → Expert 1
↓
PaliGemma.llm([prefix_tokens, suffix_tokens]) → Combined hidden states
↓
Analysis.get_hidden_states() → Merged tensor (18, seq_len, 2048)
```

## Understanding `sum(axis=1)`

### Tensor Shapes Analysis:
```python
# hidden_states shape: (18, batch_size, seq_len, 2048)
# 18 = number of layers
# batch_size = number of samples in batch
# seq_len = sequence length (816 for prefix)
# 2048 = hidden dimension

# hidden_states.sum(axis=1) → (18, seq_len, 2048)
# This sums across the batch dimension, reducing batch_size to 1
```

**Why `sum(axis=1)`?**
- **Axis 0**: Layer dimension (18 layers) - keep separate
- **Axis 1**: Batch dimension - sum across all samples in batch
- **Axis 2**: Sequence dimension (seq_len) - keep separate  
- **Axis 3**: Hidden dimension (2048) - keep separate

This operation **accumulates hidden states across all samples in the batch** while preserving layer-wise and token-wise information.

## Key Insight: Inference vs Training Data Collection

### **Training Time (Current `text_latent.py`)**:
- **No rollout context**: Training dataloader processes individual timesteps without episode context
- **No rollout step info**: Cannot track which step in the episode we're at
- **Batch processing**: Processes multiple episodes/timesteps together

### **Inference Time (Like `main.py`)**:
- **Full episode context**: Has access to complete rollout information
- **Rollout step tracking**: Can track `t` (timestep) in the episode
- **Real-time collection**: Can collect latent states at each inference step
- **Episode-level organization**: Can organize data by episode and rollout step

## Step-by-Step TODO

### **Step 1: Modify `src/openpi/models/gemma.py` - Module Class**

**File**: `src/openpi/models/gemma.py`  
**Class**: `Module`  
**Method**: `__call__` (lines ~461-624)

**Why**: The Module class currently returns combined hidden states. We need to modify it to return separate expert states.

**Changes Needed**:
```python
# Current return structure:
attn_output = (mlp_activations, attention, output_pre_attn_scales, output_pre_mlp_scales,
               output_norm, (all_post_attn_1, all_post_attn_2), layer_hidden_states, post_attn_embeddings,
               text_representation)

# New return structure:
attn_output = (mlp_activations, attention, output_pre_attn_scales, output_pre_mlp_scales,
               output_norm, (all_post_attn_1, all_post_attn_2), 
               layer_hidden_states,           # Combined (existing)
               expert_0_hidden_states,        # PaliGemma only
               expert_1_hidden_states,        # Action expert only
               post_attn_embeddings,
               text_representation)
```

**Implementation**:
1. Track `expert_0_hidden_states` and `expert_1_hidden_states` separately in the layer loop
2. Extract expert-specific states from `layer_hidden_states` based on token positions
3. Return both combined and separate expert states

---

### **Step 2: Modify `src/openpi/models/gemma.py` - Analysis Class**

**File**: `src/openpi/models/gemma.py`  
**Class**: `Analysis`  
**Methods**: Add new methods for expert-specific extraction

**Why**: Need new methods to extract expert-specific hidden states from the modified output.

**Changes Needed**:
```python
@staticmethod
def get_expert_0_hidden_states(layer_output, layer_index=None):
    """Extract PaliGemma expert hidden states (image + text tokens)."""
    layer_index = layer_index or [i for i in range(len(layer_output[7]))]  # New index
    return layer_output[7][jnp.asarray(layer_index)]

@staticmethod
def get_expert_1_hidden_states(layer_output, layer_index=None):
    """Extract Action expert hidden states (state + action tokens)."""
    layer_index = layer_index or [i for i in range(len(layer_output[8]))]  # New index
    return layer_output[8][jnp.asarray(layer_index)]

@staticmethod
def get_text_only_hidden_states(layer_output, layer_index=None):
    """Extract only text token hidden states from PaliGemma expert."""
    expert_0_states = Analysis.get_expert_0_hidden_states(layer_output, layer_index)
    # Extract text region: positions 768-815 (256*3 to 256*3+48)
    text_start = 256 * 3
    text_end = text_start + 48
    return expert_0_states[:, :, text_start:text_end, :]
```

---

### **Step 3: Modify `src/openpi/models/pi0.py` - Pi0 Class**

**File**: `src/openpi/models/pi0.py`  
**Class**: `Pi0`  
**Method**: `compute_loss_with_extra` (lines ~280-350)

**Why**: Need to extract and return separate expert states from the modified Module output.

**Changes Needed**:
```python
# Current result dict:
result = dict(
    loss_sum=loss.sum(axis=0),
    text_representation=text_representation,
    attention_score_sum=attention_score,
    hidden_states_sum=hidden_states.sum(axis=1),        # Combined
    post_attn_embedding_sum=post_attn_embedding.sum(axis=1),
    post_attn_sum=post_attn.sum(axis=1)
)

# New result dict:
result = dict(
    loss_sum=loss.sum(axis=0),
    text_representation=text_representation,
    attention_score_sum=attention_score,
    hidden_states_sum=hidden_states.sum(axis=1),        # Combined (existing)
    expert_0_hidden_states_sum=expert_0_hidden_states.sum(axis=1),  # PaliGemma
    expert_1_hidden_states_sum=expert_1_hidden_states.sum(axis=1),  # Action
    text_only_hidden_states_sum=text_only_hidden_states.sum(axis=1),  # Text only
    post_attn_embedding_sum=post_attn_embedding.sum(axis=1),
    post_attn_sum=post_attn.sum(axis=1)
)
```

**Implementation**:
1. Extract expert-specific states from `layer_output`
2. Extract text-only states from expert 0
3. Add new fields to result dict

---

### **Step 4: Add Inference Step Sampling**

**File**: `src/openpi/models/pi0.py`  
**Class**: `Pi0`  
**Methods**: `sample_actions` and `compute_loss_with_extra`

**Why**: Need to sample hidden states at different inference steps (AR process and flow matching).

**Changes Needed**:
```python
def sample_actions_with_inference_steps(self, rng, observation, num_steps=10):
    """Sample actions and collect hidden states at each inference step."""
    
    inference_steps_data = {
        "ar_process_steps": [],      # Autoregressive generation steps
        "flow_matching_steps": [],   # Flow matching denoising steps
        "final_action": None
    }
    
    # AR Process: Generate actions step by step
    for step in range(num_steps):
        # Get hidden states at this AR step
        step_result = self._get_step_hidden_states(rng, observation, step)
        inference_steps_data["ar_process_steps"].append(step_result)
    
    # Flow Matching: Denoise actions step by step  
    for step in range(num_steps):
        # Get hidden states at this flow matching step
        step_result = self._get_flow_matching_states(rng, observation, step)
        inference_steps_data["flow_matching_steps"].append(step_result)
    
    return inference_steps_data

def _get_step_hidden_states(self, rng, observation, step):
    """Get hidden states at a specific inference step."""
    # Implementation to extract hidden states at specific step
    pass
```

---

### **Step 5: Create Inference-Time Data Collection Script**

**File**: `scripts/inference_latent_collection.py` (new file)

**Why**: Need to collect latent data during inference time (like `main.py`) where we have access to rollout step information.

**Implementation**:
```python
def collect_inference_latents(model, task_suite, sampling_config):
    """Collect latent states during inference time with rollout step tracking."""
    
    all_task_data = {}
    
    for task_id in range(sampling_config["num_tasks"]):
        task = task_suite.get_task(task_id)
        task_description = task.language
        
        task_data = {
            "task_name": task_description.replace(" ", "_"),
            "task_id": task_id,
            "episodes": {}
        }
        
        for episode_idx in range(sampling_config["num_episodes_per_task"]):
            episode_data = {
                "rollout_steps": {},
                "metadata": {
                    "success": False,
                    "total_steps": 0
                }
            }
            
            # Initialize environment (like in main.py)
            env, _ = _get_libero_env(task, resolution=256, seed=sampling_config["seed"])
            obs = env.reset()
            
            t = 0
            while t < sampling_config["max_steps"]:
                # Prepare observation (like in main.py)
                element = {
                    "done": t == 0,  # reset_sever equivalent
                    "observation/image": preprocess_image(obs["agentview_image"]),
                    "observation/wrist_image": preprocess_image(obs["robot0_eye_in_hand_image"]),
                    "observation/state": get_robot_state(obs),
                    "prompt": task_description,
                }
                
                # Get action and latent states from model
                result = model.infer_with_latents(element)  # New method to return latents
                
                # Store rollout step data
                rollout_data = {
                    "timestep": t,
                    "expert_0_hidden_states": result["expert_0_hidden_states"],  # PaliGemma
                    "expert_1_hidden_states": result["expert_1_hidden_states"],  # Action
                    "text_only_hidden_states": result["text_only_hidden_states"],  # Text only
                    "action": result["actions"],
                    "observation": obs
                }
                
                episode_data["rollout_steps"][f"step_{t}"] = rollout_data
                
                # Execute action
                action = result["actions"][0]  # Take first action
                obs, reward, done, info = env.step(action.tolist())
                
                if done:
                    episode_data["metadata"]["success"] = True
                    break
                
                t += 1
            
            episode_data["metadata"]["total_steps"] = t
            task_data["episodes"][f"episode_{episode_idx}"] = episode_data
        
        all_task_data[task_id] = task_data
        
        # Save incrementally
        save_path = f"{EXP_DATA_PATH}/inference_latents/task_{task_id}_{task_data['task_name']}.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(task_data, f)
    
    return all_task_data
```

---

### **Step 6: Modify Model Interface for Latent Collection**

**File**: `src/openpi/models/pi0.py`  
**Class**: `Pi0`  
**Method**: Add new method for inference with latent collection

**Why**: Need a method that returns both actions and latent states during inference.

**Changes Needed**:
```python
def infer_with_latents(self, observation_dict):
    """Inference method that returns both actions and latent states."""
    
    # Convert observation dict to Observation object
    obs = _model.Observation.from_dict(observation_dict)
    
    # Get latent states during inference
    with jax.disable_jit():  # Ensure we can extract intermediate states
        # Forward pass to get latent states
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(obs)
        suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(obs, actions, timestep)
        
        # Get hidden states from both experts
        kv_cache, layer_output = self._encode(prefix_tokens, prefix_attn_mask, positions)
        
        # Extract expert-specific states
        expert_0_states = Analysis.get_expert_0_hidden_states(layer_output)
        expert_1_states = Analysis.get_expert_1_hidden_states(layer_output)
        text_only_states = Analysis.get_text_only_hidden_states(layer_output)
        
        # Get actions (existing logic)
        actions = self.sample_actions(rng, obs)
    
    return {
        "actions": actions,
        "expert_0_hidden_states": expert_0_states,
        "expert_1_hidden_states": expert_1_states,
        "text_only_hidden_states": text_only_states
    }
```

---

### **Step 7: Create Data Access Functions**

**File**: `scripts/expert_data_utils.py` (new file)

**Why**: Need utility functions to easily access data for different dimensions.

**Implementation**:
```python
def load_inference_task_data(task_id, task_name=None):
    """Load inference task data from pickle file."""
    if task_name is None:
        task_name = f"task_{task_id}"
    filepath = f"{EXP_DATA_PATH}/inference_latents/{task_name}.pkl"
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def get_hidden_states_for_dimensions(task_data, episode_idx, rollout_step, layer_idx, 
                                   expert_type="expert_0"):
    """Get hidden states for specific dimensions from inference data."""
    
    episode = task_data["episodes"][f"episode_{episode_idx}"]
    rollout = episode["rollout_steps"][f"step_{rollout_step}"]
    
    return rollout[f"{expert_type}_hidden_states"][layer_idx]

def analyze_language_loss_across_rollout_steps(task_data):
    """Analyze language loss across rollout steps."""
    
    results = {
        "rollout_steps": {},
        "episodes": {},
        "experts": {}
    }
    
    # Analyze across rollout steps
    for episode_idx, episode in task_data["episodes"].items():
        episode_results = {
            "step_analysis": {},
            "success": episode["metadata"]["success"],
            "total_steps": episode["metadata"]["total_steps"]
        }
        
        for step_idx, step_data in episode["rollout_steps"].items():
            # Analyze language loss at this rollout step
            step_analysis = analyze_rollout_step_language_loss(step_data)
            episode_results["step_analysis"][step_idx] = step_analysis
        
        results["episodes"][episode_idx] = episode_results
    
    return results
```

---

### **Step 8: Update Analysis Scripts**

**File**: `scripts/expert_language_analysis.py` (new file)

**Why**: Need dedicated analysis for the inference-time data format.

**Implementation**:
```python
def analyze_inference_language_loss(task_data):
    """Analyze language loss using inference-time data."""
    
    analysis_results = {}
    
    for episode_idx, episode in task_data["episodes"].items():
        episode_results = {
            "rollout_step_analysis": {},
            "language_loss_trajectory": [],
            "expert_comparison": {}
        }
        
        for step_idx, step_data in episode["rollout_steps"].items():
            # Analyze language loss at this rollout step
            step_analysis = analyze_rollout_step(step_data)
            episode_results["rollout_step_analysis"][step_idx] = step_analysis
            episode_results["language_loss_trajectory"].append(step_analysis["overall_loss"])
        
        analysis_results[episode_idx] = episode_results
    
    return analysis_results

def analyze_rollout_step(step_data):
    """Analyze language loss at a specific rollout step."""
    
    step_results = {
        "expert_0_language_loss": [],
        "expert_1_language_loss": [],
        "text_only_language_loss": [],
        "overall_loss": 0.0
    }
    
    # Analyze each layer
    for layer_idx in range(18):
        # Analyze language loss in each expert
        expert_0_loss = analyze_expert_language_loss(step_data["expert_0_hidden_states"][layer_idx])
        expert_1_loss = analyze_expert_language_loss(step_data["expert_1_hidden_states"][layer_idx])
        text_only_loss = analyze_expert_language_loss(step_data["text_only_hidden_states"][layer_idx])
        
        step_results["expert_0_language_loss"].append(expert_0_loss)
        step_results["expert_1_language_loss"].append(expert_1_loss)
        step_results["text_only_language_loss"].append(text_only_loss)
    
    # Calculate overall loss
    step_results["overall_loss"] = np.mean(step_results["text_only_language_loss"])
    
    return step_results
```

---

## Expected Benefits

### **1. Rollout Step Context**
- **Real episode progression**: Track language loss as episode unfolds
- **Temporal analysis**: See how language information degrades over time
- **Success correlation**: Relate language loss to task success/failure

### **2. Expert-Specific Analysis**
- **PaliGemma Expert**: How well it preserves language during rollout
- **Action Expert**: How it affects language information during action generation
- **Cross-expert interaction**: How experts influence each other over time

### **3. Structured Data Access**
- **Rollout-based organization**: Data organized by episode and rollout step
- **Easy access**: `task_data["episodes"]["episode_0"]["rollout_steps"]["step_15"]["expert_0_hidden_states"]`
- **Temporal queries**: Get language loss trajectory across rollout steps

## Implementation Order

1. **Steps 1-3**: Foundation (Module, Analysis, Pi0 modifications)
2. **Step 4**: Inference step sampling (AR and flow matching)
3. **Step 5**: Inference-time data collection (like main.py)
4. **Step 6**: Model interface for latent collection
5. **Steps 7-8**: Data access and analysis utilities

This approach leverages the inference-time context (like `main.py`) to collect latent data with full rollout step information, giving you comprehensive insights into how language information flows and degrades during actual task execution. 