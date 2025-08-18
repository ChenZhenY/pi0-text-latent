# Parameter Count Analysis for Pi0 Expert Models

## Where the Configs Are Defined

### 1. Expert Configurations in `src/openpi/models/gemma.py`

```python
@dataclasses.dataclass
class Config:
    width: int          # Hidden dimension
    depth: int          # Number of layers
    mlp_dim: int        # MLP hidden dimension
    num_heads: int      # Number of attention heads
    num_kv_heads: int   # Number of KV heads
    head_dim: int       # Dimension per head
    lora_configs: dict[str, lora.LoRAConfig] = dataclasses.field(default_factory=dict)
```

### 2. Model Variants in `get_config()`

```python
def get_config(variant: Variant) -> Config:
    if variant == "gemma_300m":
        # 311M params
        return Config(
            width=1024,      # Hidden dimension
            depth=18,        # 18 layers
            mlp_dim=4096,    # MLP hidden size
            num_heads=8,     # 8 attention heads
            num_kv_heads=1,  # 1 KV head
            head_dim=256,    # 256 dims per head
        )
    
    if variant == "gemma_2b":
        return Config(
            width=2048,      # Hidden dimension
            depth=18,        # 18 layers
            mlp_dim=16_384,  # MLP hidden size
            num_heads=8,     # 8 attention heads
            num_kv_heads=1,  # 1 KV head
            head_dim=256,    # 256 dims per head
        )
```

### 3. Pi0 Model Configuration in `src/openpi/models/pi0.py`

```python
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"      # Expert 0
    action_expert_variant: _gemma.Variant = "gemma_300m" # Expert 1
```

## Parameter Count Calculation

### **Expert 0 (PaliGemma): gemma_2b**

**Config:**
- `width = 2048` (hidden dimension)
- `depth = 18` (layers)
- `mlp_dim = 16_384` (MLP hidden size)
- `num_heads = 8`
- `num_kv_heads = 1`
- `head_dim = 256`

**Parameter Count Breakdown:**

1. **Embedding Layer:**
   - `vocab_size × width = 257,152 × 2048 ≈ 526M params`

2. **Per Transformer Layer (18 layers):**
   - **Attention:**
     - Query/Key/Value projections: `3 × width × width = 3 × 2048 × 2048 = 12.6M`
     - Output projection: `width × width = 2048 × 2048 = 4.2M`
     - Total attention per layer: `16.8M`
   
   - **MLP:**
     - Up projection: `width × mlp_dim = 2048 × 16,384 = 33.6M`
     - Down projection: `mlp_dim × width = 16,384 × 2048 = 33.6M`
     - Total MLP per layer: `67.2M`
   
   - **Layer Norms:**
     - Pre-attention norm: `width = 2048`
     - Pre-MLP norm: `width = 2048`
     - Total norms per layer: `4,096`
   
   - **Total per layer: `84M`**
   - **Total for 18 layers: `84M × 18 = 1.51B`**

3. **Final Layer Norm:**
   - `width = 2048` params

**Total PaliGemma Expert: ~2.04B parameters**

### **Expert 1 (Action): gemma_300m**

**Config:**
- `width = 1024` (hidden dimension)
- `depth = 18` (layers)
- `mlp_dim = 4096` (MLP hidden size)
- `num_heads = 8`
- `num_kv_heads = 1`
- `head_dim = 256`

**Parameter Count Breakdown:**

1. **Embedding Layer:**
   - `vocab_size × width = 257,152 × 1024 ≈ 263M params`

2. **Per Transformer Layer (18 layers):**
   - **Attention:**
     - Query/Key/Value projections: `3 × width × width = 3 × 1024 × 1024 = 3.1M`
     - Output projection: `width × width = 1024 × 1024 = 1.05M`
     - Total attention per layer: `4.15M`
   
   - **MLP:**
     - Up projection: `width × mlp_dim = 1024 × 4096 = 4.2M`
     - Down projection: `mlp_dim × width = 4096 × 1024 = 4.2M`
     - Total MLP per layer: `8.4M`
   
   - **Layer Norms:**
     - Pre-attention norm: `width = 1024`
     - Pre-MLP norm: `width = 1024`
     - Total norms per layer: `2,048`
   
   - **Total per layer: `12.55M`**
   - **Total for 18 layers: `12.55M × 18 = 226M`**

3. **Final Layer Norm:**
   - `width = 1024` params

**Total Action Expert: ~489M parameters**

## **Total Model Parameters**

```
PaliGemma Expert (Expert 0): ~2.04B parameters
Action Expert (Expert 1):    ~489M parameters
Vision Encoder (SigLIP):     ~400M parameters (So400m/14)
Additional Projections:      ~few M parameters

Total Pi0 Model:             ~2.93B parameters
```

## **How I Knew the Parameter Counts**

1. **Comments in Code:**
   ```python
   if variant == "gemma_300m":
       # 311M params  # This comment shows the expected count
   ```

2. **Standard Transformer Architecture:**
   - The parameter count follows standard transformer formulas
   - Each layer has attention + MLP + norms
   - The config values directly determine parameter counts

3. **Model Variants:**
   - `gemma_300m` ≈ 311M (as noted in comment)
   - `gemma_2b` ≈ 2B (standard 2B parameter model)
   - These are well-known model sizes in the literature

4. **Architecture Analysis:**
   - The configs define the exact dimensions
   - Standard transformer parameter formulas apply
   - The expert mechanism uses the same architecture for both experts

## **Key Insights**

1. **PaliGemma Expert (2B)**: Much larger, handles vision-language understanding
2. **Action Expert (300M)**: Smaller, specialized for control tasks
3. **Total Model**: ~2.93B parameters, which is reasonable for a VLA model
4. **Efficiency**: The action expert is much smaller, reducing computational cost for control tasks

This architecture allows the model to leverage a large pre-trained vision-language model while adding a smaller, specialized action generation component. 