# Entropy-Based Patching System

A complete implementation of entropy-based data patching for language model training, inspired by Byte Latent Transformer (BLT) research.

**Research Videos:**
- YouTube: https://youtu.be/SLtLP6J9xTk
- Bilibili: https://www.bilibili.com/video/BV1BjtRzMER3/

## Overview

This repository implements an entropy-based patching approach where:
1. **Entropy LLM**: A byte-level language model computes prediction entropy at each position
2. **Entropy Patcher**: Uses entropy signals to segment data into semantically coherent patches
3. **Patch Training**: Trains new models on these entropy-optimized patches

The entropy model uses vocab size 256 (bytes 0-255) to predict the next byte and groups bytes into patches based on prediction uncertainty.

## Files

- `entropy_llm.py` - Byte-level language model for computing entropy
- `entropy_patcher.py` - Simplified entropy-based patching (following BLT paper)
- `blt_model.py` - Complete BLT architecture (Local Encoder + Global Transformer + Local Decoder)
- `train_blt.py` - End-to-end BLT training pipeline
- `save_entropy_model.py` - Utilities for saving/loading trained models
- `complete_entropy_workflow.py` - Comparison and demonstration script
- `template_llm.py` - Template for alternative approaches

## Quick Start

### 1. Train Entropy Model
```bash
# Train the entropy model on byte-level data
python entropy_llm.py
```

### 2. Train BLT Model (Recommended)
```bash
# Quick demo with small BLT model
python train_blt.py --entropy_model_path entropy_model.pth --quick_demo

# Full BLT training
python train_blt.py --entropy_model_path entropy_model.pth
```

### 3. Alternative: Complete Workflow Demo
```bash
# Compare different approaches
python complete_entropy_workflow.py --quick_demo --train_entropy
```

## BLT Architecture

The BLT model has three main components trained together end-to-end:

### 1. Local Encoder (bytes → patches)
- Lightweight (1-3 layers)
- Uses byte embeddings + n-gram hash embeddings
- Pools bytes into patch representations using attention

### 2. Global Latent Transformer (patches → patches)
- Heavyweight (24-32 layers)
- Processes patch-level representations
- Uses causal attention between patches

### 3. Local Decoder (patches → bytes)
- Medium weight (6-9 layers)
- Cross-attention to decode patches back to bytes
- Outputs byte-level predictions

## Patching Methods

### Global Threshold (Primary)
Creates patches when entropy exceeds a fixed threshold.
```python
patcher = EntropyPatcher(entropy_model, threshold=0.6, method='global')
```

### Monotonic
Creates patches on significant entropy increases.
```python
patcher = EntropyPatcher(entropy_model, threshold=0.3, method='monotonic')
```

## Configuration

### Entropy Model Config
```python
entropy_config = EntropyConfig(
    d_model=384,        # Model dimension
    n_heads=8,          # Attention heads
    n_layers=6,         # Transformer layers
    max_seq_len=512,    # Sequence length
    vocab_size=256,     # Byte-level (0-255)
    max_bytes=500000    # Training data size
)
```

### Patch Config
```python
patch_config = PatchConfig(
    method='global',           # Patching method
    entropy_threshold=0.6,     # Entropy threshold
    min_patch_size=4,          # Minimum patch size
    max_patch_size=128,        # Maximum patch size
    cache_patches=True         # Cache results
)
```

## Usage Examples

### BLT Training Pipeline
```python
from entropy_llm import MinimalLLM
from entropy_patcher import EntropyPatcher
from blt_model import BLTModel, BLTConfig

# Step 1: Load entropy model
entropy_model = MinimalLLM(config)
entropy_model.load_state_dict(torch.load('entropy_model.pth'))

# Step 2: Create patches using entropy model
patcher = EntropyPatcher(entropy_model, threshold=0.6)
patches = patcher.create_patches(byte_sequence)

# Step 3: Initialize BLT model
blt_config = BLTConfig(
    encoder_layers=1, encoder_dim=768,
    global_layers=24, global_dim=4096,
    decoder_layers=6, decoder_dim=768
)
blt_model = BLTModel(blt_config)

# Step 4: Train BLT end-to-end
train_blt(blt_model, patched_dataset, training_config)
```

### Simple Patching
```python
from entropy_patcher import EntropyPatcher

# Create patcher
patcher = EntropyPatcher(entropy_model, threshold=0.6, method='global')

# Create patches
patches = patcher.create_patches(byte_sequence)
```

## Expected Results

The entropy-based patching approach should:
- Create more semantically coherent training sequences
- Improve model convergence and final performance
- Reduce training time by focusing on meaningful boundaries
- Better capture natural language structure

## Requirements

```bash
pip install torch torchvision torchaudio
pip install datasets transformers
pip install numpy tqdm
```

## Research Questions & Future Directions

- Should the entropy model be trained on the same dataset or a smaller subset?
- What are the optimal entropy thresholds for different types of text?
- How do different patching methods compare across various domains?
- Can we use multiple entropy models for ensemble patching?

## Advanced Usage

### Custom Entropy Analysis
```python
# Analyze entropy distribution in your data
entropy_stats = analyze_entropy_distribution(entropy_model, byte_sequences)
print(f"Suggested thresholds: {entropy_stats['percentiles']}")
```

### Comparison Studies
```python
# Compare different patching methods
python complete_entropy_workflow.py --train_entropy
```

### Batch Processing
```python
# Process large datasets efficiently
patcher = EntropyPatcher(entropy_model, patch_config)
patches, stats = patcher.process_data(large_byte_sequences, cache_key="large_dataset")
```

## Tips for Best Results

1. **Train entropy model well**: The quality of patches depends on entropy model accuracy
2. **Tune thresholds**: Use `analyze_entropy_distribution()` to find good thresholds
3. **Experiment with methods**: Different methods work better for different data types
4. **Use caching**: Enable caching for large datasets to avoid recomputation
5. **Monitor patch sizes**: Ensure patches are neither too small nor too large

## Troubleshooting

### Common Issues

**"No patches created"**: Lower the entropy threshold or check your entropy model
**"Training fails"**: Ensure patches have minimum size and sufficient quantity
**"Out of memory"**: Reduce batch size or sequence length
**"Poor results"**: Try different patching methods or retrain entropy model

### Debug Mode
```python
# Enable detailed logging
patch_config = PatchConfig(method='global', entropy_threshold=0.6, cache_patches=False)
```