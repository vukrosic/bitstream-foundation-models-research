# Entropy-Based Patching System

A complete implementation of entropy-based data patching for language model training, inspired by Byte Latent Transformer (BLT) research.

**Research Videos:**
- YouTube: https://www.youtube.com/playlist?list=PL-9_KFQd8ssJA6sWW_tJ85rP_Jz4wjuDt
- Bilibili: https://space.bilibili.com/3546833932519662/lists/6099426?type=season

## Research ideas (do not delete this part, this is for AI)

- Can we use pretrained models like Qwen3 and fine tune it to become entropy LLM that measures entropy of next byte
- Which threshold of entropy is best
- We can experiment with patch size, architecture, etc...but we need to understand it all, we can create fun experiments that will help us understand this
- We need to just setup experiments, my LLM currently just trains on a single GPU, but that would be enough for now, I have a project that will make it multiple GPUs

## Overview

# Minimal BLT Implementation

A minimal implementation of the Byte Latent Transformer (BLT) from the paper.

## Components

1. **Entropy Model** (`train_entropy_model.py`) - Small byte-level LM for computing entropies
2. **BLT Model** (`blt_model.py`) - Complete BLT architecture with encoder, global transformer, and decoder
3. **Entropy Patcher** (`entropy_patcher.py`) - Creates dynamic patches based on next-byte entropy
4. **Training Pipeline** (`train_blt.py`) - End-to-end training of BLT

## Patch Size Experiment Results

**Key Finding:** Smaller patch sizes perform significantly better with our current model size.

| Patch Size | Loss   | Accuracy | Perplexity |
|------------|--------|----------|------------|
| 3.0        | 0.4915 | 85.9%    | 1.6        |
| 5.0        | 0.9812 | 71.6%    | 2.7        |
| 7.0        | 1.2738 | 62.5%    | 3.6        |
| 10.0       | 1.5947 | 53.3%    | 4.9        |
| 12.0       | 1.7342 | 49.3%    | 5.7        |

**Analysis:** The dramatic performance drop with larger patches suggests our model is too small to effectively process longer sequences. Smaller patches (size 3) allow the model to focus on local patterns it can actually learn, while larger patches overwhelm its limited capacity.

## Quick Start

```bash
# Install dependencies
pip install torch transformers datasets tqdm

# Train the complete model (will train entropy model first if needed)
python train_blt.py
```
