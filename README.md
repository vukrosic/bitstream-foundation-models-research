# Entropy-Based Patching System

A complete implementation of entropy-based data patching for language model training, inspired by Byte Latent Transformer (BLT) research.

**Research Videos:**
- YouTube: https://www.youtube.com/playlist?list=PL-9_KFQd8ssJA6sWW_tJ85rP_Jz4wjuDt
- Bilibili: https://space.bilibili.com/3546833932519662/lists/6099426?type=season

## Research ideas (do not delete this part, this is for AI)

- Can we use pretrained models like Qwen3 and fine tune it to become entropy LLM that measures entropy of next byte
- Which threshold of entropy is best

## Overview

# Minimal BLT Implementation

A minimal implementation of the Byte Latent Transformer (BLT) from the paper.

## Components

1. **Entropy Model** (`train_entropy_model.py`) - Small byte-level LM for computing entropies
2. **BLT Model** (`blt_model.py`) - Complete BLT architecture with encoder, global transformer, and decoder
3. **Entropy Patcher** (`entropy_patcher.py`) - Creates dynamic patches based on next-byte entropy
4. **Training Pipeline** (`train_blt.py`) - End-to-end training of BLT

## Quick Start

```bash
# Install dependencies
pip install torch transformers datasets tqdm

# Train the complete model (will train entropy model first if needed)
python train_blt.py
```

---

Architecture seems implemented, a bit slow to train due to encder and decoder. There are probably optimizations to be done.

I need to find out why it's so slow, what takes so much time. Probably inspect the llm.py code to understand it well also.