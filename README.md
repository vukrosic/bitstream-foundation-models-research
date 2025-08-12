# Entropy-Based Patching System

A complete implementation of entropy-based data patching for language model training, inspired by Byte Latent Transformer (BLT) research.

**Research Videos:**
- YouTube: https://youtu.be/SLtLP6J9xTk
- Bilibili: https://www.bilibili.com/video/BV1BjtRzMER3/

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