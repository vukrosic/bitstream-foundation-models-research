# Byte Latent Transformer (BLT) Research

YouTube video - https://youtu.be/SLtLP6J9xTk

Bilibili video - https://www.bilibili.com/video/BV1BjtRzMER3/

## Research Questions & Future Directions

### Research Questions Ideas
1. **Optimal Entropy Thresholds**: How do different entropy thresholds affect model performance vs computational efficiency? Can we learn adaptive thresholds during training?

2. **Cross-Attention vs Pooling**: What's the performance gap between simple mean pooling and full cross-attention in local encoder/decoder? Is the computational cost worth it?

3. **Patch Size Distribution**: How does the distribution of patch sizes affect downstream task performance? Are there optimal distributions for different types of text?

4. **Multilingual Efficiency**: Does BLT's byte-level approach provide better efficiency gains for non-English languages with complex tokenization?

5. **Scaling Laws**: How do BLT scaling laws compare to traditional token-based transformers? Do the efficiency gains hold at larger scales?

### Future Research Ideas

#### Architecture Improvements
- **Hierarchical Patching**: Multi-level patching (bytes → sub-patches → patches) for better granularity control
- **Learned Patching**: Replace entropy-based patching with learned segmentation boundaries
- **Adaptive Local Models**: Dynamic local encoder/decoder complexity based on patch difficulty
- **Memory-Efficient Attention**: Implement linear attention variants for the latent transformer

#### Training Innovations
- **Curriculum Learning**: Start with fixed patches, gradually transition to entropy-based patching
- **Multi-Task Learning**: Joint training on byte prediction, patch boundary prediction, and downstream tasks
- **Distillation**: Use larger models to teach better patching strategies to smaller entropy models
- **Reinforcement Learning**: Optimize patching strategy directly for downstream task performance

#### Applications & Evaluation
- **Code Generation**: Evaluate BLT on programming languages where byte-level understanding matters
- **Multilingual Benchmarks**: Comprehensive evaluation across languages with different writing systems
- **Long Context**: Test BLT's efficiency on very long sequences (100K+ tokens)
- **Streaming Applications**: Real-time text processing with dynamic patching

#### Theoretical Analysis
- **Information Theory**: Analyze the information-theoretic properties of entropy-based patching
- **Compression Bounds**: Theoretical limits of BLT's compression capabilities
- **Convergence Analysis**: Study training dynamics of the three-component architecture

---

## Overview

Arxiv - https://arxiv.org/pdf/2412.09871

This repository implements the **Byte Latent Transformer (BLT)**, a novel architecture that processes raw bytes through dynamic patching, achieving better efficiency than traditional token-based transformers.

### Key Innovation: Entropy-Based Dynamic Patching

Unlike fixed tokenization, BLT groups bytes into variable-length "patches" based on prediction uncertainty:
- **High entropy** (unpredictable text) → smaller patches → more computational focus
- **Low entropy** (predictable text) → larger patches → computational efficiency

## Architecture

The BLT consists of three main components:

```
Raw Bytes → [Local Encoder] → Patch Embeddings → [Latent Transformer] → Next Patch Embeddings → [Local Decoder] → Predicted Bytes
```

### 1. Local Encoder
- Small transformer that converts byte patches to fixed-size embeddings
- Uses cross-attention or mean pooling to create patch representations

### 2. Latent Transformer  
- Main transformer operating on patch embeddings (repurposed from standard LLM)
- Predicts the next patch embedding in the sequence

### 3. Local Decoder
- Small transformer that converts patch embeddings back to byte sequences
- Uses teacher forcing during training for autoregressive byte generation

## Implementation

### Files
- `llm_blt.py` - Complete BLT implementation with entropy-based patching
- `entropy_model.pt` - Pre-trained ByteLM for entropy calculation (generated during first run)

### Key Classes
- `BLT` - Main model combining all three components
- `EntropyPatcher` - Dynamic patching based on next-byte prediction entropy
- `ByteLM` - Small language model for entropy calculation
- `LocalEncoder/LocalDecoder` - Byte-to-patch and patch-to-byte conversion
- `LatentTransformer` - Core transformer operating on patch embeddings

## Usage

### Training
```bash
python llm_blt.py
```

The script will:
1. Train a ByteLM entropy model (if not exists)
2. Compare different patching strategies
3. Select optimal entropy threshold
4. Train the full BLT model

### Configuration
Key hyperparameters in `ModelConfig`:
- `patch_size`: Fixed patch size for baseline comparison
- `local_d_model`: Dimension for local encoder/decoder
- `local_n_layers`: Depth of local transformers
- `max_patch_len`: Maximum bytes per patch

## Results & Analysis

### Patch Statistics
The entropy-based patcher automatically adapts to text complexity:
- **Predictable sequences**: "...from the United States of Ameri" → large patches
- **Unpredictable sequences**: "The first word of this sentence is" → small patches

### Efficiency Gains
- Reduced sequence length (patches vs bytes)
- Adaptive computation based on text difficulty
- Better compression of predictable text patterns

## Technical Details

### Entropy Calculation
```python
def calculate_entropy(logits):
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy
```

### Dynamic Patching Algorithm
1. Use pre-trained ByteLM to predict next-byte probabilities
2. Calculate entropy of each prediction
3. Start new patch when entropy exceeds threshold
4. Prevent patches from becoming too long (max 16 bytes)

### Training Process
1. **Phase 1**: Train ByteLM entropy model on byte sequences
2. **Phase 2**: Use frozen ByteLM to create dynamic patches
3. **Phase 3**: Train BLT end-to-end on patch sequences

## Dependencies

```bash
pip install torch transformers datasets tqdm numpy
```
