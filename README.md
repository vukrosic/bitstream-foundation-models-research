# Entropy-Based Patching System

A complete implementation of entropy-based data patching for language model training, inspired by Byte Latent Transformer (BLT) research.

**I recorded multiple videos while working on this project:**
- YouTube: https://www.youtube.com/playlist?list=PL-9_KFQd8ssJA6sWW_tJ85rP_Jz4wjuDt
- Bilibili: https://space.bilibili.com/3546833932519662/lists/6099426?type=season

- It trains but need optimizations. Experiments done on llm.py, llm_large.py needs more optimization.

First run

```
python entropy_llm.py
```

then run

```
python create_patches.py
```

and finally

```
python llm.py
```

Works on Google Colab, you will likely not need to reduce size of the model. I used RTX 4090.

## Experiment Status

**Current Model (`llm.py`):** All experiments completed on the smaller model
- ✅ Patch Size Experiment - Results show smaller patches (size 3) work best
- ✅ Entropy Threshold Experiment - Lower thresholds (0.50) optimal for small model
- ✅ Model Size Experiment - Confirmed larger models perform better

**Large Model (`llm_large.py`):** Performance optimization needed
- 🔄 Future experiments will be conducted on this larger model
- 📈 Expected to handle larger patch sizes and higher entropy thresholds better
- 🎯 Will validate if model capacity was the limiting factor in previous experiments
- ⚠️ **Performance Note:** Currently utilizes compute and memory on RTX 4090 but trains very slowly - needs optimizations
- 🔧 **Optimization Needed:** Encoder and decoder could be utilizing significantly less GPU compute than available

## Overview

# Minimal BLT Implementation

A minimal implementation of the Byte Latent Transformer (BLT) from the paper.

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

## Entropy Threshold Experiment Results

**Key Finding:** Lower entropy thresholds work best, confirming our small model hypothesis.

| Threshold | Avg Size | Loss   | Accuracy | Perplexity |
|-----------|----------|--------|----------|------------|
| 0.50      | 2.2      | 0.3490 | 90.0%    | 1.4        |
| 1.00      | 3.5      | 0.6472 | 81.6%    | 1.9        |
| 1.50      | 5.9      | 1.1035 | 67.6%    | 3.0        |
| 2.00      | 10.5     | 1.5891 | 54.1%    | 4.9        |
| 2.50      | 24.3     | 2.1463 | 38.5%    | 8.6        |
| 3.00      | 131.0    | 3.5250 | 24.2%    | 34.0       |

**Analysis:** Threshold 0.50 produces the best results with very small patches (avg 2.2 bytes). This confirms our model is too small for complex patterns - it performs best when processing tiny, simple chunks. The exponential performance degradation with higher thresholds shows the model can't handle the complexity of longer sequences.

## Model Size Experiment Results

**Key Finding:** Larger models perform significantly better with optimal patch settings.

| Model  | d_model | Layers | Loss   | Accuracy | Perplexity |
|--------|---------|--------|--------|----------|------------|
| Tiny   | 256     | 4      | 0.4648 | 87.1%    | 1.6        |
| Small  | 384     | 6      | 0.3874 | 89.1%    | 1.5        |
| Medium | 512     | 8      | 0.3323 | 90.4%    | 1.4        |

**Analysis:** The consistent improvement with model size confirms our hypothesis - the tiny model was indeed the bottleneck. Medium model achieves 28% lower loss than tiny, suggesting even larger models might perform better. This validates that model capacity, not just patch size, is crucial for performance.

## Possible research ideas

- Can we use pretrained models like Qwen3 and fine tune it to become entropy LLM that measures entropy of next byte
- Which threshold of entropy is best
- Experiment with patch size, architecture, etc
- Multiple GPUs