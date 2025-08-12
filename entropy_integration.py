#!/usr/bin/env python3
"""
Entropy-based Patching Integration Script

This script demonstrates how to:
1. Load a trained entropy LLM
2. Use it to create entropy-based patches from data
3. Train a new model on the patched data

Usage:
    python entropy_integration.py --entropy_model_path entropy_model.pth
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import os
import time
from typing import List, Tuple

# Import your modules
from entropy_llm import MinimalLLM, ModelConfig as EntropyConfig, load_and_cache_data
from entropy_patcher import EntropyPatcher, PatchConfig, PatchDataset, print_patch_statistics
from template_llm import ModelConfig as TemplateConfig, train_model, evaluate_model

def load_trained_entropy_model(model_path: str, config: EntropyConfig) -> MinimalLLM:
    """Load a trained entropy model from disk."""
    print(f"📂 Loading entropy model from {model_path}")
    
    model = MinimalLLM(config)
    
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        print(f"✅ Successfully loaded entropy model")
    else:
        print(f"⚠️  Model file not found. Using randomly initialized model for demonstration.")
        print(f"   Train your entropy model first using: python entropy_llm.py")
    
    return model

def convert_bytes_to_sequences(bytes_data: List[int], max_seq_len: int = 2048) -> List[List[int]]:
    """Convert flat byte list into sequences for processing."""
    sequences = []
    
    # Split into manageable sequences
    for i in range(0, len(bytes_data), max_seq_len):
        seq = bytes_data[i:i + max_seq_len]
        if len(seq) >= 10:  # Only keep sequences with reasonable length
            sequences.append(seq)
    
    print(f"📊 Created {len(sequences)} byte sequences from {len(bytes_data)} total bytes")
    return sequences

def compare_patching_methods(entropy_model: MinimalLLM, byte_sequences: List[List[int]]) -> None:
    """Compare different patching methods on the same data."""
    print(f"\n🔬 COMPARING PATCHING METHODS")
    print(f"=" * 60)
    
    methods = [
        ('global', 0.6),
        ('percentile', 75.0),
        ('adaptive', 0.5),
        ('monotonic', 0.3)
    ]
    
    # Use a subset for comparison
    test_sequences = byte_sequences[:5] if len(byte_sequences) > 5 else byte_sequences
    
    for method, threshold in methods:
        print(f"\n🧪 Testing method: {method} (threshold: {threshold})")
        
        patch_config = PatchConfig(
            method=method,
            entropy_threshold=threshold,
            min_patch_size=4,
            max_patch_size=64,
            cache_patches=False  # Don't cache for comparison
        )
        
        patcher = EntropyPatcher(entropy_model, patch_config)
        patches, stats = patcher.process_data(test_sequences, cache_key=f"compare_{method}")
        
        print(f"   Patches created: {stats['total_patches']}")
        print(f"   Avg patch size: {stats['avg_patch_size']:.1f}")
        print(f"   Avg entropy: {stats['avg_entropy']:.3f}")

def create_entropy_patches(entropy_model_path: str, data_config: EntropyConfig) -> Tuple[List[List[int]], dict]:
    """Create entropy-based patches from data."""
    
    # Load entropy model
    entropy_model = load_trained_entropy_model(entropy_model_path, data_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    entropy_model = entropy_model.to(device)
    
    # Load the same data that was used for entropy training
    print(f"\n📊 Loading data for patching...")
    texts, bytes_data = load_and_cache_data(data_config)
    
    # Convert to sequences
    byte_sequences = convert_bytes_to_sequences(bytes_data, max_seq_len=1024)
    
    # Compare different methods (optional)
    if len(byte_sequences) > 0:
        compare_patching_methods(entropy_model, byte_sequences)
    
    # Create patcher (simplified)
    patcher = EntropyPatcher(entropy_model, threshold=0.6, method='global')
    
    # Process data into patches
    all_patches = []
    for byte_sequence in byte_sequences:
        patches = patcher.create_patches(byte_sequence)
        all_patches.extend(patches)
    
    # Create simple stats
    stats = {
        'total_patches': len(all_patches),
        'avg_patch_size': sum(len(p) for p in all_patches) / len(all_patches) if all_patches else 0
    }
    
    # Print statistics
    print_patch_statistics(stats)
    
    # Flatten patches for training
    flat_patches = []
    for seq_patches in all_patches:
        flat_patches.extend(seq_patches)
    
    print(f"📦 Created {len(flat_patches)} total patches for training")
    
    return flat_patches, stats

def train_on_patches(patches: List[List[int]], stats: dict) -> None:
    """Train a model on the entropy-based patches."""
    print(f"\n🚀 TRAINING MODEL ON ENTROPY PATCHES")
    print(f"=" * 60)
    
    # Create patch dataset
    patch_dataset = PatchDataset(patches, seq_len=256)  # Smaller seq_len for patches
    
    if len(patch_dataset) == 0:
        print("❌ No valid patches for training!")
        return
    
    # Split into train/val
    val_size = max(1, len(patch_dataset) // 10)
    train_size = len(patch_dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        patch_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create config for patch-based training
    config = TemplateConfig(
        d_model=256,  # Smaller model for patches
        n_heads=8,
        n_layers=4,
        d_ff=1024,
        batch_size=16,
        max_steps=2000,  # Fewer steps for demonstration
        max_seq_len=256,
        vocab_size=256,  # Byte-level
        eval_every=200
    )
    
    print(f"📋 Patch Training Configuration:")
    print(f"   Architecture: {config.d_model}d, {config.n_layers}L, {config.n_heads}H")
    print(f"   Training patches: {len(train_dataset)}")
    print(f"   Validation patches: {len(val_dataset)}")
    print(f"   Avg patch size: {stats['avg_patch_size']:.1f} bytes")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                             shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, 
                           shuffle=False, num_workers=2)
    
    # Train model (using the template_llm training function)
    try:
        model, final_metrics = train_model(config, train_loader, val_loader)
        
        print(f"\n🎉 PATCH-BASED TRAINING COMPLETED!")
        print(f"🏆 Results on entropy patches:")
        print(f"   Final Loss: {final_metrics['val_loss']:.4f}")
        print(f"   Final Accuracy: {final_metrics['val_accuracy']:.4f}")
        print(f"   Final Perplexity: {final_metrics['val_perplexity']:.2f}")
        
        # Save the patch-trained model
        model_save_path = "patch_trained_model.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"💾 Saved patch-trained model to {model_save_path}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print(f"   This might be due to insufficient data or other issues.")

def main():
    parser = argparse.ArgumentParser(description="Entropy-based patching integration")
    parser.add_argument("--entropy_model_path", type=str, default="entropy_model.pth",
                       help="Path to trained entropy model")
    parser.add_argument("--method", type=str, default="global", 
                       choices=['global', 'monotonic', 'adaptive', 'percentile'],
                       help="Patching method to use")
    parser.add_argument("--threshold", type=float, default=0.6,
                       help="Entropy threshold for patching")
    parser.add_argument("--min_patch_size", type=int, default=4,
                       help="Minimum patch size")
    parser.add_argument("--max_patch_size", type=int, default=128,
                       help="Maximum patch size")
    parser.add_argument("--skip_training", action="store_true",
                       help="Skip training step, only create patches")
    
    args = parser.parse_args()
    
    print(f"🎯 ENTROPY-BASED PATCHING INTEGRATION")
    print(f"=" * 60)
    print(f"Entropy model: {args.entropy_model_path}")
    print(f"Method: {args.method}")
    print(f"Threshold: {args.threshold}")
    print(f"Patch size: {args.min_patch_size}-{args.max_patch_size}")
    
    # Configuration for entropy model (should match your training config)
    entropy_config = EntropyConfig(
        d_model=384,
        n_heads=8,
        n_layers=6,
        d_ff=1536,
        max_seq_len=512,
        vocab_size=256,
        num_documents=500,  # Smaller for demonstration
        max_bytes=100000
    )
    
    # Configuration for patching
    patch_config = PatchConfig(
        method=args.method,
        entropy_threshold=args.threshold,
        min_patch_size=args.min_patch_size,
        max_patch_size=args.max_patch_size,
        batch_size=32,
        cache_patches=True
    )
    
    try:
        # Step 1: Create entropy-based patches
        start_time = time.time()
        patches, stats = create_entropy_patches(args.entropy_model_path, 
                                               entropy_config, patch_config)
        patch_time = time.time() - start_time
        
        print(f"⏱️ Patching completed in {patch_time:.1f} seconds")
        
        # Step 2: Train on patches (optional)
        if not args.skip_training and len(patches) > 0:
            train_start = time.time()
            train_on_patches(patches, stats)
            train_time = time.time() - train_start
            print(f"⏱️ Training completed in {train_time:.1f} seconds")
        
        total_time = time.time() - start_time
        print(f"\n🎉 TOTAL PROCESS COMPLETED IN {total_time/60:.1f} MINUTES")
        
    except Exception as e:
        print(f"❌ Process failed: {e}")
        print(f"\nMake sure you have:")
        print(f"   1. Trained entropy model at {args.entropy_model_path}")
        print(f"   2. Required dependencies installed")
        print(f"   3. Sufficient GPU memory")

if __name__ == "__main__":
    main()