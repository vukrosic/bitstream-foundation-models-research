#!/usr/bin/env python3
"""
Complete Entropy-Based Patching Workflow

This script demonstrates the complete workflow:
1. Train entropy LLM (or load existing)
2. Use entropy LLM to create patches
3. Train a new model on patches
4. Compare results

Usage:
    python complete_entropy_workflow.py [--train_entropy] [--quick_demo]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import os
import time
import numpy as np
from typing import List, Tuple, Dict, Any

# Import all necessary modules
from entropy_llm import (
    MinimalLLM, ModelConfig as EntropyConfig, 
    load_and_cache_data, ByteDataset, train_model as train_entropy_model,
    set_seed, evaluate_model
)
from entropy_patcher import (
    EntropyPatcher, PatchConfig, PatchDataset, print_patch_statistics
)
from template_llm import ModelConfig as TemplateConfig, train_model as train_template_model
from save_entropy_model import save_entropy_model, load_entropy_model

def quick_entropy_training(config: EntropyConfig) -> MinimalLLM:
    """Train a small entropy model quickly for demonstration."""
    print(f"🚀 Quick entropy model training...")
    
    # Use smaller config for speed
    quick_config = EntropyConfig(
        d_model=128,
        n_heads=4,
        n_layers=3,
        d_ff=512,
        batch_size=16,
        max_steps=1000,
        max_seq_len=256,
        num_documents=100,
        max_bytes=50000,
        eval_every=200
    )
    
    # Load data
    texts, bytes_data = load_and_cache_data(quick_config)
    dataset = ByteDataset(bytes_data, quick_config.max_seq_len)
    
    # Split data
    val_size = len(dataset) // 10
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=quick_config.batch_size, 
                             shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=quick_config.batch_size, 
                           shuffle=False, num_workers=2)
    
    # Train
    model, metrics = train_entropy_model(quick_config, train_loader, val_loader)
    
    # Save
    save_entropy_model(model, quick_config, "quick_entropy_model.pth", "quick_entropy_config.pth")
    
    return model, quick_config

def analyze_entropy_distribution(entropy_model: MinimalLLM, byte_sequences: List[List[int]]) -> Dict[str, Any]:
    """Analyze the distribution of entropies in the data."""
    print(f"\n🔍 ANALYZING ENTROPY DISTRIBUTION")
    print(f"=" * 50)
    
    # Create a simple patcher to compute entropies
    patch_config = PatchConfig(method='global', entropy_threshold=1.0)  # High threshold to avoid splitting
    patcher = EntropyPatcher(entropy_model, patch_config)
    
    # Compute entropies for a subset of data
    test_sequences = byte_sequences[:10] if len(byte_sequences) > 10 else byte_sequences
    all_entropies = patcher.compute_entropies_batch(test_sequences)
    
    # Flatten all entropies
    flat_entropies = []
    for entropies in all_entropies:
        flat_entropies.extend(entropies)
    
    if not flat_entropies:
        print("❌ No entropies computed!")
        return {}
    
    # Compute statistics
    entropies_array = np.array(flat_entropies)
    stats = {
        'count': len(flat_entropies),
        'mean': np.mean(entropies_array),
        'std': np.std(entropies_array),
        'min': np.min(entropies_array),
        'max': np.max(entropies_array),
        'percentiles': {
            '25': np.percentile(entropies_array, 25),
            '50': np.percentile(entropies_array, 50),
            '75': np.percentile(entropies_array, 75),
            '90': np.percentile(entropies_array, 90),
            '95': np.percentile(entropies_array, 95)
        }
    }
    
    print(f"Entropy Statistics:")
    print(f"  Count: {stats['count']:,}")
    print(f"  Mean: {stats['mean']:.3f}")
    print(f"  Std: {stats['std']:.3f}")
    print(f"  Range: {stats['min']:.3f} - {stats['max']:.3f}")
    print(f"  Percentiles:")
    for p, v in stats['percentiles'].items():
        print(f"    {p}th: {v:.3f}")
    
    # Suggest good thresholds
    print(f"\n💡 Suggested thresholds:")
    print(f"  Conservative (few patches): {stats['percentiles']['90']:.3f}")
    print(f"  Moderate (balanced): {stats['percentiles']['75']:.3f}")
    print(f"  Aggressive (many patches): {stats['percentiles']['50']:.3f}")
    
    return stats

def compare_training_approaches(byte_sequences: List[List[int]], entropy_model: MinimalLLM, 
                              entropy_config: EntropyConfig) -> Dict[str, Any]:
    """Compare training on original data vs entropy patches."""
    print(f"\n⚖️  COMPARING TRAINING APPROACHES")
    print(f"=" * 60)
    
    results = {}
    
    # 1. Train on original byte sequences (baseline)
    print(f"\n1️⃣ Training baseline model on original byte sequences...")
    
    # Flatten byte sequences for baseline training
    flat_bytes = []
    for seq in byte_sequences[:20]:  # Use subset for speed
        flat_bytes.extend(seq)
    
    if len(flat_bytes) > 10000:
        flat_bytes = flat_bytes[:10000]  # Limit for demo
    
    baseline_dataset = ByteDataset(flat_bytes, seq_len=128)
    
    if len(baseline_dataset) > 10:
        val_size = max(1, len(baseline_dataset) // 10)
        train_size = len(baseline_dataset) - val_size
        
        baseline_train, baseline_val = torch.utils.data.random_split(
            baseline_dataset, [train_size, val_size], 
            generator=torch.Generator().manual_seed(42)
        )
        
        baseline_config = TemplateConfig(
            d_model=128, n_heads=4, n_layers=3, d_ff=512,
            batch_size=8, max_steps=500, max_seq_len=128,
            vocab_size=256, eval_every=100
        )
        
        baseline_train_loader = DataLoader(baseline_train, batch_size=8, shuffle=True)
        baseline_val_loader = DataLoader(baseline_val, batch_size=8, shuffle=False)
        
        try:
            baseline_model, baseline_metrics = train_template_model(
                baseline_config, baseline_train_loader, baseline_val_loader
            )
            results['baseline'] = baseline_metrics
            print(f"✅ Baseline training completed: Loss {baseline_metrics['val_loss']:.4f}")
        except Exception as e:
            print(f"❌ Baseline training failed: {e}")
            results['baseline'] = None
    
    # 2. Train on entropy patches
    print(f"\n2️⃣ Training model on entropy patches...")
    
    # Create patches using moderate threshold
    patch_config = PatchConfig(
        method='global',
        entropy_threshold=0.6,
        min_patch_size=4,
        max_patch_size=64,
        cache_patches=False
    )
    
    patcher = EntropyPatcher(entropy_model, patch_config)
    all_patches, patch_stats = patcher.process_data(
        byte_sequences[:20], cache_key="comparison_patches"
    )
    
    # Flatten patches
    flat_patches = []
    for seq_patches in all_patches:
        flat_patches.extend(seq_patches)
    
    if len(flat_patches) > 10:
        patch_dataset = PatchDataset(flat_patches, seq_len=128)
        
        val_size = max(1, len(patch_dataset) // 10)
        train_size = len(patch_dataset) - val_size
        
        patch_train, patch_val = torch.utils.data.random_split(
            patch_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        patch_config_model = TemplateConfig(
            d_model=128, n_heads=4, n_layers=3, d_ff=512,
            batch_size=8, max_steps=500, max_seq_len=128,
            vocab_size=256, eval_every=100
        )
        
        patch_train_loader = DataLoader(patch_train, batch_size=8, shuffle=True)
        patch_val_loader = DataLoader(patch_val, batch_size=8, shuffle=False)
        
        try:
            patch_model, patch_metrics = train_template_model(
                patch_config_model, patch_train_loader, patch_val_loader
            )
            results['patches'] = patch_metrics
            print(f"✅ Patch training completed: Loss {patch_metrics['val_loss']:.4f}")
        except Exception as e:
            print(f"❌ Patch training failed: {e}")
            results['patches'] = None
    
    # Compare results
    print(f"\n📊 COMPARISON RESULTS:")
    print(f"=" * 40)
    
    if results.get('baseline') and results.get('patches'):
        baseline_loss = results['baseline']['val_loss']
        patch_loss = results['patches']['val_loss']
        
        print(f"Baseline Loss: {baseline_loss:.4f}")
        print(f"Patch Loss: {patch_loss:.4f}")
        
        if patch_loss < baseline_loss:
            improvement = ((baseline_loss - patch_loss) / baseline_loss) * 100
            print(f"🎉 Patches improved loss by {improvement:.1f}%!")
        else:
            degradation = ((patch_loss - baseline_loss) / baseline_loss) * 100
            print(f"📉 Patches increased loss by {degradation:.1f}%")
        
        print(f"\nBaseline Accuracy: {results['baseline']['val_accuracy']:.4f}")
        print(f"Patch Accuracy: {results['patches']['val_accuracy']:.4f}")
    else:
        print("❌ Could not complete comparison due to training failures")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Complete entropy patching workflow")
    parser.add_argument("--train_entropy", action="store_true",
                       help="Train new entropy model (otherwise try to load existing)")
    parser.add_argument("--quick_demo", action="store_true",
                       help="Run quick demo with small models and data")
    parser.add_argument("--skip_comparison", action="store_true",
                       help="Skip training comparison step")
    
    args = parser.parse_args()
    
    print(f"🎯 COMPLETE ENTROPY PATCHING WORKFLOW")
    print(f"=" * 60)
    
    set_seed(42)
    
    # Configuration
    if args.quick_demo:
        print("🚀 Running quick demo mode...")
        entropy_config = EntropyConfig(
            d_model=128, n_heads=4, n_layers=3, d_ff=512,
            batch_size=8, max_steps=500, max_seq_len=256,
            num_documents=50, max_bytes=25000
        )
    else:
        entropy_config = EntropyConfig(
            d_model=256, n_heads=8, n_layers=4, d_ff=1024,
            batch_size=16, max_steps=2000, max_seq_len=512,
            num_documents=200, max_bytes=100000
        )
    
    total_start_time = time.time()
    
    try:
        # Step 1: Get entropy model
        if args.train_entropy:
            print(f"\n1️⃣ Training entropy model...")
            entropy_model, entropy_config = quick_entropy_training(entropy_config)
        else:
            print(f"\n1️⃣ Loading entropy model...")
            entropy_model, loaded_config = load_entropy_model(
                "quick_entropy_model.pth", "quick_entropy_config.pth"
            )
            if entropy_model is None:
                print("❌ No saved model found. Training new one...")
                entropy_model, entropy_config = quick_entropy_training(entropy_config)
            else:
                entropy_config = loaded_config
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        entropy_model = entropy_model.to(device)
        
        # Step 2: Load data for patching
        print(f"\n2️⃣ Loading data for patching...")
        texts, bytes_data = load_and_cache_data(entropy_config)
        
        # Convert to sequences
        byte_sequences = []
        seq_len = 512 if not args.quick_demo else 256
        
        for i in range(0, len(bytes_data), seq_len):
            seq = bytes_data[i:i + seq_len]
            if len(seq) >= 20:
                byte_sequences.append(seq)
        
        print(f"📊 Created {len(byte_sequences)} byte sequences")
        
        # Step 3: Analyze entropy distribution
        print(f"\n3️⃣ Analyzing entropy distribution...")
        entropy_stats = analyze_entropy_distribution(entropy_model, byte_sequences)
        
        # Step 4: Create patches with different methods
        print(f"\n4️⃣ Creating patches with different methods...")
        
        methods_to_test = [
            ('global', 0.6),
            ('percentile', 75.0),
        ]
        
        if not args.quick_demo:
            methods_to_test.extend([
                ('adaptive', 0.5),
                ('monotonic', 0.3)
            ])
        
        best_patches = None
        best_stats = None
        
        for method, threshold in methods_to_test:
            print(f"\n   Testing {method} method (threshold: {threshold})...")
            
            patch_config = PatchConfig(
                method=method,
                entropy_threshold=threshold,
                min_patch_size=4,
                max_patch_size=64 if args.quick_demo else 128,
                cache_patches=True
            )
            
            patcher = EntropyPatcher(entropy_model, patch_config)
            patches, stats = patcher.process_data(
                byte_sequences, cache_key=f"workflow_{method}_{threshold}"
            )
            
            print(f"     Created {stats['total_patches']} patches")
            print(f"     Avg size: {stats['avg_patch_size']:.1f} bytes")
            
            # Use the first method's patches for training
            if best_patches is None:
                best_patches = patches
                best_stats = stats
        
        # Step 5: Compare training approaches (optional)
        if not args.skip_comparison and best_patches:
            print(f"\n5️⃣ Comparing training approaches...")
            comparison_results = compare_training_approaches(
                byte_sequences, entropy_model, entropy_config
            )
        
        # Step 6: Final summary
        total_time = time.time() - total_start_time
        
        print(f"\n🎉 WORKFLOW COMPLETED!")
        print(f"=" * 40)
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Entropy model: {entropy_config.d_model}d, {entropy_config.n_layers}L")
        print(f"Data processed: {len(byte_sequences)} sequences")
        
        if best_stats:
            print(f"Best patches: {best_stats['total_patches']} patches")
            print(f"Avg patch size: {best_stats['avg_patch_size']:.1f} bytes")
        
        print(f"\n💡 Next steps:")
        print(f"   1. Use entropy_integration.py for full training")
        print(f"   2. Experiment with different thresholds")
        print(f"   3. Try different patching methods")
        print(f"   4. Scale up to larger datasets")
        
    except Exception as e:
        print(f"❌ Workflow failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()