#!/usr/bin/env python3
"""
Simple experiment: Test if medium model can handle larger patches better than tiny model
"""

import torch
import numpy as np
import pickle
import os
from typing import List, Dict
from tqdm import tqdm
from entropy_llm import EntropyLLM, load_and_cache_data, compute_entropy
from llm import BLT_LLM, PatchDataset, train_model, collate_batch, ModelConfig
from torch.utils.data import DataLoader

def create_patches_with_threshold(byte_sequence: List[int], entropies: np.ndarray, threshold: float) -> List[List[int]]:
    """Create patches using specific threshold"""
    patches = []
    current_patch = []
    
    for byte_val, entropy in zip(byte_sequence, entropies):
        if entropy > threshold and len(current_patch) > 0:
            patches.append(current_patch)
            current_patch = [byte_val]
        else:
            current_patch.append(byte_val)
    
    if current_patch:
        patches.append(current_patch)
    
    return patches

def generate_patch_datasets() -> Dict[float, List[List[int]]]:
    """Generate patches with different thresholds for medium model"""
    print("🔄 Loading entropy model...")
    
    checkpoint = torch.load("entropy_model.pt", map_location='cpu', weights_only=False)
    config = checkpoint['config']
    model = EntropyLLM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Load data
    config.num_documents = 400
    config.max_tokens = 80000
    texts, bytes_data = load_and_cache_data(config)
    
    # Compute entropies
    chunk_size = 512
    all_entropies = []
    all_byte_chunks = []
    
    print("🧮 Computing entropies...")
    with torch.no_grad():
        for i in tqdm(range(0, min(40000, len(bytes_data) - chunk_size), chunk_size)):
            chunk = bytes_data[i:i + chunk_size]
            byte_tensor = torch.tensor(chunk, dtype=torch.long).unsqueeze(0).to(device)
            entropy_values = compute_entropy(model, byte_tensor)
            entropy_np = entropy_values.cpu().numpy().flatten()
            all_entropies.append(entropy_np)
            all_byte_chunks.append(chunk)
    
    # Test thresholds that gave different patch sizes before
    thresholds = [0.5, 1.0, 1.5, 2.0]  # 0.5=tiny patches, 2.0=larger patches
    patch_datasets = {}
    
    for threshold in thresholds:
        print(f"✂️ Creating patches for threshold {threshold:.1f}")
        all_patches = []
        
        for chunk_bytes, entropies in zip(all_byte_chunks, all_entropies):
            patches = create_patches_with_threshold(chunk_bytes, entropies, threshold)
            all_patches.extend(patches)
        
        avg_size = np.mean([len(p) for p in all_patches])
        print(f"   Threshold: {threshold:.1f}, Avg size: {avg_size:.1f}, Patches: {len(all_patches)}")
        
        patch_datasets[threshold] = all_patches
    
    return patch_datasets

def get_medium_config() -> ModelConfig:
    """Medium model config (best from previous experiment)"""
    config = ModelConfig()
    config.d_model = 512
    config.n_layers = 8
    config.n_heads = 8
    config.h_encoder = 256
    config.h_decoder = 256
    config.max_steps = 600
    config.batch_size = 12
    config.eval_every = 150
    config.eval_steps = 15
    return config

def quick_train_medium(patches: List[List[int]], threshold: float) -> Dict[str, float]:
    """Train medium model and return metrics"""
    print(f"🚀 Training medium model with threshold {threshold:.1f}")
    
    config = get_medium_config()
    dataset = PatchDataset(patches, max_patches=12)
    
    if len(dataset) < 10:
        return {'val_loss': float('inf'), 'val_accuracy': 0.0, 'val_perplexity': float('inf')}
    
    # Split data
    val_size = max(1, len(dataset) // 10)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_batch)
    
    try:
        model, metrics = train_model(config, train_loader, val_loader)
        return metrics
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return {'val_loss': float('inf'), 'val_accuracy': 0.0, 'val_perplexity': float('inf')}

def analyze_medium_model_patches():
    """Test if medium model can handle larger patches"""
    print("🔬 MEDIUM MODEL + PATCH SIZE EXPERIMENT")
    print("=" * 60)
    
    # Generate patch datasets
    if not os.path.exists("medium_patch_datasets.pkl"):
        patch_datasets = generate_patch_datasets()
        with open("medium_patch_datasets.pkl", 'wb') as f:
            pickle.dump(patch_datasets, f)
        print("💾 Saved patch datasets")
    else:
        print("📦 Loading cached patch datasets...")
        with open("medium_patch_datasets.pkl", 'rb') as f:
            patch_datasets = pickle.load(f)
    
    # Test each threshold with medium model
    results = {}
    
    for threshold in [0.5, 1.0, 1.5, 2.0]:
        if threshold not in patch_datasets:
            continue
            
        patches = patch_datasets[threshold]
        avg_size = np.mean([len(p) for p in patches])
        
        print(f"\n📊 Testing threshold {threshold:.1f} (avg size: {avg_size:.1f})")
        
        metrics = quick_train_medium(patches, threshold)
        
        results[threshold] = {
            'avg_size': avg_size,
            'num_patches': len(patches),
            'val_loss': metrics['val_loss'],
            'val_accuracy': metrics['val_accuracy'],
            'val_perplexity': metrics['val_perplexity']
        }
        
        print(f"   Loss: {metrics['val_loss']:.4f}, Acc: {metrics['val_accuracy']:.3f}, PPL: {metrics['val_perplexity']:.2f}")
    
    # Results summary
    print(f"\n📈 MEDIUM MODEL RESULTS")
    print("=" * 60)
    print(f"{'Threshold':<10} {'Avg Size':<9} {'Loss':<8} {'Acc':<6} {'PPL'}")
    print("-" * 60)
    
    best_loss = float('inf')
    best_threshold = None
    
    for threshold in sorted(results.keys()):
        r = results[threshold]
        print(f"{threshold:<10.1f} {r['avg_size']:<9.1f} {r['val_loss']:<8.4f} {r['val_accuracy']:<6.3f} {r['val_perplexity']:<6.1f}")
        
        if r['val_loss'] < best_loss:
            best_loss = r['val_loss']
            best_threshold = threshold
    
    if best_threshold:
        print(f"\n🏆 Best threshold for medium model: {best_threshold:.1f} with loss {best_loss:.4f}")
        
        # Compare to tiny model results (from previous experiment)
        tiny_best = 0.4915  # From patch size experiment
        medium_best = best_loss
        improvement = (tiny_best - medium_best) / tiny_best * 100
        
        print(f"📊 Improvement over tiny model: {improvement:.1f}% better")
        
        if best_threshold > 0.5:
            print(f"✅ Medium model can handle larger patches! (threshold {best_threshold:.1f} vs 0.5)")
        else:
            print(f"🤔 Medium model still prefers tiny patches (threshold {best_threshold:.1f})")
    
    # Save results
    with open("medium_patch_results.pkl", 'wb') as f:
        pickle.dump(results, f)
    
    return results

if __name__ == "__main__":
    if not os.path.exists("entropy_model.pt"):
        print("❌ entropy_model.pt not found. Please run entropy_llm.py first.")
        exit(1)
    
    results = analyze_medium_model_patches()
    print("\n✅ Medium model patch experiment complete!")