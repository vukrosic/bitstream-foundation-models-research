#!/usr/bin/env python3
"""
GPU Scale Experiment: Test much larger models that actually utilize RTX 4090 properly
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

def create_optimal_patches(byte_sequence: List[int], entropies: np.ndarray) -> List[List[int]]:
    """Create patches using optimal threshold=0.5"""
    patches = []
    current_patch = []
    
    for byte_val, entropy in zip(byte_sequence, entropies):
        if entropy > 0.5 and len(current_patch) > 0:
            patches.append(current_patch)
            current_patch = [byte_val]
        else:
            current_patch.append(byte_val)
    
    if current_patch:
        patches.append(current_patch)
    
    return patches

def get_gpu_scale_configs() -> Dict[str, ModelConfig]:
    """Much larger model configs to utilize RTX 4090 properly"""
    configs = {}
    
    # Large - should use ~8GB VRAM
    large = ModelConfig()
    large.d_model = 1024
    large.n_layers = 12
    large.n_heads = 16
    large.h_encoder = 512
    large.h_decoder = 512
    large.batch_size = 32  # Bigger batch
    large.max_steps = 1000
    large.eval_every = 200
    large.eval_steps = 25
    configs['large'] = large
    
    # XL - should use ~12GB VRAM
    xl = ModelConfig()
    xl.d_model = 1536
    xl.n_layers = 16
    xl.n_heads = 24
    xl.h_encoder = 768
    xl.h_decoder = 768
    xl.batch_size = 24  # Still big batch
    xl.max_steps = 1000
    xl.eval_every = 200
    xl.eval_steps = 25
    configs['xl'] = xl
    
    # XXL - should use ~18GB VRAM
    xxl = ModelConfig()
    xxl.d_model = 2048
    xxl.n_layers = 20
    xxl.n_heads = 32
    xxl.h_encoder = 1024
    xxl.h_decoder = 1024
    xxl.batch_size = 16  # Reasonable batch for huge model
    xxl.max_steps = 1000
    xxl.eval_every = 200
    xxl.eval_steps = 25
    configs['xxl'] = xxl
    
    return configs

def generate_large_dataset() -> List[List[int]]:
    """Generate larger dataset to feed bigger models"""
    print("🔄 Loading entropy model...")
    
    checkpoint = torch.load("entropy_model.pt", map_location='cpu', weights_only=False)
    config = checkpoint['config']
    model = EntropyLLM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Load MORE data for bigger models
    config.num_documents = 1000  # 2.5x more
    config.max_tokens = 200000   # 2.5x more
    texts, bytes_data = load_and_cache_data(config)
    
    # Process larger chunks
    chunk_size = 1024  # 2x bigger chunks
    all_patches = []
    
    print("🧮 Computing entropies and creating patches...")
    with torch.no_grad():
        for i in tqdm(range(0, min(100000, len(bytes_data) - chunk_size), chunk_size)):
            chunk = bytes_data[i:i + chunk_size]
            byte_tensor = torch.tensor(chunk, dtype=torch.long).unsqueeze(0).to(device)
            entropy_values = compute_entropy(model, byte_tensor)
            entropy_np = entropy_values.cpu().numpy().flatten()
            
            patches = create_optimal_patches(chunk, entropy_np)
            all_patches.extend(patches)
    
    avg_size = np.mean([len(p) for p in all_patches])
    print(f"✅ Created {len(all_patches)} patches, avg size: {avg_size:.1f}")
    
    return all_patches

def monitor_gpu_usage():
    """Print GPU utilization"""
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_used = torch.cuda.memory_allocated(0) / 1e9
        gpu_util = gpu_used / gpu_mem * 100
        print(f"🔥 GPU Memory: {gpu_used:.1f}GB / {gpu_mem:.1f}GB ({gpu_util:.1f}%)")

def train_large_model(patches: List[List[int]], model_name: str, config: ModelConfig) -> Dict[str, float]:
    """Train large model with GPU monitoring"""
    print(f"🚀 Training {model_name} model...")
    print(f"   d_model: {config.d_model}, layers: {config.n_layers}, batch: {config.batch_size}")
    
    dataset = PatchDataset(patches, max_patches=15)  # More patches per sample
    
    if len(dataset) < 20:
        return {'val_loss': float('inf'), 'val_accuracy': 0.0, 'val_perplexity': float('inf')}
    
    # Larger train/val split
    val_size = max(10, len(dataset) // 8)  # Bigger validation set
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_batch)
    
    print(f"   Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    try:
        # Monitor GPU before training
        monitor_gpu_usage()
        
        model, metrics = train_model(config, train_loader, val_loader)
        
        # Monitor GPU after training
        monitor_gpu_usage()
        
        return metrics
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return {'val_loss': float('inf'), 'val_accuracy': 0.0, 'val_perplexity': float('inf')}

def analyze_gpu_scale():
    """Test much larger models to utilize RTX 4090"""
    print("🔬 GPU SCALE EXPERIMENT (RTX 4090 Utilization)")
    print("=" * 60)
    
    # Check GPU
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"🎮 GPU: {gpu_name}")
    print(f"💾 VRAM: {gpu_mem:.1f}GB")
    
    # Generate large dataset
    if not os.path.exists("large_patches.pkl"):
        patches = generate_large_dataset()
        with open("large_patches.pkl", 'wb') as f:
            pickle.dump(patches, f)
        print("💾 Saved large patch dataset")
    else:
        print("📦 Loading cached large patches...")
        with open("large_patches.pkl", 'rb') as f:
            patches = pickle.load(f)
    
    print(f"📊 Dataset: {len(patches)} patches")
    
    # Test large model configs
    configs = get_gpu_scale_configs()
    results = {}
    
    for model_name, config in configs.items():
        print(f"\n📊 Testing {model_name} model")
        
        metrics = train_large_model(patches, model_name, config)
        
        results[model_name] = {
            'd_model': config.d_model,
            'n_layers': config.n_layers,
            'n_heads': config.n_heads,
            'batch_size': config.batch_size,
            'val_loss': metrics['val_loss'],
            'val_accuracy': metrics['val_accuracy'],
            'val_perplexity': metrics['val_perplexity']
        }
        
        print(f"   Loss: {metrics['val_loss']:.4f}, Acc: {metrics['val_accuracy']:.3f}, PPL: {metrics['val_perplexity']:.2f}")
        
        # Clear GPU cache between models
        torch.cuda.empty_cache()
    
    # Results summary
    print(f"\n📈 GPU SCALE RESULTS")
    print("=" * 80)
    print(f"{'Model':<6} {'d_model':<8} {'Layers':<7} {'Heads':<6} {'Batch':<6} {'Loss':<8} {'Acc':<6} {'PPL'}")
    print("-" * 80)
    
    best_loss = float('inf')
    best_model = None
    
    for model_name in ['large', 'xl', 'xxl']:
        if model_name not in results:
            continue
        r = results[model_name]
        print(f"{model_name:<6} {r['d_model']:<8} {r['n_layers']:<7} {r['n_heads']:<6} {r['batch_size']:<6} {r['val_loss']:<8.4f} {r['val_accuracy']:<6.3f} {r['val_perplexity']:<6.1f}")
        
        if r['val_loss'] < best_loss:
            best_loss = r['val_loss']
            best_model = model_name
    
    if best_model:
        print(f"\n🏆 Best large model: {best_model} with loss {best_loss:.4f}")
        
        # Compare to previous best (medium model)
        medium_best = 0.3323  # From model size experiment
        improvement = (medium_best - best_loss) / medium_best * 100
        print(f"📊 Improvement over medium model: {improvement:.1f}% better")
    
    # Save results
    with open("gpu_scale_results.pkl", 'wb') as f:
        pickle.dump(results, f)
    
    return results

if __name__ == "__main__":
    if not os.path.exists("entropy_model.pt"):
        print("❌ entropy_model.pt not found. Please run entropy_llm.py first.")
        exit(1)
    
    results = analyze_gpu_scale()
    print("\n✅ GPU scale experiment complete!")
    print("🔥 Your RTX 4090 should now be properly utilized!")