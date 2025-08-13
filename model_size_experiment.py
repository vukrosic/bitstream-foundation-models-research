#!/usr/bin/env python3
"""
Simple experiment: Test different model sizes with optimal patch settings (threshold=0.5, size~2.2)
"""

import torch
import numpy as np
import pickle
import os
from typing import List, Dict, Tuple
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

def get_model_configs() -> Dict[str, ModelConfig]:
    """Different model sizes to test"""
    configs = {}
    
    # Tiny (current)
    tiny = ModelConfig()
    tiny.d_model = 256
    tiny.n_layers = 4
    tiny.n_heads = 4
    tiny.h_encoder = 128
    tiny.h_decoder = 128
    configs['tiny'] = tiny
    
    # Small
    small = ModelConfig()
    small.d_model = 384
    small.n_layers = 6
    small.n_heads = 6
    small.h_encoder = 192
    small.h_decoder = 192
    configs['small'] = small
    
    # Medium
    medium = ModelConfig()
    medium.d_model = 512
    medium.n_layers = 8
    medium.n_heads = 8
    medium.h_encoder = 256
    medium.h_decoder = 256
    configs['medium'] = medium
    
    # Set common training params
    for config in configs.values():
        config.max_steps = 600
        config.batch_size = 12  # Smaller batch for larger models
        config.eval_every = 150
        config.eval_steps = 15
    
    return configs

def generate_patches() -> List[List[int]]:
    """Generate patches using optimal settings"""
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
    all_patches = []
    
    print("🧮 Computing entropies and creating patches...")
    with torch.no_grad():
        for i in tqdm(range(0, min(40000, len(bytes_data) - chunk_size), chunk_size)):
            chunk = bytes_data[i:i + chunk_size]
            byte_tensor = torch.tensor(chunk, dtype=torch.long).unsqueeze(0).to(device)
            entropy_values = compute_entropy(model, byte_tensor)
            entropy_np = entropy_values.cpu().numpy().flatten()
            
            patches = create_optimal_patches(chunk, entropy_np)
            all_patches.extend(patches)
    
    avg_size = np.mean([len(p) for p in all_patches])
    print(f"✅ Created {len(all_patches)} patches, avg size: {avg_size:.1f}")
    
    return all_patches

def quick_train_and_evaluate(patches: List[List[int]], model_name: str, config: ModelConfig) -> Dict[str, float]:
    """Train model and return metrics"""
    print(f"🚀 Training {model_name} model...")
    
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

def analyze_model_sizes():
    """Main experiment function"""
    print("🔬 MODEL SIZE EXPERIMENT (Optimal settings: threshold=0.5)")
    print("=" * 60)
    
    # Generate patches once
    if not os.path.exists("optimal_patches.pkl"):
        patches = generate_patches()
        with open("optimal_patches.pkl", 'wb') as f:
            pickle.dump(patches, f)
        print("💾 Saved optimal patches")
    else:
        print("📦 Loading cached optimal patches...")
        with open("optimal_patches.pkl", 'rb') as f:
            patches = pickle.load(f)
    
    # Test different model sizes
    configs = get_model_configs()
    results = {}
    
    for model_name, config in configs.items():
        print(f"\n📊 Testing {model_name} model")
        print(f"   d_model: {config.d_model}, layers: {config.n_layers}, heads: {config.n_heads}")
        
        metrics = quick_train_and_evaluate(patches, model_name, config)
        
        results[model_name] = {
            'd_model': config.d_model,
            'n_layers': config.n_layers,
            'n_heads': config.n_heads,
            'val_loss': metrics['val_loss'],
            'val_accuracy': metrics['val_accuracy'],
            'val_perplexity': metrics['val_perplexity']
        }
        
        print(f"   Loss: {metrics['val_loss']:.4f}, Acc: {metrics['val_accuracy']:.3f}, PPL: {metrics['val_perplexity']:.2f}")
    
    # Results summary
    print(f"\n📈 RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Model':<8} {'d_model':<8} {'Layers':<7} {'Loss':<8} {'Acc':<6} {'PPL':<6}")
    print("-" * 70)
    
    best_loss = float('inf')
    best_model = None
    
    for model_name in ['tiny', 'small', 'medium']:
        if model_name not in results:
            continue
        r = results[model_name]
        print(f"{model_name:<8} {r['d_model']:<8} {r['n_layers']:<7} {r['val_loss']:<8.4f} {r['val_accuracy']:<6.3f} {r['val_perplexity']:<6.1f}")
        
        if r['val_loss'] < best_loss:
            best_loss = r['val_loss']
            best_model = model_name
    
    if best_model:
        print(f"\n🏆 Best model: {best_model} with loss {best_loss:.4f}")
    
    # Save results
    with open("model_size_results.pkl", 'wb') as f:
        pickle.dump(results, f)
    
    return results

if __name__ == "__main__":
    if not os.path.exists("entropy_model.pt"):
        print("❌ entropy_model.pt not found. Please run entropy_llm.py first.")
        exit(1)
    
    results = analyze_model_sizes()
    print("\n✅ Model size experiment complete!")