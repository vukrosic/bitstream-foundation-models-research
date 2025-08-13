#!/usr/bin/env python3
"""
Simple experiment: Fix patch size at 3.0 and test different entropy thresholds
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

def create_patches_with_fixed_threshold(byte_sequence: List[int], entropies: np.ndarray, threshold: float) -> List[List[int]]:
    """Create patches using fixed entropy threshold"""
    patches = []
    current_patch = []
    
    for i, (byte_val, entropy) in enumerate(zip(byte_sequence, entropies)):
        if entropy > threshold and len(current_patch) > 0:
            patches.append(current_patch)
            current_patch = [byte_val]
        else:
            current_patch.append(byte_val)
    
    if current_patch:
        patches.append(current_patch)
    
    return patches

def generate_threshold_datasets(thresholds: List[float]) -> Dict[float, List[List[int]]]:
    """Generate patch datasets with different entropy thresholds"""
    print("🔄 Loading entropy model...")
    
    # Load entropy model
    checkpoint = torch.load("entropy_model.pt", map_location='cpu', weights_only=False)
    config = checkpoint['config']
    model = EntropyLLM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Load data (same as patch size experiment)
    config.num_documents = 500
    config.max_tokens = 100000
    texts, bytes_data = load_and_cache_data(config)
    
    # Compute entropies
    chunk_size = 512
    all_entropies = []
    all_byte_chunks = []
    
    print("🧮 Computing entropies...")
    with torch.no_grad():
        for i in tqdm(range(0, min(50000, len(bytes_data) - chunk_size), chunk_size)):
            chunk = bytes_data[i:i + chunk_size]
            byte_tensor = torch.tensor(chunk, dtype=torch.long).unsqueeze(0).to(device)
            entropy_values = compute_entropy(model, byte_tensor)
            entropy_np = entropy_values.cpu().numpy().flatten()
            all_entropies.append(entropy_np)
            all_byte_chunks.append(chunk)
    
    # Create patch datasets for each threshold
    patch_datasets = {}
    
    for threshold in thresholds:
        print(f"✂️ Creating patches for threshold {threshold:.2f}")
        all_patches = []
        
        for chunk_bytes, entropies in zip(all_byte_chunks, all_entropies):
            patches = create_patches_with_fixed_threshold(chunk_bytes, entropies, threshold)
            all_patches.extend(patches)
        
        avg_size = np.mean([len(p) for p in all_patches])
        print(f"   Threshold: {threshold:.2f}, Avg size: {avg_size:.1f}, Patches: {len(all_patches)}")
        
        patch_datasets[threshold] = all_patches
    
    return patch_datasets

def quick_train_and_evaluate(patches: List[List[int]], threshold: float) -> Tuple[Dict[str, float], str]:
    """Quick training - same as patch size experiment"""
    print(f"🚀 Training for threshold {threshold:.2f}")
    
    config = ModelConfig()
    config.max_steps = 800
    config.batch_size = 16
    config.d_model = 256
    config.n_layers = 4
    config.n_heads = 4
    config.h_encoder = 128
    config.h_decoder = 128
    config.eval_every = 200
    config.eval_steps = 20
    
    dataset = PatchDataset(patches, max_patches=12)
    
    if len(dataset) < 10:
        return {'val_loss': float('inf'), 'val_accuracy': 0.0, 'val_perplexity': float('inf')}, "[No data]"
    
    val_size = max(1, len(dataset) // 10)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_batch)
    
    try:
        model, metrics = train_model(config, train_loader, val_loader)
        
        # Simple text generation
        device = next(model.parameters()).device
        model.eval()
        
        with torch.no_grad():
            # Generate from "The"
            tokens = list("The".encode('utf-8'))
            x = torch.tensor([tokens + [0] * 97], dtype=torch.long, device=device)  # pad to 100
            boundaries = torch.zeros(1, 15, 2, dtype=torch.long, device=device)
            valid_patches = torch.zeros(1, 15, dtype=torch.bool, device=device)
            boundaries[0, 0] = torch.tensor([0, len(tokens)])
            valid_patches[0, 0] = True
            
            logits = model(x, boundaries, valid_patches)
            next_logits = logits[0, len(tokens)-1, :256] / 0.8
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            try:
                sample_text = "The" + chr(next_token) if 32 <= next_token <= 126 else f"The[{next_token}]"
            except:
                sample_text = f"The[{next_token}]"
        
        return metrics, sample_text
    except Exception as e:
        return {'val_loss': float('inf'), 'val_accuracy': 0.0, 'val_perplexity': float('inf')}, f"[Error: {e}]"

def analyze_entropy_thresholds():
    """Main experiment function"""
    print("🔬 ENTROPY THRESHOLD EXPERIMENT (Fixed ~3.0 patch size)")
    print("=" * 60)
    
    # Test different entropy thresholds
    thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    
    # Generate datasets
    if not os.path.exists("threshold_datasets.pkl"):
        print("📦 Generating threshold datasets...")
        patch_datasets = generate_threshold_datasets(thresholds)
        
        with open("threshold_datasets.pkl", 'wb') as f:
            pickle.dump(patch_datasets, f)
        print("💾 Saved threshold datasets")
    else:
        print("📦 Loading cached threshold datasets...")
        with open("threshold_datasets.pkl", 'rb') as f:
            patch_datasets = pickle.load(f)
    
    # Run experiments
    results = {}
    
    for threshold in thresholds:
        if threshold not in patch_datasets:
            continue
            
        patches = patch_datasets[threshold]
        avg_size = np.mean([len(p) for p in patches])
        
        print(f"\n📊 Testing threshold {threshold:.2f} (avg size: {avg_size:.1f})")
        
        metrics, sample_text = quick_train_and_evaluate(patches, threshold)
        
        results[threshold] = {
            'avg_size': avg_size,
            'num_patches': len(patches),
            'val_loss': metrics['val_loss'],
            'val_accuracy': metrics['val_accuracy'],
            'val_perplexity': metrics['val_perplexity'],
            'sample_text': sample_text
        }
        
        print(f"   Loss: {metrics['val_loss']:.4f}, Acc: {metrics['val_accuracy']:.3f}, PPL: {metrics['val_perplexity']:.2f}")
        print(f"   Sample: '{sample_text}'")
    
    # Results summary
    print(f"\n📈 RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Threshold':<10} {'Avg Size':<9} {'Loss':<8} {'Acc':<6} {'PPL':<6} {'Sample'}")
    print("-" * 70)
    
    best_loss = float('inf')
    best_threshold = None
    
    for threshold in sorted(results.keys()):
        r = results[threshold]
        print(f"{threshold:<10.2f} {r['avg_size']:<9.1f} {r['val_loss']:<8.4f} {r['val_accuracy']:<6.3f} {r['val_perplexity']:<6.1f} '{r['sample_text']}'")
        
        if r['val_loss'] < best_loss:
            best_loss = r['val_loss']
            best_threshold = threshold
    
    if best_threshold is not None:
        print(f"\n🏆 Best threshold: {best_threshold:.2f} with loss {best_loss:.4f}")
    
    # Save results
    with open("threshold_results.pkl", 'wb') as f:
        pickle.dump(results, f)
    
    return results

if __name__ == "__main__":
    if not os.path.exists("entropy_model.pt"):
        print("❌ entropy_model.pt not found. Please run entropy_llm.py first.")
        exit(1)
    
    results = analyze_entropy_thresholds()
    print("\n✅ Entropy threshold experiment complete!")