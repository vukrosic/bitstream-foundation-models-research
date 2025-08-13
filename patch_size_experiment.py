#!/usr/bin/env python3
"""
Minimal experiment to study patch size distribution vs model performance
"""

import torch
import numpy as np
import pickle
import os
from typing import List, Dict, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
from entropy_llm import EntropyLLM, ModelConfig, load_and_cache_data, compute_entropy
from llm import BLT_LLM, PatchDataset, train_model, collate_batch
from torch.utils.data import DataLoader

def create_patches_with_target_size(byte_sequence: List[int], entropies: np.ndarray, target_avg_size: float, show_examples: bool = False) -> List[List[int]]:
    """Create patches targeting specific average size using dynamic threshold"""
    # Binary search for threshold that gives target size
    low, high = 0.0, 5.0
    final_threshold = 0.0
    
    for iteration in range(15):
        threshold = (low + high) / 2
        patches = []
        current_patch = []
        
        for i, (byte_val, entropy) in enumerate(zip(byte_sequence, entropies)):
            # Split condition: high entropy AND we have content in current patch
            if entropy > threshold and len(current_patch) > 0:
                patches.append(current_patch)
                current_patch = [byte_val]
            else:
                current_patch.append(byte_val)
        
        if current_patch:
            patches.append(current_patch)
        
        avg_size = len(byte_sequence) / len(patches) if patches else float('inf')
        
        if avg_size < target_avg_size:
            low = threshold
        else:
            high = threshold
        
        final_threshold = threshold
    
    # Show examples if requested
    if show_examples:
        print(f"   🎯 Final threshold: {final_threshold:.3f}")
        print(f"   📊 Created {len(patches)} patches, avg size {avg_size:.2f}")
        print(f"   🔍 First 8 patches:")
        
        for i, patch in enumerate(patches[:8]):
            # Convert patch bytes to text
            try:
                patch_text = bytes(patch).decode('utf-8', errors='replace')
                # Clean up text for display
                display_text = repr(patch_text)[1:-1]  # Remove outer quotes
                if len(display_text) > 25:
                    display_text = display_text[:22] + "..."
                
                print(f"      Patch {i+1}: size={len(patch):2d} | {patch[:6]}... | '{display_text}'")
            except:
                print(f"      Patch {i+1}: size={len(patch):2d} | {patch[:6]}... | [decode error]")
    
    return patches

def generate_patch_datasets(target_sizes: List[float]) -> Dict[float, List[List[int]]]:
    """Generate patch datasets with different target sizes"""
    print("🔄 Loading entropy model...")
    
    # Load entropy model
    checkpoint = torch.load("entropy_model.pt", map_location='cpu', weights_only=False)
    config = checkpoint['config']
    model = EntropyLLM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Load data (smaller subset for quick experiment)
    config.num_documents = 500  # Reduce for speed
    config.max_tokens = 100000  # Reduce for speed
    texts, bytes_data = load_and_cache_data(config)
    
    # Compute entropies on subset
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
    
    # Create patch datasets for each target size
    patch_datasets = {}
    
    for target_size in target_sizes:
        print(f"✂️ Creating patches for target size {target_size}")
        all_patches = []
        
        show_first_chunk = True
        for chunk_idx, (chunk_bytes, entropies) in enumerate(zip(all_byte_chunks, all_entropies)):
            show_examples = show_first_chunk and chunk_idx == 0  # Show examples for first chunk only
            patches = create_patches_with_target_size(chunk_bytes, entropies, target_size, show_examples)
            all_patches.extend(patches)
            show_first_chunk = False
        
        actual_avg_size = np.mean([len(p) for p in all_patches])
        print(f"   Target: {target_size:.1f}, Actual: {actual_avg_size:.1f}, Patches: {len(all_patches)}")
        
        patch_datasets[target_size] = all_patches
    
    return patch_datasets

def generate_sample_text(model, config, prompt: str = "The", max_length: int = 80) -> str:
    """Generate sample text from trained model"""
    device = next(model.parameters()).device
    model.eval()
    
    # Convert prompt to bytes
    tokens = list(prompt.encode('utf-8'))
    max_bytes = 100
    generated_tokens = []
    
    with torch.no_grad():
        for i in range(max_length):
            # Current sequence
            current_seq = tokens + generated_tokens
            
            # Pad to max_bytes
            if len(current_seq) > max_bytes:
                current_seq = current_seq[-max_bytes:]
            
            padded_seq = current_seq + [0] * (max_bytes - len(current_seq))
            
            # Create boundaries
            boundaries = torch.zeros(1, 15, 2, dtype=torch.long, device=device)
            valid_patches = torch.zeros(1, 15, dtype=torch.bool, device=device)
            seq_len = len(current_seq)
            boundaries[0, 0] = torch.tensor([0, seq_len])
            valid_patches[0, 0] = True
            
            # Get next token
            x = torch.tensor([padded_seq], dtype=torch.long, device=device)
            logits = model(x, boundaries, valid_patches)
            
            next_pos = min(seq_len - 1, max_bytes - 1)
            next_token_logits = logits[0, next_pos, :] / 0.8  # temperature
            
            # Sample from valid bytes only
            probs = torch.softmax(next_token_logits[:256], dim=-1)
            if probs[0] > 0.5:  # Reduce padding probability
                probs[0] = 0.1
                probs = probs / probs.sum()
            
            next_token = torch.multinomial(probs, 1).item()
            
            if next_token == 0 or next_token > 255:
                break
                
            generated_tokens.append(next_token)
    
    # Convert to text
    try:
        full_sequence = tokens + generated_tokens
        valid_bytes = [b for b in full_sequence if 0 < b < 256]
        text = bytes(valid_bytes).decode('utf-8', errors='replace')
        return text
    except:
        return f"[Generated {len(generated_tokens)} bytes - decode failed]"

def quick_train_and_evaluate(patches: List[List[int]], target_size: float, steps: int = 1000) -> Tuple[Dict[str, float], str]:
    """Quick training run to measure performance and generate sample text"""
    print(f"🚀 Quick training for patch size {target_size:.1f}")
    
    # Create config for quick training
    config = ModelConfig()
    config.max_steps = steps
    config.batch_size = 16  # Smaller batch
    config.d_model = 256    # Smaller model
    config.n_layers = 4     # Fewer layers
    config.n_heads = 4      # Fewer heads
    config.h_encoder = 128  # Smaller encoder
    config.h_decoder = 128  # Smaller decoder
    config.eval_every = steps // 4
    config.eval_steps = 20
    
    # Create dataset
    dataset = PatchDataset(patches, max_patches=12)
    
    if len(dataset) < 10:
        return {'val_loss': float('inf'), 'val_accuracy': 0.0, 'val_perplexity': float('inf')}, "[No data]"
    
    # Split train/val
    val_size = max(1, len(dataset) // 10)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_batch)
    
    # Train model
    try:
        model, metrics = train_model(config, train_loader, val_loader)
        
        # Generate sample text
        sample_text = generate_sample_text(model, config, "The", max_length=60)
        
        return metrics, sample_text
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return {'val_loss': float('inf'), 'val_accuracy': 0.0, 'val_perplexity': float('inf')}, f"[Training failed: {e}]"

def analyze_patch_size_performance():
    """Main experiment function"""
    print("🔬 PATCH SIZE vs PERFORMANCE EXPERIMENT")
    print("=" * 60)
    
    # Test different target patch sizes
    target_sizes = [3.0, 5.0, 7.0, 10.0, 12.0]
    
    # Generate patch datasets
    if not os.path.exists("patch_size_datasets.pkl"):
        print("📦 Generating patch datasets...")
        patch_datasets = generate_patch_datasets(target_sizes)
        
        with open("patch_size_datasets.pkl", 'wb') as f:
            pickle.dump(patch_datasets, f)
        print("💾 Saved patch datasets")
    else:
        print("📦 Loading cached patch datasets...")
        with open("patch_size_datasets.pkl", 'rb') as f:
            patch_datasets = pickle.load(f)
    
    # Run experiments
    results = {}
    
    for target_size in target_sizes:
        if target_size not in patch_datasets:
            continue
            
        patches = patch_datasets[target_size]
        actual_avg_size = np.mean([len(p) for p in patches])
        
        print(f"\n📊 Testing patch size {target_size:.1f} (actual: {actual_avg_size:.1f})")
        print(f"   Total patches: {len(patches)}")
        
        # Quick training
        metrics, sample_text = quick_train_and_evaluate(patches, target_size, steps=800)
        
        results[target_size] = {
            'actual_avg_size': actual_avg_size,
            'num_patches': len(patches),
            'val_loss': metrics['val_loss'],
            'val_accuracy': metrics['val_accuracy'],
            'val_perplexity': metrics['val_perplexity'],
            'sample_text': sample_text
        }
        
        print(f"   Results: Loss {metrics['val_loss']:.4f}, Acc {metrics['val_accuracy']:.3f}, PPL {metrics['val_perplexity']:.2f}")
        print(f"   Sample: '{sample_text[:60]}{'...' if len(sample_text) > 60 else ''}'")
    
    # Analyze results
    print(f"\n📈 RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Size':<6} {'Actual':<7} {'Patches':<8} {'Loss':<8} {'Acc':<6} {'PPL':<6} {'Sample Text':<30}")
    print("-" * 80)
    
    best_loss = float('inf')
    best_size = None
    
    for target_size in sorted(results.keys()):
        r = results[target_size]
        sample_preview = r['sample_text'][:28] + ".." if len(r['sample_text']) > 30 else r['sample_text']
        print(f"{target_size:<6.1f} {r['actual_avg_size']:<7.1f} {r['num_patches']:<8} {r['val_loss']:<8.4f} {r['val_accuracy']:<6.3f} {r['val_perplexity']:<6.1f} '{sample_preview}'")
        
        if r['val_loss'] < best_loss:
            best_loss = r['val_loss']
            best_size = target_size
    
    print(f"\n🏆 Best performance: patch size {best_size:.1f} with loss {best_loss:.4f}")
    
    # Show detailed text samples
    print(f"\n📝 DETAILED TEXT SAMPLES")
    print("=" * 80)
    for target_size in sorted(results.keys()):
        r = results[target_size]
        print(f"\nPatch size {target_size:.1f} (actual {r['actual_avg_size']:.1f}):")
        print(f"  '{r['sample_text']}'")
        print(f"  [Loss: {r['val_loss']:.4f}, PPL: {r['val_perplexity']:.1f}]")
    
    # Save results
    with open("patch_size_results.pkl", 'wb') as f:
        pickle.dump(results, f)
    
    return results

def plot_results(results: Dict[float, Dict]):
    """Plot the relationship between patch size and performance"""
    sizes = sorted(results.keys())
    losses = [results[s]['val_loss'] for s in sizes]
    accuracies = [results[s]['val_accuracy'] for s in sizes]
    perplexities = [results[s]['val_perplexity'] for s in sizes]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Loss vs patch size
    ax1.plot(sizes, losses, 'bo-')
    ax1.set_xlabel('Target Patch Size')
    ax1.set_ylabel('Validation Loss')
    ax1.set_title('Loss vs Patch Size')
    ax1.grid(True)
    
    # Accuracy vs patch size
    ax2.plot(sizes, accuracies, 'ro-')
    ax2.set_xlabel('Target Patch Size')
    ax2.set_ylabel('Validation Accuracy')
    ax2.set_title('Accuracy vs Patch Size')
    ax2.grid(True)
    
    # Perplexity vs patch size
    ax3.plot(sizes, perplexities, 'go-')
    ax3.set_xlabel('Target Patch Size')
    ax3.set_ylabel('Validation Perplexity')
    ax3.set_title('Perplexity vs Patch Size')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig('patch_size_analysis.png', dpi=150, bbox_inches='tight')
    print("📊 Plots saved to patch_size_analysis.png")

if __name__ == "__main__":
    # Check if entropy model exists
    if not os.path.exists("entropy_model.pt"):
        print("❌ entropy_model.pt not found. Please run entropy_llm.py first.")
        exit(1)
    
    # Run experiment
    results = analyze_patch_size_performance()
    
    # Plot results if matplotlib available
    try:
        plot_results(results)
    except ImportError:
        print("📊 Install matplotlib to generate plots: pip install matplotlib")
    
    print("\n✅ Experiment complete!")
    print("📁 Results saved to patch_size_results.pkl")