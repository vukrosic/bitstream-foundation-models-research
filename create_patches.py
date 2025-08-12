import torch
import torch.nn.functional as F
import numpy as np
import pickle
import os
from tqdm import tqdm
from typing import List
from entropy_llm import EntropyLLM, ModelConfig, load_and_cache_data, text_to_bytes, bytes_to_text, compute_entropy

def create_patches_global_threshold(byte_sequence: List[int], entropies: np.ndarray, threshold: float = 1.5) -> List[List[int]]:
    """Create patches when entropy exceeds global threshold. Returns list of byte chunks."""
    patches = []
    current_patch = []
    
    for i, (byte_val, entropy) in enumerate(zip(byte_sequence, entropies)):
        # Start new patch if entropy exceeds threshold
        if entropy > threshold and len(current_patch) > 0:
            patches.append(current_patch)
            current_patch = [byte_val]
        else:
            current_patch.append(byte_val)
    
    # Add final patch
    if current_patch:
        patches.append(current_patch)
    
    return patches

def find_threshold_for_target_patch_size(all_entropies: List[np.ndarray], target_size: float = 6.0) -> float:
    """Binary search to find threshold that gives desired average patch size."""
    low, high = 0.0, 5.0
    
    for _ in range(20):  # Binary search iterations
        threshold = (low + high) / 2
        
        # Count patches with this threshold
        total_bytes = 0
        total_patches = 0
        
        for entropy_seq in all_entropies:
            patches_in_seq = 1 + (entropy_seq > threshold).sum()
            total_patches += patches_in_seq
            total_bytes += len(entropy_seq)
        
        avg_patch_size = total_bytes / total_patches
        
        if avg_patch_size < target_size:
            low = threshold  # Need higher threshold for larger patches
        else:
            high = threshold
    
    return threshold

def load_model_and_create_patches():
    """Load entropy model and create patches from training data"""
    print("🔄 Loading entropy model...")
    
    # Load saved model
    model_path = "entropy_model.pt"
    if not os.path.exists(model_path):
        print(f"❌ Model file {model_path} not found. Please train the model first.")
        return
    
    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint['config']
    
    # Initialize model
    model = EntropyLLM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded on {device}")
    
    # Load same data used for training
    print("📦 Loading training data...")
    texts, bytes_data = load_and_cache_data(config)
    
    # Calculate how much data for 5000 steps
    steps_data_size = config.batch_size * config.max_seq_len * config.max_steps
    actual_data_size = min(steps_data_size, len(bytes_data))
    bytes_for_steps = bytes_data[:actual_data_size]
    
    print(f"📊 Using {len(bytes_for_steps):,} bytes ({actual_data_size:,} for {config.max_steps} steps)")
    
    # Process data in chunks to compute entropies
    print("🧮 Computing entropies...")
    chunk_size = config.max_seq_len
    all_entropies = []
    all_byte_chunks = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(bytes_for_steps) - chunk_size, chunk_size), desc="Computing entropies"):
            chunk = bytes_for_steps[i:i + chunk_size]
            byte_tensor = torch.tensor(chunk, dtype=torch.long).unsqueeze(0).to(device)
            
            entropy_values = compute_entropy(model, byte_tensor)
            entropy_np = entropy_values.cpu().numpy().flatten()
            
            all_entropies.append(entropy_np)
            all_byte_chunks.append(chunk)
    
    print(f"✅ Computed entropies for {len(all_entropies)} chunks")
    
    # Find optimal threshold for target patch size
    target_patch_size = 6.0  # Middle range as requested
    print(f"🎯 Finding threshold for target patch size: {target_patch_size} bytes")
    
    threshold = find_threshold_for_target_patch_size(all_entropies, target_patch_size)
    print(f"📏 Optimal threshold: {threshold:.3f}")
    
    # Create patches for all chunks
    print("✂️ Creating patches...")
    all_patches = []
    total_patches = 0
    total_bytes_processed = 0
    
    for chunk_bytes, entropies in tqdm(zip(all_byte_chunks, all_entropies), desc="Creating patches", total=len(all_byte_chunks)):
        patches = create_patches_global_threshold(chunk_bytes, entropies, threshold)
        all_patches.extend(patches)
        total_patches += len(patches)
        total_bytes_processed += len(chunk_bytes)
    
    avg_patch_size = total_bytes_processed / total_patches
    print(f"✅ Created {total_patches:,} patches from {total_bytes_processed:,} bytes")
    print(f"📊 Average patch size: {avg_patch_size:.2f} bytes")
    
    # Show some example patches
    print(f"\n🔍 PATCH EXAMPLES:")
    print("=" * 60)
    
    for i in range(min(10, len(all_patches))):
        patch = all_patches[i]
        patch_text = bytes_to_text(patch)
        print(f"Patch {i+1:2d} (size {len(patch):2d}): {patch[:10]} -> '{patch_text[:30]}{'...' if len(patch_text) > 30 else ''}'")
    
    # Show patch size distribution
    patch_sizes = [len(patch) for patch in all_patches]
    print(f"\n📈 PATCH SIZE DISTRIBUTION:")
    print(f"   Min size: {min(patch_sizes)}")
    print(f"   Max size: {max(patch_sizes)}")
    print(f"   Mean size: {np.mean(patch_sizes):.2f}")
    print(f"   Std size: {np.std(patch_sizes):.2f}")
    
    # Show size histogram
    size_counts = {}
    for size in patch_sizes:
        size_counts[size] = size_counts.get(size, 0) + 1
    
    print(f"\n📊 SIZE HISTOGRAM:")
    for size in sorted(size_counts.keys())[:15]:  # Show first 15 sizes
        count = size_counts[size]
        bar = "█" * min(50, count * 50 // max(size_counts.values()))
        print(f"   Size {size:2d}: {count:6d} patches {bar}")
    
    # Save patches to file
    patch_data = {
        'patches': all_patches,
        'threshold': threshold,
        'avg_patch_size': avg_patch_size,
        'total_patches': total_patches,
        'total_bytes': total_bytes_processed,
        'config': config
    }
    
    save_path = "patches_data.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(patch_data, f)
    
    print(f"\n💾 Patches saved to {save_path}")
    print(f"📦 File contains {total_patches:,} patches ready for BLT training")

if __name__ == "__main__":
    load_model_and_create_patches()