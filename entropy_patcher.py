import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import pickle
import os
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import math

# Import from your entropy LLM
from entropy_llm import MinimalLLM, ModelConfig, load_and_cache_data

@dataclass
class PatchConfig:
    """Configuration for entropy-based patching"""
    # Patching method: 'global', 'monotonic', 'adaptive', 'percentile'
    method: str = 'global'
    
    # Threshold parameters
    entropy_threshold: float = 0.6
    percentile_threshold: float = 75.0  # For percentile method
    
    # Patch size constraints
    min_patch_size: int = 4
    max_patch_size: int = 128
    
    # Processing parameters
    batch_size: int = 32
    overlap_size: int = 16  # Overlap between chunks for context
    
    # Caching
    cache_patches: bool = True
    cache_dir: str = "patch_cache"

class EntropyPatcher:
    """
    Entropy-based data patching using a trained byte-level language model.
    Simplified version following BLT paper approach.
    """
    
    def __init__(self, entropy_model, threshold=0.6, method='global'):
        """
        Initialize the entropy patcher.
        
        Args:
            entropy_model: Trained byte-level language model
            threshold: Entropy threshold for creating patch boundaries
            method: 'global' or 'monotonic' patching
        """
        self.entropy_model = entropy_model
        self.threshold = threshold
        self.method = method
        self.device = next(entropy_model.parameters()).device
        
        # Set model to eval mode
        self.entropy_model.eval()
    
    def compute_entropies(self, byte_sequence):
        """Compute next-byte entropies for a sequence"""
        entropies = []
        context = []
        
        with torch.no_grad():
            for byte in byte_sequence:
                if len(context) > 0:
                    # Get entropy for predicting this byte
                    context_tensor = torch.tensor([context], dtype=torch.long, device=self.device)
                    logits = self.entropy_model(context_tensor)
                    probs = F.softmax(logits[0, -1], dim=-1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-8))
                    entropies.append(entropy.item())
                
                context.append(byte)
                if len(context) > self.entropy_model.config.max_seq_len:
                    context = context[-self.entropy_model.config.max_seq_len:]
        
        return entropies
    
    def create_patches(self, byte_sequence):
        """Create patches based on entropy thresholds"""
        entropies = self.compute_entropies(byte_sequence)
        patches = []
        current_patch = [byte_sequence[0]]
        
        for i in range(1, len(byte_sequence)):
            if self.method == 'global':
                # Create new patch if entropy exceeds global threshold
                if i < len(entropies) and entropies[i-1] > self.threshold:
                    patches.append(current_patch)
                    current_patch = [byte_sequence[i]]
                else:
                    current_patch.append(byte_sequence[i])
                    
            elif self.method == 'monotonic':
                # Create patch if entropy increases significantly
                if i > 1 and i < len(entropies):
                    if entropies[i-1] - entropies[i-2] > self.threshold:
                        patches.append(current_patch)
                        current_patch = [byte_sequence[i]]
                    else:
                        current_patch.append(byte_sequence[i])
                else:
                    current_patch.append(byte_sequence[i])
        
        if current_patch:
            patches.append(current_patch)
        
        return patches
    
    def process_data(self, byte_sequences: List[List[int]], cache_key: str = None) -> Tuple[List[List[List[int]]], Dict[str, Any]]:
        """
        Process multiple byte sequences into patches.
        
        Args:
            byte_sequences: List of byte sequences to process
            cache_key: Optional cache key for saving/loading results
            
        Returns:
            Tuple of (patches_per_sequence, statistics)
        """
        # Check cache first
        if cache_key and self.config.cache_patches:
            cache_file = os.path.join(self.config.cache_dir, f"{cache_key}_patches.pkl")
            if os.path.exists(cache_file):
                print(f"📦 Loading cached patches from {cache_file}")
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                return cached_data['patches'], cached_data['stats']
        
        print(f"🔄 Processing {len(byte_sequences)} sequences into patches...")
        
        # Compute entropies for all sequences
        all_entropies = self.compute_entropies_batch(byte_sequences)
        
        # Create patches for each sequence
        all_patches = []
        total_patches = 0
        total_bytes = 0
        patch_sizes = []
        entropy_stats = []
        
        print(f"✂️ Creating patches using method: {self.config.method}")
        
        for seq_idx, (byte_sequence, entropies) in enumerate(zip(byte_sequences, all_entropies)):
            if not byte_sequence:
                all_patches.append([])
                continue
            
            patches = self.create_patches(byte_sequence, entropies)
            all_patches.append(patches)
            
            # Collect statistics
            total_patches += len(patches)
            total_bytes += len(byte_sequence)
            patch_sizes.extend([len(patch) for patch in patches])
            if entropies:
                entropy_stats.extend(entropies)
        
        # Compile statistics
        stats = {
            'total_sequences': len(byte_sequences),
            'total_patches': total_patches,
            'total_bytes': total_bytes,
            'avg_patches_per_sequence': total_patches / len(byte_sequences) if byte_sequences else 0,
            'avg_patch_size': np.mean(patch_sizes) if patch_sizes else 0,
            'median_patch_size': np.median(patch_sizes) if patch_sizes else 0,
            'min_patch_size': min(patch_sizes) if patch_sizes else 0,
            'max_patch_size': max(patch_sizes) if patch_sizes else 0,
            'avg_entropy': np.mean(entropy_stats) if entropy_stats else 0,
            'entropy_std': np.std(entropy_stats) if entropy_stats else 0,
            'config': self.config
        }
        
        # Cache results
        if cache_key and self.config.cache_patches:
            cache_file = os.path.join(self.config.cache_dir, f"{cache_key}_patches.pkl")
            cached_data = {'patches': all_patches, 'stats': stats}
            with open(cache_file, 'wb') as f:
                pickle.dump(cached_data, f)
            print(f"💾 Cached patches to {cache_file}")
        
        return all_patches, stats

class PatchDataset(Dataset):
    """Dataset for training on entropy-based patches."""
    
    def __init__(self, patches: List[List[int]], seq_len: int = 512):
        """
        Initialize patch dataset.
        
        Args:
            patches: List of patches (each patch is a list of bytes)
            seq_len: Maximum sequence length for training
        """
        self.patches = patches
        self.seq_len = seq_len
        
        # Filter patches that are too small
        self.valid_patches = [p for p in patches if len(p) >= 2]
        
        print(f"📊 PatchDataset: {len(self.valid_patches)} valid patches from {len(patches)} total")
    
    def __len__(self):
        return len(self.valid_patches)
    
    def __getitem__(self, idx):
        patch = self.valid_patches[idx]
        
        # If patch is longer than seq_len, take a random subsequence
        if len(patch) > self.seq_len:
            start_idx = torch.randint(0, len(patch) - self.seq_len + 1, (1,)).item()
            patch = patch[start_idx:start_idx + self.seq_len]
        
        # Create input and target sequences
        if len(patch) == 1:
            # Handle single-byte patches
            x = torch.tensor([patch[0]], dtype=torch.long)
            y = torch.tensor([patch[0]], dtype=torch.long)  # Self-prediction
        else:
            x = torch.tensor(patch[:-1], dtype=torch.long)
            y = torch.tensor(patch[1:], dtype=torch.long)
        
        # Pad if necessary
        if len(x) < self.seq_len:
            pad_len = self.seq_len - len(x)
            x = F.pad(x, (0, pad_len), value=0)
            y = F.pad(y, (0, pad_len), value=0)
        
        return x, y

def print_patch_statistics(stats: Dict[str, Any]):
    """Print detailed statistics about the patching process."""
    print(f"\n📈 PATCHING STATISTICS")
    print(f"=" * 50)
    print(f"Method: {stats['config'].method}")
    print(f"Threshold: {stats['config'].entropy_threshold}")
    print(f"Total sequences: {stats['total_sequences']:,}")
    print(f"Total patches: {stats['total_patches']:,}")
    print(f"Total bytes: {stats['total_bytes']:,}")
    print(f"Avg patches per sequence: {stats['avg_patches_per_sequence']:.1f}")
    print(f"Avg patch size: {stats['avg_patch_size']:.1f} bytes")
    print(f"Median patch size: {stats['median_patch_size']:.1f} bytes")
    print(f"Patch size range: {stats['min_patch_size']} - {stats['max_patch_size']} bytes")
    print(f"Avg entropy: {stats['avg_entropy']:.3f}")
    print(f"Entropy std: {stats['entropy_std']:.3f}")
    print(f"=" * 50)

def demonstrate_patching():
    """Demonstrate the entropy patching system."""
    print("🎯 ENTROPY PATCHING DEMONSTRATION")
    print("=" * 60)
    
    # This would typically load your trained entropy model
    # For demonstration, we'll create a dummy setup
    print("⚠️  This is a demonstration. You need to:")
    print("   1. Train your entropy LLM first using entropy_llm.py")
    print("   2. Load the trained model here")
    print("   3. Use your actual data")
    
    # Example of how to use the system:
    print(f"\n📝 Example usage:")
    print(f"```python")
    print(f"# 1. Load your trained entropy model")
    print(f"entropy_model = MinimalLLM(config)")
    print(f"entropy_model.load_state_dict(torch.load('entropy_model.pth'))")
    print(f"")
    print(f"# 2. Create patcher with configuration")
    print(f"patch_config = PatchConfig(")
    print(f"    method='global',")
    print(f"    entropy_threshold=0.6,")
    print(f"    min_patch_size=4,")
    print(f"    max_patch_size=128")
    print(f")")
    print(f"patcher = EntropyPatcher(entropy_model, patch_config)")
    print(f"")
    print(f"# 3. Process your data")
    print(f"patches, stats = patcher.process_data(byte_sequences)")
    print(f"")
    print(f"# 4. Create dataset for training")
    print(f"flat_patches = [patch for seq_patches in patches for patch in seq_patches]")
    print(f"patch_dataset = PatchDataset(flat_patches)")
    print(f"```")

if __name__ == "__main__":
    demonstrate_patching()