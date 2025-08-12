import torch
import torch.nn.functional as F
import numpy as np
from typing import List

class EntropyPatcher:
    """Creates patches based on next-byte entropy"""
    def __init__(self, entropy_model, threshold=0.6, method='global', 
                 min_patch_size=1, max_patch_size=32):
        self.entropy_model = entropy_model
        self.entropy_model.eval()
        self.threshold = threshold
        self.method = method
        self.min_patch_size = min_patch_size
        self.max_patch_size = max_patch_size
        
    def compute_entropies(self, byte_sequence):
        """Compute next-byte entropies"""
        device = next(self.entropy_model.parameters()).device
        entropies = []
        
        with torch.no_grad():
            # Process in chunks for efficiency
            chunk_size = 512
            for i in range(0, len(byte_sequence), chunk_size):
                chunk = byte_sequence[i:i+chunk_size]
                
                # Prepare input
                if len(chunk) < chunk_size:
                    chunk = [0] * (chunk_size - len(chunk)) + chunk
                
                x = torch.tensor([chunk], dtype=torch.long, device=device)
                
                # Get entropies
                logits = self.entropy_model(x)
                probs = F.softmax(logits, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
                
                entropies.extend(entropy[0].cpu().numpy())
        
        return entropies[:len(byte_sequence)]
    
    def create_patches(self, byte_sequence):
        """Create patches based on entropy thresholds"""
        if len(byte_sequence) == 0:
            return []
        
        entropies = self.compute_entropies(byte_sequence)
        patches = []
        current_patch = []
        
        for i, byte_val in enumerate(byte_sequence):
            current_patch.append(byte_val)
            
            # Check if we should start new patch
            should_split = False
            
            if len(current_patch) >= self.min_patch_size:
                if self.method == 'global':
                    # Split if entropy exceeds global threshold
                    if i < len(entropies) - 1 and entropies[i] > self.threshold:
                        should_split = True
                        
                elif self.method == 'monotonic':
                    # Split if entropy increases significantly
                    if i > 0 and i < len(entropies) - 1:
                        entropy_diff = entropies[i] - entropies[i-1]
                        if entropy_diff > self.threshold:
                            should_split = True
            
            # Force split if patch too large
            if len(current_patch) >= self.max_patch_size:
                should_split = True
            
            if should_split and current_patch:
                patches.append(current_patch)
                current_patch = []
        
        # Add remaining bytes
        if current_patch:
            patches.append(current_patch)
        
        return patches
    
    def create_boundaries(self, byte_sequence):
        """Create boundary mask for patches"""
        patches = self.create_patches(byte_sequence)
        boundaries = np.zeros(len(byte_sequence), dtype=np.int32)
        
        idx = 0
        for patch in patches:
            boundaries[idx] = 1  # Mark start of patch
            idx += len(patch)
        
        return boundaries
    
    def find_optimal_threshold(self, byte_sequences, target_patch_size=6.0):
        """Find threshold that gives desired average patch size"""
        thresholds = np.linspace(0.1, 2.0, 20)
        best_threshold = self.threshold
        best_diff = float('inf')
        
        for thresh in thresholds:
            self.threshold = thresh
            
            total_bytes = 0
            total_patches = 0
            
            for seq in byte_sequences[:100]:  # Sample
                patches = self.create_patches(seq)
                total_bytes += len(seq)
                total_patches += len(patches)
            
            avg_patch_size = total_bytes / max(total_patches, 1)
            diff = abs(avg_patch_size - target_patch_size)
            
            if diff < best_diff:
                best_diff = diff
                best_threshold = thresh
        
        self.threshold = best_threshold
        print(f"Optimal threshold: {best_threshold:.3f} for target patch size {target_patch_size}")
        return best_threshold