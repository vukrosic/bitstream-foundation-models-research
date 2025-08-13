import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import math
import random
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import time
from transformers import AutoTokenizer
from dataclasses import dataclass
from typing import List, Optional
import warnings
import os
import pickle
warnings.filterwarnings('ignore')

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🌱 Set all seeds to {seed}")

@dataclass
class ModelConfig:
    # Model architecture - Scaled up for RTX 4090
    d_model: int = 1024  # Increased from 384
    n_heads: int = 16    # Increased from 8
    n_layers: int = 12   # Increased from 6
    d_ff: int = 4096     # Increased from 1536
    batch_size: int = 16 # Reduced slightly to fit larger model
    max_steps: int = 400

    # Training parameters
    gradient_accumulation_steps: int = 8  # Increased to maintain effective batch size
    muon_lr: float = 0.008  # Slightly reduced for larger model

    # Data parameters
    max_seq_len: int = 1024  # Increased from 512
    num_documents: int = 2000
    max_tokens: int = 500000

    # Evaluation
    eval_every: int = 500
    eval_steps: int = 50

    # Regularization
    weight_decay: float = 0.1
    dropout: float = 0.1
    grad_clip: float = 1.0

    # Technical
    use_amp: bool = True
    vocab_size: int = 256  # Fixed for bytes
    
    # Encoder/Decoder dimensions - Scaled proportionally
    h_encoder: int = 512  # Increased from 192
    h_decoder: int = 512  # Increased from 192
    n_encoder_layers: int = 2  # Increased from 1
    n_decoder_layers: int = 3  # Increased from 2
    
    # Cross-attention parameters
    cross_attn_encoder: bool = True
    cross_attn_decoder: bool = True
    cross_attn_nheads: int = 8  # Increased from 4
    cross_attn_all_layers_encoder: bool = False
    cross_attn_all_layers_decoder: bool = False
    norm_eps: float = 1e-5

    def __post_init__(self):
        self.d_k = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"

@torch.compile
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz iteration to compute the zeroth power / orthogonalization of G."""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()

    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT

    return X

class Muon(torch.optim.Optimizer):
    """Muon - MomentUm Orthogonalized by Newton-schulz"""
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                p.add_(g.view_as(p), alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)
	
def load_patches_data():
    """Load patches created by entropy model"""
    patches_file = "patches_data.pkl"
    if not os.path.exists(patches_file):
        print(f"❌ Patches file {patches_file} not found. Please run create_patches.py first.")
        return None
    
    print(f"📦 Loading patches from {patches_file}")
    with open(patches_file, 'rb') as f:
        patch_data = pickle.load(f)
    
    patches = patch_data['patches']
    print(f"✅ Loaded {len(patches):,} patches")
    print(f"   Average patch size: {patch_data['avg_patch_size']:.2f} bytes")
    print(f"   Total bytes: {patch_data['total_bytes']:,}")
    
    # Show first few patches
    print(f"🔍 First 5 patches:")
    for i in range(min(5, len(patches))):
        patch = patches[i]
        patch_text = bytes(patch).decode('utf-8', errors='ignore')
        print(f"   Patch {i+1}: {patch} -> '{patch_text[:20]}{'...' if len(patch_text) > 20 else ''}'")
    
    return patches

def collate_batch(batch):
    """Custom collate function to handle boundaries as tensors"""
    bytes_list = [item['bytes'] for item in batch]
    targets_list = [item['targets'] for item in batch]
    boundaries_list = [item['boundaries'] for item in batch]
    valid_patches_list = [item['valid_patches'] for item in batch]
    
    # Stack all tensors
    bytes_batch = torch.stack(bytes_list)
    targets_batch = torch.stack(targets_list)
    boundaries_batch = torch.stack(boundaries_list)
    valid_patches_batch = torch.stack(valid_patches_list)
    
    return {
        'bytes': bytes_batch,
        'targets': targets_batch,
        'boundaries': boundaries_batch,
        'valid_patches': valid_patches_batch
    }

class PatchDataset(Dataset):
    def __init__(self, patches: List[List[int]], max_patches: int = 15):
        self.patches = patches
        self.max_patches = max_patches
        
        # Create sequences of patches
        self.sequences = []
        for i in range(0, len(patches) - max_patches, max_patches // 2):  # Overlapping windows
            seq = patches[i:i + max_patches]
            if len(seq) == max_patches:
                self.sequences.append(seq)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        patch_seq = self.sequences[idx]
        
        # Convert patches to padded byte sequences and boundaries
        all_bytes = []
        boundaries = []
        
        for patch in patch_seq:
            start = len(all_bytes)
            all_bytes.extend(patch)
            end = len(all_bytes)
            boundaries.append((start, end))
        
        # Pad to max length
        max_bytes = 100  # Reasonable max for ~15 patches of ~6 bytes each
        if len(all_bytes) > max_bytes:
            all_bytes = all_bytes[:max_bytes]
            # Adjust boundaries
            new_boundaries = []
            for start, end in boundaries:
                if start < max_bytes:
                    new_boundaries.append((start, min(end, max_bytes)))
            boundaries = new_boundaries
        else:
            # Pad with zeros
            all_bytes.extend([0] * (max_bytes - len(all_bytes)))
        
        # Convert boundaries to tensor format
        boundary_tensor = torch.zeros(self.max_patches, 2, dtype=torch.long)
        valid_patches = torch.zeros(self.max_patches, dtype=torch.bool)
        
        for i, (start, end) in enumerate(boundaries[:self.max_patches]):
            boundary_tensor[i] = torch.tensor([start, end])
            valid_patches[i] = True
        
        # Create targets (next byte prediction)
        targets = all_bytes[1:] + [0]  # Shift by 1
        
        return {
            'bytes': torch.tensor(all_bytes, dtype=torch.long),
            'targets': torch.tensor(targets, dtype=torch.long),
            'boundaries': boundary_tensor,
            'valid_patches': valid_patches
        }

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        angular_freq = (1 / 10000) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.register_buffer('cos', theta.cos(), persistent=False)
        self.register_buffer('sin', theta.sin(), persistent=False)

    def forward(self, x_BTHD: torch.Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.rotary = Rotary(self.d_k, max_seq_len)
        self.dropout = dropout

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)

        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        Q = self.rotary(Q)
        K = self.rotary(K)

        attn_output = F.scaled_dot_product_attention(
            Q, K, V, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.w_o(attn_output)

class CrossAttention(nn.Module):
    def __init__(self, dim: int, head_dim: int, n_heads: int, n_kv_heads: int, norm_eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        
        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)
        
        self.norm = nn.RMSNorm(dim, eps=norm_eps)
        
    def forward(self, x: torch.Tensor, kv: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # x: query [batch, seq_len, dim]
        # kv: key/value [batch, kv_seq_len, dim]
        batch_size, seq_len, _ = x.shape
        kv_seq_len = kv.shape[1]
        
        # Apply norm to query
        x_norm = self.norm(x)
        
        # Compute Q, K, V
        q = self.wq(x_norm).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(kv).view(batch_size, kv_seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(kv).view(batch_size, kv_seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.wo(attn_output)

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.silu(self.linear1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, max_seq_len, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x

@torch.compile
def create_patch_mask(boundaries: torch.Tensor, valid_patches: torch.Tensor, max_bytes: int):
    """Create attention masks from boundary tensors"""
    batch_size, max_patches, _ = boundaries.shape
    mask = torch.zeros(batch_size, max_patches, max_bytes, device=boundaries.device)
    
    for b in range(batch_size):
        for p in range(max_patches):
            if valid_patches[b, p]:
                start, end = boundaries[b, p]
                if start < end and end <= max_bytes:
                    mask[b, p, start:end] = 1
    return mask

class LocalEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.cross_attn_encoder = config.cross_attn_encoder
        self.cross_attn_all_layers_encoder = config.cross_attn_all_layers_encoder
        
        # Byte embedding
        self.byte_embed = nn.Embedding(config.vocab_size, config.h_encoder)
        
        # Transformer layers for processing bytes
        self.layers = nn.ModuleList([
            TransformerBlock(
                config.h_encoder, 
                max(1, config.h_encoder // 64),
                config.h_encoder * 2, 
                100,
                config.dropout
            )
            for _ in range(config.n_encoder_layers)
        ])
        
        # Cross-attention layers
        if self.cross_attn_encoder:
            layers_to_add = config.n_encoder_layers if self.cross_attn_all_layers_encoder else 1
            self.cross_attn_layers = nn.ModuleList([
                CrossAttention(
                    dim=config.h_encoder,
                    head_dim=config.h_encoder // config.cross_attn_nheads,
                    n_heads=config.cross_attn_nheads,
                    n_kv_heads=config.cross_attn_nheads,
                    norm_eps=config.norm_eps
                )
                for _ in range(layers_to_add)
            ])
        
        # Project to main model dimension
        self.patch_proj = nn.Linear(config.h_encoder, config.d_model)
        
    def forward(self, bytes_tensor, boundaries, valid_patches, first_call=False):
        batch_size, max_bytes = bytes_tensor.shape
        max_patches = boundaries.size(1)
        
        if first_call:
            print(f"🔧 Encoder input: bytes shape {bytes_tensor.shape}")
            print(f"   Boundaries shape: {boundaries.shape}")
            print(f"   Valid patches: {valid_patches.sum().item()}")
        
        # 1. Embed bytes
        h = self.byte_embed(bytes_tensor)
        h = F.dropout(h, p=self.config.dropout, training=self.training)
        
        # 2. Apply transformer layers
        for i, layer in enumerate(self.layers):
            h = layer(h)
            
            # Apply cross-attention if enabled
            if self.cross_attn_encoder and (i == len(self.layers) - 1 or self.cross_attn_all_layers_encoder):
                # Create patch embeddings by pooling
                patch_embeds = self._create_patch_embeddings(h, boundaries, valid_patches, batch_size, max_patches, max_bytes)
                
                # Apply cross-attention
                layer_idx = i if self.cross_attn_all_layers_encoder else 0
                patch_embeds_cross = self.cross_attn_layers[layer_idx](
                    x=patch_embeds,
                    kv=h
                )
                patch_embeds = patch_embeds + patch_embeds_cross
        
        # 3. Create final patch embeddings
        if not self.cross_attn_encoder:
            patch_embeds = self._create_patch_embeddings(h, boundaries, valid_patches, batch_size, max_patches, max_bytes)
        
        # 4. Project to main model dimension
        patch_vecs_projected = self.patch_proj(patch_embeds)
        
        if first_call:
            print(f"   Patch vectors shape: {patch_vecs_projected.shape}")
            print(f"   First patch vector norm: {patch_vecs_projected[0, 0].norm().item():.3f}")
        
        return patch_vecs_projected, h
    
    def _create_patch_embeddings(self, h, boundaries, valid_patches, batch_size, max_patches, max_bytes):
        """Create patch embeddings by pooling byte representations"""
        patch_embeds = torch.zeros(batch_size, max_patches, self.config.h_encoder, device=h.device)
        
        for b in range(batch_size):
            for p in range(max_patches):
                if valid_patches[b, p]:
                    start, end = boundaries[b, p]
                    if start < end and end <= max_bytes:
                        # Mean pooling over the patch
                        patch_embeds[b, p] = h[b, start:end].mean(dim=0)
        
        return patch_embeds

class LocalDecoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.cross_attn_decoder = config.cross_attn_decoder
        self.cross_attn_all_layers_decoder = config.cross_attn_all_layers_decoder
        
        # Normalization
        self.norm = nn.RMSNorm(config.h_decoder, eps=config.norm_eps)
        
        # Cross-attention layers
        if self.cross_attn_decoder:
            layers_to_add = config.n_decoder_layers if self.cross_attn_all_layers_decoder else 1
            self.cross_attn_layers = nn.ModuleList([
                CrossAttention(
                    dim=config.h_decoder,
                    head_dim=config.h_decoder // config.cross_attn_nheads,
                    n_heads=config.cross_attn_nheads,
                    n_kv_heads=config.cross_attn_nheads,
                    norm_eps=config.norm_eps
                )
                for _ in range(layers_to_add)
            ])
        
        # Transformer layers for bytes
        self.layers = nn.ModuleList([
            TransformerBlock(
                config.h_decoder, 
                max(1, config.h_decoder // 64),
                config.h_decoder * 2, 
                100,
                config.dropout
            )
            for _ in range(config.n_decoder_layers)
        ])
        
        # Project patch vectors to decoder dimension
        self.patch_proj = nn.Linear(config.d_model, config.h_decoder)
        
        # Project encoder hidden if needed
        self.byte_proj = nn.Linear(config.h_encoder, config.h_decoder)
        
        # Output projection to byte vocabulary
        self.output = nn.Linear(config.h_decoder, config.vocab_size, bias=False)
        
    def forward(self, patch_vecs, byte_hidden_from_encoder, boundaries, valid_patches, first_call=False):
        batch_size = patch_vecs.size(0)
        max_bytes = byte_hidden_from_encoder.size(1)
        
        if first_call:
            print(f"🔧 Decoder input: patch_vecs shape {patch_vecs.shape}")
            print(f"   Byte hidden shape: {byte_hidden_from_encoder.shape}")
        
        # Project inputs to decoder dimension
        patch_embeds = self.patch_proj(patch_vecs)
        h = self.byte_proj(byte_hidden_from_encoder)
        
        # Add patch embeddings if not using cross-attention
        if not self.cross_attn_decoder:
            # Map patch embeddings to byte positions
            h = h + self._map_patches_to_bytes(patch_embeds, boundaries, valid_patches, batch_size, max_bytes)
        
        # Apply dropout
        h = F.dropout(h, p=self.config.dropout, training=self.training)
        
        # Apply decoder layers
        for i, layer in enumerate(self.layers):
            # Apply cross-attention if enabled
            if self.cross_attn_decoder and (i == 0 or self.cross_attn_all_layers_decoder):
                layer_idx = i if self.cross_attn_all_layers_decoder else 0
                h_cross = self.cross_attn_layers[layer_idx](
                    x=h,
                    kv=patch_embeds
                )
                h = h + h_cross
            
            # Apply transformer layer
            h = layer(h)
            
            if first_call and i == 0:
                print(f"   After decoder layer {i}: {h.shape}")
        
        # Final normalization and projection
        h_preds = self.norm(h)
        h_preds = F.dropout(h_preds, p=self.config.dropout, training=self.training)
        logits = self.output(h_preds)
        
        if first_call:
            print(f"   Final logits shape: {logits.shape}")
        
        return logits.float()
    
    def _map_patches_to_bytes(self, patch_embeds, boundaries, valid_patches, batch_size, max_bytes):
        """Map patch embeddings to byte positions"""
        byte_patch_embeds = torch.zeros(batch_size, max_bytes, self.config.h_decoder, device=patch_embeds.device)
        
        for b in range(batch_size):
            for p in range(patch_embeds.size(1)):
                if valid_patches[b, p]:
                    start, end = boundaries[b, p]
                    if start < end and end <= max_bytes:
                        # Broadcast patch embedding to all bytes in the patch
                        byte_patch_embeds[b, start:end] = patch_embeds[b, p]
        
        return byte_patch_embeds

class BLT_LLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Encoder: bytes -> patch vectors
        self.encoder = LocalEncoder(config)
        
        # Main LLM: processes patch vectors
        self.position_dropout = nn.Dropout(config.dropout)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, 20, config.dropout)  # Max ~20 patches
            for _ in range(config.n_layers)
        ])

        self.norm = nn.RMSNorm(config.d_model)
        self.output_dropout = nn.Dropout(config.dropout)
        
        # Decoder: patch vectors -> byte predictions
        self.decoder = LocalDecoder(config)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, bytes_tensor, boundaries, valid_patches, first_call=False):
        if first_call:
            print(f"🚀 BLT Forward pass:")
        
        # 1. Encoder: bytes -> patch vectors
        patch_vecs, byte_hidden = self.encoder(bytes_tensor, boundaries, valid_patches, first_call)
        
        if first_call:
            print(f"   Encoded to {patch_vecs.shape[1]} patch vectors")
        
        # 2. Main LLM: process patch vectors
        x = patch_vecs * math.sqrt(self.config.d_model)
        x = self.position_dropout(x)

        for i, block in enumerate(self.transformer_blocks):
            x = block(x)
            if first_call and i == 0:
                print(f"   After LLM layer {i}: {x.shape}")

        x = self.norm(x)
        x = self.output_dropout(x)
        
        if first_call:
            print(f"   LLM output: {x.shape}")
        
        # 3. Decoder: patch vectors -> byte predictions
        logits = self.decoder(x, byte_hidden, boundaries, valid_patches, first_call)
        
        return logits

def evaluate_model(model: nn.Module, val_loader: DataLoader, config: ModelConfig):
    """Evaluate model performance"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_correct = 0

    device = next(model.parameters()).device

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= config.eval_steps:
                break
            
            bytes_tensor = batch['bytes'].to(device)
            targets = batch['targets'].to(device)
            boundaries = batch['boundaries'].to(device)
            valid_patches = batch['valid_patches'].to(device)

            with autocast(enabled=config.use_amp):
                logits = model(bytes_tensor, boundaries, valid_patches)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), targets.view(-1))

            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()

            predictions = logits.argmax(dim=-1)
            total_correct += (predictions == targets).sum().item()

    avg_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens
    perplexity = math.exp(min(avg_loss, 20))

    model.train()
    return {'val_loss': avg_loss, 'val_accuracy': accuracy, 'val_perplexity': perplexity}

def generate_text(model: nn.Module, config: ModelConfig, prompt: str = "", max_length: int = 200, temperature: float = 0.8):
    """Generate text from the model"""
    model.eval()
    device = next(model.parameters()).device
    
    # Convert prompt to bytes if provided
    if prompt:
        tokens = list(prompt.encode('utf-8'))
    else:
        tokens = [ord('T')]  # Start with 'T'
    
    # Pad to max_bytes (100)
    max_bytes = 100
    if len(tokens) > max_bytes:
        tokens = tokens[:max_bytes]
    else:
        tokens.extend([0] * (max_bytes - len(tokens)))
    
    # Create dummy boundaries and valid_patches for generation
    # Treat the whole sequence as one patch
    boundaries = torch.zeros(1, 15, 2, dtype=torch.long, device=device)  # max_patches=15
    valid_patches = torch.zeros(1, 15, dtype=torch.bool, device=device)
    boundaries[0, 0] = torch.tensor([0, len(tokens)])
    valid_patches[0, 0] = True
    
    generated_tokens = tokens.copy()
    
    with torch.no_grad():
        for _ in range(max_length):
            # Prepare input
            x = torch.tensor([generated_tokens], dtype=torch.long, device=device)
            
            # Get logits
            logits = model(x, boundaries, valid_patches)
            next_token_logits = logits[0, -1, :] / temperature
            
            # Sample next token (only valid bytes 0-255)
            probs = F.softmax(next_token_logits[:256], dim=-1)  # Only first 256 for valid bytes
            next_token = torch.multinomial(probs, 1).item()
            
            # Stop if we hit padding or invalid byte
            if next_token == 0:
                break
                
            generated_tokens.append(next_token)
            
            # Update for next iteration - keep last max_bytes tokens
            if len(generated_tokens) > max_bytes:
                generated_tokens = generated_tokens[-max_bytes:]
    
    # Convert back to text
    try:
        # Only use valid byte values, skip padding zeros
        valid_bytes = [b for b in generated_tokens if 0 < b < 256]
        text = bytes(valid_bytes).decode('utf-8', errors='ignore')
        return text
    except:
        return "Generated invalid bytes"

def setup_muon_optimizer(model: nn.Module, config: ModelConfig):
    """Setup Muon optimizer with hybrid approach"""
    muon_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if (param.ndim == 2 and 
            'byte_embed' not in name and 
            'norm' not in name and 
            param.requires_grad):
            muon_params.append(param)
        else:
            adamw_params.append(param)

    print(f"  Muon parameters: {sum(p.numel() for p in muon_params):,}")
    print(f"  AdamW parameters: {sum(p.numel() for p in adamw_params):,}")

    muon_optimizer = Muon(muon_params, lr=config.muon_lr, momentum=0.95)
    adamw_optimizer = torch.optim.AdamW(adamw_params, lr=config.muon_lr*0.1, weight_decay=config.weight_decay)

    return [muon_optimizer, adamw_optimizer]

def train_model(config: ModelConfig, train_loader: DataLoader, val_loader: DataLoader):
    """Train the BLT model with Muon optimizer"""
    print(f"\n🚀 Training BLT model with Muon optimizer")

    # Initialize model
    set_seed(42)
    model = BLT_LLM(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    llm_params = total_params - encoder_params - decoder_params
    
    print(f"  📊 Total parameters: {total_params:,}")
    print(f"     Encoder: {encoder_params:,}")
    print(f"     LLM: {llm_params:,}")
    print(f"     Decoder: {decoder_params:,}")

    # Setup optimizers
    optimizers = setup_muon_optimizer(model, config)

    # Learning rate schedule
    schedulers = []
    for optimizer in optimizers:
        warmup_steps = config.max_steps // 20
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (config.max_steps - warmup_steps)
                return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        schedulers.append(scheduler)

    scaler = GradScaler() if config.use_amp else None

    # Generate initial text (before training)
    print(f"\n🎯 INITIAL TEXT GENERATION (Step 0):")
    print("=" * 60)
    initial_text = generate_text(model, config, prompt="The", max_length=150)
    print(f"Generated: {initial_text}")
    print("=" * 60)

    # Training loop
    model.train()
    step = 0
    start_time = time.time()
    best_val_loss = float('inf')
    first_batch = True

    pbar = tqdm(total=config.max_steps, desc="Training")

    while step < config.max_steps:
        for batch_idx, batch in enumerate(train_loader):
            if step >= config.max_steps:
                break

            bytes_tensor = batch['bytes'].to(device)
            targets = batch['targets'].to(device)
            boundaries = batch['boundaries'].to(device)
            valid_patches = batch['valid_patches'].to(device)

            # Forward pass with gradient accumulation
            if config.use_amp:
                with autocast():
                    logits = model(bytes_tensor, boundaries, valid_patches, first_call=first_batch)
                    loss = F.cross_entropy(logits.view(-1, config.vocab_size), targets.view(-1))
                    loss = loss / config.gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                logits = model(bytes_tensor, boundaries, valid_patches, first_call=first_batch)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), targets.view(-1))
                loss = loss / config.gradient_accumulation_steps
                loss.backward()
            
            first_batch = False

            # Optimizer step after accumulation
            if (step + 1) % config.gradient_accumulation_steps == 0:
                if config.use_amp:
                    for optimizer in optimizers:
                        scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

                    for optimizer in optimizers:
                        scaler.step(optimizer)
                        optimizer.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()
                    scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    for optimizer in optimizers:
                        optimizer.step()
                        optimizer.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()

            # Logging
            if step % 50 == 0:  # More frequent for shorter training
                with torch.no_grad():
                    predictions = logits.argmax(dim=-1)
                    accuracy = (predictions == targets).float().mean().item()
                    current_loss = loss.item() * config.gradient_accumulation_steps
                    perplexity = math.exp(min(current_loss, 20))

                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'acc': f'{accuracy:.3f}',
                    'ppl': f'{perplexity:.1f}',
                    'lr': f'{optimizers[0].param_groups[0]["lr"]:.2e}'
                })

            # Evaluation
            if step % config.eval_every == 0 and step > 0:
                eval_metrics = evaluate_model(model, val_loader, config)
                print(f"\nStep {step}: Val Loss: {eval_metrics['val_loss']:.4f}, "
                      f"Val Acc: {eval_metrics['val_accuracy']:.4f}, "
                      f"Val PPL: {eval_metrics['val_perplexity']:.2f}")

                if eval_metrics['val_loss'] < best_val_loss:
                    best_val_loss = eval_metrics['val_loss']

                # Generate text at middle of training (around step 2500)
                if step == config.max_steps // 2:
                    print(f"\n🎯 MIDDLE TEXT GENERATION (Step {step}):")
                    print("=" * 60)
                    middle_text = generate_text(model, config, prompt="The", max_length=150)
                    print(f"Generated: {middle_text}")
                    print("=" * 60)

            step += 1
            if step % 50 == 0:
                pbar.update(50)

    pbar.close()

    training_time = time.time() - start_time
    print(f"  ⏱️ Training completed in {training_time:.1f} seconds")

    # Generate final text (after training)
    print(f"\n🎯 FINAL TEXT GENERATION (Step {config.max_steps}):")
    print("=" * 60)
    final_text = generate_text(model, config, prompt="The", max_length=150)
    print(f"Generated: {final_text}")
    print("=" * 60)

    # Final evaluation
    final_eval = evaluate_model(model, val_loader, config)
    print(f"  📊 Final - Loss: {final_eval['val_loss']:.4f}, "
          f"Acc: {final_eval['val_accuracy']:.4f}, PPL: {final_eval['val_perplexity']:.2f}")

    return model, final_eval

if __name__ == "__main__":
    # Check system
    print(f"🔍 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Set seed
    set_seed(42)

    # Create config for BLT model
    config = ModelConfig()
    print(f"\n📋 BLT Model Configuration:")
    print(f"   Main LLM: {config.d_model}d, {config.n_layers}L, {config.n_heads}H, {config.d_ff}ff")
    print(f"   Encoder: {config.h_encoder}d, {config.n_encoder_layers}L")
    print(f"   Decoder: {config.h_decoder}d, {config.n_decoder_layers}L")
    print(f"   Training: {config.max_steps} steps, batch size {config.batch_size}")

    # Load patches data
    patches = load_patches_data()
    if patches is None:
        exit(1)
    
    dataset = PatchDataset(patches, max_patches=15)

    # Train/val split
    val_size = len(dataset) // 10
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size * 2,  # Increase batch size
        shuffle=True, 
        num_workers=2,  # Use multiple workers
        pin_memory=True,  # Pin memory for faster GPU transfer
        collate_fn=collate_batch
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size * 2, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True,
        collate_fn=collate_batch
    )

    print(f"📊 Dataset: {len(train_dataset)} train, {len(val_dataset)} val samples")

    # Train model
    start_time = time.time()
    model, final_metrics = train_model(config, train_loader, val_loader)
    total_time = time.time() - start_time

    print(f"\n🎉 BLT TRAINING COMPLETED!")
    print(f"⏱️ Total time: {total_time/60:.1f} minutes")
    print(f"🏆 Final Results:")
    print(f"   Validation Loss: {final_metrics['val_loss']:.4f}")
    print(f"   Validation Accuracy: {final_metrics['val_accuracy']:.4f}")
    print(f"   Validation Perplexity: {final_metrics['val_perplexity']:.2f}")
    
    # Save the trained BLT model
    model_save_path = "blt_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'final_metrics': final_metrics
    }, model_save_path)
    print(f"\n💾 BLT Model saved to {model_save_path}")
    
    # Show some final statistics
    print(f"\n📈 TRAINING SUMMARY:")
    print(f"   🔧 Architecture: Encoder -> LLM -> Decoder")
    print(f"   📊 Processes patches of ~6 bytes each")
    print(f"   🎯 Predicts individual bytes from patch representations")
    print(f"   ⚡ Trained for {config.max_steps} steps in {total_time:.1f}s")