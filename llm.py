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
    # Model architecture
    d_model: int = 384
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1536
    batch_size: int = 24
    max_steps: int = 300  # Reduced for speed

    # Training parameters
    gradient_accumulation_steps: int = 4
    muon_lr: float = 0.01

    # Data parameters
    max_seq_len: int = 512
    num_documents: int = 2000
    max_tokens: int = 500000

    # Evaluation
    eval_every: int = 100  # More frequent for shorter training
    eval_steps: int = 50

    # Regularization
    weight_decay: float = 0.1
    dropout: float = 0.1
    grad_clip: float = 1.0

    # Technical
    use_amp: bool = True
    vocab_size: int = 256  # Fixed for bytes
    
    # Encoder/Decoder dimensions (smaller than main model)
    h_encoder: int = 192  # Half of d_model
    h_decoder: int = 192  # Half of d_model
    n_encoder_layers: int = 1
    n_decoder_layers: int = 2

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
        
        # Byte embedding
        self.byte_embed = nn.Embedding(config.vocab_size, config.h_encoder)
        
        # Small transformer for processing bytes
        self.byte_transformer = TransformerBlock(
            config.h_encoder, 
            max(1, config.h_encoder // 64),  # At least 1 head
            config.h_encoder * 2, 
            100,  # Max bytes
            config.dropout
        )
        
        # Cross-attention to pool bytes into patch vectors
        self.cross_attn = nn.MultiheadAttention(
            config.h_encoder, 
            num_heads=max(1, config.h_encoder // 64), 
            batch_first=True
        )
        
        # Project to main model dimension
        self.patch_proj = nn.Linear(config.h_encoder, config.d_model)
        
    def forward(self, bytes_tensor, boundaries, valid_patches, first_call=False):
        batch_size, max_bytes = bytes_tensor.shape
        max_patches = boundaries.size(1)
        
        if first_call:
            print(f"🔧 Encoder input: bytes shape {bytes_tensor.shape}")
            print(f"   Boundaries shape: {boundaries.shape}")
            print(f"   Valid patches: {valid_patches.sum().item()}")
        
        # 1. Embed and transform bytes
        byte_embeds = self.byte_embed(bytes_tensor)
        byte_hidden = self.byte_transformer(byte_embeds)
        
        # 2. Create attention mask for all patches at once
        attn_mask = create_patch_mask(boundaries, valid_patches, max_bytes)
        
        # 3. Create queries by masked pooling
        queries = torch.zeros(batch_size, max_patches, self.config.h_encoder, device=bytes_tensor.device)
        
        for b in range(batch_size):
            for p in range(max_patches):
                if valid_patches[b, p]:
                    start, end = boundaries[b, p]
                    if start < end and end <= max_bytes:
                        queries[b, p] = byte_hidden[b, start:end].mean(dim=0)
        
        # 4. Batched cross-attention
        # Reshape for batched processing
        queries_flat = queries.view(-1, 1, self.config.h_encoder)  # [batch*max_patches, 1, h_encoder]
        
        # Expand byte_hidden for each patch
        byte_hidden_expanded = byte_hidden.unsqueeze(1).expand(-1, max_patches, -1, -1)
        keys_flat = byte_hidden_expanded.reshape(-1, max_bytes, self.config.h_encoder)
        values_flat = keys_flat
        
        # Create attention mask
        attn_mask_flat = attn_mask.view(-1, 1, max_bytes)
        attn_mask_flat = attn_mask_flat.masked_fill(attn_mask_flat == 0, float('-inf'))
        attn_mask_flat = attn_mask_flat.masked_fill(attn_mask_flat == 1, 0.0)
        
        # Single cross-attention call
        patch_vecs, _ = self.cross_attn(
            query=queries_flat,
            key=keys_flat,
            value=values_flat,
            attn_mask=attn_mask_flat
        )
        
        # Reshape back
        patch_vecs = patch_vecs.view(batch_size, max_patches, self.config.h_encoder)
        
        # 5. Project to main model dimension
        patch_vecs_projected = self.patch_proj(patch_vecs)
        
        if first_call:
            print(f"   Patch vectors shape: {patch_vecs_projected.shape}")
            print(f"   First patch vector norm: {patch_vecs_projected[0, 0].norm().item():.3f}")
        
        return patch_vecs_projected, byte_hidden

class LocalDecoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                config.h_decoder, 
                num_heads=max(1, config.h_decoder // 64), 
                batch_first=True
            )
            for _ in range(config.n_decoder_layers)
        ])
        
        # Transformer layers for bytes
        self.byte_transformer_layers = nn.ModuleList([
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
        self.output_proj = nn.Linear(config.h_decoder, config.vocab_size)
        
    def forward(self, patch_vecs, byte_hidden_from_encoder, boundaries, valid_patches, first_call=False):
        batch_size = patch_vecs.size(0)
        max_bytes = byte_hidden_from_encoder.size(1)
        max_patches = boundaries.size(1)
        
        if first_call:
            print(f"🔧 Decoder input: patch_vecs shape {patch_vecs.shape}")
            print(f"   Byte hidden shape: {byte_hidden_from_encoder.shape}")
        
        # Project patch vectors and byte hidden to decoder dimension
        patch_vecs_dec = self.patch_proj(patch_vecs)
        byte_reprs = self.byte_proj(byte_hidden_from_encoder)
        
        # Create byte-to-patch mapping
        byte_to_patch = torch.zeros(batch_size, max_bytes, dtype=torch.long, device=patch_vecs.device)
        
        for b in range(batch_size):
            for p in range(max_patches):
                if valid_patches[b, p]:
                    start, end = boundaries[b, p]
                    if start < end and end <= max_bytes:
                        byte_to_patch[b, start:end] = p
        
        # Apply decoder layers
        for i, (cross_attn, transformer) in enumerate(zip(self.cross_attn_layers, self.byte_transformer_layers)):
            # Get patch vectors for each byte position
            patch_vecs_for_bytes = torch.gather(
                patch_vecs_dec, 
                dim=1, 
                index=byte_to_patch.unsqueeze(-1).expand(-1, -1, self.config.h_decoder)
            )
            
            # Batched cross-attention
            enhanced_bytes, _ = cross_attn(
                query=byte_reprs,
                key=patch_vecs_for_bytes,
                value=patch_vecs_for_bytes
            )
            
            # Residual connection and transformer
            byte_reprs = enhanced_bytes + byte_reprs
            byte_reprs = transformer(byte_reprs)
            
            if first_call and i == 0:
                print(f"   After decoder layer {i}: {byte_reprs.shape}")
        
        # Project to byte vocabulary
        logits = self.output_proj(byte_reprs)
        
        if first_call:
            print(f"   Final logits shape: {logits.shape}")
        
        return logits

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

            step += 1
            if step % 50 == 0:
                pbar.update(50)

    pbar.close()

    training_time = time.time() - start_time
    print(f"  ⏱️ Training completed in {training_time:.1f} seconds")

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