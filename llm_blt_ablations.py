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
from typing import List, Optional, Dict, Tuple
import warnings
import os
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
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
    batch_size: int = 8  # Reduced for BLT complexity
    max_steps: int = 5000

    # Training parameters
    gradient_accumulation_steps: int = 4
    muon_lr: float = 0.01

    # Data parameters
    max_seq_len: int = 128  # In patches, not bytes
    num_documents: int = 1000  # Reduced for initial testing
    max_tokens: int = 200000

    # BLT specific parameters
    patch_size: int = 8  # Fixed patch size for simplicity
    local_d_model: int = 192  # Smaller dimension for local models
    local_n_layers: int = 2  # Lightweight local transformers
    max_patch_len: int = 16  # Maximum bytes per patch

    # Evaluation
    eval_every: int = 500
    eval_steps: int = 50

    # Regularization
    weight_decay: float = 0.1
    dropout: float = 0.1
    grad_clip: float = 1.0

    # Technical
    use_amp: bool = True
    vocab_size: int = 258  # 256 bytes + PAD + EOS

    # Ablation parameters
    track_metrics_every: int = 50  # Track detailed metrics every N steps
    save_checkpoints: bool = True
    checkpoint_dir: str = "ablation_checkpoints"

    def __post_init__(self):
        self.d_k = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"

@dataclass
class AblationMetrics:
    """Container for tracking ablation metrics"""
    step: int
    threshold: float
    
    # Training dynamics
    train_loss: float
    val_loss: float
    grad_norm: float
    learning_rate: float
    steps_per_second: float
    peak_memory_mb: float
    
    # Patch statistics
    avg_patch_size: float
    std_patch_size: float
    min_patch_size: int
    max_patch_size: int
    total_patches: int
    patches_per_1000_bytes: float
    
    # Component losses
    encoder_loss: float
    latent_loss: float
    decoder_loss: float

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
	
def load_and_cache_data(config: ModelConfig, cache_dir: str = "data_cache"):
    """Load and cache tokenized data to avoid reprocessing"""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/tokenized_data_{config.num_documents}_{config.max_tokens}.pkl"

    # Check if cached data exists
    if os.path.exists(cache_file):
        print(f"📦 Loading cached data from {cache_file}")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)

        texts = cached_data['texts']
        tokenizer = cached_data['tokenizer']
        tokens = cached_data['tokens']
        config.vocab_size = tokenizer.vocab_size

        print(f"✅ Loaded {len(texts)} documents, {len(tokens):,} tokens from cache")
        return texts, tokenizer, tokens

    print(f"🔄 Processing new data (will cache for future use)")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", token=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True, token=False)

    texts = []
    for i, item in enumerate(dataset):
        if i >= config.num_documents:
            break
        texts.append(item["text"][:3000])

    print(f"Loaded {len(texts)} documents")

    # Tokenize
    print("Tokenizing texts...")
    all_tokens = []
    for text in tqdm(texts, desc="Tokenizing"):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(tokens)

    tokens = all_tokens[:config.max_tokens]
    print(f"Using {len(tokens):,} tokens")
    config.vocab_size = tokenizer.vocab_size

    # Cache the processed data
    cached_data = {'texts': texts, 'tokenizer': tokenizer, 'tokens': tokens}
    with open(cache_file, 'wb') as f:
        pickle.dump(cached_data, f)

    print(f"💾 Cached data to {cache_file}")
    return texts, tokenizer, tokens

class BytePatchDataset(Dataset):
    """Dataset that works with byte patches for BLT"""
    def __init__(self, texts: List[str], patcher: FixedSizePatcher, seq_len: int = 128):
        self.texts = texts
        self.patcher = patcher
        self.seq_len = seq_len  # sequence length in patches
        
        # Pre-patch all texts
        print("Creating patches from texts...")
        self.all_patches = []
        for text in tqdm(texts, desc="Patching texts"):
            byte_sequence = list(text.encode('utf-8', errors='ignore'))
            if len(byte_sequence) > 0:  # Skip empty texts
                patches = self.patcher.patch(byte_sequence)
                self.all_patches.extend(patches)
        
        print(f"Created {len(self.all_patches)} total patches")
        
    def __len__(self):
        return max(0, len(self.all_patches) - self.seq_len - 1)
    
    def __getitem__(self, idx):
        # Get sequence of patches
        input_patches = self.all_patches[idx:idx + self.seq_len]
        target_patches = self.all_patches[idx + 1:idx + self.seq_len + 1]
        return input_patches, target_patches

def collate_fn(batch):
    """Custom collate function for BLT - just return first item since we use batch_size=1"""
    return batch[0]

def train_entropy_model(config: ModelConfig, texts: List[str], save_path: str = "entropy_model.pt"):
    """Train a small ByteLM model for entropy calculation"""
    print(f"\n🧠 Training ByteLM entropy model...")
    
    # Create smaller config for entropy model
    entropy_config = ModelConfig()
    entropy_config.d_model = 256
    entropy_config.n_layers = 4
    entropy_config.n_heads = 4
    entropy_config.d_ff = 1024
    entropy_config.max_seq_len = 512
    entropy_config.batch_size = 16
    entropy_config.max_steps = 2000  # Don't need to train for long
    entropy_config.vocab_size = 258
    
    # Create entropy model
    entropy_model = ByteLM(
        vocab_size=entropy_config.vocab_size,
        d_model=entropy_config.d_model,
        n_heads=entropy_config.n_heads,
        n_layers=entropy_config.n_layers,
        d_ff=entropy_config.d_ff,
        max_seq_len=entropy_config.max_seq_len
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    entropy_model = entropy_model.to(device)
    
    print(f"  📊 Entropy model parameters: {sum(p.numel() for p in entropy_model.parameters()):,}")
    
    # Create byte-level dataset
    all_bytes = []
    for text in texts[:500]:  # Use subset for entropy model training
        byte_sequence = list(text.encode('utf-8', errors='ignore'))
        all_bytes.extend(byte_sequence)
        all_bytes.append(257)  # EOS token
    
    # Limit to reasonable size
    all_bytes = all_bytes[:100000]
    byte_dataset = TextTokenDataset(all_bytes, entropy_config.max_seq_len)
    
    train_loader = DataLoader(byte_dataset, batch_size=entropy_config.batch_size, shuffle=True)
    
    # Simple optimizer
    optimizer = torch.optim.AdamW(entropy_model.parameters(), lr=1e-3, weight_decay=0.1)
    
    # Training loop
    entropy_model.train()
    step = 0
    pbar = tqdm(total=entropy_config.max_steps, desc="Training ByteLM")
    
    while step < entropy_config.max_steps:
        for x, y in train_loader:
            if step >= entropy_config.max_steps:
                break
            
            x, y = x.to(device), y.to(device)
            
            logits = entropy_model(x)
            loss = F.cross_entropy(logits.view(-1, entropy_config.vocab_size), y.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if step % 100 == 0:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            step += 1
            if step % 100 == 0:
                pbar.update(100)
    
    pbar.close()
    
    # Save the model
    torch.save({
        'model_state_dict': entropy_model.state_dict(),
        'vocab_size': entropy_config.vocab_size,
        'd_model': entropy_config.d_model,
        'n_heads': entropy_config.n_heads,
        'n_layers': entropy_config.n_layers,
        'd_ff': entropy_config.d_ff,
        'max_seq_len': entropy_config.max_seq_len
    }, save_path)
    
    print(f"💾 Entropy model saved to {save_path}")
    return entropy_model

def load_entropy_model(model_path: str = "entropy_model.pt"):
    """Load pre-trained entropy model"""
    checkpoint = torch.load(model_path, map_location='cpu')
    
    entropy_model = ByteLM(
        vocab_size=checkpoint['vocab_size'],
        d_model=checkpoint['d_model'],
        n_heads=checkpoint['n_heads'],
        n_layers=checkpoint['n_layers'],
        d_ff=checkpoint['d_ff'],
        max_seq_len=checkpoint['max_seq_len']
    )
    
    entropy_model.load_state_dict(checkpoint['model_state_dict'])
    entropy_model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    entropy_model = entropy_model.to(device)
    
    return entropy_model

def analyze_patch_statistics(texts: List[str], patcher, num_samples: int = 100, verbose: bool = True) -> Dict:
    """Analyze patch size distribution for a given patcher"""
    if verbose:
        print(f"\n📊 Analyzing patch statistics...")
    
    patch_lengths = []
    total_patches = 0
    total_bytes = 0
    
    for i, text in enumerate(texts[:num_samples]):
        byte_sequence = list(text.encode('utf-8', errors='ignore'))
        if len(byte_sequence) > 0:
            patches = patcher.patch(byte_sequence)
            lengths = [len(patch) for patch in patches]
            patch_lengths.extend(lengths)
            total_patches += len(patches)
            total_bytes += len(byte_sequence)
    
    stats = {}
    if patch_lengths:
        stats = {
            'avg_patch_size': np.mean(patch_lengths),
            'std_patch_size': np.std(patch_lengths),
            'min_patch_size': min(patch_lengths),
            'max_patch_size': max(patch_lengths),
            'total_patches': total_patches,
            'patches_per_1000_bytes': (total_patches / total_bytes) * 1000 if total_bytes > 0 else 0,
            'patch_lengths': patch_lengths
        }
        
        if verbose:
            print(f"  Average patch length: {stats['avg_patch_size']:.2f} ± {stats['std_patch_size']:.2f}")
            print(f"  Min/Max patch length: {stats['min_patch_size']}/{stats['max_patch_size']}")
            print(f"  Total patches analyzed: {stats['total_patches']}")
            print(f"  Patches per 1000 bytes: {stats['patches_per_1000_bytes']:.1f}")
            
            # Show distribution
            hist, bins = np.histogram(patch_lengths, bins=range(1, stats['max_patch_size'] + 2))
            print(f"  Length distribution:")
            for i, count in enumerate(hist[:10]):  # Show first 10 bins
                if count > 0:
                    print(f"    Length {i+1}: {count} patches ({count/len(patch_lengths)*100:.1f}%)")
    
    return stats

def generate_with_blt(model: BLT, patcher, prompt: str = "The", max_patches: int = 20, temperature: float = 0.8):
    """Generate text using the BLT model"""
    model.eval()
    device = next(model.parameters()).device
    
    # Convert prompt to patches
    prompt_bytes = list(prompt.encode('utf-8'))
    input_patches = patcher.patch(prompt_bytes)
    
    if not input_patches:
        return prompt
    
    generated_patches = input_patches.copy()
    
    with torch.no_grad():
        for _ in range(max_patches):
            # Encode current patches
            patch_embeddings = model.local_encoder(generated_patches)
            
            # Get next patch embedding prediction
            predicted_embeddings = model.latent_transformer(patch_embeddings)
            next_patch_embedding = predicted_embeddings[0, -1:, :]  # Last position
            
            # Decode to get next patch bytes
            # For simplicity, we'll use the average patch size from input
            avg_patch_size = int(np.mean([len(p) for p in input_patches]))
            target_size = min(avg_patch_size, 8)  # Cap at 8 bytes
            
            # Create dummy target for decoding (we'll ignore the loss)
            dummy_target = torch.zeros(target_size, dtype=torch.long, device=device)
            byte_logits = model.local_decoder(next_patch_embedding, dummy_target)
            
            # Sample bytes for the next patch
            next_patch_bytes = []
            for i in range(target_size):
                logits = byte_logits[0, i, :] / temperature
                probs = F.softmax(logits, dim=-1)
                next_byte = torch.multinomial(probs, 1).item()
                
                # Stop if we hit EOS or PAD
                if next_byte >= 256:
                    break
                next_patch_bytes.append(next_byte)
            
            if not next_patch_bytes:
                break
                
            # Add the new patch
            generated_patches.append(torch.tensor(next_patch_bytes, dtype=torch.long))
            
            # Keep only recent patches to avoid memory issues
            if len(generated_patches) > 50:
                generated_patches = generated_patches[-25:]
    
    # Convert patches back to text
    all_bytes = []
    for patch in generated_patches:
        all_bytes.extend(patch.tolist())
    
    try:
        # Filter out special tokens and convert to text
        valid_bytes = [b for b in all_bytes if b < 256]
        return bytes(valid_bytes).decode('utf-8', errors='replace')
    except:
        return prompt + "[decode error]"

class TextTokenDataset(Dataset):
    def __init__(self, tokens: List[int], seq_len: int = 512):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx:idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.tokens[idx + 1:idx + self.seq_len + 1], dtype=torch.long)
        return x, y

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

# BLT Components

class ByteLM(nn.Module):
    """Small byte-level language model for entropy calculation"""
    def __init__(self, vocab_size=258, d_model=256, n_heads=4, n_layers=4, d_ff=1024, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_dropout = nn.Dropout(dropout)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, max_seq_len, dropout)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # Weight tying
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, byte_ids):
        x = self.token_embedding(byte_ids) * math.sqrt(self.d_model)
        x = self.position_dropout(x)
        
        for block in self.transformer_blocks:
            x = block(x)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

def calculate_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Calculate entropy of a distribution from logits"""
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy

class EntropyPatcher:
    """Dynamic patcher based on next-byte prediction entropy"""
    def __init__(self, entropy_model: nn.Module, threshold: float = 2.0):
        self.entropy_model = entropy_model
        self.threshold = threshold
        self.entropy_model.eval()
        self.device = next(entropy_model.parameters()).device
    
    def patch(self, byte_sequence: List[int]) -> List[torch.Tensor]:
        """Split byte sequence based on prediction entropy"""
        if not byte_sequence:
            return []
        
        # Limit sequence length to avoid memory issues
        max_len = min(len(byte_sequence), 1024)
        byte_sequence = byte_sequence[:max_len]
        
        byte_tensor = torch.tensor(byte_sequence, dtype=torch.long, device=self.device).unsqueeze(0)
        
        # Get entropy predictions
        with torch.no_grad():
            logits = self.entropy_model(byte_tensor)
            entropies = calculate_entropy(logits).squeeze(0)
        
        patches = []
        current_patch = []
        
        # First byte always starts a patch
        current_patch.append(byte_sequence[0])
        
        # Iterate through remaining bytes
        for i in range(1, len(byte_sequence)):
            # Check entropy of predicting current byte
            if i-1 < len(entropies) and entropies[i-1].item() > self.threshold:
                # High entropy - start new patch
                if current_patch:
                    patches.append(torch.tensor(current_patch, dtype=torch.long))
                current_patch = []
            
            current_patch.append(byte_sequence[i])
            
            # Prevent patches from getting too long
            if len(current_patch) >= 16:
                patches.append(torch.tensor(current_patch, dtype=torch.long))
                current_patch = []
        
        # Add final patch
        if current_patch:
            patches.append(torch.tensor(current_patch, dtype=torch.long))
        
        return patches

class FixedSizePatcher:
    """Simple fixed-size patcher for BLT"""
    def __init__(self, patch_size=8):
        self.patch_size = patch_size

    def patch(self, byte_sequence: List[int]) -> List[torch.Tensor]:
        """Splits a sequence of byte IDs into patches"""
        patches = []
        for i in range(0, len(byte_sequence), self.patch_size):
            patch_bytes = byte_sequence[i:i + self.patch_size]
            # Pad short patches with 256 (PAD token)
            while len(patch_bytes) < self.patch_size:
                patch_bytes.append(256)
            patches.append(torch.tensor(patch_bytes, dtype=torch.long))
        return patches

class LocalEncoder(nn.Module):
    """Local encoder that converts byte patches to patch embeddings"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Byte embedding (256 bytes + PAD + EOS)
        self.byte_embedding = nn.Embedding(config.vocab_size, config.local_d_model)
        
        # Small transformer for processing bytes within a patch
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config.local_d_model, config.n_heads, config.d_ff // 2, 
                           config.max_patch_len, config.dropout)
            for _ in range(config.local_n_layers)
        ])
        
        self.norm = nn.RMSNorm(config.local_d_model)
        
        # Project to patch embedding dimension
        self.to_patch_embedding = nn.Linear(config.local_d_model, config.d_model)
        
    def forward(self, byte_patches: List[torch.Tensor]) -> torch.Tensor:
        """Convert list of byte patches to patch embeddings"""
        device = self.byte_embedding.weight.device
        patch_embeddings = []
        
        for patch in byte_patches:
            patch = patch.to(device)
            # Embed bytes and add batch dimension
            embedded_bytes = self.byte_embedding(patch.unsqueeze(0))  # (1, patch_len, local_d_model)
            
            # Process with local transformer
            x = embedded_bytes
            for block in self.transformer_blocks:
                x = block(x)
            
            # Mean pooling to get single vector per patch
            pooled = self.norm(x).mean(dim=1)  # (1, local_d_model)
            
            # Project to patch embedding
            patch_emb = self.to_patch_embedding(pooled)  # (1, d_model)
            patch_embeddings.append(patch_emb)
        
        # Stack into sequence: (1, num_patches, d_model)
        return torch.cat(patch_embeddings, dim=0).unsqueeze(0)

class LocalDecoder(nn.Module):
    """Local decoder that converts patch embeddings back to bytes"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Project patch embedding to local dimension
        self.from_patch_embedding = nn.Linear(config.d_model, config.local_d_model)
        
        # Byte embedding for autoregressive decoding
        self.byte_embedding = nn.Embedding(config.vocab_size, config.local_d_model)
        
        # Small transformer for generating bytes
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config.local_d_model, config.n_heads, config.d_ff // 2,
                           config.max_patch_len, config.dropout)
            for _ in range(config.local_n_layers)
        ])
        
        self.norm = nn.RMSNorm(config.local_d_model)
        self.to_byte_logits = nn.Linear(config.local_d_model, config.vocab_size)
        
    def forward(self, patch_embedding: torch.Tensor, target_bytes: torch.Tensor):
        """Decode patch embedding to byte sequence with teacher forcing"""
        # patch_embedding: (1, d_model)
        # target_bytes: (patch_len,)
        
        # Project patch embedding to local space
        patch_context = self.from_patch_embedding(patch_embedding)  # (1, local_d_model)
        
        # Embed target bytes (for teacher forcing)
        target_embedded = self.byte_embedding(target_bytes.unsqueeze(0))  # (1, patch_len, local_d_model)
        
        # Add patch context to each position
        x = target_embedded + patch_context.unsqueeze(1)  # Broadcast patch context
        
        # Process with transformer
        for block in self.transformer_blocks:
            x = block(x)
        
        # Generate byte logits
        logits = self.to_byte_logits(self.norm(x))  # (1, patch_len, vocab_size)
        return logits

class LatentTransformer(nn.Module):
    """The main transformer that operates on patch embeddings (repurposed MinimalLLM)"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # No token embedding - we receive dense patch embeddings directly
        self.position_dropout = nn.Dropout(config.dropout)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.max_seq_len, config.dropout)
            for _ in range(config.n_layers)
        ])

        self.norm = nn.RMSNorm(config.d_model)
        self.output_dropout = nn.Dropout(config.dropout)

        # Output head predicts next patch embedding
        self.lm_head = nn.Linear(config.d_model, config.d_model, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, patch_embeddings):
        # patch_embeddings: (batch_size, num_patches, d_model)
        x = patch_embeddings * math.sqrt(self.config.d_model)
        x = self.position_dropout(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)
        x = self.output_dropout(x)
        predicted_patch_embeddings = self.lm_head(x)
        return predicted_patch_embeddings

class BLT(nn.Module):
    """Byte Latent Transformer - combines local encoder, latent transformer, and local decoder"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # The three main components
        self.local_encoder = LocalEncoder(config)
        self.latent_transformer = LatentTransformer(config)
        self.local_decoder = LocalDecoder(config)
        
    def forward(self, input_patches: List[torch.Tensor], target_patches: List[torch.Tensor], return_component_losses: bool = False):
        """Forward pass for training"""
        device = next(self.parameters()).device
        
        # Move patches to device
        input_patches = [p.to(device) for p in input_patches]
        target_patches = [p.to(device) for p in target_patches]
        
        # Step 1: Encode input patches to embeddings
        patch_embeddings = self.local_encoder(input_patches)  # (1, num_patches, d_model)
        
        # Step 2: Latent transformer predicts next patch embeddings
        predicted_embeddings = self.latent_transformer(patch_embeddings)  # (1, num_patches, d_model)
        
        # Step 3: Decode each predicted embedding and compute loss
        total_loss = 0
        decoder_losses = []
        num_patches = len(target_patches)
        
        for i in range(num_patches):
            # Get predicted embedding for patch i
            pred_emb = predicted_embeddings[0, i:i+1, :]  # (1, d_model)
            
            # Get target bytes for patch i
            target_bytes = target_patches[i]
            
            # Decode to byte logits
            byte_logits = self.local_decoder(pred_emb, target_bytes)  # (1, patch_len, vocab_size)
            
            # Compute cross-entropy loss
            loss = F.cross_entropy(
                byte_logits.squeeze(0),  # (patch_len, vocab_size)
                target_bytes  # (patch_len,)
            )
            total_loss += loss
            decoder_losses.append(loss.item())
        
        avg_loss = total_loss / num_patches
        
        if return_component_losses:
            # For ablation analysis, compute component-specific losses
            # Encoder loss: reconstruction quality (simplified as embedding variance)
            encoder_loss = torch.var(patch_embeddings).item()
            
            # Latent loss: prediction consistency (MSE between consecutive predictions)
            if num_patches > 1:
                latent_loss = F.mse_loss(predicted_embeddings[0, :-1, :], predicted_embeddings[0, 1:, :]).item()
            else:
                latent_loss = 0.0
            
            # Decoder loss: average of all patch decoding losses
            decoder_loss = np.mean(decoder_losses)
            
            return avg_loss, {
                'encoder_loss': encoder_loss,
                'latent_loss': latent_loss,
                'decoder_loss': decoder_loss
            }
        
        return avg_loss

# Keep the original MinimalLLM for comparison
class MinimalLLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_dropout = nn.Dropout(config.dropout)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.max_seq_len, config.dropout)
            for _ in range(config.n_layers)
        ])

        self.norm = nn.RMSNorm(config.d_model)
        self.output_dropout = nn.Dropout(config.dropout)

        # Tie weights
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        x = self.token_embedding(x) * math.sqrt(self.config.d_model)
        x = self.position_dropout(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)
        x = self.output_dropout(x)
        logits = self.lm_head(x)
        return logits

def evaluate_model(model: nn.Module, val_loader: DataLoader, config: ModelConfig, is_blt: bool = False):
    """Evaluate model performance"""
    model.eval()
    total_loss = 0
    total_samples = 0

    device = next(model.parameters()).device

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= config.eval_steps:
                break
            
            if is_blt:
                input_patches, target_patches = batch
                with autocast(enabled=config.use_amp):
                    loss = model(input_patches, target_patches)
                total_loss += loss.item()
                total_samples += 1
            else:
                x, y = batch
                x, y = x.to(device), y.to(device)
                with autocast(enabled=config.use_amp):
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
                total_loss += loss.item()
                total_samples += 1

    avg_loss = total_loss / total_samples
    perplexity = math.exp(min(avg_loss, 20))

    model.train()
    return {'val_loss': avg_loss, 'val_perplexity': perplexity}

def setup_muon_optimizer(model: nn.Module, config: ModelConfig):
    """Setup Muon optimizer with hybrid approach"""
    muon_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if (param.ndim == 2 and 
            'token_embedding' not in name and 
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

def train_blt_model_with_metrics(config: ModelConfig, train_loader: DataLoader, val_loader: DataLoader, 
                                threshold: float, patcher, texts: List[str]) -> Tuple[nn.Module, Dict, List[AblationMetrics]]:
    """Train BLT model with detailed metric tracking for ablation study"""
    print(f"\n🚀 Training BLT model (threshold={threshold}) with detailed metrics")

    # Initialize model
    set_seed(42)
    model = BLT(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.local_encoder.parameters())
    latent_params = sum(p.numel() for p in model.latent_transformer.parameters())
    decoder_params = sum(p.numel() for p in model.local_decoder.parameters())
    
    print(f"  📊 Total parameters: {total_params:,}")
    print(f"     Local Encoder: {encoder_params:,}")
    print(f"     Latent Transformer: {latent_params:,}")
    print(f"     Local Decoder: {decoder_params:,}")

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

    # Metrics tracking
    metrics_history = []
    patch_stats = analyze_patch_statistics(texts, patcher, num_samples=100, verbose=False)

    # Training loop
    model.train()
    step = 0
    start_time = time.time()
    best_val_loss = float('inf')
    step_times = []

    pbar = tqdm(total=config.max_steps, desc=f"Training BLT (T={threshold})")

    while step < config.max_steps:
        step_start_time = time.time()
        
        for batch_idx, (input_patches, target_patches) in enumerate(train_loader):
            if step >= config.max_steps:
                break

            # Forward pass with gradient accumulation
            if config.use_amp:
                with autocast():
                    if step % config.track_metrics_every == 0:
                        loss, component_losses = model(input_patches, target_patches, return_component_losses=True)
                    else:
                        loss = model(input_patches, target_patches)
                        component_losses = None
                    loss = loss / config.gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                if step % config.track_metrics_every == 0:
                    loss, component_losses = model(input_patches, target_patches, return_component_losses=True)
                else:
                    loss = model(input_patches, target_patches)
                    component_losses = None
                loss = loss / config.gradient_accumulation_steps
                loss.backward()

            # Optimizer step after accumulation
            grad_norm = 0
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

            # Track detailed metrics
            if step % config.track_metrics_every == 0:
                step_end_time = time.time()
                step_times.append(step_end_time - step_start_time)
                
                # Get validation loss
                val_metrics = evaluate_model(model, val_loader, config, is_blt=True)
                
                # Memory usage
                if torch.cuda.is_available():
                    peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                    torch.cuda.reset_peak_memory_stats()
                else:
                    peak_memory_mb = 0
                
                # Steps per second
                recent_steps_per_sec = len(step_times) / sum(step_times) if step_times else 0
                
                # Create metrics object
                metrics = AblationMetrics(
                    step=step,
                    threshold=threshold,
                    train_loss=loss.item() * config.gradient_accumulation_steps,
                    val_loss=val_metrics['val_loss'],
                    grad_norm=grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    learning_rate=optimizers[0].param_groups[0]["lr"],
                    steps_per_second=recent_steps_per_sec,
                    peak_memory_mb=peak_memory_mb,
                    avg_patch_size=patch_stats['avg_patch_size'],
                    std_patch_size=patch_stats['std_patch_size'],
                    min_patch_size=patch_stats['min_patch_size'],
                    max_patch_size=patch_stats['max_patch_size'],
                    total_patches=patch_stats['total_patches'],
                    patches_per_1000_bytes=patch_stats['patches_per_1000_bytes'],
                    encoder_loss=component_losses['encoder_loss'] if component_losses else 0,
                    latent_loss=component_losses['latent_loss'] if component_losses else 0,
                    decoder_loss=component_losses['decoder_loss'] if component_losses else 0
                )
                
                metrics_history.append(metrics)
                step_times = []  # Reset for next batch

            # Logging
            if step % 100 == 0:
                current_loss = loss.item() * config.gradient_accumulation_steps
                perplexity = math.exp(min(current_loss, 20))

                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'ppl': f'{perplexity:.1f}',
                    'lr': f'{optimizers[0].param_groups[0]["lr"]:.2e}'
                })

            # Evaluation
            if step % config.eval_every == 0 and step > 0:
                eval_metrics = evaluate_model(model, val_loader, config, is_blt=True)
                if eval_metrics['val_loss'] < best_val_loss:
                    best_val_loss = eval_metrics['val_loss']

            step += 1
            if step % 100 == 0:
                pbar.update(100)

    pbar.close()

    training_time = time.time() - start_time
    print(f"  ⏱️ Training completed in {training_time:.1f} seconds")

    # Final evaluation
    final_eval = evaluate_model(model, val_loader, config, is_blt=True)
    print(f"  📊 Final - Loss: {final_eval['val_loss']:.4f}, PPL: {final_eval['val_perplexity']:.2f}")

    return model, final_eval, metrics_history

def train_model(config: ModelConfig, train_loader: DataLoader, val_loader: DataLoader):
    """Train the model with Muon optimizer"""
    print(f"\n🚀 Training Small model with Muon optimizer")

    # Initialize model
    set_seed(42)
    model = MinimalLLM(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  📊 Total parameters: {total_params:,}")

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

    pbar = tqdm(total=config.max_steps, desc="Training")

    while step < config.max_steps:
        for batch_idx, (x, y) in enumerate(train_loader):
            if step >= config.max_steps:
                break

            x, y = x.to(device), y.to(device)

            # Forward pass with gradient accumulation
            if config.use_amp:
                with autocast():
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
                    loss = loss / config.gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
                loss = loss / config.gradient_accumulation_steps
                loss.backward()

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
            if step % 100 == 0:
                with torch.no_grad():
                    predictions = logits.argmax(dim=-1)
                    accuracy = (predictions == y).float().mean().item()
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
                      f"Val PPL: {eval_metrics['val_perplexity']:.2f}")

                if eval_metrics['val_loss'] < best_val_loss:
                    best_val_loss = eval_metrics['val_loss']

            step += 1
            if step % 100 == 0:
                pbar.update(100)

    pbar.close()

    training_time = time.time() - start_time
    print(f"  ⏱️ Training completed in {training_time:.1f} seconds")

    # Final evaluation
    final_eval = evaluate_model(model, val_loader, config)
    print(f"  📊 Final - Loss: {final_eval['val_loss']:.4f}, PPL: {final_eval['val_perplexity']:.2f}")

    return model, final_eval

def save_ablation_results(all_metrics: Dict[float, List[AblationMetrics]], 
                         all_final_results: Dict[float, Dict], 
                         config: ModelConfig):
    """Save ablation results to files"""
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Save raw metrics
    metrics_data = {}
    for threshold, metrics_list in all_metrics.items():
        metrics_data[threshold] = [
            {
                'step': m.step,
                'threshold': m.threshold,
                'train_loss': m.train_loss,
                'val_loss': m.val_loss,
                'grad_norm': m.grad_norm,
                'learning_rate': m.learning_rate,
                'steps_per_second': m.steps_per_second,
                'peak_memory_mb': m.peak_memory_mb,
                'avg_patch_size': m.avg_patch_size,
                'std_patch_size': m.std_patch_size,
                'min_patch_size': m.min_patch_size,
                'max_patch_size': m.max_patch_size,
                'total_patches': m.total_patches,
                'patches_per_1000_bytes': m.patches_per_1000_bytes,
                'encoder_loss': m.encoder_loss,
                'latent_loss': m.latent_loss,
                'decoder_loss': m.decoder_loss
            }
            for m in metrics_list
        ]
    
    with open(f"{config.checkpoint_dir}/ablation_metrics.json", 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    # Save final results summary
    with open(f"{config.checkpoint_dir}/final_results.json", 'w') as f:
        json.dump(all_final_results, f, indent=2)
    
    print(f"💾 Ablation results saved to {config.checkpoint_dir}/")

def plot_ablation_results(all_metrics: Dict[float, List[AblationMetrics]], config: ModelConfig):
    """Create comprehensive plots for ablation study"""
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('BLT Ablation Study: Entropy Threshold Comparison', fontsize=16, fontweight='bold')
    
    thresholds = sorted(all_metrics.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(thresholds)))
    
    for i, threshold in enumerate(thresholds):
        metrics = all_metrics[threshold]
        steps = [m.step for m in metrics]
        color = colors[i]
        label = f'T={threshold}'
        
        # Training dynamics
        axes[0, 0].plot(steps, [m.train_loss for m in metrics], color=color, label=label, alpha=0.8)
        axes[0, 1].plot(steps, [m.val_loss for m in metrics], color=color, label=label, alpha=0.8)
        axes[0, 2].plot(steps, [m.grad_norm for m in metrics], color=color, label=label, alpha=0.8)
        
        # Performance metrics
        axes[1, 0].plot(steps, [m.steps_per_second for m in metrics], color=color, label=label, alpha=0.8)
        axes[1, 1].plot(steps, [m.peak_memory_mb for m in metrics], color=color, label=label, alpha=0.8)
        axes[1, 2].plot(steps, [m.learning_rate for m in metrics], color=color, label=label, alpha=0.8)
        
        # Component losses
        axes[2, 0].plot(steps, [m.encoder_loss for m in metrics], color=color, label=label, alpha=0.8)
        axes[2, 1].plot(steps, [m.latent_loss for m in metrics], color=color, label=label, alpha=0.8)
        axes[2, 2].plot(steps, [m.decoder_loss for m in metrics], color=color, label=label, alpha=0.8)
    
    # Set titles and labels
    titles = [
        ['Training Loss', 'Validation Loss', 'Gradient Norm'],
        ['Steps/Second', 'Peak Memory (MB)', 'Learning Rate'],
        ['Encoder Loss', 'Latent Loss', 'Decoder Loss']
    ]
    
    for i in range(3):
        for j in range(3):
            axes[i, j].set_title(titles[i][j], fontweight='bold')
            axes[i, j].set_xlabel('Training Step')
            axes[i, j].legend()
            axes[i, j].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{config.checkpoint_dir}/ablation_training_dynamics.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Patch statistics comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Patch Statistics Comparison', fontsize=16, fontweight='bold')
    
    patch_stats = {}
    for threshold in thresholds:
        metrics = all_metrics[threshold][0]  # Use first metric for patch stats
        patch_stats[threshold] = {
            'avg_size': metrics.avg_patch_size,
            'std_size': metrics.std_patch_size,
            'min_size': metrics.min_patch_size,
            'max_size': metrics.max_patch_size,
            'patches_per_1000_bytes': metrics.patches_per_1000_bytes
        }
    
    # Bar plots for patch statistics
    x_pos = np.arange(len(thresholds))
    
    axes[0, 0].bar(x_pos, [patch_stats[t]['avg_size'] for t in thresholds], color=colors)
    axes[0, 0].set_title('Average Patch Size')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels([f'T={t}' for t in thresholds])
    
    axes[0, 1].bar(x_pos, [patch_stats[t]['std_size'] for t in thresholds], color=colors)
    axes[0, 1].set_title('Patch Size Std Dev')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels([f'T={t}' for t in thresholds])
    
    axes[1, 0].bar(x_pos, [patch_stats[t]['max_size'] - patch_stats[t]['min_size'] for t in thresholds], color=colors)
    axes[1, 0].set_title('Patch Size Range')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels([f'T={t}' for t in thresholds])
    
    axes[1, 1].bar(x_pos, [patch_stats[t]['patches_per_1000_bytes'] for t in thresholds], color=colors)
    axes[1, 1].set_title('Patches per 1000 Bytes')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels([f'T={t}' for t in thresholds])
    
    plt.tight_layout()
    plt.savefig(f"{config.checkpoint_dir}/patch_statistics.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Plots saved to {config.checkpoint_dir}/")

def run_ablation_study():
    """Run comprehensive ablation study with 4 different entropy thresholds"""
    # Check system
    print(f"🔍 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Set seed
    set_seed(42)

    # Create config for BLT model
    config = ModelConfig()
    config.max_steps = 3000  # Reduced for ablation study
    config.track_metrics_every = 50
    
    print(f"\n📋 BLT Ablation Study Configuration:")
    print(f"   Architecture: {config.d_model}d, {config.n_layers}L, {config.n_heads}H, {config.d_ff}ff")
    print(f"   Local models: {config.local_d_model}d, {config.local_n_layers}L")
    print(f"   Training: {config.max_steps} steps, batch size {config.batch_size}")
    print(f"   Metrics tracked every {config.track_metrics_every} steps")

    # Load data
    texts, _, _ = load_and_cache_data(config)
    
    # Train or load entropy model
    entropy_model_path = "entropy_model.pt"
    if not os.path.exists(entropy_model_path):
        print(f"\n🔄 Training ByteLM entropy model...")
        entropy_model = train_entropy_model(config, texts, entropy_model_path)
    else:
        print(f"\n📦 Loading existing entropy model...")
        entropy_model = load_entropy_model(entropy_model_path)
    
    # Define ablation thresholds
    ablation_thresholds = [1.5, 2.0, 2.5, 3.0]
    model_names = ['Model A', 'Model B', 'Model C', 'Model D']
    
    print(f"\n🧪 ABLATION STUDY: Training {len(ablation_thresholds)} BLT models")
    print(f"   Model A: Threshold = 1.5 (frequent splits, small patches)")
    print(f"   Model B: Threshold = 2.0 (balanced splits)")
    print(f"   Model C: Threshold = 2.5 (occasional splits)")
    print(f"   Model D: Threshold = 3.0 (rare splits, large patches)")
    
    # Storage for results
    all_models = {}
    all_final_results = {}
    all_metrics = {}
    
    # Run ablation for each threshold
    for i, threshold in enumerate(ablation_thresholds):
        model_name = model_names[i]
        print(f"\n{'='*60}")
        print(f"🚀 TRAINING {model_name} (Threshold = {threshold})")
        print(f"{'='*60}")
        
        # Create patcher for this threshold
        patcher = EntropyPatcher(entropy_model, threshold=threshold)
        
        # Analyze patch statistics
        patch_stats = analyze_patch_statistics(texts, patcher, num_samples=100, verbose=True)
        
        # Create dataset
        dataset = BytePatchDataset(texts, patcher, config.max_seq_len)
        
        # Train/val split (use same seed for consistency)
        val_size = len(dataset) // 10
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )
        
        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)
        
        print(f"📊 Dataset: {len(train_dataset)} train, {len(val_dataset)} val samples")
        
        # Train model with detailed metrics
        start_time = time.time()
        model, final_results, metrics_history = train_blt_model_with_metrics(
            config, train_loader, val_loader, threshold, patcher, texts
        )
        total_time = time.time() - start_time
        
        # Store results
        all_models[threshold] = model
        all_final_results[threshold] = {
            'model_name': model_name,
            'threshold': threshold,
            'final_val_loss': final_results['val_loss'],
            'final_val_perplexity': final_results['val_perplexity'],
            'training_time_minutes': total_time / 60,
            'patch_stats': patch_stats
        }
        all_metrics[threshold] = metrics_history
        
        print(f"\n✅ {model_name} COMPLETED!")
        print(f"   Final Val Loss: {final_results['val_loss']:.4f}")
        print(f"   Final Val Perplexity: {final_results['val_perplexity']:.2f}")
        print(f"   Training Time: {total_time/60:.1f} minutes")
        print(f"   Avg Patch Size: {patch_stats['avg_patch_size']:.2f}")
        
        # Save individual model
        if config.save_checkpoints:
            os.makedirs(config.checkpoint_dir, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'threshold': threshold,
                'final_results': final_results,
                'metrics_history': metrics_history,
                'patch_stats': patch_stats
            }, f'{config.checkpoint_dir}/model_{model_name.lower().replace(" ", "_")}_threshold_{threshold}.pt')
    
    # Final analysis and comparison
    print(f"\n{'='*80}")
    print(f"🎉 ABLATION STUDY COMPLETED!")
    print(f"{'='*80}")
    
    print(f"\n📊 FINAL RESULTS SUMMARY:")
    print(f"{'Model':<10} {'Threshold':<10} {'Val Loss':<10} {'Val PPL':<10} {'Avg Patch':<12} {'Time (min)':<10}")
    print(f"{'-'*70}")
    
    for i, threshold in enumerate(ablation_thresholds):
        results = all_final_results[threshold]
        print(f"{results['model_name']:<10} {threshold:<10.1f} {results['final_val_loss']:<10.4f} "
              f"{results['final_val_perplexity']:<10.2f} {results['patch_stats']['avg_patch_size']:<12.2f} "
              f"{results['training_time_minutes']:<10.1f}")
    
    # Find best model
    best_threshold = min(ablation_thresholds, key=lambda t: all_final_results[t]['final_val_loss'])
    best_model_name = all_final_results[best_threshold]['model_name']
    
    print(f"\n🏆 BEST MODEL: {best_model_name} (Threshold = {best_threshold})")
    print(f"   Validation Loss: {all_final_results[best_threshold]['final_val_loss']:.4f}")
    print(f"   Validation Perplexity: {all_final_results[best_threshold]['final_val_perplexity']:.2f}")
    
    # Save results and create plots
    save_ablation_results(all_metrics, all_final_results, config)
    plot_ablation_results(all_metrics, config)
    
    print(f"\n📈 Key Insights:")
    patch_sizes = [all_final_results[t]['patch_stats']['avg_patch_size'] for t in ablation_thresholds]
    val_losses = [all_final_results[t]['final_val_loss'] for t in ablation_thresholds]
    
    print(f"   Patch size range: {min(patch_sizes):.2f} - {max(patch_sizes):.2f}")
    print(f"   Val loss range: {min(val_losses):.4f} - {max(val_losses):.4f}")
    print(f"   Best threshold appears to be: {best_threshold}")
    
    return all_models, all_final_results, all_metrics

if __name__ == "__main__":
    run_ablation_study()