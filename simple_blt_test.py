import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import random
import numpy as np
from tqdm import tqdm
import time
from dataclasses import dataclass
from typing import List
import warnings
warnings.filterwarnings('ignore')

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"🌱 Set all seeds to {seed}")

@dataclass
class SimpleConfig:
    # Small model for testing
    d_model: int = 128  # Must be divisible by n_heads
    n_heads: int = 4    # 128 / 4 = 32 (d_k)
    n_layers: int = 2
    d_ff: int = 512
    max_steps: int = 200
    
    # Local models (even smaller)
    local_d_model: int = 64  # Must be divisible by local n_heads (2)
    local_n_layers: int = 1
    patch_size: int = 4
    max_patch_len: int = 8
    
    # Training
    lr: float = 1e-3
    batch_size: int = 1
    max_seq_len: int = 32  # Very short sequences
    
    # Data
    vocab_size: int = 258
    dropout: float = 0.1
    
    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        assert self.local_d_model % 2 == 0, f"local_d_model ({self.local_d_model}) must be divisible by 2 (for local attention)"

class SimpleRotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        # Create rotary frequencies
        angular_freq = (1 / 10000) ** torch.linspace(0, 1, steps=dim//2, dtype=torch.float32)
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.register_buffer('cos', theta.cos(), persistent=False)
        self.register_buffer('sin', theta.sin(), persistent=False)

    def forward(self, x):
        # x shape: (batch_size, n_heads, seq_len, d_k)
        seq_len = x.size(-2)
        cos = self.cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, d_k//2)
        sin = self.sin[:seq_len, :].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, d_k//2)
        
        # Split x into two halves
        x1, x2 = x.chunk(2, dim=-1)
        
        # Apply rotation
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        
        return torch.cat((y1, y2), dim=-1).type_as(x)

class SimpleAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.rotary = SimpleRotary(self.d_k, max_seq_len)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv(x)  # (batch_size, seq_len, d_model * 3)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, n_heads, seq_len, d_k)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        # Apply rotary embeddings
        Q = self.rotary(Q)
        K = self.rotary(K)
        
        # Attention
        attn_output = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()  # (batch_size, seq_len, n_heads, d_k)
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)
        
        return self.w_o(attn_output)

class SimpleFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.linear2(F.silu(self.linear1(x)))

class SimpleTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int):
        super().__init__()
        self.attention = SimpleAttention(d_model, n_heads, max_seq_len)
        self.feed_forward = SimpleFeedForward(d_model, d_ff)
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x

class FixedSizePatcher:
    def __init__(self, patch_size=4):
        self.patch_size = patch_size

    def patch(self, byte_sequence: List[int]) -> List[torch.Tensor]:
        patches = []
        for i in range(0, len(byte_sequence), self.patch_size):
            patch_bytes = byte_sequence[i:i + self.patch_size]
            while len(patch_bytes) < self.patch_size:
                patch_bytes.append(256)  # PAD token
            patches.append(torch.tensor(patch_bytes, dtype=torch.long))
        return patches

class SimpleLocalEncoder(nn.Module):
    def __init__(self, config: SimpleConfig):
        super().__init__()
        self.byte_embedding = nn.Embedding(config.vocab_size, config.local_d_model)
        self.transformer = SimpleTransformerBlock(config.local_d_model, 2, config.local_d_model * 2, config.max_patch_len)
        self.norm = nn.RMSNorm(config.local_d_model)
        self.to_patch_embedding = nn.Linear(config.local_d_model, config.d_model)
        
    def forward(self, byte_patches: List[torch.Tensor]) -> torch.Tensor:
        device = self.byte_embedding.weight.device
        patch_embeddings = []
        
        for patch in byte_patches:
            patch = patch.to(device)
            embedded_bytes = self.byte_embedding(patch.unsqueeze(0))
            x = self.transformer(embedded_bytes)
            pooled = self.norm(x).mean(dim=1)
            patch_emb = self.to_patch_embedding(pooled)
            patch_embeddings.append(patch_emb)
        
        return torch.cat(patch_embeddings, dim=0).unsqueeze(0)

class SimpleLatentTransformer(nn.Module):
    def __init__(self, config: SimpleConfig):
        super().__init__()
        self.blocks = nn.ModuleList([
            SimpleTransformerBlock(config.d_model, config.n_heads, config.d_ff, config.max_seq_len)
            for _ in range(config.n_layers)
        ])
        self.norm = nn.RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, patch_embeddings):
        x = patch_embeddings
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.lm_head(x)

class SimpleLocalDecoder(nn.Module):
    def __init__(self, config: SimpleConfig):
        super().__init__()
        self.from_patch_embedding = nn.Linear(config.d_model, config.local_d_model)
        self.byte_embedding = nn.Embedding(config.vocab_size, config.local_d_model)
        self.transformer = SimpleTransformerBlock(config.local_d_model, 2, config.local_d_model * 2, config.max_patch_len)
        self.norm = nn.RMSNorm(config.local_d_model)
        self.to_byte_logits = nn.Linear(config.local_d_model, config.vocab_size)
        
    def forward(self, patch_embedding: torch.Tensor, target_bytes: torch.Tensor):
        patch_context = self.from_patch_embedding(patch_embedding)
        target_embedded = self.byte_embedding(target_bytes.unsqueeze(0))
        x = target_embedded + patch_context.unsqueeze(1)
        x = self.transformer(x)
        logits = self.to_byte_logits(self.norm(x))
        return logits

class SimpleBLT(nn.Module):
    def __init__(self, config: SimpleConfig):
        super().__init__()
        self.config = config
        self.local_encoder = SimpleLocalEncoder(config)
        self.latent_transformer = SimpleLatentTransformer(config)
        self.local_decoder = SimpleLocalDecoder(config)
        
    def forward(self, input_patches: List[torch.Tensor], target_patches: List[torch.Tensor]):
        device = next(self.parameters()).device
        input_patches = [p.to(device) for p in input_patches]
        target_patches = [p.to(device) for p in target_patches]
        
        # Encode patches
        patch_embeddings = self.local_encoder(input_patches)
        
        # Latent transformer
        predicted_embeddings = self.latent_transformer(patch_embeddings)
        
        # Decode and compute loss
        total_loss = 0
        num_patches = min(len(target_patches), predicted_embeddings.size(1))
        
        for i in range(num_patches):
            pred_emb = predicted_embeddings[0, i:i+1, :]
            target_bytes = target_patches[i]
            byte_logits = self.local_decoder(pred_emb, target_bytes)
            loss = F.cross_entropy(byte_logits.squeeze(0), target_bytes)
            total_loss += loss
        
        return total_loss / num_patches if num_patches > 0 else total_loss

def simple_generate(model: SimpleBLT, patcher, prompt: str = "Hello", max_patches: int = 8):
    """Simple generation function"""
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
            try:
                # Encode current patches
                patch_embeddings = model.local_encoder(generated_patches[-8:])  # Use last 8 patches
                
                # Get next patch prediction
                predicted_embeddings = model.latent_transformer(patch_embeddings)
                next_patch_embedding = predicted_embeddings[0, -1:, :]
                
                # Decode to get next patch bytes
                target_size = patcher.patch_size
                dummy_target = torch.zeros(target_size, dtype=torch.long, device=device)
                byte_logits = model.local_decoder(next_patch_embedding, dummy_target)
                
                # Sample bytes for the next patch
                next_patch_bytes = []
                for i in range(target_size):
                    logits = byte_logits[0, i, :] / 0.8  # temperature
                    probs = F.softmax(logits, dim=-1)
                    next_byte = torch.multinomial(probs, 1).item()
                    
                    if next_byte >= 256:  # Stop at special tokens
                        break
                    next_patch_bytes.append(next_byte)
                
                if not next_patch_bytes:
                    break
                    
                generated_patches.append(torch.tensor(next_patch_bytes, dtype=torch.long))
                
            except Exception as e:
                break
    
    # Convert back to text
    all_bytes = []
    for patch in generated_patches:
        all_bytes.extend([b for b in patch.tolist() if b < 256])
    
    try:
        return bytes(all_bytes).decode('utf-8', errors='replace')
    except:
        return prompt + "[decode error]"

class SimpleDataset(Dataset):
    def __init__(self, texts: List[str], patcher, seq_len: int = 16):
        self.patcher = patcher
        self.seq_len = seq_len
        
        # Create patches from texts
        self.all_patches = []
        for text in texts:
            byte_sequence = list(text.encode('utf-8', errors='ignore'))
            if len(byte_sequence) > 8:  # Skip very short texts
                patches = patcher.patch(byte_sequence)
                self.all_patches.extend(patches)
        
        print(f"Created {len(self.all_patches)} patches from {len(texts)} texts")
        
    def __len__(self):
        return max(0, len(self.all_patches) - self.seq_len - 1)
    
    def __getitem__(self, idx):
        input_patches = self.all_patches[idx:idx + self.seq_len]
        target_patches = self.all_patches[idx + 1:idx + self.seq_len + 1]
        return input_patches, target_patches

def test_simple_blt():
    print("🧪 Testing Simple BLT Model")
    set_seed(42)
    
    # Create simple config
    config = SimpleConfig()
    print(f"📋 Config: {config.d_model}d, {config.n_layers}L, {config.max_steps} steps")
    
    # Create simple training data
    texts = [
        "Hello world! This is a test.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is fascinating.",
        "Python is a great programming language.",
        "Deep learning models are powerful.",
        "Natural language processing is important.",
        "Transformers changed everything.",
        "Attention is all you need.",
        "Large language models are impressive.",
        "Artificial intelligence is the future."
    ] * 10  # Repeat for more data
    
    print(f"📚 Using {len(texts)} training texts")
    
    # Create patcher and dataset
    patcher = FixedSizePatcher(patch_size=config.patch_size)
    dataset = SimpleDataset(texts, patcher, config.max_seq_len)
    
    # Split data
    val_size = len(dataset) // 10
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    print(f"📊 Dataset: {len(train_dataset)} train, {len(val_dataset)} val")
    
    # Create model
    model = SimpleBLT(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Total parameters: {total_params:,}")
    
    # Simple optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    
    # Training loop
    model.train()
    step = 0
    start_time = time.time()
    
    print(f"\n🚀 Starting training...")
    
    # Initial generation
    print(f"\n🎯 Step 0 - Initial generation:")
    sample_text = simple_generate(model, patcher, prompt="Hello", max_patches=6)
    print(f"   Generated: '{sample_text}'")
    
    pbar = tqdm(total=config.max_steps, desc="Training Simple BLT")
    
    while step < config.max_steps:
        for input_patches, target_patches in train_loader:
            if step >= config.max_steps:
                break
            
            # Forward pass
            loss = model(input_patches, target_patches)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Logging and generation
            if step % 50 == 0 and step > 0:
                current_loss = loss.item()
                pbar.set_postfix({'loss': f'{current_loss:.4f}'})
                
                # Generate sample text
                print(f"\n🎯 Step {step} - Generation:")
                sample_text = simple_generate(model, patcher, prompt="Hello", max_patches=6)
                print(f"   Generated: '{sample_text}'")
                
                # Validation
                model.eval()
                val_loss = 0
                val_count = 0
                with torch.no_grad():
                    for val_input, val_target in val_loader:
                        if val_count >= 10:  # Quick validation
                            break
                        val_loss += model(val_input, val_target).item()
                        val_count += 1
                
                avg_val_loss = val_loss / val_count if val_count > 0 else 0
                print(f"   Val Loss: {avg_val_loss:.4f}")
                model.train()
            
            step += 1
            if step % 10 == 0:
                pbar.update(10)
    
    pbar.close()
    
    training_time = time.time() - start_time
    print(f"\n✅ Training completed in {training_time:.1f} seconds")
    
    # Final generation test
    print(f"\n🎉 Final generation tests:")
    test_prompts = ["Hello", "The", "Python", "AI"]
    
    for prompt in test_prompts:
        generated = simple_generate(model, patcher, prompt=prompt, max_patches=8)
        print(f"   '{prompt}' -> '{generated}'")
    
    return model

if __name__ == "__main__":
    test_simple_blt()