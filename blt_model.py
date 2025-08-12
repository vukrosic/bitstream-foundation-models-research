import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple
import hashlib

@dataclass
class BLTConfig:
    # Local Encoder
    encoder_layers: int = 1
    encoder_dim: int = 768
    encoder_heads: int = 12
    encoder_window: int = 512
    
    # Global Latent Transformer
    global_layers: int = 24
    global_dim: int = 1280
    global_heads: int = 10
    
    # Local Decoder  
    decoder_layers: int = 7
    decoder_dim: int = 768
    decoder_heads: int = 12
    decoder_window: int = 512
    
    # Cross-attention
    cross_attn_heads: int = 8
    k_factor: int = 2  # ratio of global_dim to encoder/decoder_dim
    
    # N-gram embeddings
    ngram_sizes: list = None
    ngram_vocab_size: int = 500000
    
    # Training
    max_seq_len: int = 8192  # in bytes
    dropout: float = 0.1
    
    def __post_init__(self):
        if self.ngram_sizes is None:
            self.ngram_sizes = [3, 4, 5, 6, 7, 8]
        assert self.global_dim == self.encoder_dim * self.k_factor

class RollingHash:
    """Rolling polynomial hash for n-grams"""
    def __init__(self, vocab_size=500000):
        self.vocab_size = vocab_size
        self.prime = 31
        
    def hash(self, bytes_seq):
        """Compute hash for byte sequence"""
        h = 0
        for b in bytes_seq:
            h = (h * self.prime + b) % self.vocab_size
        return h

class HashNgramEmbeddings(nn.Module):
    """Hash-based n-gram embeddings"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hasher = RollingHash(config.ngram_vocab_size)
        
        # Create embedding tables for each n-gram size
        self.embeddings = nn.ModuleDict({
            str(n): nn.Embedding(config.ngram_vocab_size, config.encoder_dim)
            for n in config.ngram_sizes
        })
        
    def forward(self, byte_ids):
        """
        byte_ids: [batch, seq_len]
        Returns: [batch, seq_len, dim]
        """
        B, T = byte_ids.shape
        device = byte_ids.device
        
        # Base byte embeddings
        base_embed = torch.zeros(B, T, self.config.encoder_dim, device=device)
        
        # Add n-gram embeddings
        for n in self.config.ngram_sizes:
            n_str = str(n)
            ngram_embeds = torch.zeros(B, T, self.config.encoder_dim, device=device)
            
            for b in range(B):
                for t in range(T):
                    if t >= n - 1:
                        # Get n-gram
                        ngram = byte_ids[b, t-n+1:t+1].cpu().numpy()
                        hash_idx = self.hasher.hash(ngram)
                        ngram_embeds[b, t] = self.embeddings[n_str](
                            torch.tensor(hash_idx, device=device)
                        )
            
            base_embed += ngram_embeds
        
        # Normalize by number of n-gram sizes + 1
        return base_embed / (len(self.config.ngram_sizes) + 1)

class CrossAttention(nn.Module):
    """Cross-attention for byte-patch conversion"""
    def __init__(self, query_dim, kv_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        
        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(kv_dim, query_dim)  
        self.v_proj = nn.Linear(kv_dim, query_dim)
        self.out_proj = nn.Linear(query_dim, query_dim)
        
        self.norm_q = nn.LayerNorm(query_dim)
        self.norm_kv = nn.LayerNorm(kv_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, queries, keys_values, mask=None):
        """
        queries: [batch, num_queries, query_dim]
        keys_values: [batch, seq_len, kv_dim]
        mask: [batch, num_queries, seq_len]
        """
        B, NQ, _ = queries.shape
        _, T, _ = keys_values.shape
        
        # Pre-norm
        queries = self.norm_q(queries)
        keys_values = self.norm_kv(keys_values)
        
        # Project
        Q = self.q_proj(queries).reshape(B, NQ, self.num_heads, self.head_dim)
        K = self.k_proj(keys_values).reshape(B, T, self.num_heads, self.head_dim)
        V = self.v_proj(keys_values).reshape(B, T, self.num_heads, self.head_dim)
        
        # Transpose for attention
        Q = Q.transpose(1, 2)  # [B, heads, NQ, head_dim]
        K = K.transpose(1, 2)  # [B, heads, T, head_dim]
        V = V.transpose(1, 2)  # [B, heads, T, head_dim]
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            mask = mask.unsqueeze(1)  # Add head dimension
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).reshape(B, NQ, -1)
        
        return self.out_proj(out)

class LocalEncoder(nn.Module):
    """Encodes bytes into patch representations"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Byte embeddings
        self.byte_embed = nn.Embedding(256, config.encoder_dim)
        self.ngram_embed = HashNgramEmbeddings(config)
        
        # Transformer layers with cross-attention
        self.layers = nn.ModuleList()
        for _ in range(config.encoder_layers):
            self.layers.append(nn.ModuleDict({
                'self_attn': nn.TransformerEncoderLayer(
                    d_model=config.encoder_dim,
                    nhead=config.encoder_heads,
                    dim_feedforward=config.encoder_dim * 4,
                    dropout=config.dropout,
                    batch_first=True
                ),
                'cross_attn': CrossAttention(
                    query_dim=config.encoder_dim,
                    kv_dim=config.encoder_dim,
                    num_heads=config.cross_attn_heads,
                    dropout=config.dropout
                ),
                'norm': nn.LayerNorm(config.encoder_dim)
            }))
        
    def forward(self, byte_ids, patch_boundaries):
        """
        byte_ids: [batch, seq_len]
        patch_boundaries: [batch, seq_len] binary mask where 1 indicates patch start
        Returns: [batch, num_patches, encoder_dim * k_factor]
        """
        B, T = byte_ids.shape
        
        # Embed bytes
        byte_embeds = self.byte_embed(byte_ids)
        ngram_embeds = self.ngram_embed(byte_ids)
        x = byte_embeds + ngram_embeds
        
        # Process through encoder layers
        for layer in self.layers:
            # Self-attention over bytes
            x = layer['self_attn'](x)
            
            # Cross-attention to create patches
            # Initialize patch queries by pooling bytes in each patch
            patch_queries = self._init_patch_queries(x, patch_boundaries)
            
            # Create attention mask
            attn_mask = self._create_patch_mask(patch_boundaries)
            
            # Cross-attention
            patch_reprs = layer['cross_attn'](patch_queries, x, attn_mask)
            patch_reprs = layer['norm'](patch_reprs + patch_queries)
        
        # Expand to global dimension
        patch_reprs = patch_reprs.repeat(1, 1, self.config.k_factor)
        
        return patch_reprs
    
    def _init_patch_queries(self, byte_reprs, patch_boundaries):
        """Initialize patch queries by pooling bytes"""
        B, T, D = byte_reprs.shape
        device = byte_reprs.device
        
        # Find patch indices
        patch_starts = []
        for b in range(B):
            starts = torch.where(patch_boundaries[b] == 1)[0]
            if len(starts) == 0:
                starts = torch.tensor([0], device=device)
            patch_starts.append(starts)
        
        # Pool bytes for each patch
        max_patches = max(len(starts) for starts in patch_starts)
        queries = torch.zeros(B, max_patches, D, device=device)
        
        for b in range(B):
            starts = patch_starts[b]
            for i, start_idx in enumerate(starts):
                if i < len(starts) - 1:
                    end_idx = starts[i + 1]
                else:
                    end_idx = T
                    
                # Max pooling
                queries[b, i] = byte_reprs[b, start_idx:end_idx].max(dim=0)[0]
        
        return queries
    
    def _create_patch_mask(self, patch_boundaries):
        """Create attention mask for patches to bytes"""
        B, T = patch_boundaries.shape
        device = patch_boundaries.device
        
        # Find patch assignments for each byte
        patch_ids = torch.cumsum(patch_boundaries, dim=1) - 1
        max_patches = patch_ids.max().item() + 1
        
        # Create mask
        mask = torch.zeros(B, max_patches, T, device=device)
        for b in range(B):
            for t in range(T):
                patch_id = patch_ids[b, t]
                mask[b, patch_id, t] = 1
        
        return mask

class GlobalTransformer(nn.Module):
    """Large transformer over patch representations"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.global_dim,
                nhead=config.global_heads,
                dim_feedforward=config.global_dim * 4,
                dropout=config.dropout,
                batch_first=True
            ) for _ in range(config.global_layers)
        ])
        
        self.norm = nn.LayerNorm(config.global_dim)
        
    def forward(self, x):
        """
        x: [batch, num_patches, global_dim]
        """
        # Create causal mask for patches
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        
        for layer in self.layers:
            x = layer(x, src_mask=mask)
        
        return self.norm(x)

class LocalDecoder(nn.Module):
    """Decodes patch representations back to bytes"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Initial byte representations from encoder
        self.layers = nn.ModuleList()
        for _ in range(config.decoder_layers):
            self.layers.append(nn.ModuleDict({
                'cross_attn': CrossAttention(
                    query_dim=config.decoder_dim,
                    kv_dim=config.global_dim,
                    num_heads=config.cross_attn_heads,
                    dropout=config.dropout
                ),
                'self_attn': nn.TransformerEncoderLayer(
                    d_model=config.decoder_dim,
                    nhead=config.decoder_heads,
                    dim_feedforward=config.decoder_dim * 4,
                    dropout=config.dropout,
                    batch_first=True
                ),
                'norm': nn.LayerNorm(config.decoder_dim)
            }))
        
        self.output_proj = nn.Linear(config.decoder_dim, 256)
        
    def forward(self, patch_reprs, encoder_hidden, patch_boundaries):
        """
        patch_reprs: [batch, num_patches, global_dim]
        encoder_hidden: [batch, seq_len, encoder_dim]
        patch_boundaries: [batch, seq_len]
        """
        B, T, D = encoder_hidden.shape
        
        # Start with encoder hidden states
        x = encoder_hidden
        
        # Create patch-to-byte mask
        attn_mask = self._create_byte_to_patch_mask(patch_boundaries)
        
        for layer in self.layers:
            # Cross-attention: bytes attend to patches
            patch_info = layer['cross_attn'](x, patch_reprs, attn_mask)
            x = layer['norm'](x + patch_info)
            
            # Self-attention over bytes
            x = layer['self_attn'](x)
        
        # Project to vocabulary
        logits = self.output_proj(x)
        
        return logits
    
    def _create_byte_to_patch_mask(self, patch_boundaries):
        """Create mask for bytes attending to their patch"""
        B, T = patch_boundaries.shape
        
        # Assign patch IDs to bytes
        patch_ids = torch.cumsum(patch_boundaries, dim=1) - 1
        max_patches = patch_ids.max().item() + 1
        
        # Create mask [batch, seq_len, num_patches]
        mask = torch.zeros(B, T, max_patches, device=patch_boundaries.device)
        for b in range(B):
            for t in range(T):
                patch_id = patch_ids[b, t]
                # Each byte can attend to its patch and all previous patches
                mask[b, t, :patch_id+1] = 1
        
        return mask

class BLTModel(nn.Module):
    """Complete Byte Latent Transformer"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.local_encoder = LocalEncoder(config)
        self.global_transformer = GlobalTransformer(config)
        self.local_decoder = LocalDecoder(config)
        
    def forward(self, byte_ids, patch_boundaries):
        """
        byte_ids: [batch, seq_len] input bytes
        patch_boundaries: [batch, seq_len] patch boundaries (1 at start of patch)
        Returns: [batch, seq_len, 256] byte predictions
        """
        # Encode bytes to patches
        patch_reprs = self.local_encoder(byte_ids, patch_boundaries)
        
        # Store encoder hidden states for decoder
        encoder_hidden = self.local_encoder.byte_embed(byte_ids) + \
                        self.local_encoder.ngram_embed(byte_ids)
        
        # Process patches through global transformer  
        global_outputs = self.global_transformer(patch_reprs)
        
        # Decode patches back to bytes
        byte_logits = self.local_decoder(global_outputs, encoder_hidden, patch_boundaries)
        
        return byte_logits
    
    def generate(self, prompt_bytes, max_length=100, temperature=0.8):
        """Generate bytes autoregressively"""
        self.eval()
        device = next(self.parameters()).device
        
        generated = prompt_bytes.copy()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Prepare input
                x = torch.tensor([generated[-self.config.max_seq_len:]], 
                               dtype=torch.long, device=device)
                
                # Create dummy patch boundaries (for simplicity, patch every 8 bytes)
                boundaries = torch.zeros_like(x)
                boundaries[0, ::8] = 1
                
                # Forward pass
                logits = self.forward(x, boundaries)
                
                # Sample next byte
                next_byte_logits = logits[0, -1] / temperature
                probs = F.softmax(next_byte_logits, dim=-1)
                next_byte = torch.multinomial(probs, 1).item()
                
                generated.append(next_byte)
                
                # Try to decode
                try:
                    text = bytes(generated).decode('utf-8')
                    if text.endswith(('.', '!', '?', '\n')):
                        break
                except:
                    continue
        
        return bytes(generated).decode('utf-8', errors='ignore')