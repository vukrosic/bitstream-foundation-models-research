import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class BLTConfig:
    # Local Encoder (lightweight)
    encoder_layers: int = 1
    encoder_dim: int = 768
    encoder_heads: int = 12
    
    # Global Latent Transformer (heavyweight)
    global_layers: int = 24
    global_dim: int = 4096
    global_heads: int = 32
    
    # Local Decoder
    decoder_layers: int = 6
    decoder_dim: int = 768
    decoder_heads: int = 12
    
    # General
    vocab_size: int = 256  # byte vocabulary
    max_seq_len: int = 2048
    dropout: float = 0.1
    
    # N-gram embeddings
    use_ngram_embeddings: bool = True
    ngram_sizes: List[int] = None
    hash_vocab_size: int = 100000
    
    def __post_init__(self):
        if self.ngram_sizes is None:
            self.ngram_sizes = [3, 4, 5, 6, 7, 8]

class NGramHashEmbedding(nn.Module):
    """N-gram hash embeddings for byte sequences"""
    
    def __init__(self, ngram_sizes: List[int], hash_vocab_size: int, embed_dim: int):
        super().__init__()
        self.ngram_sizes = ngram_sizes
        self.hash_vocab_size = hash_vocab_size
        
        # Separate embedding for each n-gram size
        self.embeddings = nn.ModuleDict({
            f'ngram_{n}': nn.Embedding(hash_vocab_size, embed_dim)
            for n in ngram_sizes
        })
        
        self.projection = nn.Linear(len(ngram_sizes) * embed_dim, embed_dim)
    
    def _hash_ngram(self, ngram: List[int]) -> int:
        """Simple hash function for n-grams"""
        hash_val = 0
        for byte_val in ngram:
            hash_val = (hash_val * 256 + byte_val) % self.hash_vocab_size
        return hash_val
    
    def forward(self, byte_sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            byte_sequence: [batch_size, seq_len]
        Returns:
            embeddings: [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len = byte_sequence.shape
        device = byte_sequence.device
        
        # Collect embeddings for all n-gram sizes
        all_embeddings = []
        
        for n in self.ngram_sizes:
            ngram_embeddings = torch.zeros(batch_size, seq_len, self.embeddings[f'ngram_{n}'].embedding_dim, device=device)
            
            for i in range(seq_len - n + 1):
                # Extract n-gram
                ngram = byte_sequence[:, i:i+n].cpu().numpy()
                
                # Hash each n-gram in the batch
                hash_ids = []
                for b in range(batch_size):
                    hash_id = self._hash_ngram(ngram[b].tolist())
                    hash_ids.append(hash_id)
                
                hash_tensor = torch.tensor(hash_ids, device=device)
                embed = self.embeddings[f'ngram_{n}'](hash_tensor)
                
                # Assign to center position of n-gram
                center_pos = i + n // 2
                if center_pos < seq_len:
                    ngram_embeddings[:, center_pos] = embed
            
            all_embeddings.append(ngram_embeddings)
        
        # Concatenate and project
        combined = torch.cat(all_embeddings, dim=-1)
        return self.projection(combined)

class LocalEncoder(nn.Module):
    """Local encoder: bytes → patches"""
    
    def __init__(self, config: BLTConfig):
        super().__init__()
        self.config = config
        
        # Byte embeddings
        self.byte_embedding = nn.Embedding(config.vocab_size, config.encoder_dim)
        
        # N-gram hash embeddings
        if config.use_ngram_embeddings:
            self.ngram_embedding = NGramHashEmbedding(
                config.ngram_sizes, config.hash_vocab_size, config.encoder_dim
            )
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.encoder_dim,
            nhead=config.encoder_heads,
            dim_feedforward=config.encoder_dim * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config.encoder_layers)
        
        # Patch pooling
        self.patch_query = nn.Parameter(torch.randn(config.encoder_dim))
        self.patch_attention = nn.MultiheadAttention(
            config.encoder_dim, config.encoder_heads, batch_first=True
        )
    
    def forward(self, bytes_input: torch.Tensor, patch_boundaries: List[List[Tuple[int, int]]]) -> torch.Tensor:
        """
        Args:
            bytes_input: [batch_size, seq_len] byte sequences
            patch_boundaries: List of patch boundaries for each sequence
        Returns:
            patch_representations: [batch_size, num_patches, encoder_dim]
        """
        batch_size, seq_len = bytes_input.shape
        
        # Byte embeddings
        byte_embeds = self.byte_embedding(bytes_input)
        
        # Add n-gram embeddings if enabled
        if self.config.use_ngram_embeddings:
            ngram_embeds = self.ngram_embedding(bytes_input)
            byte_embeds = byte_embeds + ngram_embeds
        
        # Apply transformer
        encoded = self.transformer(byte_embeds)
        
        # Pool into patches
        patch_representations = []
        
        for b in range(batch_size):
            batch_patches = []
            boundaries = patch_boundaries[b]
            
            for start, end in boundaries:
                # Extract patch bytes
                patch_bytes = encoded[b, start:end]  # [patch_len, encoder_dim]
                
                # Pool using attention with learnable query
                query = self.patch_query.unsqueeze(0).unsqueeze(0)  # [1, 1, encoder_dim]
                patch_repr, _ = self.patch_attention(
                    query, patch_bytes.unsqueeze(0), patch_bytes.unsqueeze(0)
                )
                batch_patches.append(patch_repr.squeeze(0).squeeze(0))
            
            if batch_patches:
                patch_representations.append(torch.stack(batch_patches))
            else:
                # Handle empty case
                patch_representations.append(torch.zeros(1, self.config.encoder_dim, device=bytes_input.device))
        
        # Pad to same length
        max_patches = max(p.size(0) for p in patch_representations)
        padded_patches = []
        
        for patches in patch_representations:
            if patches.size(0) < max_patches:
                padding = torch.zeros(max_patches - patches.size(0), self.config.encoder_dim, device=bytes_input.device)
                patches = torch.cat([patches, padding], dim=0)
            padded_patches.append(patches)
        
        return torch.stack(padded_patches)

class GlobalTransformer(nn.Module):
    """Global latent transformer: patches → patches"""
    
    def __init__(self, config: BLTConfig):
        super().__init__()
        self.config = config
        
        # Project encoder dim to global dim if different
        if config.encoder_dim != config.global_dim:
            self.input_projection = nn.Linear(config.encoder_dim, config.global_dim)
        else:
            self.input_projection = nn.Identity()
        
        # Global transformer layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.global_dim,
            nhead=config.global_heads,
            dim_feedforward=config.global_dim * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, config.global_layers)
        
        # Position embeddings for patches
        self.pos_embedding = nn.Embedding(1000, config.global_dim)  # Support up to 1000 patches
    
    def forward(self, patch_representations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_representations: [batch_size, num_patches, encoder_dim]
        Returns:
            global_outputs: [batch_size, num_patches, global_dim]
        """
        batch_size, num_patches, _ = patch_representations.shape
        
        # Project to global dimension
        x = self.input_projection(patch_representations)
        
        # Add position embeddings
        positions = torch.arange(num_patches, device=x.device)
        pos_embeds = self.pos_embedding(positions).unsqueeze(0).expand(batch_size, -1, -1)
        x = x + pos_embeds
        
        # Create causal mask for patches
        causal_mask = torch.triu(torch.ones(num_patches, num_patches, device=x.device), diagonal=1).bool()
        
        # Apply transformer with causal attention
        output = self.transformer(x, x, tgt_mask=causal_mask)
        
        return output

class LocalDecoder(nn.Module):
    """Local decoder: patches → bytes"""
    
    def __init__(self, config: BLTConfig):
        super().__init__()
        self.config = config
        
        # Project global dim to decoder dim if different
        if config.global_dim != config.decoder_dim:
            self.input_projection = nn.Linear(config.global_dim, config.decoder_dim)
        else:
            self.input_projection = nn.Identity()
        
        # Decoder transformer layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.decoder_dim,
            nhead=config.decoder_heads,
            dim_feedforward=config.decoder_dim * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, config.decoder_layers)
        
        # Output projection to vocabulary
        self.output_projection = nn.Linear(config.decoder_dim, config.vocab_size)
        
        # Byte position embeddings
        self.byte_pos_embedding = nn.Embedding(config.max_seq_len, config.decoder_dim)
    
    def forward(self, patch_representations: torch.Tensor, 
                encoder_hidden_states: torch.Tensor,
                patch_boundaries: List[List[Tuple[int, int]]]) -> torch.Tensor:
        """
        Args:
            patch_representations: [batch_size, num_patches, global_dim]
            encoder_hidden_states: [batch_size, seq_len, encoder_dim] from local encoder
            patch_boundaries: List of patch boundaries for each sequence
        Returns:
            byte_predictions: [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len, _ = encoder_hidden_states.shape
        
        # Project patch representations
        patch_repr = self.input_projection(patch_representations)
        
        # Create byte-level representations using cross-attention
        byte_outputs = []
        
        for b in range(batch_size):
            boundaries = patch_boundaries[b]
            byte_sequence = torch.zeros(seq_len, self.config.decoder_dim, device=patch_repr.device)
            
            for patch_idx, (start, end) in enumerate(boundaries):
                if patch_idx < patch_repr.size(1):
                    # Get patch representation
                    patch_vec = patch_repr[b, patch_idx].unsqueeze(0).unsqueeze(0)  # [1, 1, decoder_dim]
                    
                    # Get byte positions for this patch
                    patch_len = end - start
                    if patch_len > 0:
                        # Create queries for each byte position in patch
                        byte_positions = torch.arange(start, end, device=patch_repr.device)
                        pos_embeds = self.byte_pos_embedding(byte_positions)
                        
                        # Use cross-attention to decode bytes from patch
                        decoded_bytes, _ = nn.functional.multi_head_attention_forward(
                            pos_embeds, patch_vec.expand(-1, patch_len, -1), patch_vec.expand(-1, patch_len, -1),
                            self.config.decoder_dim, self.config.decoder_heads,
                            torch.zeros(self.config.decoder_dim * 3, device=patch_repr.device),
                            None, None, False, 0.0, 
                            nn.Linear(self.config.decoder_dim, self.config.decoder_dim).weight,
                            nn.Linear(self.config.decoder_dim, self.config.decoder_dim).weight,
                            nn.Linear(self.config.decoder_dim, self.config.decoder_dim).weight,
                            training=self.training
                        )
                        
                        byte_sequence[start:end] = decoded_bytes
            
            byte_outputs.append(byte_sequence)
        
        byte_representations = torch.stack(byte_outputs)
        
        # Apply decoder transformer
        decoded = self.transformer(byte_representations, patch_repr)
        
        # Project to vocabulary
        logits = self.output_projection(decoded)
        
        return logits

class BLTModel(nn.Module):
    """Complete BLT model with all three components"""
    
    def __init__(self, config: BLTConfig):
        super().__init__()
        self.config = config
        
        # Component 1: Local Encoder (bytes → patches)
        self.local_encoder = LocalEncoder(config)
        
        # Component 2: Global Latent Transformer (patches → patches)
        self.global_transformer = GlobalTransformer(config)
        
        # Component 3: Local Decoder (patches → bytes)
        self.local_decoder = LocalDecoder(config)
    
    def forward(self, bytes_input: torch.Tensor, patch_boundaries: List[List[Tuple[int, int]]]) -> torch.Tensor:
        """
        End-to-end forward pass
        
        Args:
            bytes_input: [batch_size, seq_len] byte sequences
            patch_boundaries: List of patch boundaries for each sequence
        Returns:
            byte_predictions: [batch_size, seq_len, vocab_size]
        """
        # Step 1: Encode bytes to patches
        patch_representations = self.local_encoder(bytes_input, patch_boundaries)
        
        # Step 2: Process patches through global transformer
        global_outputs = self.global_transformer(patch_representations)
        
        # Step 3: Decode patches back to bytes
        byte_predictions = self.local_decoder(
            global_outputs, 
            patch_representations,  # Use as encoder hidden states
            patch_boundaries
        )
        
        return byte_predictions

def train_blt(model: BLTModel, patched_data, config, optimizer):
    """Train the complete BLT model end-to-end"""
    model.train()
    
    for batch in patched_data:
        # batch contains: bytes, patch_boundaries, target_bytes
        bytes_input = batch['bytes']  # [batch_size, seq_len]
        patch_boundaries = batch['patch_boundaries']  # List of boundaries
        target_bytes = batch['target_bytes']  # [batch_size, seq_len]
        
        # Forward pass through entire model
        byte_predictions = model(bytes_input, patch_boundaries)
        
        # Compute loss on byte predictions
        loss = F.cross_entropy(
            byte_predictions.view(-1, model.config.vocab_size), 
            target_bytes.view(-1)
        )
        
        # Backprop through entire model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        yield loss.item()

# Example usage
if __name__ == "__main__":
    # Create BLT configuration
    config = BLTConfig(
        encoder_layers=1,
        encoder_dim=768,
        global_layers=24,
        global_dim=4096,
        decoder_layers=6,
        decoder_dim=768
    )
    
    # Initialize model
    model = BLTModel(config)
    
    print(f"BLT Model created with:")
    print(f"  Encoder: {config.encoder_layers}L, {config.encoder_dim}d")
    print(f"  Global: {config.global_layers}L, {config.global_dim}d") 
    print(f"  Decoder: {config.decoder_layers}L, {config.decoder_dim}d")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")