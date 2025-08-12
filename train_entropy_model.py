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
from dataclasses import dataclass
import pickle
import os

@dataclass
class EntropyModelConfig:
    # Small byte-level LM for entropy computation
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 14
    d_ff: int = 2048
    max_seq_len: int = 512
    dropout: float = 0.1
    vocab_size: int = 256  # Byte vocabulary
    
    # Training
    batch_size: int = 32
    learning_rate: float = 3e-4
    max_steps: int = 10000
    warmup_steps: int = 1000
    
    # Data
    num_documents: int = 5000
    max_bytes: int = 2000000

class ByteDataset(Dataset):
    def __init__(self, bytes_data, seq_len=512):
        self.bytes_data = bytes_data
        self.seq_len = seq_len
    
    def __len__(self):
        return max(0, len(self.bytes_data) - self.seq_len)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.bytes_data[idx:idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.bytes_data[idx + 1:idx + self.seq_len + 1], dtype=torch.long)
        return x, y

class EntropyModel(nn.Module):
    """Small byte-level model for computing entropies"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(256, config.d_model)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.n_heads,
                dim_feedforward=config.d_ff,
                dropout=config.dropout,
                batch_first=True
            ) for _ in range(config.n_layers)
        ])
        
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, 256)
        
    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        
        x = self.embedding(x) + self.pos_embedding(pos)
        x = self.dropout(x)
        
        # Create causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        
        for block in self.blocks:
            x = block(x, src_mask=mask)
        
        x = self.norm(x)
        return self.head(x)
    
    def compute_entropy(self, x):
        """Compute entropy for next-byte prediction"""
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            return entropy

def train_entropy_model(config):
    """Train the entropy model"""
    print("🔄 Training Entropy Model...")
    
    # Load data
    dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", 
                          split="train", streaming=True)
    
    bytes_data = []
    for i, item in enumerate(tqdm(dataset, desc="Loading data", total=config.num_documents)):
        if i >= config.num_documents:
            break
        text_bytes = item["text"][:3000].encode('utf-8')
        bytes_data.extend(list(text_bytes))
        if len(bytes_data) >= config.max_bytes:
            break
    
    bytes_data = bytes_data[:config.max_bytes]
    print(f"Loaded {len(bytes_data):,} bytes")
    
    # Create dataset
    dataset = ByteDataset(bytes_data, config.max_seq_len)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EntropyModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Training loop
    model.train()
    step = 0
    pbar = tqdm(total=config.max_steps, desc="Training Entropy Model")
    
    while step < config.max_steps:
        for x, y in train_loader:
            if step >= config.max_steps:
                break
                
            x, y = x.to(device), y.to(device)
            
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, 256), y.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if step % 100 == 0:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            step += 1
            pbar.update(1)
    
    pbar.close()
    
    # Save model
    torch.save(model.state_dict(), 'entropy_model.pt')
    print("✅ Entropy model saved to entropy_model.pt")
    
    return model

if __name__ == "__main__":
    config = EntropyModelConfig()
    model = train_entropy_model(config)