import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import math
import time
from tqdm import tqdm
from dataclasses import dataclass
import numpy as np

from blt_model import BLTModel, BLTConfig
from entropy_patcher import EntropyPatcher
from train_entropy_model import EntropyModel, EntropyModelConfig

@dataclass 
class TrainingConfig:
    # Model scale
    model_size: str = "small"  # small, medium, large
    
    # Training
    batch_size: int = 16
    learning_rate: float = 3e-4
    max_steps: int = 10000
    warmup_steps: int = 1000
    gradient_accumulation: int = 4
    
    # Data
    max_seq_len: int = 2048  # in bytes
    num_documents: int = 5000
    
    # Patching
    target_patch_size: float = 6.0
    patching_method: str = "global"
    
    # Evaluation
    eval_every: int = 500
    save_every: int = 1000

def get_model_config(size="small"):
    """Get BLT config for different model sizes"""
    if size == "small":
        return BLTConfig(
            encoder_layers=1,
            encoder_dim=512,
            global_layers=12,
            global_dim=1024,
            global_heads=8,
            decoder_layers=6,
            decoder_dim=512,
            k_factor=2
        )
    elif size == "medium":
        return BLTConfig(
            encoder_layers=1,
            encoder_dim=768,
            global_layers=24,
            global_dim=1536,
            global_heads=12,
            decoder_layers=9,
            decoder_dim=768,
            k_factor=2
        )
    else:  # large
        return BLTConfig(
            encoder_layers=3,
            encoder_dim=1024,
            global_layers=32,
            global_dim=4096,
            global_heads=32,
            decoder_layers=9,
            decoder_dim=1024,
            k_factor=4
        )

class BLTDataset(Dataset):
    """Dataset for BLT training with pre-computed patches"""
    def __init__(self, byte_sequences, patcher, max_seq_len=2048):
        self.byte_sequences = byte_sequences
        self.patcher = patcher
        self.max_seq_len = max_seq_len
        
        # Pre-compute patches for all sequences
        print("Pre-computing patches...")
        self.data = []
        for seq in tqdm(byte_sequences, desc="Creating patches"):
            if len(seq) > max_seq_len:
                # Split long sequences
                for i in range(0, len(seq) - max_seq_len, max_seq_len // 2):
                    chunk = seq[i:i + max_seq_len]
                    boundaries = patcher.create_boundaries(chunk)
                    self.data.append((chunk, boundaries))
            else:
                boundaries = patcher.create_boundaries(seq)
                self.data.append((seq, boundaries))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        bytes_seq, boundaries = self.data[idx]
        
        # Pad to max_seq_len
        if len(bytes_seq) < self.max_seq_len:
            pad_len = self.max_seq_len - len(bytes_seq)
            bytes_seq = bytes_seq + [0] * pad_len
            boundaries = np.concatenate([boundaries, np.zeros(pad_len)])
        
        # Convert to tensors
        x = torch.tensor(bytes_seq[:-1], dtype=torch.long)
        y = torch.tensor(bytes_seq[1:], dtype=torch.long)
        boundaries = torch.tensor(boundaries[:-1], dtype=torch.long)
        
        return x, y, boundaries

def train_blt(config: TrainingConfig):
    """Main training function"""
    print("🚀 Starting BLT Training")
    
    # Load entropy model
    print("Loading entropy model...")
    entropy_config = EntropyModelConfig()
    entropy_model = EntropyModel(entropy_config)
    entropy_model.load_state_dict(torch.load('entropy_model.pt'))
    entropy_model.eval()
    
    # Create patcher
    patcher = EntropyPatcher(
        entropy_model, 
        threshold=0.6,
        method=config.patching_method
    )
    
    # Load data
    print("Loading training data...")
    from datasets import load_dataset
    dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", 
                          split="train", streaming=True)
    
    byte_sequences = []
    for i, item in enumerate(tqdm(dataset, desc="Loading data", total=config.num_documents)):
        if i >= config.num_documents:
            break
        text_bytes = list(item["text"][:3000].encode('utf-8'))
        byte_sequences.append(text_bytes)
    
    # Find optimal threshold
    patcher.find_optimal_threshold(byte_sequences[:100], config.target_patch_size)
    
    # Create dataset
    train_dataset = BLTDataset(byte_sequences, patcher, config.max_seq_len)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2
    )
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_config = get_model_config(config.model_size)
    model = BLTModel(model_config).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"  Encoder: {sum(p.numel() for p in model.local_encoder.parameters()):,}")
    print(f"  Global: {sum(p.numel() for p in model.global_transformer.parameters()):,}")
    print(f"  Decoder: {sum(p.numel() for p in model.local_decoder.parameters()):,}")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Cosine schedule with warmup
    def lr_lambda(step):
        if step < config.warmup_steps:
            return step / config.warmup_steps
        progress = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    model.train()
    step = 0
    accumulation_counter = 0
    accumulated_loss = 0
    
    pbar = tqdm(total=config.max_steps, desc="Training BLT")
    
    while step < config.max_steps:
        for x, y, boundaries in train_loader:
            if step >= config.max_steps:
                break
            
            x = x.to(device)
            y = y.to(device) 
            boundaries = boundaries.to(device)
            
            # Forward pass
            logits = model(x, boundaries)
            loss = F.cross_entropy(logits.view(-1, 256), y.view(-1), ignore_index=0)
            loss = loss / config.gradient_accumulation
            
            loss.backward()
            accumulated_loss += loss.item()
            accumulation_counter += 1
            
            # Optimizer step
            if accumulation_counter >= config.gradient_accumulation:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Update progress
                if step % 10 == 0:
                    avg_loss = accumulated_loss * config.gradient_accumulation
                    pbar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                    })
                
                accumulated_loss = 0
                accumulation_counter = 0
                step += 1
                pbar.update(1)
                
                # Evaluation
                if step % config.eval_every == 0:
                    model.eval()
                    
                    # Generate sample
                    prompt = "The future of AI is"
                    prompt_bytes = list(prompt.encode('utf-8'))
                    generated = model.generate(prompt_bytes, max_length=100)
                    
                    print(f"\n[Step {step}] Generated: {generated}")
                    
                    model.train()
                
                # Save checkpoint
                if step % config.save_every == 0:
                    checkpoint = {
                        'step': step,
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'scheduler_state': scheduler.state_dict(),
                        'config': config
                    }
                    torch.save(checkpoint, f'blt_checkpoint_{step}.pt')
                    print(f"Saved checkpoint at step {step}")
    
    pbar.close()
    print("✅ Training complete!")
    
    # Save final model
    torch.save(model.state_dict(), 'blt_final.pt')
    
    return model

def main():
    """Main entry point"""
    # First train entropy model if needed
    import os
    if not os.path.exists('entropy_model.pt'):
        print("Training entropy model first...")
        from train_entropy_model import train_entropy_model, EntropyModelConfig
        entropy_config = EntropyModelConfig()
        train_entropy_model(entropy_config)
    
    # Train BLT
    config = TrainingConfig(
        model_size="small",
        batch_size=16,
        max_steps=10000,
        target_patch_size=6.0
    )
    
    model = train_blt(config)
    
    # Final generation test
    print("\n🎯 Final Generation Test:")
    test_prompts = [
        "Once upon a time",
        "The meaning of life is",
        "In the year 2050"
    ]
    
    model.eval()
    for prompt in test_prompts:
        prompt_bytes = list(prompt.encode('utf-8'))
        generated = model.generate(prompt_bytes, max_length=150)
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated}")

if __name__ == "__main__":
    main()