#!/usr/bin/env python3
"""
Complete BLT Training Pipeline

This script implements the full BLT training approach:
1. Train entropy model (or load existing)
2. Use entropy model to create patches
3. Train BLT model end-to-end on patched data

Usage:
    python train_blt.py [--entropy_model_path entropy_model.pth]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import time
from typing import List, Tuple, Dict, Any
from tqdm import tqdm

# Import modules
from entropy_llm import MinimalLLM, ModelConfig as EntropyConfig, load_and_cache_data, set_seed
from entropy_patcher import EntropyPatcher
from blt_model import BLTModel, BLTConfig, train_blt
from save_entropy_model import load_entropy_model

class BLTDataset(Dataset):
    """Dataset for BLT training with patches"""
    
    def __init__(self, texts: List[str], entropy_patcher: EntropyPatcher, max_seq_len: int = 512):
        self.texts = texts
        self.patcher = entropy_patcher
        self.max_seq_len = max_seq_len
        
        # Pre-process all texts into byte sequences and patches
        self.data = []
        print("🔄 Pre-processing texts into patches...")
        
        for text in tqdm(texts, desc="Processing texts"):
            # Convert to bytes
            byte_sequence = list(text.encode('utf-8'))
            
            # Limit length
            if len(byte_sequence) > max_seq_len:
                byte_sequence = byte_sequence[:max_seq_len]
            
            if len(byte_sequence) < 10:  # Skip very short sequences
                continue
            
            # Create patches
            patches = self.patcher.create_patches(byte_sequence)
            
            if not patches:  # Skip if no patches created
                continue
            
            # Create patch boundaries
            boundaries = []
            current_pos = 0
            for patch in patches:
                start = current_pos
                end = current_pos + len(patch)
                boundaries.append((start, end))
                current_pos = end
            
            # Reconstruct full sequence from patches (should match original)
            reconstructed = []
            for patch in patches:
                reconstructed.extend(patch)
            
            # Pad if necessary
            if len(reconstructed) < max_seq_len:
                reconstructed.extend([0] * (max_seq_len - len(reconstructed)))
            else:
                reconstructed = reconstructed[:max_seq_len]
            
            self.data.append({
                'bytes': reconstructed,
                'boundaries': boundaries,
                'original_length': len(byte_sequence)
            })
        
        print(f"✅ Processed {len(self.data)} sequences with patches")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        bytes_tensor = torch.tensor(item['bytes'], dtype=torch.long)
        
        # Create input (bytes[:-1]) and target (bytes[1:])
        input_bytes = bytes_tensor[:-1]
        target_bytes = bytes_tensor[1:]
        
        # Adjust boundaries for input sequence
        boundaries = [(max(0, start-1), max(0, end-1)) for start, end in item['boundaries']]
        boundaries = [(s, e) for s, e in boundaries if s < len(input_bytes) and e <= len(input_bytes)]
        
        return {
            'bytes': input_bytes,
            'target_bytes': target_bytes,
            'boundaries': boundaries
        }

def collate_blt_batch(batch):
    """Custom collate function for BLT batches"""
    bytes_batch = torch.stack([item['bytes'] for item in batch])
    target_batch = torch.stack([item['target_bytes'] for item in batch])
    boundaries_batch = [item['boundaries'] for item in batch]
    
    return {
        'bytes': bytes_batch,
        'target_bytes': target_batch,
        'patch_boundaries': boundaries_batch
    }

def setup_blt_optimizer(model: BLTModel, lr: float = 1e-4):
    """Setup optimizer for BLT model"""
    # Different learning rates for different components
    encoder_params = list(model.local_encoder.parameters())
    global_params = list(model.global_transformer.parameters())
    decoder_params = list(model.local_decoder.parameters())
    
    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': lr * 0.5},  # Encoder gets lower LR
        {'params': global_params, 'lr': lr},         # Global gets base LR
        {'params': decoder_params, 'lr': lr * 0.5}   # Decoder gets lower LR
    ], weight_decay=0.01)
    
    return optimizer

def evaluate_blt_model(model: BLTModel, val_loader: DataLoader, device: torch.device):
    """Evaluate BLT model"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_correct = 0
    
    with torch.no_grad():
        for batch in val_loader:
            bytes_input = batch['bytes'].to(device)
            target_bytes = batch['target_bytes'].to(device)
            patch_boundaries = batch['patch_boundaries']
            
            try:
                # Forward pass
                predictions = model(bytes_input, patch_boundaries)
                
                # Compute loss
                loss = F.cross_entropy(
                    predictions.view(-1, model.config.vocab_size),
                    target_bytes.view(-1),
                    ignore_index=0  # Ignore padding
                )
                
                # Accumulate metrics
                total_loss += loss.item() * target_bytes.numel()
                total_tokens += target_bytes.numel()
                
                # Accuracy
                pred_tokens = predictions.argmax(dim=-1)
                correct = (pred_tokens == target_bytes).sum().item()
                total_correct += correct
                
            except Exception as e:
                print(f"⚠️ Evaluation batch failed: {e}")
                continue
    
    if total_tokens > 0:
        avg_loss = total_loss / total_tokens
        accuracy = total_correct / total_tokens
        perplexity = torch.exp(torch.tensor(min(avg_loss, 20))).item()
    else:
        avg_loss = float('inf')
        accuracy = 0.0
        perplexity = float('inf')
    
    model.train()
    return {
        'val_loss': avg_loss,
        'val_accuracy': accuracy,
        'val_perplexity': perplexity
    }

def train_blt_pipeline(entropy_model_path: str, config: Dict[str, Any]):
    """Complete BLT training pipeline"""
    
    print(f"🚀 BLT TRAINING PIPELINE")
    print(f"=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Step 1: Load entropy model
    print(f"\n1️⃣ Loading entropy model...")
    config_path = entropy_model_path.replace('.pth', '_config.pth')
    entropy_model, entropy_config = load_entropy_model(entropy_model_path, config_path)
    
    if entropy_model is None:
        print(f"❌ Could not load entropy model from {entropy_model_path}")
        print(f"   Train entropy model first: python entropy_llm.py")
        return
    
    entropy_model = entropy_model.to(device)
    print(f"✅ Loaded entropy model with {sum(p.numel() for p in entropy_model.parameters()):,} parameters")
    
    # Step 2: Load data
    print(f"\n2️⃣ Loading training data...")
    texts, bytes_data = load_and_cache_data(entropy_config)
    
    # Convert bytes back to texts for BLT processing
    if not texts:
        # If we only have bytes, convert back to text (lossy but necessary)
        chunk_size = 1000
        texts = []
        for i in range(0, len(bytes_data), chunk_size):
            chunk = bytes_data[i:i+chunk_size]
            try:
                text = bytes(chunk).decode('utf-8', errors='ignore')
                if len(text.strip()) > 10:
                    texts.append(text)
            except:
                continue
    
    print(f"✅ Loaded {len(texts)} text documents")
    
    # Step 3: Create entropy patcher
    print(f"\n3️⃣ Creating entropy patcher...")
    patcher = EntropyPatcher(entropy_model, threshold=config['entropy_threshold'], method='global')
    
    # Step 4: Create BLT dataset
    print(f"\n4️⃣ Creating BLT dataset...")
    dataset = BLTDataset(texts[:config['max_documents']], patcher, config['max_seq_len'])
    
    if len(dataset) == 0:
        print(f"❌ No valid data for training!")
        return
    
    # Split dataset
    val_size = max(1, len(dataset) // 10)
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        collate_fn=collate_blt_batch,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        collate_fn=collate_blt_batch,
        num_workers=2
    )
    
    print(f"✅ Created datasets: {len(train_dataset)} train, {len(val_dataset)} val")
    
    # Step 5: Initialize BLT model
    print(f"\n5️⃣ Initializing BLT model...")
    blt_config = BLTConfig(
        encoder_layers=config['encoder_layers'],
        encoder_dim=config['encoder_dim'],
        global_layers=config['global_layers'],
        global_dim=config['global_dim'],
        decoder_layers=config['decoder_layers'],
        decoder_dim=config['decoder_dim'],
        vocab_size=256,
        max_seq_len=config['max_seq_len']
    )
    
    model = BLTModel(blt_config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"✅ BLT Model initialized:")
    print(f"   Encoder: {blt_config.encoder_layers}L, {blt_config.encoder_dim}d")
    print(f"   Global: {blt_config.global_layers}L, {blt_config.global_dim}d")
    print(f"   Decoder: {blt_config.decoder_layers}L, {blt_config.decoder_dim}d")
    print(f"   Total parameters: {total_params:,}")
    
    # Step 6: Setup training
    print(f"\n6️⃣ Setting up training...")
    optimizer = setup_blt_optimizer(model, config['learning_rate'])
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['max_steps'], eta_min=config['learning_rate'] * 0.1
    )
    
    # Step 7: Training loop
    print(f"\n7️⃣ Starting BLT training...")
    model.train()
    step = 0
    best_val_loss = float('inf')
    
    pbar = tqdm(total=config['max_steps'], desc="Training BLT")
    
    while step < config['max_steps']:
        for batch in train_loader:
            if step >= config['max_steps']:
                break
            
            try:
                bytes_input = batch['bytes'].to(device)
                target_bytes = batch['target_bytes'].to(device)
                patch_boundaries = batch['patch_boundaries']
                
                # Forward pass
                predictions = model(bytes_input, patch_boundaries)
                
                # Compute loss
                loss = F.cross_entropy(
                    predictions.view(-1, model.config.vocab_size),
                    target_bytes.view(-1),
                    ignore_index=0
                )
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                # Logging
                if step % 100 == 0:
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
                    })
                
                # Evaluation
                if step % config['eval_every'] == 0 and step > 0:
                    eval_metrics = evaluate_blt_model(model, val_loader, device)
                    print(f"\nStep {step}: Val Loss: {eval_metrics['val_loss']:.4f}, "
                          f"Val Acc: {eval_metrics['val_accuracy']:.4f}")
                    
                    if eval_metrics['val_loss'] < best_val_loss:
                        best_val_loss = eval_metrics['val_loss']
                        torch.save(model.state_dict(), 'best_blt_model.pth')
                
                step += 1
                if step % 100 == 0:
                    pbar.update(100)
                    
            except Exception as e:
                print(f"⚠️ Training step failed: {e}")
                step += 1
                continue
    
    pbar.close()
    
    # Final evaluation
    print(f"\n8️⃣ Final evaluation...")
    final_metrics = evaluate_blt_model(model, val_loader, device)
    
    print(f"\n🎉 BLT TRAINING COMPLETED!")
    print(f"Final Results:")
    print(f"   Loss: {final_metrics['val_loss']:.4f}")
    print(f"   Accuracy: {final_metrics['val_accuracy']:.4f}")
    print(f"   Perplexity: {final_metrics['val_perplexity']:.2f}")
    
    # Save final model
    torch.save(model.state_dict(), 'final_blt_model.pth')
    torch.save(blt_config, 'blt_config.pth')
    print(f"💾 Saved model to final_blt_model.pth")

def main():
    parser = argparse.ArgumentParser(description="Train BLT model")
    parser.add_argument("--entropy_model_path", type=str, default="entropy_model.pth",
                       help="Path to trained entropy model")
    parser.add_argument("--quick_demo", action="store_true",
                       help="Run quick demo with small model")
    
    args = parser.parse_args()
    
    set_seed(42)
    
    # Configuration
    if args.quick_demo:
        config = {
            'entropy_threshold': 0.6,
            'max_documents': 50,
            'max_seq_len': 256,
            'batch_size': 4,
            'max_steps': 500,
            'eval_every': 100,
            'learning_rate': 1e-4,
            
            # BLT architecture (small)
            'encoder_layers': 1,
            'encoder_dim': 256,
            'global_layers': 6,
            'global_dim': 1024,
            'decoder_layers': 3,
            'decoder_dim': 256
        }
    else:
        config = {
            'entropy_threshold': 0.6,
            'max_documents': 500,
            'max_seq_len': 512,
            'batch_size': 8,
            'max_steps': 5000,
            'eval_every': 500,
            'learning_rate': 1e-4,
            
            # BLT architecture (full)
            'encoder_layers': 1,
            'encoder_dim': 768,
            'global_layers': 24,
            'global_dim': 4096,
            'decoder_layers': 6,
            'decoder_dim': 768
        }
    
    try:
        train_blt_pipeline(args.entropy_model_path, config)
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()