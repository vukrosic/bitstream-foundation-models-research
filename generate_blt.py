#!/usr/bin/env python3
"""
Load trained BLT model and generate text
"""

import torch
import torch.nn.functional as F
import math
from llm import BLT_LLM, ModelConfig

def load_blt_model(model_path: str = "blt_model.pt"):
    """Load the trained BLT model"""
    print(f"📦 Loading BLT model from {model_path}")
    
    # Load checkpoint with weights_only=False for compatibility
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    
    # Create model
    model = BLT_LLM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded on {device}")
    print(f"   Config: {config.d_model}d, {config.n_layers}L, {config.n_heads}H")
    
    if 'final_metrics' in checkpoint:
        metrics = checkpoint['final_metrics']
        print(f"   Final metrics: Loss {metrics['val_loss']:.4f}, PPL {metrics['val_perplexity']:.2f}")
    
    return model, config

def generate_text_blt(model, config, prompt: str = "", max_length: int = 200, temperature: float = 0.8):
    """Generate text from BLT model"""
    device = next(model.parameters()).device
    
    # Convert prompt to bytes
    if prompt:
        tokens = list(prompt.encode('utf-8'))
        print(f"🎯 Generating from prompt: '{prompt}' ({len(tokens)} bytes)")
    else:
        tokens = [ord('T')]  # Start with 'T'
        print(f"🎯 Generating from default start")
    
    max_bytes = 100
    generated_tokens = []
    
    print(f"🔄 Generating {max_length} tokens...")
    
    with torch.no_grad():
        for i in range(max_length):
            # Current sequence
            current_seq = tokens + generated_tokens
            
            # Pad to max_bytes
            if len(current_seq) > max_bytes:
                current_seq = current_seq[-max_bytes:]  # Keep last max_bytes
            
            padded_seq = current_seq + [0] * (max_bytes - len(current_seq))
            
            # Create boundaries for current sequence
            boundaries = torch.zeros(1, 15, 2, dtype=torch.long, device=device)
            valid_patches = torch.zeros(1, 15, dtype=torch.bool, device=device)
            seq_len = len(current_seq)
            boundaries[0, 0] = torch.tensor([0, seq_len])
            valid_patches[0, 0] = True
            
            # Prepare input
            x = torch.tensor([padded_seq], dtype=torch.long, device=device)
            
            # Get logits
            logits = model(x, boundaries, valid_patches)
            
            # Get logits for the next position after current sequence
            next_pos = min(seq_len - 1, max_bytes - 1)  # Position of last real token
            next_token_logits = logits[0, next_pos, :] / temperature
            
            # Sample next token (only valid bytes 0-255)
            probs = F.softmax(next_token_logits[:256], dim=-1)
            
            # Avoid sampling 0 (padding) too often
            if probs[0] > 0.5:  # If padding is too likely, reduce it
                probs[0] = 0.1
                probs = probs / probs.sum()
            
            next_token = torch.multinomial(probs, 1).item()
            
            # Stop conditions
            if next_token == 0 or next_token > 255:
                break
            
            # Add new token
            generated_tokens.append(next_token)
            
            # Stop if we've generated enough or hit reasonable stopping points
            if len(generated_tokens) >= max_length:
                break
    
    # Convert back to text
    try:
        full_sequence = tokens + generated_tokens
        # Only use valid byte values
        valid_bytes = [b for b in full_sequence if 0 < b < 256]
        text = bytes(valid_bytes).decode('utf-8', errors='ignore')
        return text
    except Exception as e:
        print(f"❌ Error decoding: {e}")
        return f"Generated {len(generated_tokens)} new bytes (decode failed)"

def main():
    """Main generation function"""
    print("🚀 BLT Text Generation")
    print("=" * 50)
    
    # Load model
    try:
        model, config = load_blt_model()
    except FileNotFoundError:
        print("❌ Model file 'blt_model.pt' not found!")
        print("   Please run llm.py first to train the model.")
        return
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Generate with different prompts
    prompts = [
        "",  # No prompt
        "The",
        "Hello",
        "In the beginning",
        "Once upon a time"
    ]
    
    temperatures = [0.5, 0.8, 1.0]
    
    for temp in temperatures:
        print(f"\n🌡️ Temperature: {temp}")
        print("-" * 30)
        
        for prompt in prompts:
            print(f"\n📝 Prompt: '{prompt}' (temp={temp})")
            try:
                generated = generate_text_blt(model, config, prompt, max_length=100, temperature=temp)
                print(f"Generated: {generated}")
            except Exception as e:
                print(f"❌ Generation failed: {e}")
        
        print()

if __name__ == "__main__":
    main()