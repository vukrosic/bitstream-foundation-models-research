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
    
    # Pad to max_bytes (100)
    max_bytes = 100
    original_len = len(tokens)
    if len(tokens) > max_bytes:
        tokens = tokens[:max_bytes]
    else:
        tokens.extend([0] * (max_bytes - len(tokens)))
    
    # Create dummy boundaries and valid_patches
    boundaries = torch.zeros(1, 15, 2, dtype=torch.long, device=device)
    valid_patches = torch.zeros(1, 15, dtype=torch.bool, device=device)
    boundaries[0, 0] = torch.tensor([0, original_len])
    valid_patches[0, 0] = True
    
    generated_tokens = tokens.copy()
    
    print(f"🔄 Generating {max_length} tokens...")
    
    with torch.no_grad():
        for i in range(max_length):
            # Prepare input
            x = torch.tensor([generated_tokens], dtype=torch.long, device=device)
            
            # Get logits
            logits = model(x, boundaries, valid_patches)
            
            # Get last position logits
            if len(generated_tokens) < max_bytes:
                next_pos = len([t for t in generated_tokens if t != 0]) - 1
            else:
                next_pos = max_bytes - 1
            
            next_token_logits = logits[0, next_pos, :] / temperature
            
            # Sample next token (only valid bytes 0-255)
            probs = F.softmax(next_token_logits[:256], dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            # Stop if we hit padding
            if next_token == 0:
                break
            
            # Add to generated tokens
            if len([t for t in generated_tokens if t != 0]) < max_bytes:
                # Find first zero and replace it
                for j, t in enumerate(generated_tokens):
                    if t == 0:
                        generated_tokens[j] = next_token
                        break
            else:
                # Shift and add new token
                generated_tokens = generated_tokens[1:] + [next_token]
            
            # Update boundaries for next iteration
            current_len = len([t for t in generated_tokens if t != 0])
            boundaries[0, 0] = torch.tensor([0, min(current_len, max_bytes)])
    
    # Convert back to text
    try:
        valid_bytes = [b for b in generated_tokens if 0 < b < 256]
        text = bytes(valid_bytes).decode('utf-8', errors='ignore')
        return text
    except Exception as e:
        print(f"❌ Error decoding: {e}")
        return f"Generated {len(valid_bytes)} bytes (decode failed)"

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