#!/usr/bin/env python3
"""
Load trained BLT model and generate text
"""

import torch
import torch.nn.functional as F
import math
from llm_large import BLT_LLM, ModelConfig

def load_blt_model(model_path: str = None):
    """Load the trained BLT model"""
    # Try different model file names
    possible_paths = [
        "blt_model.pt",
        "model.pt", 
        "trained_model.pt",
        "llm_model.pt"
    ]
    
    if model_path:
        possible_paths.insert(0, model_path)
    
    checkpoint = None
    used_path = None
    
    for path in possible_paths:
        try:
            print(f"📦 Trying to load model from {path}")
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            used_path = path
            break
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"   ❌ Error loading {path}: {e}")
            continue
    
    if checkpoint is None:
        raise FileNotFoundError(f"No model found. Tried: {possible_paths}")
    
    print(f"✅ Found model at {used_path}")
    
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
            if len(current_seq) >= max_bytes:
                current_seq = current_seq[-(max_bytes-1):]  # Keep space for next token
            
            padded_seq = current_seq + [0] * (max_bytes - len(current_seq))
            
            # Create boundaries - treat whole sequence as patches
            boundaries = torch.zeros(1, 15, 2, dtype=torch.long, device=device)
            valid_patches = torch.zeros(1, 15, dtype=torch.bool, device=device)
            
            # Create multiple patches from the sequence
            seq_len = len(current_seq)
            patch_size = max(1, seq_len // 10)  # Divide into ~10 patches
            patch_idx = 0
            
            for start in range(0, seq_len, patch_size):
                if patch_idx >= 15:  # Max patches
                    break
                end = min(start + patch_size, seq_len)
                boundaries[0, patch_idx] = torch.tensor([start, end])
                valid_patches[0, patch_idx] = True
                patch_idx += 1
            
            # If no patches created, create one for whole sequence
            if patch_idx == 0:
                boundaries[0, 0] = torch.tensor([0, seq_len])
                valid_patches[0, 0] = True
            
            # Prepare input
            x = torch.tensor([padded_seq], dtype=torch.long, device=device)
            
            # Get logits
            logits = model(x, boundaries, valid_patches)
            
            # Get logits for the next position after current sequence
            next_pos = min(seq_len, max_bytes - 1)  # Position to predict
            next_token_logits = logits[0, next_pos, :] / temperature
            
            # Sample next token (only valid bytes 1-255, avoid 0 padding)
            probs = F.softmax(next_token_logits[1:256], dim=-1)  # Skip 0
            next_token = torch.multinomial(probs, 1).item() + 1  # Add 1 to offset
            
            # Stop conditions - check for reasonable text bytes
            if next_token > 255:
                break
                
            # Check if it's a reasonable text character
            if next_token < 32 and next_token not in [9, 10, 13]:  # Tab, newline, carriage return
                # If it's a control character, maybe stop or replace
                if len(generated_tokens) > 10:  # Only stop if we've generated something
                    break
            
            # Add new token
            generated_tokens.append(next_token)
            
            # Show progress every 20 tokens
            if i % 20 == 0 and i > 0:
                try:
                    partial_text = bytes(tokens + generated_tokens).decode('utf-8', errors='ignore')
                    print(f"   Step {i}: '{partial_text[-50:]}'")  # Show last 50 chars
                except:
                    pass
    
    # Convert back to text
    try:
        full_sequence = tokens + generated_tokens
        text = bytes(full_sequence).decode('utf-8', errors='ignore')
        print(f"✅ Generated {len(generated_tokens)} new bytes")
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
        print("   Please run llm_large.py first to train the model.")
        return
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Generate with different prompts
    prompts = [
        "The",
        "Hello world",
        "In the beginning",
        "Once upon a time",
        "Python is"
    ]
    
    temperatures = [0.7, 1.0]
    
    for temp in temperatures:
        print(f"\n🌡️ Temperature: {temp}")
        print("=" * 60)
        
        for prompt in prompts:
            print(f"\n📝 Prompt: '{prompt}' (temp={temp})")
            try:
                generated = generate_text_blt(model, config, prompt, max_length=150, temperature=temp)
                print(f"📖 Full text: {generated}")
                print(f"📏 Length: {len(generated)} characters")
            except Exception as e:
                print(f"❌ Generation failed: {e}")
                import traceback
                traceback.print_exc()
        
        print()

if __name__ == "__main__":
    main()