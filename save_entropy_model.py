#!/usr/bin/env python3
"""
Script to save the trained entropy model for use with patching.

Add this to the end of your entropy_llm.py training script to save the model.
"""

import torch
import os

def save_entropy_model(model, config, save_path="entropy_model.pth", config_path="entropy_config.pth"):
    """
    Save the trained entropy model and its configuration.
    
    Args:
        model: Trained MinimalLLM model
        config: ModelConfig used for training
        save_path: Path to save model weights
        config_path: Path to save model configuration
    """
    print(f"💾 Saving entropy model...")
    
    # Save model state dict
    torch.save(model.state_dict(), save_path)
    print(f"   Model weights saved to: {save_path}")
    
    # Save configuration
    torch.save(config, config_path)
    print(f"   Model config saved to: {config_path}")
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {total_params:,}")
    print(f"   Model architecture: {config.d_model}d, {config.n_layers}L, {config.n_heads}H")
    
    print(f"✅ Entropy model saved successfully!")
    print(f"\nTo use with patching:")
    print(f"   python entropy_integration.py --entropy_model_path {save_path}")

def load_entropy_model(save_path="entropy_model.pth", config_path="entropy_config.pth"):
    """
    Load a saved entropy model.
    
    Returns:
        Tuple of (model, config)
    """
    from entropy_llm import MinimalLLM
    
    print(f"📂 Loading entropy model...")
    
    # Load configuration
    if os.path.exists(config_path):
        config = torch.load(config_path, map_location='cpu')
        print(f"   Config loaded from: {config_path}")
    else:
        print(f"❌ Config file not found: {config_path}")
        return None, None
    
    # Create model
    model = MinimalLLM(config)
    
    # Load weights
    if os.path.exists(save_path):
        state_dict = torch.load(save_path, map_location='cpu')
        model.load_state_dict(state_dict)
        print(f"   Weights loaded from: {save_path}")
    else:
        print(f"❌ Model file not found: {save_path}")
        return None, None
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {total_params:,}")
    print(f"✅ Entropy model loaded successfully!")
    
    return model, config

# Example usage code to add to entropy_llm.py
EXAMPLE_SAVE_CODE = '''
# Add this to the end of your entropy_llm.py main() function:

if __name__ == "__main__":
    # ... existing training code ...
    
    # After training is complete:
    model, final_metrics = train_model(config, train_loader, val_loader)
    
    # Save the trained model
    from save_entropy_model import save_entropy_model
    save_entropy_model(model, config)
    
    print(f"\\n🎉 TRAINING AND SAVING COMPLETED!")
'''

if __name__ == "__main__":
    print("🔧 ENTROPY MODEL SAVE/LOAD UTILITIES")
    print("=" * 50)
    print("This script provides utilities to save and load your trained entropy model.")
    print("\nTo save your model, add this to entropy_llm.py:")
    print(EXAMPLE_SAVE_CODE)
    
    # Test loading if files exist
    if os.path.exists("entropy_model.pth") and os.path.exists("entropy_config.pth"):
        print("\n🧪 Testing model loading...")
        model, config = load_entropy_model()
        if model is not None:
            print("✅ Model loading test successful!")
    else:
        print("\n⚠️  No saved model found. Train your entropy model first!")