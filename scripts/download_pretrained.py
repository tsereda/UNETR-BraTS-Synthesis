#!/usr/bin/env python3
"""
Download pre-trained models for UNETR synthesis.

This script downloads pre-trained models from various sources including
MONAI Model Zoo and other repositories.
"""

import os
import argparse
import torch
import urllib.request
from pathlib import Path
from typing import Dict, Any

# Model URLs and configurations
PRETRAINED_MODELS = {
    'brats_segmentation': {
        'description': 'BraTS segmentation model from MONAI',
        'url': None,  # Will be downloaded via MONAI bundle
        'monai_bundle': 'brats_mri_segmentation',
        'file_name': 'brats_segmentation.pth'
    },
    'imagenet_vit': {
        'description': 'ImageNet pre-trained Vision Transformer',
        'url': 'https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz',
        'file_name': 'vit_base_patch16_224.pth'
    }
}


def download_monai_bundle(bundle_name: str, output_dir: Path) -> str:
    """Download a MONAI bundle."""
    try:
        from monai.bundle import download
        
        print(f"Downloading MONAI bundle: {bundle_name}")
        bundle_path = download(name=bundle_name, bundle_dir=str(output_dir))
        
        # Find the model file
        model_files = list(Path(bundle_path).glob('**/*.pth'))
        if model_files:
            return str(model_files[0])
        else:
            print(f"Warning: No .pth file found in bundle {bundle_name}")
            return None
            
    except ImportError:
        print("MONAI not available. Install with: pip install monai[all]")
        return None
    except Exception as e:
        print(f"Error downloading MONAI bundle: {e}")
        return None


def download_url(url: str, output_path: Path) -> bool:
    """Download a file from URL."""
    try:
        print(f"Downloading from: {url}")
        urllib.request.urlretrieve(url, output_path)
        print(f"Downloaded to: {output_path}")
        return True
    except Exception as e:
        print(f"Error downloading from {url}: {e}")
        return False


def convert_vit_weights(input_path: Path, output_path: Path) -> bool:
    """Convert ImageNet ViT weights to PyTorch format."""
    try:
        import numpy as np
        
        # Load numpy weights
        weights = np.load(input_path)
        
        # Convert to PyTorch format (simplified)
        # This is a placeholder - actual conversion would need proper mapping
        pytorch_weights = {}
        for key, value in weights.items():
            if value.ndim > 0:
                pytorch_weights[key] = torch.from_numpy(value)
        
        # Save as PyTorch checkpoint
        torch.save(pytorch_weights, output_path)
        print(f"Converted weights saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error converting weights: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Download pre-trained models')
    parser.add_argument('--model', type=str, required=True,
                       choices=list(PRETRAINED_MODELS.keys()),
                       help='Model to download')
    parser.add_argument('--output_dir', type=str, default='pretrained',
                       help='Output directory for downloaded models')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get model configuration
    model_config = PRETRAINED_MODELS[args.model]
    output_path = output_dir / model_config['file_name']
    
    print(f"Downloading model: {args.model}")
    print(f"Description: {model_config['description']}")
    
    # Download based on source
    if 'monai_bundle' in model_config:
        # Download from MONAI
        bundle_path = download_monai_bundle(
            model_config['monai_bundle'], 
            output_dir
        )
        if bundle_path:
            # Copy/link to expected location
            import shutil
            shutil.copy2(bundle_path, output_path)
            print(f"Model saved to: {output_path}")
        else:
            print("Failed to download MONAI bundle")
            return
            
    elif 'url' in model_config and model_config['url']:
        # Download from URL
        temp_path = output_dir / f"temp_{model_config['file_name']}"
        
        if download_url(model_config['url'], temp_path):
            # Handle different file formats
            if model_config['file_name'].endswith('.pth'):
                if temp_path.suffix == '.npz':
                    # Convert numpy to PyTorch
                    convert_vit_weights(temp_path, output_path)
                    temp_path.unlink()  # Remove temp file
                else:
                    temp_path.rename(output_path)
            else:
                temp_path.rename(output_path)
        else:
            print("Failed to download from URL")
            return
    
    else:
        print("No valid download source specified")
        return
    
    print(f"Successfully downloaded {args.model} to {output_path}")


if __name__ == "__main__":
    main()
