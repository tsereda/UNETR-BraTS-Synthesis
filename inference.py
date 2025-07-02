#!/usr/bin/env python3
"""
Simple inference script for UNETR synthesis model.

This script performs inference on test data using a trained model.
"""

import os
import sys
import argparse
import torch
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from models.unetr_synthesis import create_model
    from data.brats_dataset import BraTSDataset
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure to install dependencies: pip install -r requirements.txt")
    sys.exit(1)

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    print("Warning: nibabel not available. Install with: pip install nibabel")


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Load trained model from checkpoint."""
    print(f"Loading model from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Create model
    model = create_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully (epoch {checkpoint['epoch']})")
    return model, config


def inference_single(
    model: torch.nn.Module,
    input_data: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """Perform inference on a single volume."""
    with torch.no_grad():
        input_data = input_data.unsqueeze(0).to(device)  # Add batch dimension
        output = model(input_data)
        output = output.squeeze(0).cpu()  # Remove batch dimension
    return output


def save_nifti(data: np.ndarray, output_path: Path, reference_path: Path = None):
    """Save data as NIfTI file."""
    if not NIBABEL_AVAILABLE:
        print("Cannot save NIfTI: nibabel not available")
        return
    
    try:
        if reference_path and reference_path.exists():
            # Use reference image for header info
            ref_img = nib.load(str(reference_path))
            new_img = nib.Nifti1Image(data, ref_img.affine, ref_img.header)
        else:
            # Create simple image
            new_img = nib.Nifti1Image(data, np.eye(4))
        
        nib.save(new_img, str(output_path))
        print(f"Saved: {output_path}")
        
    except Exception as e:
        print(f"Error saving {output_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description='UNETR synthesis inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory with BraTS data')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for synthetic images')
    parser.add_argument('--target_modality', type=str, required=True,
                       choices=['t1n', 't1c', 't2w', 't2f'],
                       help='Target modality to synthesize')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for inference')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model, config = load_model(args.checkpoint, device)
    
    # Create dataset for inference
    dataset = BraTSDataset(
        data_root=args.input_dir,
        target_modality=args.target_modality,
        phase='test',
        transforms=None  # Use minimal transforms for inference
    )
    
    print(f"Found {len(dataset)} subjects for inference")
    
    # Process each subject
    for i in range(len(dataset)):
        try:
            sample = dataset[i]
            
            input_data = sample['input']
            subject_name = sample['subject_name']
            
            print(f"Processing {subject_name} ({i+1}/{len(dataset)})...")
            
            # Perform inference
            synthetic = inference_single(model, input_data, device)
            
            # Convert to numpy
            synthetic_np = synthetic.squeeze().numpy()  # Remove channel dimension
            
            # Denormalize (assuming [-1, 1] to original range)
            synthetic_np = (synthetic_np + 1.0) * 500.0 - 1000.0  # Simple denormalization
            
            # Save result
            output_path = output_dir / f"{subject_name}-{args.target_modality}_synthetic.nii.gz"
            
            # Try to find reference image for header
            reference_path = None
            subject_dir = Path(args.input_dir) / subject_name
            if subject_dir.exists():
                for mod in ['t1n', 't1c', 't2w', 't2f']:
                    if mod != args.target_modality:
                        ref_file = subject_dir / f"{subject_name}-{mod}.nii.gz"
                        if ref_file.exists():
                            reference_path = ref_file
                            break
            
            save_nifti(synthetic_np, output_path, reference_path)
            
        except Exception as e:
            print(f"Error processing subject {i}: {e}")
            continue
    
    print("Inference completed!")


if __name__ == "__main__":
    main()
