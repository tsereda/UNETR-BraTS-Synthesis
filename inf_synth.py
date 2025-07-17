#!/usr/bin/env python3
"""
Synthesis Inference for BraSyn Pipeline
Uses trained UNETR models to synthesize missing modalities

Usage:
python synthesis_inference.py --input_dir pseudo_validation --output_dir completed_cases --models_dir /data
"""

import os
import argparse
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import shutil
from pathlib import Path
from monai.inferers import sliding_window_inference
from monai import transforms
from monai.networks.nets import SwinUNETR
import warnings

warnings.filterwarnings("ignore")


class SynthesisModel(nn.Module):
    """UNETR synthesis model (same as your training setup)"""
    
    def __init__(self, output_channels=1):
        super().__init__()
        # Handle MONAI version compatibility
        try:
            # Try with img_size (newer MONAI versions)
            self.backbone = SwinUNETR(
                img_size=(128, 128, 128),  # Match your training ROI size
                in_channels=4,  # 4 input channels as in training
                out_channels=3,
                feature_size=48,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                dropout_path_rate=0.0,
                use_checkpoint=True,
            )
        except TypeError:
            # Fallback for older MONAI versions (like your training)
            self.backbone = SwinUNETR(
                in_channels=4,
                out_channels=3,
                feature_size=48,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                dropout_path_rate=0.0,
                use_checkpoint=True,
            )
        
        # Replace output head for synthesis
        in_channels = self.backbone.out.conv.in_channels
        self.backbone.out = nn.Conv3d(
            in_channels,
            output_channels,
            kernel_size=1,
            padding=0
        )

    def forward(self, x):
        return self.backbone(x)


def detect_missing_modality(case_dir):
    """Detect which modality is missing using marker files"""
    case_name = os.path.basename(case_dir)
    
    # Check for marker files first
    modality_map = {
        't2f': 'FLAIR',
        't1c': 'T1CE', 
        't1n': 'T1',
        't2w': 'T2'
    }
    
    for suffix, name in modality_map.items():
        marker_file = os.path.join(case_dir, f"missing_{suffix}.txt")
        if os.path.exists(marker_file):
            return name, suffix
    
    # Fallback: check which modality file is missing
    for suffix, name in modality_map.items():
        expected_file = os.path.join(case_dir, f"{case_name}-{suffix}.nii.gz")
        if not os.path.exists(expected_file):
            return name, suffix
    
    return None, None


def load_synthesis_model(target_modality, models_dir, device):
    """Load the appropriate synthesis model"""
    
    model_files = {
        'FLAIR': '10_flair.pt',
        'T1CE': '10_t1ce.pt',
        'T1': '10_t1.pt',
        'T2': '10_t2.pt'
    }
    
    model_path = os.path.join(models_dir, model_files[target_modality])
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"Loading {target_modality} synthesis model: {model_path}")
    
    # Create and load model
    model = SynthesisModel(output_channels=1).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded (epoch {checkpoint['epoch']})")
    return model


def prepare_input_data(case_dir, case_name, target_modality_suffix):
    """Prepare input data for synthesis"""
    
    # Available modality suffixes (excluding the missing one)
    all_suffixes = ['t2f', 't1c', 't1n', 't2w']  # FLAIR, T1CE, T1, T2
    available_suffixes = [s for s in all_suffixes if s != target_modality_suffix]
    
    # Get the 3 available modality files
    input_files = []
    for suffix in available_suffixes:
        file_path = os.path.join(case_dir, f"{case_name}-{suffix}.nii.gz")
        if os.path.exists(file_path):
            input_files.append(file_path)
        else:
            raise FileNotFoundError(f"Expected input file not found: {file_path}")
    
    # Add duplicate to make 4 channels (as in training)
    if len(input_files) == 3:
        input_files.append(input_files[0])  # Duplicate first modality
    
    return {"input_image": input_files}


def synthesize_modality(input_data, model, device):
    """Run synthesis inference"""
    
    # Store original image shape for later cropping
    original_img = nib.load(input_data["input_image"][0])
    original_shape = original_img.shape
    
    # Transforms (same as validation in training)
    transform = transforms.Compose([
        transforms.LoadImaged(keys=["input_image"]),
        transforms.NormalizeIntensityd(keys=["input_image"], nonzero=True, channel_wise=True),
        transforms.DivisiblePadd(
            keys=["input_image"],
            k=32,
            mode="constant",
            constant_values=0,
        ),
    ])
    
    # Apply transforms
    transformed = transform(input_data)
    input_tensor = transformed["input_image"].unsqueeze(0).to(device)
    
    # Run sliding window inference
    roi = (128, 128, 128)
    with torch.no_grad():
        prediction = sliding_window_inference(
            inputs=input_tensor,
            roi_size=roi,
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
            mode="gaussian",
            sigma_scale=0.125,
            padding_mode="constant",
            cval=0.0,
        )
    
    # Get result as numpy array
    result = prediction[0, 0].cpu().numpy()  # Remove batch and channel dims
    
    # Crop back to original dimensions
    if result.shape != original_shape:
        print(f"    Cropping from {result.shape} to {original_shape}")
        # Calculate crop indices
        crop_slices = []
        for i in range(3):
            if result.shape[i] >= original_shape[i]:
                start = (result.shape[i] - original_shape[i]) // 2
                end = start + original_shape[i]
                crop_slices.append(slice(start, end))
            else:
                crop_slices.append(slice(None))
        
        result = result[tuple(crop_slices)]
    
    return result


def process_single_case(case_dir, output_dir, models_dir, device):
    """Process a single case: detect missing modality and synthesize it"""
    
    case_name = os.path.basename(case_dir)
    print(f"\nProcessing case: {case_name}")
    
    try:
        # Detect missing modality
        missing_modality, missing_suffix = detect_missing_modality(case_dir)
        
        if missing_modality is None:
            print(f"  No missing modality detected - skipping")
            return False
        
        print(f"  Missing modality: {missing_modality} ({missing_suffix})")
        
        # Create case output directory
        case_output_dir = os.path.join(output_dir, case_name)
        os.makedirs(case_output_dir, exist_ok=True)
        
        # Copy existing modalities first
        print(f"  Copying existing modalities...")
        for filename in os.listdir(case_dir):
            if filename.endswith('.nii.gz'):
                src_file = os.path.join(case_dir, filename)
                dst_file = os.path.join(case_output_dir, filename)
                shutil.copy2(src_file, dst_file)
        
        # Load synthesis model
        model = load_synthesis_model(missing_modality, models_dir, device)
        
        # Prepare input data
        input_data = prepare_input_data(case_dir, case_name, missing_suffix)
        
        # Synthesize missing modality
        print(f"  Synthesizing {missing_modality}...")
        synthesized = synthesize_modality(input_data, model, device)
        
        # Save synthesized modality
        synthesized_filename = f"{case_name}-{missing_suffix}.nii.gz"
        synthesized_path = os.path.join(case_output_dir, synthesized_filename)
        
        # Get reference image for header/affine
        existing_files = [f for f in os.listdir(case_dir) if f.endswith('.nii.gz')]
        if existing_files:
            reference_path = os.path.join(case_dir, existing_files[0])
            reference_img = nib.load(reference_path)
            
            # Ensure the synthesized data matches the reference shape exactly
            if synthesized.shape != reference_img.shape:
                print(f"    Warning: Shape mismatch - synthesized: {synthesized.shape}, reference: {reference_img.shape}")
                # This should not happen with the fix above, but as a safety measure
                synthesized = synthesized[:reference_img.shape[0], :reference_img.shape[1], :reference_img.shape[2]]
            
            synthesized_img = nib.Nifti1Image(
                synthesized, 
                reference_img.affine, 
                reference_img.header
            )
        else:
            synthesized_img = nib.Nifti1Image(synthesized, np.eye(4))
        
        nib.save(synthesized_img, synthesized_path)
        print(f"  ✓ Saved: {synthesized_filename}")
        
        # Verify dimensional consistency
        print(f"    Verifying dimensional consistency...")
        all_files = [f for f in os.listdir(case_output_dir) if f.endswith('.nii.gz')]
        shapes = {}
        for filename in all_files:
            img = nib.load(os.path.join(case_output_dir, filename))
            shapes[filename] = img.shape
        
        # Check if all shapes are the same
        unique_shapes = set(shapes.values())
        if len(unique_shapes) == 1:
            print(f"    ✅ All modalities have consistent shape: {list(unique_shapes)[0]}")
        else:
            print(f"    ⚠️  Shape inconsistency detected:")
            for filename, shape in shapes.items():
                print(f"      {filename}: {shape}")
        
        # Verify complete case
        expected_files = [f"{case_name}-{s}.nii.gz" for s in ['t2f', 't1c', 't1n', 't2w']]
        actual_files = [f for f in os.listdir(case_output_dir) if f.endswith('.nii.gz')]
        
        if len(actual_files) == 4:
            print(f"  ✅ Complete case ready for FeTS")
        else:
            print(f"  ⚠️  Warning: Expected 4 files, found {len(actual_files)}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Synthesis Inference for BraSyn")
    parser.add_argument("--input_dir", type=str, default="pseudo_validation",
                       help="Directory containing cases with missing modalities")
    parser.add_argument("--output_dir", type=str, default="completed_cases",
                       help="Output directory for completed cases")
    parser.add_argument("--models_dir", type=str, default="/data",
                       help="Directory containing synthesis models")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device for inference")
    parser.add_argument("--max_cases", type=int, default=None,
                       help="Maximum number of cases to process")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check if models exist
    required_models = ['10_flair.pt', '10_t1ce.pt', '10_t1.pt', '10_t2.pt']
    missing_models = []
    for model_file in required_models:
        model_path = os.path.join(args.models_dir, model_file)
        if not os.path.exists(model_path):
            missing_models.append(model_path)
    
    if missing_models:
        print(f"❌ Missing required models:")
        for model in missing_models:
            print(f"   {model}")
        print(f"Please ensure all synthesis models are in {args.models_dir}")
        return
    
    print(f"✓ All required models found in {args.models_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all case directories
    case_dirs = [d for d in os.listdir(args.input_dir) 
                if os.path.isdir(os.path.join(args.input_dir, d)) and 'BraTS' in d]
    case_dirs.sort()
    
    if args.max_cases:
        case_dirs = case_dirs[:args.max_cases]
    
    print(f"\n{'='*60}")
    print(f"SYNTHESIS INFERENCE FOR BRASYN")
    print(f"{'='*60}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Found {len(case_dirs)} cases to process")
    
    # Process each case
    successful = 0
    failed = 0
    
    for i, case_dir_name in enumerate(case_dirs):
        case_path = os.path.join(args.input_dir, case_dir_name)
        print(f"\n[{i+1}/{len(case_dirs)}]", end="")
        
        success = process_single_case(case_path, args.output_dir, args.models_dir, device)
        
        if success:
            successful += 1
        else:
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"SYNTHESIS COMPLETE")
    print(f"{'='*60}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {len(case_dirs)}")
    
    if successful > 0:
        print(f"\n🎯 Dataset ready for FeTS segmentation: {args.output_dir}")
        print(f"\nNext steps:")
        print(f"1. Convert to FeTS format: ./convert_to_fets_format.sh")
        print(f"2. Run FeTS segmentation: ./squashfs-root/usr/bin/fets_cli_segment -d fets_formatted/ -a deepMedic -g 0 -t 0")
        print(f"3. Convert to BraSyn submission format")


if __name__ == "__main__":
    main()