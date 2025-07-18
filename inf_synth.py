#!/usr/bin/env python3
"""
FIXED Synthesis Inference for BraSyn Pipeline
Addresses file size, data type, and intensity scaling issues
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


def analyze_reference_modalities(reference_files, target_modality):
    """Analyze reference modalities to determine appropriate target characteristics"""
    
    modality_characteristics = {
        'FLAIR': {'typical_range': (0, 2000), 'prefer_dtype': np.int16},
        'T1CE': {'typical_range': (0, 3000), 'prefer_dtype': np.int16}, 
        'T1': {'typical_range': (0, 2500), 'prefer_dtype': np.int16},
        'T2': {'typical_range': (0, 3500), 'prefer_dtype': np.int16}
    }
    
    # Analyze reference files
    ref_stats = []
    ref_dtypes = []
    ref_ranges = []
    
    for ref_file in reference_files:
        img = nib.load(ref_file)
        data = img.get_fdata()
        ref_dtypes.append(img.header.get_data_dtype())
        
        # Get brain tissue intensities (exclude background)
        brain_mask = data > np.percentile(data[data > 0], 5) if np.any(data > 0) else data > 0
        brain_data = data[brain_mask]
        
        if len(brain_data) > 0:
            stats = {
                'min': np.percentile(brain_data, 1),
                'max': np.percentile(brain_data, 99),
                'median': np.median(brain_data),
                'p25': np.percentile(brain_data, 25),
                'p75': np.percentile(brain_data, 75)
            }
            ref_stats.append(stats)
            ref_ranges.append((stats['min'], stats['max']))
    
    # Determine target characteristics
    if ref_stats:
        # Use median range from reference modalities
        all_mins = [s['min'] for s in ref_stats]
        all_maxs = [s['max'] for s in ref_stats]
        target_min = np.median(all_mins) * 0.8  # Slightly lower
        target_max = np.median(all_maxs) * 1.1  # Slightly higher
    else:
        # Fallback to typical ranges
        target_min, target_max = modality_characteristics[target_modality]['typical_range']
    
    # Determine target dtype (prefer int16 for compression)
    target_dtype = np.int16
    
    # Check if reference files use consistent dtype
    unique_dtypes = list(set(ref_dtypes))
    if len(unique_dtypes) == 1 and 'int' in str(unique_dtypes[0]):
        target_dtype = unique_dtypes[0]
    
    return {
        'target_range': (target_min, target_max),
        'target_dtype': target_dtype,
        'reference_stats': ref_stats
    }


def synthesize_modality(input_data, model, device, reference_files, target_modality):
    """Run synthesis inference with improved intensity scaling and data type handling"""
    
    # Store original image info
    reference_img = nib.load(input_data["input_image"][0])
    original_shape = reference_img.shape
    original_header = reference_img.header.copy()
    original_affine = reference_img.affine.copy()
    
    print(f"    Original image info:")
    print(f"      Shape: {original_shape}")
    print(f"      Reference dtype: {reference_img.get_fdata().dtype}")
    
    # Analyze reference modalities for target characteristics
    target_info = analyze_reference_modalities(reference_files, target_modality)
    target_min, target_max = target_info['target_range']
    target_dtype = target_info['target_dtype']
    
    print(f"    Target range: [{target_min:.1f}, {target_max:.1f}]")
    print(f"    Target dtype: {target_dtype}")
    
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
    
    print(f"    Input tensor shape: {input_tensor.shape}")
    
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
    print(f"    Raw prediction shape: {result.shape}, dtype: {result.dtype}")
    print(f"    Raw prediction range: [{result.min():.6f}, {result.max():.6f}]")
    
    # Crop back to original dimensions
    if result.shape != original_shape:
        print(f"    Cropping from {result.shape} to {original_shape}")
        crop_slices = []
        for i in range(3):
            if result.shape[i] >= original_shape[i]:
                start = (result.shape[i] - original_shape[i]) // 2
                end = start + original_shape[i]
                crop_slices.append(slice(start, end))
            else:
                crop_slices.append(slice(None))
        result = result[tuple(crop_slices)]
    
    print(f"    After cropping: {result.shape}")
    
    # IMPROVED intensity scaling and data type conversion
    
    # Step 1: Ensure result is in [0,1] range and apply sigmoid-like scaling for better distribution
    result = np.clip(result, 0.0, 1.0)
    
    # Apply smooth scaling to improve intensity distribution
    result = np.power(result, 0.8)  # Slightly compress high intensities
    
    # Step 2: Scale to target range with proper handling
    result_scaled = result * (target_max - target_min) + target_min
    
    # Step 3: Ensure values are within reasonable bounds
    result_scaled = np.clip(result_scaled, 0, target_max * 1.2)
    
    print(f"    Scaled range: [{result_scaled.min():.1f}, {result_scaled.max():.1f}]")
    
    # Step 4: Convert to target data type with proper handling
    if target_dtype == np.int16:
        # For int16, ensure values are within valid range
        result_scaled = np.clip(result_scaled, -32768, 32767)
        result_final = result_scaled.astype(np.int16)
    elif target_dtype == np.uint16:
        result_scaled = np.clip(result_scaled, 0, 65535)
        result_final = result_scaled.astype(np.uint16)
    else:
        # For float types, use float32 for consistency
        result_final = result_scaled.astype(np.float32)
    
    print(f"    Final dtype: {result_final.dtype}")
    print(f"    Final range: [{result_final.min():.1f}, {result_final.max():.1f}]")
    
    # Update header to match target data type
    updated_header = original_header.copy()
    updated_header.set_data_dtype(result_final.dtype)
    
    return result_final, updated_header, original_affine


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
        
        # Copy existing modalities first and collect reference files
        print(f"  Copying existing modalities...")
        reference_files = []
        for filename in os.listdir(case_dir):
            if filename.endswith('.nii.gz'):
                src_file = os.path.join(case_dir, filename)
                dst_file = os.path.join(case_output_dir, filename)
                shutil.copy2(src_file, dst_file)
                reference_files.append(src_file)
        
        # Load synthesis model
        model = load_synthesis_model(missing_modality, models_dir, device)
        
        # Prepare input data
        input_data = prepare_input_data(case_dir, case_name, missing_suffix)
        
        # Synthesize missing modality
        print(f"  Synthesizing {missing_modality}...")
        synthesized, updated_header, original_affine = synthesize_modality(
            input_data, model, device, reference_files, missing_modality
        )
        
        # Save synthesized modality with optimized settings
        synthesized_filename = f"{case_name}-{missing_suffix}.nii.gz"
        synthesized_path = os.path.join(case_output_dir, synthesized_filename)
        
        # Create NIfTI image with updated header
        synthesized_img = nib.Nifti1Image(
            synthesized, 
            original_affine, 
            updated_header
        )
        
        # Save with compression optimization
        nib.save(synthesized_img, synthesized_path)
        print(f"  ✓ Saved: {synthesized_filename}")
        
        # Verify file size and consistency
        file_size = os.path.getsize(synthesized_path) / (1024 * 1024)  # MB
        print(f"    File size: {file_size:.1f} MB")
        
        # Verify dimensional consistency
        print(f"    Verifying consistency...")
        all_files = [f for f in os.listdir(case_output_dir) if f.endswith('.nii.gz')]
        
        # Check shapes, dtypes, and file sizes
        reference_shape = None
        file_info = {}
        
        for filename in all_files:
            filepath = os.path.join(case_output_dir, filename)
            img = nib.load(filepath)
            shape = img.shape
            dtype = img.header.get_data_dtype()
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            
            file_info[filename] = {
                'shape': shape,
                'dtype': dtype,
                'size_mb': size_mb
            }
            
            if reference_shape is None:
                reference_shape = shape
        
        # Report consistency
        shape_consistent = all(info['shape'] == reference_shape for info in file_info.values())
        size_reasonable = all(info['size_mb'] < 8.0 for info in file_info.values())
        
        if shape_consistent:
            print(f"    ✅ Shape consistent: {reference_shape}")
        else:
            print(f"    ⚠️  Shape inconsistency detected")
            
        if size_reasonable:
            print(f"    ✅ File sizes reasonable (all < 8MB)")
        else:
            print(f"    ⚠️  Large file sizes detected")
        
        # Print summary
        print(f"    File summary:")
        for filename, info in file_info.items():
            status = "✅" if info['size_mb'] < 6.0 else "⚠️"
            print(f"      {filename}: {info['size_mb']:.1f}MB, {info['dtype']} {status}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="FIXED Synthesis Inference for BraSyn")
    parser.add_argument("--input_dir", type=str, default="pseudo_validation",
                       help="Directory containing cases with missing modalities")
    parser.add_argument("--output_dir", type=str, default="completed_cases_fixed",
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
    print(f"FIXED SYNTHESIS INFERENCE FOR BRASYN")
    print(f"{'='*60}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Found {len(case_dirs)} cases to process")
    print(f"✅ KEY FIXES:")
    print(f"  • Improved intensity scaling with power law adjustment")
    print(f"  • Proper data type handling and header updates")
    print(f"  • File size optimization (target < 6MB)")
    print(f"  • Better compression through int16 preference")
    print(f"  • Reference-based target range determination")
    
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
    print(f"FIXED SYNTHESIS COMPLETE")
    print(f"{'='*60}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {len(case_dirs)}")
    
    if successful > 0:
        print(f"\n🎯 FIXED dataset ready: {args.output_dir}")
        print(f"✅ All files should now be consistently sized (< 6MB each)")


if __name__ == "__main__":
    main()