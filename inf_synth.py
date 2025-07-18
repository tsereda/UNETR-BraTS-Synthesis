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
from scipy import ndimage
from skimage.filters import threshold_otsu

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
    """Analyze reference modalities with improved scaling strategy"""
    
    modality_characteristics = {
        'FLAIR': {'typical_range': (100, 2500), 'prefer_dtype': np.int16},
        'T1CE': {'typical_range': (100, 3000), 'prefer_dtype': np.int16}, 
        'T1': {'typical_range': (100, 2800), 'prefer_dtype': np.int16},
        'T2': {'typical_range': (100, 3200), 'prefer_dtype': np.int16}
    }
    
    # Analyze reference files
    ref_intensities = []
    ref_dtypes = []
    
    for ref_file in reference_files:
        img = nib.load(ref_file)
        data = img.get_fdata()
        ref_dtypes.append(img.header.get_data_dtype())
        
        # Get meaningful intensity values (exclude zeros and extreme outliers)
        nonzero_data = data[data > 0]
        if len(nonzero_data) > 1000:  # Ensure we have enough data points
            # Remove extreme outliers
            p1, p99 = np.percentile(nonzero_data, [1, 99])
            filtered_data = nonzero_data[(nonzero_data >= p1) & (nonzero_data <= p99)]
            ref_intensities.extend(filtered_data.flatten())
    
    # Determine target characteristics
    if len(ref_intensities) > 0:
        ref_intensities = np.array(ref_intensities)
        
        # Use more conservative percentiles to avoid extreme values
        target_min = max(np.percentile(ref_intensities, 5), 50)  # At least 50
        target_max = min(np.percentile(ref_intensities, 95), 5000)  # At most 5000
        
        # Ensure reasonable range
        if target_max - target_min < 500:
            # Expand range if too narrow
            center = (target_min + target_max) / 2
            target_min = max(center - 750, 50)
            target_max = center + 750
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
        'ref_intensities': ref_intensities
    }

def create_brain_mask(reference_files, target_shape):
    """Create brain mask from reference modalities using Otsu thresholding"""
    
    combined_mask = np.zeros(target_shape, dtype=bool)

    for ref_file in reference_files:
        img = nib.load(ref_file)
        data = img.get_fdata()

        # Remove extreme outliers
        nonzero = data[data > 0]
        if nonzero.size == 0:
            continue
        p1, p99 = np.percentile(nonzero, [1, 99])
        data_clean = np.clip(data, p1, p99)

        # Otsu thresholding
        try:
            threshold = threshold_otsu(data_clean[data_clean > 0])
        except Exception:
            threshold = np.mean(data_clean[data_clean > 0])
        brain_mask = data_clean > threshold * 0.3  # Lower threshold for inclusivity

        # Morphological cleanup
        brain_mask = ndimage.binary_fill_holes(brain_mask)
        brain_mask = ndimage.binary_erosion(brain_mask, iterations=1)
        brain_mask = ndimage.binary_dilation(brain_mask, iterations=2)

        combined_mask |= brain_mask

    combined_mask = ndimage.binary_fill_holes(combined_mask)
    return combined_mask.astype(np.float32)

def analyze_reference_modalities_enhanced(reference_files, target_modality, brain_mask):
    """Enhanced analysis with brain mask consideration"""
    import numpy as np

    modality_characteristics = {
        'FLAIR': {'typical_range': (100, 2500), 'prefer_dtype': np.int16},
        'T1CE': {'typical_range': (100, 3000), 'prefer_dtype': np.int16},
        'T1': {'typical_range': (100, 2800), 'prefer_dtype': np.int16},
        'T2': {'typical_range': (100, 3200), 'prefer_dtype': np.int16}
    }

    ref_intensities = []
    ref_dtypes = []

    for ref_file in reference_files:
        img = nib.load(ref_file)
        data = img.get_fdata()
        ref_dtypes.append(img.header.get_data_dtype())

        brain_data = data[brain_mask > 0]
        if len(brain_data) > 100:
            p5, p95 = np.percentile(brain_data, [5, 95])
            filtered_data = brain_data[(brain_data >= p5) & (brain_data <= p95)]
            ref_intensities.extend(filtered_data.flatten())

    if len(ref_intensities) > 0:
        ref_intensities = np.array(ref_intensities)
        target_min = max(np.percentile(ref_intensities, 10), 50)
        target_max = min(np.percentile(ref_intensities, 90), 4000)
        if target_max - target_min < 300:
            center = (target_min + target_max) / 2
            target_min = max(center - 400, 50)
            target_max = center + 400
    else:
        target_min, target_max = modality_characteristics[target_modality]['typical_range']

    target_dtype = np.int16
    return {
        'target_range': (target_min, target_max),
        'target_dtype': target_dtype,
        'ref_intensities': ref_intensities
    }

def synthesize_modality_enhanced(input_data, model, device, reference_files, target_modality):
    """Enhanced synthesis with proper background masking and intensity control"""

    reference_img = nib.load(input_data["input_image"][0])
    original_shape = reference_img.shape
    original_header = reference_img.header.copy()
    original_affine = reference_img.affine.copy()

    print(f"    Original image info:")
    print(f"      Shape: {original_shape}")
    print(f"      Reference dtype: {reference_img.get_fdata().dtype}")

    brain_mask = create_brain_mask(reference_files, original_shape)

    target_info = analyze_reference_modalities_enhanced(reference_files, target_modality, brain_mask)
    target_min, target_max = target_info['target_range']
    target_dtype = target_info['target_dtype']

    print(f"    Target range: [{target_min:.1f}, {target_max:.1f}]")
    print(f"    Target dtype: {target_dtype}")
    print(f"    Brain mask coverage: {brain_mask.sum() / brain_mask.size:.1%}")

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

    transformed = transform(input_data)
    input_tensor = transformed["input_image"].unsqueeze(0).to(device)

    print(f"    Input tensor shape: {input_tensor.shape}")

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

    result = prediction[0, 0].cpu().numpy()
    print(f"    Raw prediction shape: {result.shape}, dtype: {result.dtype}")
    print(f"    Raw prediction range: [{result.min():.6f}, {result.max():.6f}]")

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

    # Step 1: Apply brain mask
    result_masked = result * brain_mask
    brain_voxels = result_masked[brain_mask > 0]
    if len(brain_voxels) == 0:
        print("    ⚠️  Warning: No brain voxels detected!")
        brain_voxels = result.flatten()

    # Step 2: Normalize brain tissue intensities
    brain_min, brain_max = brain_voxels.min(), brain_voxels.max()
    if brain_max > brain_min:
        result_norm = np.zeros_like(result)
        result_norm[brain_mask > 0] = (result_masked[brain_mask > 0] - brain_min) / (brain_max - brain_min)
        result_enhanced = np.sqrt(np.clip(result_norm, 0, 1))
    else:
        result_enhanced = np.zeros_like(result)

    print(f"    Brain tissue range: [{brain_min:.6f}, {brain_max:.6f}]")
    print(f"    After enhancement: [{result_enhanced.min():.6f}, {result_enhanced.max():.6f}]")

    # Step 3: Scale to target range (brain tissue only)
    intensity_range = target_max - target_min
    result_scaled = np.zeros_like(result_enhanced)
    result_scaled[brain_mask > 0] = (result_enhanced[brain_mask > 0] * intensity_range + target_min)

    # Ensure minimum intensity for brain tissue
    brain_tissue_mask = result_scaled > 0 # This ensures we don't apply floor to background
    if np.sum(brain_tissue_mask) > 0:
        min_brain_intensity = target_min + intensity_range * 0.1
        result_scaled[brain_tissue_mask] = np.maximum(
            result_scaled[brain_tissue_mask],
            min_brain_intensity
        )

    print(f"    Scaled range: [{result_scaled.min():.1f}, {result_scaled.max():.1f}]")

    # Step 4: Convert to target data type with proper clipping
    if target_dtype == np.int16:
        result_scaled = np.clip(result_scaled, -32768, 32767)
        result_final = result_scaled.astype(np.int16)
    elif target_dtype == np.uint16:
        result_scaled = np.clip(result_scaled, 0, 65535)
        result_final = result_scaled.astype(np.uint16)
    else:
        result_final = result_scaled.astype(np.float32)

    print(f"    Final dtype: {result_final.dtype}")
    print(f"    Final range: [{result_final.min():.1f}, {result_final.max():.1f}]")

    nonzero_voxels = np.sum(result_final > 0)
    total_voxels = result_final.size
    nonzero_fraction = nonzero_voxels / total_voxels
    brain_coverage = np.sum(result_final[brain_mask > 0] > 0) / np.sum(brain_mask > 0) if np.sum(brain_mask) > 0 else 0

    print(f"    Non-zero voxels: {nonzero_fraction:.1%} ({nonzero_voxels}/{total_voxels})")
    print(f"    Brain coverage: {brain_coverage:.1%}")

    if nonzero_fraction > 0.8:
        print(f"    ⚠️  Warning: Too many non-zero voxels ({nonzero_fraction:.1%})")
    if brain_coverage < 0.9:
        print(f"    ⚠️  Warning: Incomplete brain coverage ({brain_coverage:.1%})")

    updated_header = original_header.copy()
    updated_header.set_data_dtype(result_final.dtype)

    return result_final, updated_header, original_affine


def synthesize_modality(input_data, model, device, reference_files, target_modality):
    """Run synthesis inference with improved intensity scaling"""
    
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
    
    # IMPROVED intensity scaling to prevent tiny files
    
    # Step 1: Handle negative predictions properly
    # Some models output negative values; need to handle this carefully
    raw_min, raw_max = result.min(), result.max()
    
    if raw_min < 0:
        # Shift to positive range but preserve relative intensities
        result_shifted = result - raw_min  # Now min is 0
        result_norm = result_shifted / (raw_max - raw_min) if (raw_max - raw_min) > 0 else result_shifted
    else:
        # Already positive, just normalize
        result_norm = result / raw_max if raw_max > 0 else result
    
    # Ensure we have a reasonable distribution (not all zeros)
    result_norm = np.clip(result_norm, 0.0, 1.0)
    
    # Apply gentler transformation to preserve more detail
    # Use square root instead of power to expand low intensities
    result_enhanced = np.sqrt(result_norm)
    
    print(f"    After enhancement: [{result_enhanced.min():.6f}, {result_enhanced.max():.6f}]")
    
    # Step 2: Scale to target range with minimum floor
    intensity_range = target_max - target_min
    result_scaled = result_enhanced * intensity_range + target_min
    
    # Step 3: Add some minimum intensity to ensure file isn't too sparse
    # This prevents overly compressed files
    brain_mask = result_scaled > (target_min + intensity_range * 0.1)  # Basic brain mask
    if np.sum(brain_mask) > 0:
        # Ensure brain tissue has reasonable minimum intensity
        min_brain_intensity = target_min + intensity_range * 0.15
        result_scaled[brain_mask] = np.maximum(result_scaled[brain_mask], min_brain_intensity)
    
    print(f"    Scaled range: [{result_scaled.min():.1f}, {result_scaled.max():.1f}]")
    
    # Step 4: Convert to target data type with proper clipping
    if target_dtype == np.int16:
        result_scaled = np.clip(result_scaled, -32768, 32767)
        result_final = result_scaled.astype(np.int16)
    elif target_dtype == np.uint16:
        result_scaled = np.clip(result_scaled, 0, 65535)
        result_final = result_scaled.astype(np.uint16)
    else:
        result_final = result_scaled.astype(np.float32)
    
    print(f"    Final dtype: {result_final.dtype}")
    print(f"    Final range: [{result_final.min():.1f}, {result_final.max():.1f}]")
    
    # Quality check: ensure we have reasonable data distribution
    nonzero_voxels = np.sum(result_final > 0)
    total_voxels = result_final.size
    nonzero_fraction = nonzero_voxels / total_voxels
    
    print(f"    Non-zero voxels: {nonzero_fraction:.1%} ({nonzero_voxels}/{total_voxels})")
    
    if nonzero_fraction < 0.1:
        print(f"    ⚠️  Warning: Very sparse result ({nonzero_fraction:.1%} non-zero)")
    
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
            print(f"    No missing modality detected - skipping")
            return False
        
        print(f"    Missing modality: {missing_modality} ({missing_suffix})")
        
        # Create case output directory
        case_output_dir = os.path.join(output_dir, case_name)
        os.makedirs(case_output_dir, exist_ok=True)
        
        # Copy existing modalities first and collect reference files
        print(f"    Copying existing modalities...")
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
        
        # Synthesize missing modality using the ENHANCED function
        print(f"    Synthesizing {missing_modality}...")
        synthesized, updated_header, original_affine = synthesize_modality_enhanced(
            input_data, model, device, reference_files, missing_modality
        )
        
        # Save synthesized modality
        synthesized_filename = f"{case_name}-{missing_suffix}.nii.gz"
        synthesized_path = os.path.join(case_output_dir, synthesized_filename)
        
        # Create NIfTI image with updated header
        synthesized_img = nib.Nifti1Image(
            synthesized, 
            original_affine, 
            updated_header
        )
        
        # Save with compression
        nib.save(synthesized_img, synthesized_path)
        print(f"    ✓ Saved: {synthesized_filename}")
        
        # Verify file size and consistency
        file_size = os.path.getsize(synthesized_path) / (1024 * 1024)  # MB
        print(f"      File size: {file_size:.1f} MB")
        
        # File size check
        if file_size < 1.0:
            print(f"      ⚠️  Warning: File size is very small ({file_size:.1f} MB)")
        elif file_size > 8.0: # Adjusted upper threshold slightly
            print(f"      ⚠️  Warning: File size is large ({file_size:.1f} MB)")
        else:
            print(f"      ✅ File size is reasonable")
        
        # Verify dimensional consistency
        print(f"      Verifying consistency...")
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
        size_reasonable = all(1.0 <= info['size_mb'] <= 8.0 for info in file_info.values())
        
        if shape_consistent:
            print(f"      ✅ Shape consistent: {reference_shape}")
        else:
            print(f"      ⚠️  Shape inconsistency detected")
            
        if size_reasonable:
            print(f"      ✅ All file sizes reasonable (1-8MB)")
        else:
            print(f"      ⚠️  Some file sizes outside normal range")
        
        # Print summary
        print(f"      File summary:")
        for filename, info in file_info.items():
            if info['size_mb'] < 1.0:
                status = "⚠️  (small)"
            elif info['size_mb'] > 8.0:
                status = "⚠️  (large)" 
            else:
                status = "✅"
            print(f"        {filename}: {info['size_mb']:.1f}MB, {info['dtype']} {status}")
        
        return True
        
    except Exception as e:
        print(f"    ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="IMPROVED Synthesis Inference for BraSyn")
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
            print(f"    {model}")
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
    print(f"IMPROVED SYNTHESIS INFERENCE FOR BRASYN")
    print(f"{'='*60}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Found {len(case_dirs)} cases to process")
    print(f"✅ IMPROVEMENTS:")
    print(f"    • Fixed small file issue (prevents <1MB files)")
    print(f"    • Better negative value handling")
    print(f"    • Improved intensity distribution")
    print(f"    • Enhanced brain tissue preservation")
    print(f"    • Target file size: 2-8MB per modality (adjusted target)")
    
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
    print(f"IMPROVED SYNTHESIS COMPLETE")
    print(f"{'='*60}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {len(case_dirs)}")
    
    if successful > 0:
        print(f"\n🎯 IMPROVED dataset ready: {args.output_dir}")
        print(f"✅ Files should be consistently sized (2-8MB each)")
        print(f"\nNext: Run the fixed conversion script to prepare for FeTS")


if __name__ == "__main__":
    main()