#!/usr/bin/env python3
"""
Quick BraTS Training Data Visualization with WandB
Shows all 4 modalities + segmentation for first 5 training cases
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import wandb
from pathlib import Path


def find_brats_cases(data_dir, max_cases=5):
    """Find first 5 BraTS training cases"""
    cases = []
    
    print(f"Scanning {data_dir} for BraTS cases...")
    
    if not os.path.exists(data_dir):
        print(f"ERROR: Directory {data_dir} does not exist!")
        return cases
    
    # Get all BraTS directories
    brats_dirs = [d for d in os.listdir(data_dir) if 'BraTS' in d and os.path.isdir(os.path.join(data_dir, d))]
    brats_dirs.sort()  # Sort for consistent ordering
    
    for item in brats_dirs[:max_cases]:  # Only take first 5
        case_path = os.path.join(data_dir, item)
        
        # Check all required files exist
        flair_file = os.path.join(case_path, f"{item}-t2f.nii.gz")
        t1ce_file = os.path.join(case_path, f"{item}-t1c.nii.gz")
        t1_file = os.path.join(case_path, f"{item}-t1n.nii.gz")
        t2_file = os.path.join(case_path, f"{item}-t2w.nii.gz")
        seg_file = os.path.join(case_path, f"{item}-seg.nii.gz")
        
        required_files = [flair_file, t1ce_file, t1_file, t2_file, seg_file]
        
        if all(os.path.exists(f) for f in required_files):
            case_data = {
                "case_id": item,
                "flair": flair_file,
                "t1ce": t1ce_file,
                "t1": t1_file,
                "t2": t2_file,
                "seg": seg_file
            }
            cases.append(case_data)
            print(f"✓ Found case: {item}")
        else:
            print(f"✗ Missing files for: {item}")
    
    print(f"Total cases found: {len(cases)}")
    return cases


def load_and_normalize_image(file_path):
    """Load and normalize a NIfTI image"""
    try:
        img = nib.load(file_path)
        data = img.get_fdata()
        
        # Normalize to 0-1 range
        if data.max() > data.min():
            data = (data - data.min()) / (data.max() - data.min())
        
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def create_segmentation_overlay(seg_data):
    """Create colored segmentation overlay with distinct colors"""
    # Convert to RGB
    seg_colored = np.zeros((*seg_data.shape, 3), dtype=np.uint8)
    
    # Background: black (only this should be black)
    seg_colored[seg_data == 0] = [0, 0, 0]
    # NCR/NET: bright red
    seg_colored[seg_data == 1] = [255, 0, 0]
    # Edema: bright green  
    seg_colored[seg_data == 2] = [0, 255, 0]
    # Enhancing tumor: bright cyan (more visible than dark blue)
    seg_colored[seg_data == 4] = [0, 255, 255]
    
    # Handle any unexpected labels (like 3) - make them yellow so we can see them
    unexpected_labels = ~np.isin(seg_data, [0, 1, 2, 4])
    if np.any(unexpected_labels):
        seg_colored[unexpected_labels] = [255, 255, 0]  # Yellow for debugging
    
    return seg_colored


def find_best_slice(seg_data, slice_range=(70, 85)):
    """Find the slice with most tumor content in the given range"""
    start_slice, end_slice = slice_range
    max_slice = min(end_slice, seg_data.shape[2])
    start_slice = max(start_slice, 0)
    
    best_slice = start_slice
    max_tumor_area = 0
    
    for slice_idx in range(start_slice, max_slice):
        tumor_area = np.sum(seg_data[:, :, slice_idx] > 0)
        if tumor_area > max_tumor_area:
            max_tumor_area = tumor_area
            best_slice = slice_idx
    
    return best_slice


def visualize_case(case_data, slice_range=(70, 85)):
    """Create a simple visualization of one case - just 5 images"""
    case_id = case_data["case_id"]
    
    print(f"Processing case: {case_id}")
    
    # Load all modalities
    flair_data = load_and_normalize_image(case_data["flair"])
    t1ce_data = load_and_normalize_image(case_data["t1ce"])
    t1_data = load_and_normalize_image(case_data["t1"])
    t2_data = load_and_normalize_image(case_data["t2"])
    
    # Load segmentation
    seg_img = nib.load(case_data["seg"])
    seg_data = seg_img.get_fdata()
    
    if any(data is None for data in [flair_data, t1ce_data, t1_data, t2_data]):
        print(f"Failed to load data for {case_id}")
        return None
    
    # Find best slice with tumor content
    best_slice = find_best_slice(seg_data, slice_range)
    print(f"  Best slice: {best_slice}")
    
    # Extract slices
    flair_slice = flair_data[:, :, best_slice]
    t1ce_slice = t1ce_data[:, :, best_slice]
    t1_slice = t1_data[:, :, best_slice]
    t2_slice = t2_data[:, :, best_slice]
    seg_slice = seg_data[:, :, best_slice]
    
    # Debug: show what labels are present
    unique_labels_slice = np.unique(seg_slice)
    unique_labels_volume = np.unique(seg_data)
    print(f"  Labels in this slice: {unique_labels_slice}")
    print(f"  Labels in full volume: {unique_labels_volume}")
    
    # Create colored segmentation
    seg_colored = create_segmentation_overlay(seg_slice)
    
    # Create simple 1x5 layout
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    # All 5 images in a row
    axes[0].imshow(flair_slice, cmap='gray')
    axes[0].set_title('FLAIR', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(t1ce_slice, cmap='gray')
    axes[1].set_title('T1CE', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(t1_slice, cmap='gray')
    axes[2].set_title('T1', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    axes[3].imshow(t2_slice, cmap='gray')
    axes[3].set_title('T2', fontsize=14, fontweight='bold')
    axes[3].axis('off')
    
    axes[4].imshow(seg_colored)
    axes[4].set_title('Seg (1=Red, 2=Green, 4=Cyan)', fontsize=14, fontweight='bold')
    axes[4].axis('off')
    
    # Simple title
    plt.suptitle(f'{case_id} - Slice {best_slice}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig, {"case_id": case_id, "slice_idx": best_slice}


def main():
    # Initialize W&B
    wandb.init(
        project="BraTS-Data-Visualization",
        name="simple_5_images",
        config={
            "dataset": "BraTS2023-GLI-Challenge-TrainingData",
            "num_cases": 5,
            "slice_range": [70, 85],
            "layout": "1x5 (FLAIR, T1CE, T1, T2, Segmentation)",
            "description": "Simple showcase: 5 images per case"
        }
    )
    
    print("🎯 BraTS Training Data Visualization")
    print("=" * 50)
    
    # Find data directory
    possible_dirs = [
        "/app/UNETR-BraTS-Synthesis/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData",
        "/data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData",
        "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    ]
    
    data_dir = None
    for dir_path in possible_dirs:
        if os.path.exists(dir_path):
            data_dir = dir_path
            break
    
    if not data_dir:
        print("❌ Could not find BraTS training data directory")
        print("Tried:")
        for dir_path in possible_dirs:
            print(f"  - {dir_path}")
        return
    
    print(f"✓ Found data directory: {data_dir}")
    
    # Find cases
    cases = find_brats_cases(data_dir, max_cases=5)
    
    if not cases:
        print("❌ No cases found!")
        return
    
    print(f"\n📊 Processing {len(cases)} cases...")
    
    # Process each case
    for i, case_data in enumerate(cases):
        print(f"\n[{i+1}/{len(cases)}] Processing {case_data['case_id']}...")
        
        try:
            fig, stats = visualize_case(case_data, slice_range=(70, 85))
            
            if fig is not None:
                # Log to W&B
                wandb.log({
                    f"case_{i+1}_{case_data['case_id']}": wandb.Image(fig)
                })
                
                print(f"  ✓ Logged case {case_data['case_id']} (slice {stats['slice_idx']})")
                
                # Close figure to save memory
                plt.close(fig)
            else:
                print(f"  ❌ Failed to process {case_data['case_id']}")
                
        except Exception as e:
            print(f"  ❌ Error processing {case_data['case_id']}: {e}")
    
    print(f"\n🎉 VISUALIZATION COMPLETE!")
    print(f"✓ Check your W&B project for the 5 clean visualizations!")
    
    wandb.finish()


if __name__ == "__main__":
    main()