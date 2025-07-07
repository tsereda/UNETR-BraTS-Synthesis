#!/usr/bin/env python3
"""
Inference script for modality synthesis using trained Swin UNETR model.
This script loads a trained model and performs inference on test cases.
"""

import os
import json
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import torch
from functools import partial

from monai import transforms
from monai.data import Dataset, DataLoader
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR


def load_model(model_path, device):
    """Load trained model from checkpoint"""
    # Create model architecture (should match training config)
    model = SwinUNETR(
        in_channels=3,      # 3 input modalities
        out_channels=1,     # 1 output modality
        feature_size=48,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=True,
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    
    print(f"Model loaded from {model_path}")
    print(f"Best training MSE: {checkpoint.get('best_acc', 'Unknown')}")
    
    return model


def create_test_transforms():
    """Create transforms for test data"""
    return transforms.Compose([
        transforms.LoadImaged(keys=["image", "target"]),
        transforms.EnsureChannelFirstd(keys=["image", "target"]),
        transforms.NormalizeIntensityd(keys=["image", "target"], nonzero=True, channel_wise=True),
    ])


def create_test_case(data_dir, case_name):
    """Create test case data structure"""
    case_dir = os.path.join(data_dir, case_name)
    
    test_case = {
        "image": [
            os.path.join(case_dir, f"{case_name}-t1c.nii.gz"),
            os.path.join(case_dir, f"{case_name}-t1n.nii.gz"),
            os.path.join(case_dir, f"{case_name}-t2w.nii.gz"),
        ],
        "target": os.path.join(case_dir, f"{case_name}-t2f.nii.gz"),
        "case_name": case_name
    }
    
    # Check if all files exist
    all_files = test_case["image"] + [test_case["target"]]
    missing_files = [f for f in all_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"Warning: Missing files for {case_name}:")
        for f in missing_files:
            print(f"  - {f}")
        return None
    
    return test_case


def perform_inference(model, data, model_inferer, device):
    """Perform inference on a single case"""
    with torch.no_grad():
        image = data["image"].to(device)
        target = data["target"].to(device) if "target" in data else None
        
        # Perform inference
        pred = model_inferer(image)
        
        # Convert to numpy
        pred_np = pred[0, 0].detach().cpu().numpy()
        input_np = image[0].detach().cpu().numpy()
        target_np = target[0, 0].detach().cpu().numpy() if target is not None else None
        
        return pred_np, input_np, target_np


def calculate_metrics(pred, target):
    """Calculate synthesis metrics"""
    mse = np.mean((pred - target) ** 2)
    mae = np.mean(np.abs(pred - target))
    psnr = 20 * np.log10(np.max(target) / np.sqrt(mse)) if mse > 0 else float('inf')
    
    return {
        "MSE": mse,
        "MAE": mae,
        "PSNR": psnr
    }


def visualize_results(input_imgs, pred, target, case_name, slice_idx=None, save_path=None):
    """Visualize synthesis results"""
    # Auto-select middle slice if not specified
    if slice_idx is None:
        slice_idx = pred.shape[2] // 2
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Input modalities
    modality_names = ["T1c", "T1n", "T2w"]
    for i, (img, name) in enumerate(zip(input_imgs, modality_names)):
        axes[0, i].imshow(img[:, :, slice_idx], cmap="gray")
        axes[0, i].set_title(f"Input: {name}")
        axes[0, i].axis('off')
    
    # Prediction and target
    axes[1, 0].imshow(pred[:, :, slice_idx], cmap="gray")
    axes[1, 0].set_title("Synthesized FLAIR")
    axes[1, 0].axis('off')
    
    if target is not None:
        axes[1, 1].imshow(target[:, :, slice_idx], cmap="gray")
        axes[1, 1].set_title("Ground Truth FLAIR")
        axes[1, 1].axis('off')
        
        # Difference map
        diff = np.abs(pred - target)
        im = axes[1, 2].imshow(diff[:, :, slice_idx], cmap="hot")
        axes[1, 2].set_title("Absolute Difference")
        axes[1, 2].axis('off')
        plt.colorbar(im, ax=axes[1, 2], shrink=0.8)
    else:
        axes[1, 1].axis('off')
        axes[1, 2].axis('off')
    
    plt.suptitle(f"Modality Synthesis Results: {case_name} (Slice {slice_idx})")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()


def save_prediction(pred, reference_path, output_path):
    """Save prediction as NIfTI file"""
    # Load reference to get header and affine
    ref_img = nib.load(reference_path)
    
    # Create new NIfTI image with prediction data
    pred_img = nib.Nifti1Image(pred, ref_img.affine, ref_img.header)
    nib.save(pred_img, output_path)
    print(f"Prediction saved to: {output_path}")


def main():
    """Main inference function"""
    # Configuration - UPDATE THESE PATHS
    model_path = "./model.pt"  # Path to trained model
    data_dir = "ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData"  # Path to validation data
    output_dir = "./inference_results"  # Where to save results
    
    # Test case to run inference on
    case_name = "BraTS-GLI-00001-000"  # Update with actual case name
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        print("Please make sure you have trained a model first.")
        return
    
    # Load model
    model = load_model(model_path, device)
    
    # Set up inference
    roi = (128, 128, 128)
    model_inferer = partial(
        sliding_window_inference,
        roi_size=roi,
        sw_batch_size=1,
        predictor=model,
        overlap=0.6,
    )
    
    # Create test case
    test_case = create_test_case(data_dir, case_name)
    if test_case is None:
        print(f"Could not create test case for {case_name}")
        return
    
    # Set up data loader
    test_transforms = create_test_transforms()
    test_ds = Dataset(data=[test_case], transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    # Perform inference
    print(f"Running inference on {case_name}...")
    for batch_data in test_loader:
        pred, input_imgs, target = perform_inference(
            model, batch_data, model_inferer, device
        )
        
        # Calculate metrics if target is available
        if target is not None:
            metrics = calculate_metrics(pred, target)
            print(f"Synthesis metrics for {case_name}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        # Visualize results
        viz_path = os.path.join(output_dir, f"{case_name}_synthesis_results.png")
        visualize_results(
            input_imgs, pred, target, case_name, 
            save_path=viz_path
        )
        
        # Save prediction
        pred_path = os.path.join(output_dir, f"{case_name}_synthesized_flair.nii.gz")
        save_prediction(pred, test_case["target"], pred_path)
        
        break  # Only process first (and only) batch
    
    print(f"Inference completed! Results saved in: {output_dir}")


if __name__ == "__main__":
    main()