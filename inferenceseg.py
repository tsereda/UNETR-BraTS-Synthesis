#!/usr/bin/env python3
"""
BraTS Segmentation Inference - Load pretrained weights and run segmentation
"""

import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import time
import warnings
import nibabel as nib

import torch
import wandb

from monai.inferers import sliding_window_inference
from monai import transforms
from monai.transforms import AsDiscrete, Activations
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from monai.networks.nets import SwinUNETR
from monai.data import Dataset, DataLoader
from monai.data import decollate_batch

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def find_brats_cases(data_dir, dataset_type="train"):
    """Find BraTS cases for 2023 GLI format"""
    cases = []
    
    print(f"Scanning {data_dir} for BraTS {dataset_type} cases...")
    
    if not os.path.exists(data_dir):
        print(f"ERROR: Directory {data_dir} does not exist!")
        return cases
    
    for item in os.listdir(data_dir):
        if 'BraTS' in item and os.path.isdir(os.path.join(data_dir, item)):
            case_path = os.path.join(data_dir, item)
            
            # Simple pattern: {case_name}-{modality}.nii.gz
            flair_file = f"{item}-t2f.nii.gz"
            t1ce_file = f"{item}-t1c.nii.gz"
            t1_file = f"{item}-t1n.nii.gz"
            t2_file = f"{item}-t2w.nii.gz"
            
            # For training data, we need segmentation files
            if dataset_type == "train":
                seg_file = f"{item}-seg.nii.gz"
                required_files = [flair_file, t1ce_file, t1_file, t2_file, seg_file]
            else:
                # For validation data, no segmentation file required
                required_files = [flair_file, t1ce_file, t1_file, t2_file]
            
            # Check if all files exist
            if all(os.path.exists(os.path.join(case_path, f)) for f in required_files):
                case_data = {
                    "image": [
                        os.path.join(case_path, flair_file),
                        os.path.join(case_path, t1ce_file), 
                        os.path.join(case_path, t1_file),
                        os.path.join(case_path, t2_file)
                    ],
                    "case_id": item
                }
                
                # Add label only for training data
                if dataset_type == "train":
                    case_data["label"] = os.path.join(case_path, seg_file)
                
                cases.append(case_data)
                
                # Progress update every 50 cases
                if len(cases) % 50 == 0:
                    print(f"Found {len(cases)} valid {dataset_type} cases so far...")
    
    print(f"Finished scanning {dataset_type} data. Total cases found: {len(cases)}")
    return cases


def convert_predictions_to_original_labels(pred_tensor):
    """Convert model predictions back to original BraTS label format"""
    # pred_tensor shape: [3, H, W, D] with classes [TC, WT, ET]
    pred = pred_tensor.cpu().numpy()
    
    # Initialize output with background (label 0)
    output = np.zeros((pred.shape[1], pred.shape[2], pred.shape[3]), dtype=np.uint8)
    
    # Apply labels in correct order (ET takes priority over other overlapping regions)
    output[pred[1] == 1] = 2  # WT (Whole Tumor) = label 2
    output[pred[0] == 1] = 1  # TC (Tumor Core) = label 1  
    output[pred[2] == 1] = 4  # ET (Enhancing Tumor) = label 4
    
    return output


def save_nifti_prediction(prediction, reference_path, output_path):
    """Save prediction as NIfTI file with same affine/header as reference"""
    try:
        # Load reference image to get affine transformation and header
        ref_img = nib.load(reference_path)
        
        # Create new NIfTI image with prediction data
        pred_img = nib.Nifti1Image(prediction, ref_img.affine, ref_img.header)
        
        # Save prediction
        nib.save(pred_img, output_path)
        print(f"✓ Saved prediction: {output_path}")
        
    except Exception as e:
        print(f"Error saving NIfTI: {e}")


def visualize_segmentation(images, prediction, ground_truth, case_id, slice_idx=None):
    """Create visualization of segmentation results"""
    if slice_idx is None:
        slice_idx = images.shape[-1] // 2
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Top row - input images
    axes[0, 0].imshow(images[0, :, :, slice_idx], cmap='gray')
    axes[0, 0].set_title('FLAIR')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(images[1, :, :, slice_idx], cmap='gray')
    axes[0, 1].set_title('T1CE')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(images[2, :, :, slice_idx], cmap='gray')
    axes[0, 2].set_title('T1')
    axes[0, 2].axis('off')
    
    # Bottom row - segmentations
    axes[1, 0].imshow(images[1, :, :, slice_idx], cmap='gray', alpha=0.7)
    axes[1, 0].imshow(ground_truth[:, :, slice_idx], alpha=0.5, cmap='jet', vmin=0, vmax=4)
    axes[1, 0].set_title('Ground Truth Overlay')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(images[1, :, :, slice_idx], cmap='gray', alpha=0.7)
    axes[1, 1].imshow(prediction[:, :, slice_idx], alpha=0.5, cmap='jet', vmin=0, vmax=4)
    axes[1, 1].set_title('Prediction Overlay')
    axes[1, 1].axis('off')
    
    # Side by side comparison
    axes[1, 2].imshow(np.concatenate([ground_truth[:, :, slice_idx], prediction[:, :, slice_idx]], axis=1))
    axes[1, 2].set_title('GT (left) vs Pred (right)')
    axes[1, 2].axis('off')
    
    plt.suptitle(f'Case: {case_id}, Slice: {slice_idx}')
    plt.tight_layout()
    
    return fig


def run_inference(model, loader, model_inferer, post_sigmoid, post_pred, output_dir, visualize=True):
    """Run inference on the dataset"""
    model.eval()
    
    dice_metric = DiceMetric(
        include_background=True, 
        reduction=MetricReduction.MEAN_BATCH, 
        get_not_nans=True,
        ignore_empty=False
    )
    
    all_dice_scores = []
    processed_cases = 0
    
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            try:
                data = batch_data["image"].cuda()
                case_id = batch_data.get("case_id", [f"case_{idx}"])[0]
                
                print(f"\nProcessing case {idx+1}/{len(loader)}: {case_id}")
                
                # Run inference
                start_time = time.time()
                logits = model_inferer(data)
                inference_time = time.time() - start_time
                
                # Post-process predictions
                pred_sigmoid = post_sigmoid(logits)
                pred_discrete = post_pred(pred_sigmoid)
                
                # Convert to original BraTS label format
                prediction = convert_predictions_to_original_labels(pred_discrete[0])
                
                print(f"  Inference time: {inference_time:.2f}s")
                print(f"  Prediction shape: {prediction.shape}")
                print(f"  Unique labels in prediction: {np.unique(prediction)}")
                
                # Save prediction as NIfTI
                case_output_dir = os.path.join(output_dir, case_id)
                os.makedirs(case_output_dir, exist_ok=True)
                
                # Use first input image path as reference for affine transformation
                reference_path = None
                if "image" in batch_data:
                    # batch_data["image"] is a list of file paths (strings)
                    if isinstance(batch_data["image"][0], str):
                        reference_path = batch_data["image"][0]
                    elif isinstance(batch_data["image"][0], (list, tuple)) and isinstance(batch_data["image"][0][0], str):
                        reference_path = batch_data["image"][0][0]
                if reference_path is None:
                    print(f"Error processing case {idx}: Cannot determine reference_path for NIfTI saving.")
                    continue
                pred_path = os.path.join(case_output_dir, f"{case_id}_pred.nii.gz")
                save_nifti_prediction(prediction, reference_path, pred_path)
                
                # If ground truth is available, compute metrics and visualize
                if "label" in batch_data:
                    target = batch_data["label"].cuda()
                    
                    # Compute dice scores
                    val_labels_list = decollate_batch(target)
                    val_outputs_list = decollate_batch(logits)
                    val_output_convert = [post_pred(post_sigmoid(val_pred_tensor)) for val_pred_tensor in val_outputs_list]
                    
                    dice_metric.reset()
                    dice_metric(y_pred=val_output_convert, y=val_labels_list)
                    dice_scores, not_nans = dice_metric.aggregate()
                    
                    dice_tc = dice_scores[0].item() if len(dice_scores) > 0 else 0.0
                    dice_wt = dice_scores[1].item() if len(dice_scores) > 1 else 0.0
                    dice_et = dice_scores[2].item() if len(dice_scores) > 2 else 0.0
                    dice_avg = np.mean([dice_tc, dice_wt, dice_et])
                    
                    all_dice_scores.append([dice_tc, dice_wt, dice_et, dice_avg])
                    
                    print(f"  Dice TC: {dice_tc:.4f}, WT: {dice_wt:.4f}, ET: {dice_et:.4f}, Avg: {dice_avg:.4f}")
                    
                    # Create visualization
                    if visualize and idx < 10:  # Only visualize first 10 cases
                        ground_truth_orig = target[0].cpu().numpy()
                        gt_labels = np.zeros_like(prediction)
                        gt_labels[ground_truth_orig[1] == 1] = 2  # WT
                        gt_labels[ground_truth_orig[0] == 1] = 1  # TC
                        gt_labels[ground_truth_orig[2] == 1] = 4  # ET
                        
                        fig = visualize_segmentation(
                            data[0].cpu().numpy(), 
                            prediction, 
                            gt_labels, 
                            case_id
                        )
                        
                        viz_path = os.path.join(case_output_dir, f"{case_id}_visualization.png")
                        fig.savefig(viz_path, dpi=150, bbox_inches='tight')
                        plt.close(fig)
                        print(f"  ✓ Saved visualization: {viz_path}")
                
                processed_cases += 1
                
            except Exception as e:
                print(f"Error processing case {idx}: {e}")
                continue
    
    # Print summary statistics
    if all_dice_scores:
        dice_array = np.array(all_dice_scores)
        print(f"\n=== INFERENCE SUMMARY ===")
        print(f"Processed cases: {processed_cases}")
        print(f"Mean Dice TC: {dice_array[:, 0].mean():.4f} ± {dice_array[:, 0].std():.4f}")
        print(f"Mean Dice WT: {dice_array[:, 1].mean():.4f} ± {dice_array[:, 1].std():.4f}")
        print(f"Mean Dice ET: {dice_array[:, 2].mean():.4f} ± {dice_array[:, 2].std():.4f}")
        print(f"Mean Dice Avg: {dice_array[:, 3].mean():.4f} ± {dice_array[:, 3].std():.4f}")
        
        return dice_array
    else:
        print(f"\n=== INFERENCE SUMMARY ===")
        print(f"Processed cases: {processed_cases}")
        print("No ground truth available - only predictions saved")
        return None


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='BraTS Segmentation Inference')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model weights (.pt file)')
    parser.add_argument('--data_dir', type=str, default="ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData",
                        help='Directory containing BraTS data')
    parser.add_argument('--output_dir', type=str, default="/data/inference_results",
                        help='Directory to save inference results')
    parser.add_argument('--subset', type=int, default=None,
                        help='Number of cases to process (default: all)')
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='Generate visualization images')
    parser.add_argument('--use_val_data', action='store_true',
                        help='Use validation data instead of training data')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading model from: {args.model_path}")
    print(f"Output directory: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find data
    base_dir = "/app/UNETR-BraTS-Synthesis"
    if args.use_val_data:
        data_dir = os.path.join(base_dir, "ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData")
        dataset_type = "val"
    else:
        data_dir = os.path.join(base_dir, args.data_dir)
        dataset_type = "train"
    
    print(f"Data directory: {data_dir}")
    
    # Load cases
    all_cases = find_brats_cases(data_dir, dataset_type)
    
    if args.subset:
        all_cases = all_cases[:args.subset]
        print(f"Using subset of {args.subset} cases")
    
    print(f"Total cases to process: {len(all_cases)}")
    
    if not all_cases:
        print("No cases found!")
        return
    
    # Setup transforms
    roi = (128, 128, 128)
    
    if dataset_type == "train":
        # Training data has labels
        transform = transforms.Compose([
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ])
    else:
        # Validation data has no labels
        transform = transforms.Compose([
            transforms.LoadImaged(keys=["image"]),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ])
    
    # Create data loader
    dataset = Dataset(data=all_cases, transform=transform)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    
    # Load model
    model = SwinUNETR(
        in_channels=4,
        out_channels=3,
        feature_size=48,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=True,
    ).cuda()
    
    # Load weights
    print("Loading model weights...")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✓ Loaded weights from epoch {checkpoint['epoch']}")
    print(f"✓ Training loss: {checkpoint.get('train_loss', 'N/A')}")
    print(f"✓ Best validation dice: {checkpoint.get('val_acc_max', 'N/A')}")
    
    # Setup inference
    post_sigmoid = Activations(sigmoid=True)
    post_pred = AsDiscrete(argmax=False, threshold=0.5)
    
    model_inferer = partial(
        sliding_window_inference,
        roi_size=[roi[0], roi[1], roi[2]],
        sw_batch_size=2,
        predictor=model,
        overlap=0.5,
    )
    
    print(f"\nStarting inference on {len(all_cases)} cases...")
    
    # Run inference
    dice_scores = run_inference(
        model=model,
        loader=data_loader,
        model_inferer=model_inferer,
        post_sigmoid=post_sigmoid,
        post_pred=post_pred,
        output_dir=args.output_dir,
        visualize=args.visualize
    )
    
    print(f"\n✓ Inference completed!")
    print(f"✓ Results saved to: {args.output_dir}")
    if dice_scores is not None:
        print(f"✓ Overall performance: {dice_scores[:, 3].mean():.4f} ± {dice_scores[:, 3].std():.4f}")


if __name__ == "__main__":
    main()