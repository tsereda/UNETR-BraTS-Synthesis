#!/usr/bin/env python3
"""
Multi-task UNETR Inference: Synthesis + Segmentation
Simultaneously synthesize missing modality and perform segmentation
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from monai import transforms
from monai.networks.nets import SwinUNETR
from monai.data import Dataset, DataLoader
from monai.transforms import Activations, AsDiscrete
import warnings

warnings.filterwarnings("ignore")


class MultiTaskSwinUNETR(torch.nn.Module):
    """Multi-task SwinUNETR for synthesis + segmentation"""
    
    def __init__(self):
        super().__init__()
        self.backbone = SwinUNETR(
            in_channels=3,
            out_channels=4,  # 1 synthesis + 3 segmentation
            feature_size=48,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=True,
        )

    def forward(self, x):
        return self.backbone(x)


def load_multitask_model(model_path, device):
    """Load trained multi-task model"""
    model = MultiTaskSwinUNETR().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Loaded multi-task model from: {model_path}")
    print(f"  Target modality: {checkpoint['target_modality']}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Combined score: {checkpoint['best_combined_score']:.6f}")
    
    return model, checkpoint['target_modality']


def find_test_cases(data_dir):
    """Find test cases for inference"""
    cases = []
    
    modality_files = {
        "FLAIR": "t2f.nii.gz",
        "T1CE": "t1c.nii.gz",
        "T1": "t1n.nii.gz",
        "T2": "t2w.nii.gz"
    }

    for item in os.listdir(data_dir):
        if 'BraTS' in item and os.path.isdir(os.path.join(data_dir, item)):
            case_path = os.path.join(data_dir, item)
            
            # Build file paths
            files = {}
            for modality, suffix in modality_files.items():
                files[modality] = os.path.join(case_path, f"{item}-{suffix}")
            
            # Check if all files exist
            if all(os.path.exists(f) for f in files.values()):
                case_data = {
                    "all_modalities": files,
                    "case_id": item,
                    "case_path": case_path
                }
                cases.append(case_data)
                
                if len(cases) % 50 == 0:
                    print(f"Found {len(cases)} test cases...")
    
    print(f"Total test cases found: {len(cases)}")
    return cases


def prepare_input_for_model(case_data, target_modality):
    """Prepare input modalities for the model (exclude target)"""
    modality_map = {
        "FLAIR": "FLAIR",
        "T1CE": "T1CE", 
        "T1": "T1",
        "T2": "T2"
    }
    
    all_modalities = ["FLAIR", "T1CE", "T1", "T2"]
    input_modalities = [mod for mod in all_modalities if mod != target_modality]
    
    input_files = [case_data["all_modalities"][mod] for mod in input_modalities]
    target_file = case_data["all_modalities"][target_modality]
    
    return {
        "input_image": input_files,
        "target_image": target_file,
        "case_id": case_data["case_id"],
        "input_modalities": input_modalities,
        "target_modality": target_modality
    }


def convert_segmentation_to_brats_labels(seg_tensor):
    """Convert model segmentation output to BraTS label format"""
    # seg_tensor: [3, H, W, D] with TC, WT, ET
    seg = seg_tensor.cpu().numpy()
    
    # Initialize with background
    output = np.zeros(seg.shape[1:], dtype=np.uint8)
    
    # Apply labels (ET has highest priority)
    output[seg[1] == 1] = 2  # WT (Whole Tumor) = 2
    output[seg[0] == 1] = 1  # TC (Tumor Core) = 1
    output[seg[2] == 1] = 4  # ET (Enhancing Tumor) = 4
    
    return output


def create_comparison_visualization(input_images, target_image, pred_synthesis, pred_segmentation, 
                                  input_modalities, target_modality, case_id):
    """Create comprehensive visualization"""
    slice_idx = input_images.shape[-1] // 2
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Top row: Input modalities + target
    for i, modality in enumerate(input_modalities):
        axes[0, i].imshow(input_images[i, :, :, slice_idx], cmap='gray')
        axes[0, i].set_title(f'{modality} (Input)')
        axes[0, i].axis('off')
    
    # Last column: target modality
    axes[0, 3].imshow(target_image[0, :, :, slice_idx], cmap='gray')
    axes[0, 3].set_title(f'{target_modality} (Ground Truth)')
    axes[0, 3].axis('off')
    
    # Bottom row: Predicted synthesis + segmentation
    axes[1, 0].imshow(pred_synthesis[0, :, :, slice_idx], cmap='gray')
    axes[1, 0].set_title(f'{target_modality} (Predicted)')
    axes[1, 0].axis('off')
    
    # Segmentation visualization
    axes[1, 1].imshow(input_images[1, :, :, slice_idx], cmap='gray', alpha=0.7)
    axes[1, 1].imshow(pred_segmentation[:, :, slice_idx], alpha=0.5, cmap='jet', vmin=0, vmax=4)
    axes[1, 1].set_title('Predicted Segmentation')
    axes[1, 1].axis('off')
    
    # Synthesis comparison
    comparison = np.concatenate([
        target_image[0, :, :, slice_idx], 
        pred_synthesis[0, :, :, slice_idx]
    ], axis=1)
    axes[1, 2].imshow(comparison, cmap='gray')
    axes[1, 2].set_title('GT vs Pred Synthesis')
    axes[1, 2].axis('off')
    
    # Synthesis difference map
    diff = np.abs(target_image[0, :, :, slice_idx] - pred_synthesis[0, :, :, slice_idx])
    im = axes[1, 3].imshow(diff, cmap='hot')
    axes[1, 3].set_title('Synthesis Difference')
    axes[1, 3].axis('off')
    plt.colorbar(im, ax=axes[1, 3], fraction=0.046)
    
    plt.suptitle(f'Multi-task Results: {case_id} (Missing: {target_modality})')
    plt.tight_layout()
    
    return fig


def run_multitask_inference(model, test_case, target_modality, device, output_dir):
    """Run inference on a single test case"""
    
    # Prepare transforms
    transform = transforms.Compose([
        transforms.LoadImaged(keys=["input_image", "target_image"]),
        transforms.EnsureChannelFirstd(keys=["target_image"]),
        transforms.NormalizeIntensityd(keys=["input_image", "target_image"], nonzero=True, channel_wise=True),
        transforms.DivisiblePadd(
            keys=["input_image", "target_image"],
            k=32,
            mode="constant",
            constant_values=0,
        ),
    ])
    
    # Apply transforms
    transformed_data = transform(test_case)
    
    # Move to device
    input_data = transformed_data["input_image"].unsqueeze(0).to(device)
    target_data = transformed_data["target_image"].unsqueeze(0).to(device)
    
    # Run inference
    roi = (128, 128, 128)
    with torch.no_grad():
        prediction = sliding_window_inference(
            inputs=input_data,
            roi_size=roi,
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
            mode="gaussian",
            sigma_scale=0.125,
            padding_mode="constant",
            cval=0.0,
        )
    
    # Split predictions
    pred_synthesis = prediction[:, 0:1, ...]  # First channel
    pred_segmentation = prediction[:, 1:4, ...]  # Last 3 channels
    
    # Post-process segmentation
    post_sigmoid = Activations(sigmoid=True)
    post_pred = AsDiscrete(argmax=False, threshold=0.5)
    
    pred_seg_sigmoid = post_sigmoid(pred_segmentation)
    pred_seg_discrete = post_pred(pred_seg_sigmoid)
    
    # Convert to BraTS format
    pred_seg_brats = convert_segmentation_to_brats_labels(pred_seg_discrete[0])
    
    # Calculate synthesis metrics
    synth_l1 = F.l1_loss(pred_synthesis, target_data).item()
    synth_mse = F.mse_loss(pred_synthesis, target_data).item()
    synth_psnr = 20 * torch.log10(1.0 / torch.sqrt(torch.tensor(synth_mse))).item()
    
    # Save results
    case_id = test_case["case_id"]
    case_output_dir = os.path.join(output_dir, case_id)
    os.makedirs(case_output_dir, exist_ok=True)
    
    # Save synthesized modality
    synth_path = os.path.join(case_output_dir, f"{case_id}_{target_modality.lower()}_synthesized.nii.gz")
    synth_img = nib.Nifti1Image(pred_synthesis[0, 0].cpu().numpy(), np.eye(4))
    nib.save(synth_img, synth_path)
    
    # Save segmentation
    seg_path = os.path.join(case_output_dir, f"{case_id}_segmentation.nii.gz")
    seg_img = nib.Nifti1Image(pred_seg_brats, np.eye(4))
    nib.save(seg_img, seg_path)
    
    # Create visualization
    fig = create_comparison_visualization(
        input_data[0].cpu().numpy(),
        target_data[0].cpu().numpy(),
        pred_synthesis[0].cpu().numpy(),
        pred_seg_brats,
        test_case["input_modalities"],
        target_modality,
        case_id
    )
    
    viz_path = os.path.join(case_output_dir, f"{case_id}_multitask_results.png")
    fig.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ {case_id}: L1={synth_l1:.6f}, PSNR={synth_psnr:.2f}dB")
    
    return {
        "case_id": case_id,
        "synthesis_l1": synth_l1,
        "synthesis_mse": synth_mse,
        "synthesis_psnr": synth_psnr,
        "synthesis_path": synth_path,
        "segmentation_path": seg_path,
        "visualization_path": viz_path
    }


def main():
    parser = argparse.ArgumentParser(description='Multi-task UNETR Inference')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained multi-task model')
    parser.add_argument('--data_dir', type=str, 
                        default="ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData",
                        help='Directory containing test data')
    parser.add_argument('--output_dir', type=str, default="/data/multitask_inference_results",
                        help='Directory to save results')
    parser.add_argument('--max_cases', type=int, default=None,
                        help='Maximum number of cases to process')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model, target_modality = load_multitask_model(args.model_path, device)
    
    # Find test cases
    test_cases = find_test_cases(args.data_dir)
    
    if args.max_cases:
        test_cases = test_cases[:args.max_cases]
    
    print(f"\nRunning multi-task inference on {len(test_cases)} cases")
    print(f"Target modality to synthesize: {target_modality}")
    print(f"Simultaneous segmentation will be performed")
    
    # Run inference
    all_results = []
    
    for i, case_data in enumerate(test_cases):
        try:
            print(f"\nProcessing case {i+1}/{len(test_cases)}: {case_data['case_id']}")
            
            # Prepare input for this specific model
            test_case = prepare_input_for_model(case_data, target_modality)
            
            # Run inference
            result = run_multitask_inference(
                model, test_case, target_modality, device, args.output_dir
            )
            all_results.append(result)
            
        except Exception as e:
            print(f"Error processing case {case_data['case_id']}: {e}")
            continue
    
    # Summary statistics
    if all_results:
        synthesis_l1_scores = [r["synthesis_l1"] for r in all_results]
        synthesis_psnr_scores = [r["synthesis_psnr"] for r in all_results]
        
        print(f"\n{'='*60}")
        print(f"MULTI-TASK INFERENCE SUMMARY")
        print(f"{'='*60}")
        print(f"Processed cases: {len(all_results)}")
        print(f"Target modality: {target_modality}")
        print(f"\nSYNTHESIS METRICS:")
        print(f"  Mean L1 Loss: {np.mean(synthesis_l1_scores):.6f} ± {np.std(synthesis_l1_scores):.6f}")
        print(f"  Mean PSNR: {np.mean(synthesis_psnr_scores):.2f} ± {np.std(synthesis_psnr_scores):.2f} dB")
        print(f"\nSEGMENTATION:")
        print(f"  Segmentation maps saved for all cases")
        print(f"  Use external tools to evaluate segmentation performance")
        print(f"\nOUTPUTS:")
        print(f"  Synthesized modalities: *_synthesized.nii.gz")
        print(f"  Segmentation maps: *_segmentation.nii.gz")
        print(f"  Visualizations: *_multitask_results.png")
        print(f"\nResults saved to: {args.output_dir}")
        print(f"✅ Multi-task inference completed successfully!")
    else:
        print("❌ No cases processed successfully")


if __name__ == "__main__":
    main()