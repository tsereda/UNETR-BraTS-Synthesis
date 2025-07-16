#!/usr/bin/env python3
"""
BraTS Modality Synthesis - Transfer Learning from Segmentation Model
"""

import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from monai.inferers import sliding_window_inference
from monai import transforms
from monai.transforms import AsDiscrete, Activations
from monai.networks.nets import SwinUNETR
from monai.data import Dataset, DataLoader
from monai.metrics import PSNRMetric, SSIMMetric

# Suppress numpy warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count
        else:
            self.avg = self.sum


class SynthesisModel(nn.Module):
    """Adapt SwinUNETR from segmentation to synthesis"""
    
    def __init__(self, pretrained_seg_path=None, output_channels=1):
        super().__init__()
        # Always use 4 input channels
        self.backbone = SwinUNETR(
            in_channels=4,
            out_channels=3,
            feature_size=48,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=True,
        )
        if pretrained_seg_path and os.path.exists(pretrained_seg_path):
            print(f"Loading pretrained segmentation weights from: {pretrained_seg_path}")
            checkpoint = torch.load(pretrained_seg_path, map_location='cpu', weights_only=False)
            self.backbone.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded weights from epoch {checkpoint['epoch']}")
            print(f"✓ Segmentation dice: {checkpoint.get('val_acc_max', 'N/A')}")
        # Replace output head for synthesis (1 channel output)
        in_channels = self.backbone.out.conv.in_channels
        self.backbone.out = nn.Conv3d(
            in_channels,
            output_channels,
            kernel_size=1,
            padding=0
        )
        print(f"✓ Model adapted: 4 input → {output_channels} output channels")

    def forward(self, x):
        return self.backbone(x)


class SimplePerceptualLoss(nn.Module):
    """Simple perceptual loss using L1 loss (placeholder for full perceptual loss)"""
    
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
        self.l1_loss = nn.L1Loss()
        
    def forward(self, pred, target):
        # For now, just use L1 loss
        # In a full implementation, you'd use pretrained VGG features
        return self.weight * self.l1_loss(pred, target)


from monai.losses import DiceLoss

class DiceSynthesisLoss(nn.Module):
    """Dice loss for synthesis (for single-channel output)"""
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss(include_background=True, to_onehot_y=False, sigmoid=True, reduction="mean")

    def forward(self, pred, target):
        loss = self.dice(pred, target)
        return loss, {"dice": loss.item()}


def find_brats_cases(data_dir, target_modality="T1CE"):
    """Find BraTS cases and set up for synthesis"""
    cases = []
    
    print(f"Scanning {data_dir} for BraTS synthesis cases...")
    print(f"Target modality for synthesis: {target_modality}")
    
    if not os.path.exists(data_dir):
        print(f"ERROR: Directory {data_dir} does not exist!")
        return cases
    
    # Modality mapping
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
                # Always use 4 input channels: copy one modality if needed
                input_modalities = [mod for mod in modality_files.keys() if mod != target_modality]
                input_images = [files[mod] for mod in input_modalities]
                # Copy FLAIR (or T1CE if FLAIR is target) to keep 4 channels
                if len(input_images) == 3:
                    if target_modality != "FLAIR":
                        input_images.append(files["FLAIR"])
                    else:
                        input_images.append(files["T1CE"])
                case_data = {
                    "input_image": input_images,
                    "target_image": files[target_modality],
                    "case_id": item,
                    "target_modality": target_modality
                }
                cases.append(case_data)
                if len(cases) % 50 == 0:
                    print(f"Found {len(cases)} valid synthesis cases so far...")
    print(f"Finished scanning synthesis data. Total cases found: {len(cases)}")
    print(f"Input modalities: always 4 (one copied if needed)")
    print(f"Target modality: {target_modality}")
    return cases


def log_synthesis_samples(inputs, targets, predictions, case_names, epoch=None, batch_idx=None, target_modality="T1CE"):
    """Log synthesis samples to W&B (show all three input modalities, no diff image)"""
    try:
        num_samples = len(inputs)
        fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4 * num_samples))

        # Handle single sample case
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(num_samples):
            slice_idx = inputs[i].shape[-1] // 2
            # Show all three input modalities (channels 0,1,2)
            for ch in range(3):
                axes[i, ch].imshow(inputs[i][ch, :, :, slice_idx], cmap='gray')
                axes[i, ch].set_title(f'Input Modality {ch+1}')
                axes[i, ch].axis('off')
            # Target modality
            axes[i, 3].imshow(targets[i][0, :, :, slice_idx], cmap='gray')
            axes[i, 3].set_title(f'Target ({target_modality})')
            axes[i, 3].axis('off')
            # Predicted modality
            axes[i, 4].imshow(predictions[i][0, :, :, slice_idx], cmap='gray')
            axes[i, 4].set_title(f'Predicted ({target_modality})')
            axes[i, 4].axis('off')

        plt.tight_layout()

        title = f"synthesis_samples_{target_modality.lower()}"
        if epoch is not None:
            title += f"_epoch_{epoch}"
        if batch_idx is not None:
            title += f"_batch_{batch_idx}"

        wandb.log({f"synthesis/{title}": wandb.Image(fig)})
        plt.close(fig)
    except Exception as e:
        print(f"Error logging synthesis samples: {e}")


def train_epoch(model, loader, optimizer, epoch, loss_func, max_epochs, target_modality):
    """Training epoch for synthesis"""
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    run_l1 = AverageMeter()
    run_mse = AverageMeter()
    batch_log_freq = 50
    
    for idx, batch_data in enumerate(loader):
        input_data = batch_data["input_image"].cuda()
        target_data = batch_data["target_image"].cuda()
        
        optimizer.zero_grad()
        
        # Forward pass
        predicted = model(input_data)
        
        # Compute loss
        total_loss, loss_components = loss_func(predicted, target_data)
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Update metrics
        run_loss.update(total_loss.item(), n=input_data.shape[0])
        run_l1.update(loss_components["l1"], n=input_data.shape[0])
        run_mse.update(loss_components["mse"], n=input_data.shape[0])
        
        print(
            f"Epoch {epoch + 1}/{max_epochs} Batch {idx + 1}/{len(loader)}",
            f"loss: {run_loss.avg:.4f}",
            f"l1: {run_l1.avg:.4f}",
            f"mse: {run_mse.avg:.4f}",
            f"time: {time.time() - start_time:.2f}s"
        )
        
        # Log to W&B every batch_log_freq batches
        if (idx + 1) % batch_log_freq == 0:
            wandb.log({
                "batch_step": epoch * len(loader) + idx,
                "batch_loss": total_loss.item(),
                "batch_loss_avg": run_loss.avg,
                "batch_l1": loss_components["l1"],
                "batch_mse": loss_components["mse"],
                "epoch": epoch + 1,
                "batch": idx + 1
            })
            
            # Log sample synthesis every 100 batches
            if (idx + 1) % 100 == 0:
                log_batch_synthesis(model, input_data, target_data, batch_data, epoch, idx, target_modality)
        
        start_time = time.time()
    
    return run_loss.avg


def log_batch_synthesis(model, input_data, target_data, batch_data, epoch, batch_idx, target_modality):
    """Log a quick synthesis sample during training"""
    try:
        model.eval()
        with torch.no_grad():
            # Get prediction for first sample in batch
            predicted = model(input_data[:1])
            
            # Log single sample
            case_name = f"batch_sample_{batch_idx}"
            log_synthesis_samples(
                [input_data[0].cpu().numpy()], 
                [target_data[0].cpu().numpy()], 
                [predicted[0].cpu().numpy()], 
                [case_name], 
                epoch,
                batch_idx,
                target_modality
            )
        model.train()
    except Exception as e:
        print(f"Error logging batch synthesis sample: {e}")


def val_epoch(model, loader, epoch, max_epochs, target_modality):
    """Validation epoch for synthesis"""
    model.eval()
    start_time = time.time()
    
    # Initialize metrics
    l1_loss = nn.L1Loss()
    
    run_psnr = AverageMeter()
    run_ssim = AverageMeter()
    run_l1 = AverageMeter()

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            try:
                input_data = batch_data["input_image"].cuda()
                target_data = batch_data["target_image"].cuda()
                
                # Forward pass
                predicted = model(input_data)
                
                # Compute metrics
                l1_val = l1_loss(predicted, target_data).item()
                
                # PSNR and SSIM (handle potential errors)
                try:
                    psnr_metric = PSNRMetric(max_val=1.0)
                    ssim_metric = SSIMMetric(spatial_dims=3)
                    psnr_val = psnr_metric(predicted, target_data).item()
                    ssim_val = ssim_metric(predicted, target_data).item()
                except:
                    psnr_val = 0.0
                    ssim_val = 0.0
                
                run_l1.update(l1_val, n=input_data.shape[0])
                run_psnr.update(psnr_val, n=input_data.shape[0])
                run_ssim.update(ssim_val, n=input_data.shape[0])
                
                print(
                    f"Val Epoch {epoch + 1}/{max_epochs} Batch {idx + 1}/{len(loader)}",
                    f"L1: {run_l1.avg:.6f}",
                    f"PSNR: {run_psnr.avg:.6f}",
                    f"SSIM: {run_ssim.avg:.6f}",
                    f"time: {time.time() - start_time:.2f}s"
                )
                start_time = time.time()
                
            except Exception as e:
                print(f"Error in validation step {idx}: {e}")
                continue

    return {"l1": run_l1.avg, "psnr": run_psnr.avg, "ssim": run_ssim.avg}


def log_validation_samples(model, loader, epoch, target_modality, num_samples=3):
    """Log synthesis samples during validation"""
    model.eval()
    
    inputs, targets, predictions, case_names = [], [], [], []
    
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if len(inputs) >= num_samples:
                break
                
            try:
                input_data = batch_data["input_image"].cuda()
                target_data = batch_data["target_image"].cuda()
                case_id = batch_data.get("case_id", [f"case_{idx}"])[0]
                
                # Get prediction
                predicted = model(input_data)
                
                # Collect data
                inputs.append(input_data[0].cpu().numpy())
                targets.append(target_data[0].cpu().numpy())
                predictions.append(predicted[0].cpu().numpy())
                case_names.append(case_id)
                
            except Exception as e:
                print(f"Error processing validation sample {idx}: {e}")
                continue
    
    # Log all samples in one figure
    if inputs:
        log_synthesis_samples(inputs, targets, predictions, case_names, epoch, target_modality=target_modality)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='BraTS Modality Synthesis')
    parser.add_argument('--pretrained_path', type=str, required=True,
                        help='Path to pretrained segmentation model')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Path where to save the best synthesis model')
    parser.add_argument('--target_modality', type=str, default='T1CE',
                        choices=['FLAIR', 'T1CE', 'T1', 'T2'],
                        help='Target modality to synthesize')
    parser.add_argument('--max_epochs', type=int, default=50,
                        help='Maximum number of training epochs')
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    
    # Ensure save directory exists
    save_dir = os.path.dirname(args.save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        print(f"Created save directory: {save_dir}")
    
    # Initialize W&B
    roi = (128, 128, 128)
    wandb.init(
        project="BraTS-Synthesis", 
        name=f"synthesis_{args.target_modality.lower()}_transfer_learning",
        config={
            "target_modality": args.target_modality,
            "max_epochs": args.max_epochs,
            "pretrained_path": args.pretrained_path,
            "save_path": args.save_path,
            "batch_size": 2,
            "roi": roi,
            "optimizer": "AdamW",
            "learning_rate": 5e-5,
            "weight_decay": 1e-5,
            "scheduler": "CosineAnnealingLR",
            "loss": "Combined L1 + MSE + Perceptual"
        }
    )
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Target modality: {args.target_modality}")
    print(f"Pretrained model: {args.pretrained_path}")
    print(f"Model will be saved to: {args.save_path}")
    
    # Find synthesis data
    print("Looking for BraTS data...")
    base_dir = "/app/UNETR-BraTS-Synthesis"
    data_dir = os.path.join(base_dir, "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData")
    print(f"Data directory: {data_dir}")

    # Load all cases for synthesis
    all_cases = find_brats_cases(data_dir, target_modality=args.target_modality)
    print(f"Total synthesis cases found: {len(all_cases)}")

    # Split into train/val (80% train, 20% val for synthesis)
    np.random.seed(42)
    np.random.shuffle(all_cases)
    split_idx = int(0.8 * len(all_cases))
    train_cases = all_cases[:split_idx]
    val_cases = all_cases[split_idx:]

    print(f"\n=== SYNTHESIS DATASET SUMMARY ===")
    print(f"Training cases: {len(train_cases)}")
    print(f"Validation cases: {len(val_cases)}")
    print(f"Target modality: {args.target_modality}")
    print(f"Input modalities: 3 (excluding {args.target_modality})")

    if not train_cases:
        print("No training cases found!")
        return

    # Log dataset info to W&B
    wandb.log({
        "dataset/train_cases": len(train_cases),
        "dataset/val_cases": len(val_cases),
        "dataset/target_modality": args.target_modality,
        "pretrained_model": args.pretrained_path
    })
    
    # Transforms for synthesis
    
    def debug_shape(data):
        # Print shapes for debugging
        input_img = data["input_image"]
        target_img = data["target_image"]
        if isinstance(input_img, list):
            print("input_image is a list, shapes:", [np.array(i).shape for i in input_img])
        else:
            print("input_image shape:", np.array(input_img).shape)
        print("target_image shape:", np.array(target_img).shape)
        return data

    train_transform = transforms.Compose([
        transforms.LoadImaged(keys=["input_image", "target_image"]),
        transforms.EnsureChannelFirstd(keys=["target_image"]),
        debug_shape,  # Debug print after loading
        transforms.NormalizeIntensityd(keys=["input_image", "target_image"], nonzero=True, channel_wise=True),
        transforms.CropForegroundd(
            keys=["input_image", "target_image"],
            source_key="input_image",
            k_divisible=[roi[0], roi[1], roi[2]],
            allow_smaller=True,
        ),
        transforms.RandSpatialCropd(
            keys=["input_image", "target_image"],
            roi_size=[roi[0], roi[1], roi[2]],
            random_size=False,
        ),
        # Data augmentation
        transforms.RandFlipd(keys=["input_image", "target_image"], prob=0.5, spatial_axis=0),
        transforms.RandFlipd(keys=["input_image", "target_image"], prob=0.5, spatial_axis=1),
        transforms.RandFlipd(keys=["input_image", "target_image"], prob=0.5, spatial_axis=2),
        transforms.RandRotate90d(keys=["input_image", "target_image"], prob=0.3, spatial_axes=(0, 1)),
        transforms.RandScaleIntensityd(keys=["input_image", "target_image"], factors=0.1, prob=0.5),
        transforms.RandShiftIntensityd(keys=["input_image", "target_image"], offsets=0.1, prob=0.5),
    ])
    
    val_transform = transforms.Compose([
        transforms.LoadImaged(keys=["input_image", "target_image"]),
        transforms.NormalizeIntensityd(keys=["input_image", "target_image"], nonzero=True, channel_wise=True),
    ])
    
    # Data loaders
    batch_size = 2
    train_ds = Dataset(data=train_cases, transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)
    
    val_ds = Dataset(data=val_cases, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
    
    # Removed mistaken print statement that used 'self' in main()
    # Create synthesis model with pretrained weights
    model = SynthesisModel(
        pretrained_seg_path=args.pretrained_path,
        output_channels=1
    ).cuda()
    
    # Synthesis loss function
    loss_func = DiceSynthesisLoss()
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)  # Lower LR for transfer
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-6)
    
    print(f"\n=== SYNTHESIS TRAINING CONFIGURATION ===")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Learning rate: 5e-5 (reduced for transfer learning)")
    print(f"Loss: Combined L1 + MSE + Perceptual")
    print(f"ROI size: {roi}")
    
    best_l1 = float('inf')
    
    for epoch in range(args.max_epochs):
        # Optionally log gradients and parameter histograms every epoch
        wandb.watch(model, log="all", log_freq=100)
        print(f"\n=== EPOCH {epoch+1}/{args.max_epochs} ===")
        epoch_time = time.time()
        
        # Training
        train_loss = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            loss_func=loss_func,
            max_epochs=args.max_epochs,
            target_modality=args.target_modality
        )
        
        print(f"EPOCH {epoch + 1} COMPLETE, avg_loss: {train_loss:.4f}, time: {time.time() - epoch_time:.2f}s")
        
        # Validation
        epoch_time = time.time()
        val_metrics = val_epoch(
            model=model,
            loader=val_loader,
            epoch=epoch,
            max_epochs=args.max_epochs,
            target_modality=args.target_modality
        )
        
        # Log metrics to W&B
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_l1": val_metrics["l1"],
            "val_psnr": val_metrics["psnr"],
            "val_ssim": val_metrics["ssim"],
            "learning_rate": optimizer.param_groups[0]['lr'],
            "val_time": time.time() - epoch_time
        })
        
        print(f"VALIDATION COMPLETE: L1: {val_metrics['l1']:.6f}, PSNR: {val_metrics['psnr']:.6f}, SSIM: {val_metrics['ssim']:.6f}")
        
        # Log synthesis samples
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print("Logging synthesis samples...")
            log_validation_samples(model, val_loader, epoch, args.target_modality, num_samples=3)
        
        # Save best model
        if val_metrics["l1"] < best_l1:
            print(f"NEW BEST L1 SCORE! ({best_l1:.6f} --> {val_metrics['l1']:.6f})")
            best_l1 = val_metrics["l1"]
            wandb.log({"best_val_l1": best_l1})
            
            try:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_l1': best_l1,
                    'val_metrics': val_metrics,
                    'target_modality': args.target_modality,
                    'pretrained_from': args.pretrained_path
                }, args.save_path)
                print(f"✓ Best synthesis model saved to: {args.save_path}")
            except Exception as e:
                print(f"ERROR saving model: {e}")
                
        scheduler.step()
    
    print(f"\n✓ SYNTHESIS TRAINING COMPLETED!")
    print(f"✓ Best L1 score: {best_l1:.6f}")
    print(f"✓ Target modality: {args.target_modality}")
    print(f"✓ Best model saved to: {args.save_path}")
    # Log summary to W&B
    wandb.run.summary["best_l1"] = best_l1
    wandb.run.summary["target_modality"] = args.target_modality
    wandb.run.summary["best_model_path"] = args.save_path
    wandb.finish()


if __name__ == "__main__":
    main()