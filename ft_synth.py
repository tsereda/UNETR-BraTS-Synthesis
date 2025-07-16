#!/usr/bin/env python3
"""
BraTS Modality Synthesis - Transfer Learning from Segmentation Model
ENHANCED VERSION: Frequent sample logging with all input modalities visible
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


class FrequentSampleLogger:
    """WandB logger for frequent synthesis sample tracking"""
    
    def __init__(self, target_modality):
        self.target_modality = target_modality
        self.step = 0
        self.samples_logged = 0
        
        # Define input modality names based on target
        all_modalities = ["FLAIR", "T1CE", "T1", "T2"]
        self.input_modalities = [mod for mod in all_modalities if mod != target_modality]
        print(f"Input modalities: {self.input_modalities}")
        print(f"Target modality: {target_modality}")
        
    def log_training_metrics(self, epoch, batch_idx, total_batches, loss, loss_components, lr):
        """Log training metrics - reasonable frequency"""
        # Log every 25 batches or 5% of epoch progress
        log_frequency = max(25, total_batches // 20)
        
        if batch_idx % log_frequency == 0:
            wandb.log({
                "train/loss": loss,
                "train/dice_loss": loss_components["dice"],
                "train/learning_rate": lr,
                "train/epoch": epoch + 1,
                "train/batch": batch_idx,
                "train/progress": (epoch * total_batches + batch_idx) / (50 * total_batches)
            }, step=self.step)
            self.step += 1
    
    def log_training_samples(self, model, input_data, target_data, batch_data, epoch, batch_idx):
        """Log samples DURING training - early and often!"""
        try:
            model.eval()
            with torch.no_grad():
                predicted = model(input_data[:1])  # Just first sample to save memory
                
                # Get case name
                case_name = batch_data.get("case_id", [f"epoch{epoch+1}_batch{batch_idx}"])[0]
                
                # Log single sample with all 3 input modalities
                self._log_detailed_sample(
                    input_data[0].cpu().numpy(),
                    target_data[0].cpu().numpy(), 
                    predicted[0].cpu().numpy(),
                    case_name,
                    f"TRAINING | Epoch {epoch+1} Batch {batch_idx}"
                )
            model.train()
        except Exception as e:
            print(f"Error logging training sample: {e}")
    
    def log_validation_samples(self, inputs, targets, predictions, case_names, epoch, split="val"):
        """Log validation samples - show more detail"""
        try:
            for i in range(min(5, len(inputs))):  # Show up to 5 validation samples
                self._log_detailed_sample(
                    inputs[i], targets[i], predictions[i], case_names[i],
                    f"VALIDATION | Epoch {epoch+1}"
                )
        except Exception as e:
            print(f"Error logging validation samples: {e}")
    
    def _log_detailed_sample(self, input_vol, target_vol, pred_vol, case_name, stage_info):
        """Log detailed sample showing all 3 input modalities"""
        try:
            # Get middle slice
            slice_idx = input_vol.shape[-1] // 2
            
            # Extract all slices
            input1_slice = input_vol[0, :, :, slice_idx]  # First input modality
            input2_slice = input_vol[1, :, :, slice_idx]  # Second input modality  
            input3_slice = input_vol[2, :, :, slice_idx]  # Third input modality
            target_slice = target_vol[0, :, :, slice_idx]
            pred_slice = pred_vol[0, :, :, slice_idx]
            
            # Create comprehensive visualization
            # Layout: [Input1 | Input2 | Input3 | Target | Prediction]
            all_images = [input1_slice, input2_slice, input3_slice, target_slice, pred_slice]
            
            # Normalize each image individually for better contrast
            normalized_images = []
            for img in all_images:
                img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
                normalized_images.append(img_norm)
            
            # Concatenate horizontally
            comparison = np.concatenate(normalized_images, axis=1)
            
            # Convert to RGB for wandb
            comparison_rgb = np.stack([comparison] * 3, axis=-1)
            
            # Create detailed caption
            modality_labels = " | ".join([
                f"{self.input_modalities[0]} INPUT",
                f"{self.input_modalities[1]} INPUT", 
                f"{self.input_modalities[2]} INPUT",
                f"{self.target_modality} TARGET",
                f"{self.target_modality} PREDICTED"
            ])
            
            caption = f"{stage_info} | {case_name} | {modality_labels}"
            
            # Log with sample counter for easy tracking
            wandb.log({
                f"samples/detailed_synthesis": wandb.Image(comparison_rgb, caption=caption),
                f"samples/count": self.samples_logged
            }, step=self.step)
            
            self.samples_logged += 1
            self.step += 1
            
        except Exception as e:
            print(f"Error creating detailed sample: {e}")
    
    def log_epoch_summary(self, epoch, train_loss, val_metrics, epoch_time):
        """Log epoch summary"""
        wandb.log({
            "epoch": epoch + 1,
            "summary/train_loss": train_loss,
            "summary/val_l1": val_metrics["l1"],
            "summary/val_psnr": val_metrics.get("psnr", 0.0),
            "summary/val_ssim": val_metrics.get("ssim", 0.0),
            "summary/epoch_time": epoch_time,
            "summary/samples_logged_total": self.samples_logged
        }, step=self.step)
        self.step += 1
    
    def log_best_model(self, epoch, val_l1, model_path):
        """Log when best model is saved"""
        wandb.log({
            "best_model/epoch": epoch + 1,
            "best_model/val_l1": val_l1,
            "best_model/saved": True
        }, step=self.step)
        self.step += 1


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


def frequent_sample_train_epoch(model, loader, optimizer, epoch, loss_func, max_epochs, target_modality, logger):
    """Training with frequent sample logging"""
    model.train()
    run_loss = AverageMeter()
    run_dice = AverageMeter()
    
    for idx, batch_data in enumerate(loader):
        input_data = batch_data["input_image"].cuda()
        target_data = batch_data["target_image"].cuda()
        
        optimizer.zero_grad()
        predicted = model(input_data)
        total_loss, loss_components = loss_func(predicted, target_data)
        total_loss.backward()
        optimizer.step()
        
        # Update metrics
        run_loss.update(total_loss.item(), n=input_data.shape[0])
        run_dice.update(loss_components["dice"], n=input_data.shape[0])
        
        # Log training metrics
        logger.log_training_metrics(
            epoch, idx, len(loader), 
            total_loss.item(), loss_components, 
            optimizer.param_groups[0]['lr']
        )
        
        # LOG SAMPLES FREQUENTLY - every 15 batches in early epochs, less frequent later
        if epoch < 5:
            sample_freq = 15  # Very frequent early on
        elif epoch < 20:
            sample_freq = 30  # Medium frequency
        else:
            sample_freq = 50  # Less frequent later
            
        if (idx + 1) % sample_freq == 0:
            print(f"Logging training sample at epoch {epoch+1}, batch {idx+1}")
            logger.log_training_samples(model, input_data, target_data, batch_data, epoch, idx)
        
        # Progress print
        if (idx + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{max_epochs} [{idx+1}/{len(loader)}] "
                  f"Loss: {run_loss.avg:.4f} Dice: {run_dice.avg:.4f}")
    
    return run_loss.avg


def frequent_sample_val_epoch(model, loader, epoch, max_epochs, target_modality, logger):
    """Validation with frequent sample logging"""
    model.eval()
    run_l1 = AverageMeter()
    run_psnr = AverageMeter()
    run_ssim = AverageMeter()
    
    # Collect samples for logging
    sample_inputs, sample_targets, sample_preds, sample_names = [], [], [], []
    
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            try:
                input_data = batch_data["input_image"].cuda()
                target_data = batch_data["target_image"].cuda()
                predicted = model(input_data)
                
                # Compute metrics
                l1_loss = torch.nn.functional.l1_loss(predicted, target_data)
                run_l1.update(l1_loss.item(), n=input_data.shape[0])
                
                # PSNR and SSIM (handle potential errors)
                try:
                    psnr_metric = PSNRMetric(max_val=1.0)
                    ssim_metric = SSIMMetric(spatial_dims=3)
                    psnr_val = psnr_metric(predicted, target_data).item()
                    ssim_val = ssim_metric(predicted, target_data).item()
                    run_psnr.update(psnr_val, n=input_data.shape[0])
                    run_ssim.update(ssim_val, n=input_data.shape[0])
                except:
                    pass  # Skip if metrics fail
                
                # Collect samples (more samples for better coverage)
                if len(sample_inputs) < 8:  # Increased from 3 to 8
                    sample_inputs.append(input_data[0].cpu().numpy())
                    sample_targets.append(target_data[0].cpu().numpy())
                    sample_preds.append(predicted[0].cpu().numpy())
                    sample_names.append(batch_data.get("case_id", [f"val_case_{idx}"])[0])
                
                if (idx + 1) % 10 == 0:
                    print(f"Val [{idx+1}/{len(loader)}] L1: {run_l1.avg:.6f}")
                    
            except Exception as e:
                print(f"Error in validation step {idx}: {e}")
                continue
    
    # Log validation samples EVERY epoch (not just every 5)
    print(f"Logging {len(sample_inputs)} validation samples for epoch {epoch+1}")
    logger.log_validation_samples(
        sample_inputs, sample_targets, sample_preds, sample_names, epoch, "val"
    )
    
    return {"l1": run_l1.avg, "psnr": run_psnr.avg, "ssim": run_ssim.avg}


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='BraTS Modality Synthesis with Frequent Sample Logging')
    parser.add_argument('--pretrained_path', type=str, required=True,
                        help='Path to pretrained segmentation model')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Path where to save the best synthesis model')
    parser.add_argument('--target_modality', type=str, default='T1CE',
                        choices=['FLAIR', 'T1CE', 'T1', 'T2'],
                        help='Target modality to synthesize')
    parser.add_argument('--max_epochs', type=int, default=50,
                        help='Maximum number of training epochs')
    parser.add_argument('--use_tables', action='store_true',
                        help='Also log results using WandB Tables for better organization')
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    
    # Ensure save directory exists
    save_dir = os.path.dirname(args.save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        print(f"Created save directory: {save_dir}")
    
    # Initialize W&B with frequent sample configuration
    roi = (128, 128, 128)
    wandb.init(
        project="BraTS2025",
        name=f"synthesis_{args.target_modality.lower()}",
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
            "loss": "Dice",
            "use_tables": args.use_tables,
            "logging_strategy": "frequent_samples_all_inputs",
            "sample_frequency": "every_15_batches_early_epochs",
            "validation_samples": "every_epoch_8_samples",
            "input_modalities_shown": "all_3_plus_target_and_prediction"
        }
    )
    
    print("✓ WandB initialized for frequent sample logging with all input modalities")
    
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
        "dataset/pretrained_model": args.pretrained_path
    })
    
    # Transforms for synthesis
    train_transform = transforms.Compose([
        transforms.LoadImaged(keys=["input_image", "target_image"]),
        transforms.EnsureChannelFirstd(keys=["target_image"]),
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
    
    # Create synthesis model with pretrained weights
    model = SynthesisModel(
        pretrained_seg_path=args.pretrained_path,
        output_channels=1
    ).cuda()
    
    # Synthesis loss function
    loss_func = DiceSynthesisLoss()
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-6)
    
    # Initialize frequent sample logger
    logger = FrequentSampleLogger(args.target_modality)
    
    print(f"\n=== SYNTHESIS TRAINING CONFIGURATION ===")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Learning rate: 5e-5 (reduced for transfer learning)")
    print(f"Loss: Dice")
    print(f"ROI size: {roi}")
    print(f"FREQUENT SAMPLE LOGGING ENABLED")
    print(f"Training samples: Every 15 batches (early epochs) → 50 batches (later)")
    print(f"Validation samples: Every epoch, up to 8 samples")
    print(f"Showing all input modalities: {logger.input_modalities}")
    
    best_l1 = float('inf')
    
    for epoch in range(args.max_epochs):
        print(f"\n=== EPOCH {epoch+1}/{args.max_epochs} ===")
        epoch_start = time.time()

        # Training with frequent samples
        train_loss = frequent_sample_train_epoch(
            model, train_loader, optimizer, epoch, 
            loss_func, args.max_epochs, args.target_modality, logger
        )

        print(f"EPOCH {epoch + 1} COMPLETE, avg_loss: {train_loss:.4f}, time: {time.time() - epoch_start:.2f}s")

        # Validation with samples every epoch
        epoch_time = time.time()
        val_metrics = frequent_sample_val_epoch(
            model, val_loader, epoch, args.max_epochs, args.target_modality, logger
        )

        epoch_time = time.time() - epoch_time
        
        # Log epoch summary
        logger.log_epoch_summary(epoch, train_loss, val_metrics, epoch_time)

        print(f"VALIDATION COMPLETE: L1: {val_metrics['l1']:.6f}, PSNR: {val_metrics['psnr']:.6f}, SSIM: {val_metrics['ssim']:.6f}")

        # Save best model
        if val_metrics["l1"] < best_l1:
            print(f"NEW BEST L1 SCORE! ({best_l1:.6f} --> {val_metrics['l1']:.6f})")
            best_l1 = val_metrics["l1"]
            
            logger.log_best_model(epoch, best_l1, args.save_path)

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
        print(f"Epoch {epoch+1} complete in {epoch_time:.1f}s | Samples logged: {logger.samples_logged}")
    
    print(f"\n🎉 TRAINING COMPLETE!")
    print(f"📊 Total samples logged: {logger.samples_logged}")
    print(f"🏆 Best L1: {best_l1:.6f}")
    print(f"✓ Target modality: {args.target_modality}")
    print(f"✓ Best model saved to: {args.save_path}")
    print(f"🔍 All input modalities were visible in samples")
    
    # Log summary to W&B
    wandb.run.summary["best_l1"] = best_l1
    wandb.run.summary["target_modality"] = args.target_modality
    wandb.run.summary["best_model_path"] = args.save_path
    wandb.run.summary["total_samples_logged"] = logger.samples_logged
    wandb.run.summary["logging_strategy"] = "frequent_samples_all_inputs"
    wandb.finish()


if __name__ == "__main__":
    main()