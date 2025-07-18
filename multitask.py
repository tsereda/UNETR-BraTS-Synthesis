#!/usr/bin/env python3
"""
BraTS Multi-task UNETR: Synthesis + Segmentation
Train 4 models that simultaneously synthesize missing modality and perform segmentation
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
import torch.nn as nn
import torch.nn.functional as F
import wandb

from monai.inferers import sliding_window_inference
from monai import transforms
from monai.transforms import AsDiscrete, Activations, DivisiblePadd
from monai.networks.nets import SwinUNETR
from monai.data import Dataset, DataLoader
from monai.metrics import DiceMetric, PSNRMetric, SSIMMetric
from monai.losses import DiceFocalLoss
from monai.utils.enums import MetricReduction
from monai.data import decollate_batch

# Suppress warnings
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
        if isinstance(self.count, np.ndarray):
            if (self.count > 0).all():
                self.avg = self.sum / self.count
            else:
                self.avg = self.sum
        else:
            if self.count > 0:
                self.avg = self.sum / self.count
            else:
                self.avg = self.sum


class MultiTaskSwinUNETR(nn.Module):
    """SwinUNETR for multi-task learning: synthesis + multi-class segmentation"""
    def __init__(self, num_segmentation_classes=4):
        super().__init__()
        self.num_segmentation_classes = num_segmentation_classes
        # Shared encoder-decoder backbone
        self.backbone = SwinUNETR(
            in_channels=3,
            out_channels=64,  # Shared feature output
            feature_size=48,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=True,
        )
        # Task-specific heads
        self.synth_head = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=1)
        )
        self.seg_head = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, num_segmentation_classes, kernel_size=1)
        )
        print(f"✓ Multi-task model initialized: 3 input → 1 synthesis + {num_segmentation_classes} segmentation channels")

    def forward(self, x):
        features = self.backbone(x)
        synth = self.synth_head(features)
        seg = self.seg_head(features)
        # Concatenate along channel dim: [B, 1+num_segmentation_classes, ...]
        return torch.cat([synth, seg], dim=1)


class MultiTaskLoss(nn.Module):
    """Combined loss for synthesis and multi-class segmentation"""
    def __init__(self, synthesis_weight=1.0, segmentation_weight=1.0, num_segmentation_classes=4):
        super().__init__()
        self.synthesis_weight = synthesis_weight
        self.segmentation_weight = segmentation_weight
        self.num_segmentation_classes = num_segmentation_classes
        
        # Synthesis loss (L1 + MSE)
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
        # Segmentation loss: DiceFocal + CrossEntropy
        self.dice_focal_loss = DiceFocalLoss(
            to_onehot_y=True,
            softmax=True,
            gamma=2.0,
            lambda_dice=1.0,
            lambda_focal=1.0,
        )
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, pred, target_synthesis, target_segmentation):
        # Split predictions
        pred_synthesis = pred[:, 0:1, ...]  # First channel
        pred_segmentation = pred[:, 1:1+self.num_segmentation_classes, ...]  # Next N channels
        
        # Fix target_synthesis shape handling
        if target_synthesis.dim() == 4:
            target_synthesis = target_synthesis.unsqueeze(1)
        elif target_synthesis.dim() == 5 and target_synthesis.shape[1] == 1:
            pass  # Already has correct shape
        else:
            # Ensure target_synthesis has channel dimension
            target_synthesis = target_synthesis.unsqueeze(1)
        
        # Ensure shapes match
        if pred_synthesis.shape != target_synthesis.shape:
            # Try to adjust target_synthesis shape
            if target_synthesis.dim() == 5 and target_synthesis.shape[1] != 1:
                # If target has multiple channels, take first one
                target_synthesis = target_synthesis[:, 0:1, ...]
            elif target_synthesis.dim() == 4:
                # Add channel dimension
                target_synthesis = target_synthesis.unsqueeze(1)
        
        # Synthesis loss
        synthesis_loss = self.l1_loss(pred_synthesis, target_synthesis) + \
                         0.5 * self.mse_loss(pred_synthesis, target_synthesis)
        
        # Segmentation loss
        # Ensure target_segmentation is in correct format (squeeze channel dim if present)
        if target_segmentation.dim() == 5 and target_segmentation.shape[1] == 1:
            target_segmentation = target_segmentation.squeeze(1)
        
        dice_focal = self.dice_focal_loss(pred_segmentation, target_segmentation.long())
        ce = self.ce_loss(pred_segmentation, target_segmentation.long())
        segmentation_loss = dice_focal + ce
        
        # Combined loss
        total_loss = (self.synthesis_weight * synthesis_loss + 
                      self.segmentation_weight * segmentation_loss)
        
        return total_loss, {
            "total": total_loss.item(),
            "synthesis": synthesis_loss.item(),
            "segmentation": segmentation_loss.item(),
            "seg_dicefocal": dice_focal.item(),
            "seg_ce": ce.item()
        }


class MultiTaskLogger:
    """Enhanced logger for multi-task learning"""
    
    def __init__(self, target_modality):
        self.target_modality = target_modality
        self.step = 0
        self.samples_logged = 0
        
        # Define input modalities
        all_modalities = ["FLAIR", "T1CE", "T1", "T2"]
        self.input_modalities = [mod for mod in all_modalities if mod != target_modality]
        print(f"Input modalities: {self.input_modalities}")
        print(f"Target modality: {target_modality}")
        
    def log_training_metrics(self, epoch, batch_idx, total_batches, loss_components, lr):
        """Log training metrics"""
        log_frequency = max(25, total_batches // 20)
        
        if batch_idx % log_frequency == 0:
            wandb.log({
                "train/total_loss": loss_components["total"],
                "train/synthesis_loss": loss_components["synthesis"],
                "train/segmentation_loss": loss_components["segmentation"],
                "train/learning_rate": lr,
                "train/epoch": epoch + 1,
                "train/batch": batch_idx,
            }, step=self.step)
            self.step += 1
    
    def log_training_samples(self, model, input_data, target_synth, target_seg, batch_data, epoch, batch_idx):
        """Log multi-task samples during training"""
        try:
            model.eval()
            with torch.no_grad():
                predicted = model(input_data[:1])
                pred_synth = predicted[:, 0:1, ...]
                pred_seg_raw = predicted[:, 1:, ...]
                
                # Apply softmax and get argmax for visualization
                pred_seg = torch.softmax(pred_seg_raw, dim=1)
                pred_seg = torch.argmax(pred_seg, dim=1, keepdim=True)
                
                case_name = batch_data.get("case_id", [f"epoch{epoch+1}_batch{batch_idx}"])[0]
                
                self._log_multitask_sample(
                    input_data[0].cpu().numpy(),
                    target_synth[0].cpu().numpy(),
                    target_seg[0].cpu().numpy(),
                    pred_synth[0].cpu().numpy(),
                    pred_seg[0].cpu().numpy(),
                    case_name,
                    f"TRAINING | Epoch {epoch+1} Batch {batch_idx}"
                )
            model.train()
        except Exception as e:
            print(f"Error logging training sample: {e}")
    
    def log_validation_samples(self, inputs, targets_synth, targets_seg, 
                               predictions_synth, predictions_seg, case_names, epoch):
        """Log validation samples"""
        try:
            for i in range(min(5, len(inputs))):
                self._log_multitask_sample(
                    inputs[i], targets_synth[i], targets_seg[i],
                    predictions_synth[i], predictions_seg[i], case_names[i],
                    f"VALIDATION | Epoch {epoch+1}"
                )
        except Exception as e:
            print(f"Error logging validation samples: {e}")
    
    def _log_multitask_sample(self, input_vol, target_synth, target_seg, 
                               pred_synth, pred_seg, case_name, stage_info):
        """Log detailed multi-task sample with color overlays"""
        try:
            # Handle input volume shape
            if input_vol.shape[0] == 3:
                # Input has 3 channels (3 modalities)
                pass
            elif input_vol.ndim == 4 and input_vol.shape[0] == 1:
                # Remove singleton dimension
                input_vol = input_vol[0]
            
            # Squeeze other arrays
            target_synth = np.squeeze(target_synth)
            pred_synth = np.squeeze(pred_synth)
            target_seg = np.squeeze(target_seg)
            pred_seg = np.squeeze(pred_seg)

            # Get middle slice
            slice_idx = input_vol.shape[-1] // 2
            
            # Extract slices
            if input_vol.ndim == 4:
                input1_slice = input_vol[0, :, :, slice_idx]
                input2_slice = input_vol[1, :, :, slice_idx]
                input3_slice = input_vol[2, :, :, slice_idx]
            else:
                # Fallback if shape is unexpected
                input1_slice = input_vol[:, :, slice_idx]
                input2_slice = input_vol[:, :, slice_idx]
                input3_slice = input_vol[:, :, slice_idx]
            
            target_synth_slice = target_synth[:, :, slice_idx]
            pred_synth_slice = pred_synth[:, :, slice_idx]
            target_seg_slice = target_seg[:, :, slice_idx]
            pred_seg_slice = pred_seg[:, :, slice_idx]

            # Normalize images
            def norm_img(img):
                img_min, img_max = img.min(), img.max()
                if img_max > img_min:
                    return (img - img_min) / (img_max - img_min)
                return img
            
            input1_slice = norm_img(input1_slice)
            input2_slice = norm_img(input2_slice)
            input3_slice = norm_img(input3_slice)
            target_synth_slice = norm_img(target_synth_slice)
            pred_synth_slice = norm_img(pred_synth_slice)

            # Create color map for segmentation
            def create_colored_mask(mask):
                mask = mask.astype(np.int32)
                colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
                
                # Background: black
                colored[mask == 0] = [0, 0, 0]
                # Class 1: red
                colored[mask == 1] = [255, 0, 0]
                # Class 2: green  
                colored[mask == 2] = [0, 255, 0]
                # Class 3: blue
                colored[mask == 3] = [0, 0, 255]
                
                return colored

            # Create colored segmentation masks
            target_seg_colored = create_colored_mask(target_seg_slice)
            pred_seg_colored = create_colored_mask(pred_seg_slice)

            # Convert grayscale to RGB
            def gray2rgb(img):
                img = (img * 255).astype(np.uint8)
                return np.stack([img, img, img], axis=-1)

            # Create layout
            layout = np.concatenate([
                gray2rgb(input1_slice),
                gray2rgb(input2_slice),
                gray2rgb(input3_slice),
                gray2rgb(target_synth_slice),
                gray2rgb(pred_synth_slice),
                target_seg_colored,
                pred_seg_colored
            ], axis=1)

            # Create labels
            labels = " | ".join([
                f"{self.input_modalities[0]}", f"{self.input_modalities[1]}", f"{self.input_modalities[2]}",
                f"{self.target_modality}_GT", f"{self.target_modality}_PRED",
                "SEG_GT", "SEG_PRED"
            ])
            
            caption = f"{stage_info} | {case_name} | {labels}"
            
            wandb.log({
                f"samples/multitask_{self.target_modality.lower()}": wandb.Image(layout, caption=caption),
                f"samples/count": self.samples_logged
            }, step=self.step)
            
            self.samples_logged += 1
            self.step += 1
            
        except Exception as e:
            print(f"Error creating multi-task sample: {e}")
            import traceback
            traceback.print_exc()
    
    def log_epoch_summary(self, epoch, train_losses, val_metrics, epoch_time):
        """Log epoch summary"""
        wandb.log({
            "epoch": epoch + 1,
            "summary/train_total_loss": train_losses["total"],
            "summary/train_synthesis_loss": train_losses["synthesis"],
            "summary/train_segmentation_loss": train_losses["segmentation"],
            "summary/val_synthesis_l1": val_metrics["synthesis_l1"],
            "summary/val_synthesis_psnr": val_metrics["synthesis_psnr"],
            "summary/val_synthesis_ssim": val_metrics["synthesis_ssim"],
            "summary/val_seg_dice": val_metrics["seg_dice"],
            "summary/epoch_time": epoch_time,
        }, step=self.step)
        self.step += 1


def find_multitask_cases(data_dir, target_modality="T1CE"):
    """Find BraTS cases for multi-task learning"""
    cases = []
    
    print(f"Scanning {data_dir} for multi-task cases...")
    print(f"Target modality: {target_modality}")
    
    if not os.path.exists(data_dir):
        print(f"ERROR: Directory {data_dir} does not exist!")
        return cases
    
    modality_files = {
        "FLAIR": "t2f.nii.gz",
        "T1CE": "t1c.nii.gz",
        "T1": "t1n.nii.gz", 
        "T2": "t2w.nii.gz"
    }

    # Search for BraTS cases
    for subject_dir in glob.glob(os.path.join(data_dir, 'BraTS*')):
        if not os.path.isdir(subject_dir):
            continue
            
        item = os.path.basename(subject_dir)

        # Build file paths
        files = {}
        for modality, suffix in modality_files.items():
            files[modality] = os.path.join(subject_dir, f"{item}-{suffix}")

        # Add segmentation file
        seg_file = os.path.join(subject_dir, f"{item}-seg.nii.gz")

        # Check if all files exist
        all_files_exist = all(os.path.exists(f) for f in list(files.values()) + [seg_file])

        if all_files_exist:
            # Get input modalities (exclude target)
            input_modalities = [mod for mod in modality_files.keys() if mod != target_modality]
            input_images = [files[mod] for mod in input_modalities]

            case_data = {
                "input_image": input_images,
                "target_synthesis": files[target_modality],
                "target_segmentation": seg_file,
                "case_id": item,
                "target_modality": target_modality
            }
            cases.append(case_data)

            if len(cases) % 50 == 0:
                print(f"Found {len(cases)} valid cases so far...")
    
    print(f"Total multi-task cases found: {len(cases)}")
    return cases


def multitask_train_epoch(model, loader, optimizer, epoch, loss_func, max_epochs, target_modality, logger):
    """Training epoch for multi-task learning"""
    model.train()
    run_total_loss = AverageMeter()
    run_synth_loss = AverageMeter()
    run_seg_loss = AverageMeter()
    
    for idx, batch_data in enumerate(loader):
        try:
            input_data = batch_data["input_image"].cuda()
            target_synthesis = batch_data["target_synthesis"].cuda()
            target_segmentation = batch_data["target_segmentation"].cuda()
            
            optimizer.zero_grad()
            predicted = model(input_data)
            total_loss, loss_components = loss_func(predicted, target_synthesis, target_segmentation)
            total_loss.backward()
            optimizer.step()
            
            # Update metrics
            run_total_loss.update(loss_components["total"], n=input_data.shape[0])
            run_synth_loss.update(loss_components["synthesis"], n=input_data.shape[0])
            run_seg_loss.update(loss_components["segmentation"], n=input_data.shape[0])
            
            # Log metrics
            logger.log_training_metrics(
                epoch, idx, len(loader), loss_components, 
                optimizer.param_groups[0]['lr']
            )
            
            # Log samples
            if epoch < 5:
                sample_freq = 15
            elif epoch < 20:
                sample_freq = 30
            else:
                sample_freq = 50
                
            if (idx + 1) % sample_freq == 0:
                logger.log_training_samples(
                    model, input_data, target_synthesis, target_segmentation, 
                    batch_data, epoch, idx
                )
            
            # Progress print
            if (idx + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{max_epochs} [{idx+1}/{len(loader)}] "
                      f"Total: {run_total_loss.avg:.4f} Synth: {run_synth_loss.avg:.4f} Seg: {run_seg_loss.avg:.4f}")
                
        except Exception as e:
            print(f"Error in training step {idx}: {e}")
            continue
    
    return {
        "total": run_total_loss.avg,
        "synthesis": run_synth_loss.avg,
        "segmentation": run_seg_loss.avg
    }


def multitask_val_epoch(model, loader, epoch, max_epochs, target_modality, logger):
    """Validation epoch for multi-task learning"""
    model.eval()
    
    # Initialize metrics
    run_synth_l1 = AverageMeter()
    run_synth_psnr = AverageMeter()
    run_synth_ssim = AverageMeter()
    
    # Dice metric for segmentation
    dice_metric = DiceMetric(
        include_background=False,  # Exclude background for dice calculation
        reduction=MetricReduction.MEAN,
        get_not_nans=True
    )
    
    # Collect samples for logging
    sample_inputs, sample_targets_synth, sample_targets_seg = [], [], []
    sample_preds_synth, sample_preds_seg, sample_names = [], [], []

    roi = (128, 128, 128)
    post_softmax = Activations(softmax=True)
    post_pred_seg = AsDiscrete(argmax=True)

    # Metrics calculators
    psnr_calculator = PSNRMetric(max_val=1.0)
    ssim_calculator = SSIMMetric(data_range=1.0)
    
    all_dice_scores = []

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            try:
                input_data = batch_data["input_image"].cuda()
                target_synthesis = batch_data["target_synthesis"].cuda()
                target_segmentation = batch_data["target_segmentation"].cuda()

                # Use sliding window inference
                predicted = sliding_window_inference(
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
                pred_synthesis = predicted[:, 0:1, ...]
                pred_segmentation_raw = predicted[:, 1:, ...]

                # Synthesis metrics
                synth_l1 = F.l1_loss(pred_synthesis, target_synthesis)
                run_synth_l1.update(synth_l1.item(), n=input_data.shape[0])

                # PSNR and SSIM
                psnr_val = psnr_calculator(y_pred=pred_synthesis, y=target_synthesis).mean().item()
                ssim_val = ssim_calculator(y_pred=pred_synthesis, y=target_synthesis).mean().item()
                run_synth_psnr.update(psnr_val, n=input_data.shape[0])
                run_synth_ssim.update(ssim_val, n=input_data.shape[0])

                # Segmentation metrics
                pred_seg_softmax = post_softmax(pred_segmentation_raw)
                pred_seg_discrete = post_pred_seg(pred_seg_softmax)

                # Calculate dice score
                dice_metric.reset()
                dice_metric(y_pred=pred_seg_discrete, y=target_segmentation)
                dice_scores = dice_metric.aggregate()
                
                if isinstance(dice_scores, torch.Tensor):
                    dice_avg = dice_scores.mean().item()
                else:
                    dice_avg = float(dice_scores)
                
                all_dice_scores.append(dice_avg)

                # Collect samples for logging
                if len(sample_inputs) < 5:
                    sample_inputs.append(input_data[0].cpu().numpy())
                    sample_targets_synth.append(target_synthesis[0].cpu().numpy())
                    sample_targets_seg.append(target_segmentation[0].cpu().numpy())
                    sample_preds_synth.append(pred_synthesis[0].cpu().numpy())
                    sample_preds_seg.append(pred_seg_discrete[0].cpu().numpy())
                    sample_names.append(batch_data.get("case_id", [f"val_case_{idx}"])[0])

                if (idx + 1) % 10 == 0:
                    print(f"Val [{idx+1}/{len(loader)}] "
                          f"Synth L1: {run_synth_l1.avg:.6f} "
                          f"PSNR: {run_synth_psnr.avg:.2f} "
                          f"SSIM: {run_synth_ssim.avg:.4f} "
                          f"Dice: {dice_avg:.4f}")

            except Exception as e:
                print(f"Error in validation step {idx}: {e}")
                continue

    # Calculate overall dice score
    overall_dice = np.mean(all_dice_scores) if all_dice_scores else 0.0

    # Log validation samples
    logger.log_validation_samples(
        sample_inputs, sample_targets_synth, sample_targets_seg,
        sample_preds_synth, sample_preds_seg, sample_names, epoch
    )

    return {
        "synthesis_l1": run_synth_l1.avg,
        "synthesis_psnr": run_synth_psnr.avg,
        "synthesis_ssim": run_synth_ssim.avg,
        "seg_dice": overall_dice
    }


def get_multitask_transforms(roi, num_segmentation_classes=4):
    """Get transforms for multi-task learning"""
    train_transform = transforms.Compose([
        transforms.LoadImaged(keys=["input_image", "target_synthesis", "target_segmentation"]),
        transforms.EnsureChannelFirstd(keys=["input_image", "target_synthesis", "target_segmentation"]),
        transforms.NormalizeIntensityd(keys=["input_image", "target_synthesis"], nonzero=True, channel_wise=True),
        transforms.CropForegroundd(
            keys=["input_image", "target_synthesis", "target_segmentation"],
            source_key="input_image",
            k_divisible=[roi[0], roi[1], roi[2]],
            allow_smaller=True,
        ),
        transforms.RandSpatialCropd(
            keys=["input_image", "target_synthesis", "target_segmentation"],
            roi_size=[roi[0], roi[1], roi[2]],
            random_size=False,
        ),
        transforms.RandFlipd(keys=["input_image", "target_synthesis", "target_segmentation"], prob=0.5, spatial_axis=0),
        transforms.RandFlipd(keys=["input_image", "target_synthesis", "target_segmentation"], prob=0.5, spatial_axis=1),
        transforms.RandFlipd(keys=["input_image", "target_synthesis", "target_segmentation"], prob=0.5, spatial_axis=2),
        transforms.RandRotate90d(keys=["input_image", "target_synthesis", "target_segmentation"], prob=0.3, spatial_axes=(0, 1)),
        transforms.RandScaleIntensityd(keys=["input_image", "target_synthesis"], factors=0.1, prob=0.5),
        transforms.RandShiftIntensityd(keys=["input_image", "target_synthesis"], offsets=0.1, prob=0.5),
    ])
    
    val_transform = transforms.Compose([
        transforms.LoadImaged(keys=["input_image", "target_synthesis", "target_segmentation"]),
        transforms.EnsureChannelFirstd(keys=["input_image", "target_synthesis", "target_segmentation"]),
        transforms.NormalizeIntensityd(keys=["input_image", "target_synthesis"], nonzero=True, channel_wise=True),
        transforms.CropForegroundd(
            keys=["input_image", "target_synthesis", "target_segmentation"],
            source_key="input_image",
            k_divisible=[32, 32, 32],
            allow_smaller=True,
        ),
        transforms.DivisiblePadd(
            keys=["input_image", "target_synthesis", "target_segmentation"],
            k=32,
            mode="constant",
            constant_values=0,
        ),
    ])
    
    return train_transform, val_transform


def train_single_multitask_model(target_modality, save_dir, max_epochs=50, batch_size=2, num_segmentation_classes=4):
    """Train a single multi-task model for one missing modality"""
    print(f"\n=== TRAINING MULTI-TASK MODEL FOR {target_modality} ===")
    roi = (96, 96, 96)
    
    wandb.init(
        project="BraTS2025-MultiTask",
        name=f"multitask_{target_modality.lower()}_synth_seg",
        config={
            "target_modality": target_modality,
            "max_epochs": max_epochs,
            "save_dir": save_dir,
            "batch_size": batch_size,
            "roi": roi,
            "task": "synthesis_and_segmentation",
            "num_segmentation_classes": num_segmentation_classes,
        }
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Find data
    base_dir = "/app/UNETR-BraTS-Synthesis"
    data_dir = os.path.join(base_dir, "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData")
    all_cases = find_multitask_cases(data_dir, target_modality=target_modality)
    
    if not all_cases:
        print("No cases found! Exiting training.")
        wandb.finish()
        return 0.0
    
    # Split data
    np.random.seed(42)
    np.random.shuffle(all_cases)
    split_idx = int(0.8 * len(all_cases))
    train_cases = all_cases[:split_idx]
    val_cases = all_cases[split_idx:]
    
    print(f"Training cases: {len(train_cases)}")
    print(f"Validation cases: {len(val_cases)}")
    
    # Create datasets
    train_transform, val_transform = get_multitask_transforms(roi, num_segmentation_classes)
    train_ds = Dataset(data=train_cases, transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_ds = Dataset(data=val_cases, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    
    # Initialize model
    model = MultiTaskSwinUNETR(num_segmentation_classes=num_segmentation_classes).to(device)
    loss_func = MultiTaskLoss(synthesis_weight=1.0, segmentation_weight=1.0, num_segmentation_classes=num_segmentation_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
    
    logger = MultiTaskLogger(target_modality)
    
    best_combined_score = -float('inf')
    
    for epoch in range(max_epochs):
        print(f"\n=== EPOCH {epoch+1}/{max_epochs} ===")
        epoch_start = time.time()
        
        # Training
        train_losses = multitask_train_epoch(
            model, train_loader, optimizer, epoch,
            loss_func, max_epochs, target_modality, logger
        )
        
        # Validation
        val_metrics = multitask_val_epoch(
            model, val_loader, epoch, max_epochs, target_modality, logger
        )
        
        epoch_time = time.time() - epoch_start
        
        # Log summary
        logger.log_epoch_summary(epoch, train_losses, val_metrics, epoch_time)
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train - Total: {train_losses['total']:.4f}, Synth: {train_losses['synthesis']:.4f}, Seg: {train_losses['segmentation']:.4f}")
        print(f"  Val - L1: {val_metrics['synthesis_l1']:.6f}, PSNR: {val_metrics['synthesis_psnr']:.2f}, SSIM: {val_metrics['synthesis_ssim']:.4f}, Dice: {val_metrics['seg_dice']:.4f}")
        
        # Combined score for model selection
        combined_score = (-val_metrics['synthesis_l1'] + 
                         val_metrics['synthesis_psnr']/50.0 + 
                         val_metrics['synthesis_ssim'] + 
                         val_metrics['seg_dice'])
        
        if combined_score > best_combined_score:
            best_combined_score = combined_score
            filename = f"multitask_{target_modality.lower()}_best.pt"
            save_path = os.path.join(save_dir, filename)
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_combined_score': best_combined_score,
                'val_metrics': val_metrics,
                'target_modality': target_modality,
            }, save_path)
            
            print(f"✓ New best model saved: {save_path}")
        
        scheduler.step()
    
    wandb.finish()
    return best_combined_score


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Multi-task BraTS: Synthesis + Segmentation')
    parser.add_argument('--save_dir', type=str, default='/data/multitask_models',
                        help='Directory to save the trained models')
    parser.add_argument('--max_epochs', type=int, default=50,
                        help='Maximum number of training epochs')
    parser.add_argument('--target_modality', type=str, default='all',
                        choices=['FLAIR', 'T1CE', 'T1', 'T2', 'all'],
                        help='Which modality to train (or all for all 4 models)')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for training')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    modalities = ['FLAIR', 'T1CE', 'T1', 'T2']
    if args.target_modality != 'all':
        modalities = [args.target_modality]
    
    print(f"🚀 MULTI-TASK TRAINING: Synthesis + Segmentation")
    print(f"📊 Training {len(modalities)} model(s)")
    print(f"💾 Models will be saved to: {args.save_dir}")
    
    results = {}
    for modality in modalities:
        print(f"\n{'='*60}")
        print(f"🎯 STARTING {modality} MULTI-TASK TRAINING")
        print(f"{'='*60}")
        
        try:
            score = train_single_multitask_model(
                target_modality=modality,
                save_dir=args.save_dir,
                max_epochs=args.max_epochs,
                batch_size=args.batch_size,
                num_segmentation_classes=4
            )
            results[modality] = score
            print(f"✅ {modality} completed with score: {score:.6f}")
        except Exception as e:
            print(f"❌ Error training {modality}: {e}")
            import traceback
            traceback.print_exc()
            results[modality] = 0.0
    
    print(f"\n{'='*60}")
    print(f"🏁 ALL MULTI-TASK TRAINING COMPLETE!")
    print(f"{'='*60}")
    
    for modality, score in results.items():
        print(f"🎯 {modality}: {score:.6f}")
    
    if results:
        avg_score = np.mean(list(results.values()))
        print(f"\n🏆 Average score: {avg_score:.6f}")
    
    print(f"📁 All models saved in: {args.save_dir}")


if __name__ == "__main__":
    main()