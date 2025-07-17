#!/usr/bin/env python3
"""
BraTS Multi-task UNETR: Synthesis + Segmentation
Train 4 models that simultaneously synthesize missing modality and perform segmentation
Training from scratch without pretrained weights
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
    """SwinUNETR for multi-task learning: synthesis + segmentation"""
    
    def __init__(self):
        super().__init__()
        # Always 3 input channels (3 available modalities)
        # 4 output channels: 1 for synthesis + 3 for segmentation
        self.backbone = SwinUNETR(
            in_channels=3,
            out_channels=4,  # 1 synthesis + 3 segmentation
            feature_size=48,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=True,
        )
        
        print(f"✓ Multi-task model initialized: 3 input → 4 output channels (1 synthesis + 3 segmentation)")
        print(f"✓ Training from scratch without pretrained weights")

    def forward(self, x):
        return self.backbone(x)


class MultiTaskLoss(nn.Module):
    """Combined loss for synthesis and segmentation"""
    
    def __init__(self, synthesis_weight=1.0, segmentation_weight=1.0):
        super().__init__()
        self.synthesis_weight = synthesis_weight
        self.segmentation_weight = segmentation_weight
        
        # Synthesis loss (L1 + MSE)
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
        # Segmentation loss (DiceFocal)
        self.seg_loss = DiceFocalLoss(
            to_onehot_y=False,
            sigmoid=True,
            weight=[1.0, 2.0, 3.0],  # Weight TC, WT, ET
            gamma=2.0,
            lambda_dice=1.0,
            lambda_focal=1.0,
        )
        
    def forward(self, pred, target_synthesis, target_segmentation):
        # Split predictions
        pred_synthesis = pred[:, 0:1, ...]  # First channel
        pred_segmentation = pred[:, 1:4, ...]  # Last 3 channels
        
        # Synthesis loss
        synthesis_loss = self.l1_loss(pred_synthesis, target_synthesis) + \
                        0.5 * self.mse_loss(pred_synthesis, target_synthesis)
        
        # Segmentation loss
        segmentation_loss = self.seg_loss(pred_segmentation, target_segmentation)
        
        # Combined loss
        total_loss = (self.synthesis_weight * synthesis_loss + 
                     self.segmentation_weight * segmentation_loss)
        
        return total_loss, {
            "total": total_loss.item(),
            "synthesis": synthesis_loss.item(),
            "segmentation": segmentation_loss.item()
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
                pred_seg = predicted[:, 1:4, ...]
                
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
        """Log detailed multi-task sample"""
        try:
            slice_idx = input_vol.shape[-1] // 2
            
            # Extract slices
            input1_slice = input_vol[0, :, :, slice_idx]
            input2_slice = input_vol[1, :, :, slice_idx]
            input3_slice = input_vol[2, :, :, slice_idx]
            target_synth_slice = target_synth[0, :, :, slice_idx]
            pred_synth_slice = pred_synth[0, :, :, slice_idx]
            
            # Convert segmentation to visualization format
            target_seg_viz = self._convert_seg_for_viz(target_seg, slice_idx)
            pred_seg_viz = self._convert_seg_for_viz(pred_seg, slice_idx)
            
            # Create layout: [Input1 | Input2 | Input3 | Target_Synth | Pred_Synth | Target_Seg | Pred_Seg]
            all_images = [
                input1_slice, input2_slice, input3_slice,
                target_synth_slice, pred_synth_slice,
                target_seg_viz, pred_seg_viz
            ]
            
            # Normalize images
            normalized_images = []
            for img in all_images:
                img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
                normalized_images.append(img_norm)
            
            # Concatenate
            comparison = np.concatenate(normalized_images, axis=1)
            comparison_rgb = np.stack([comparison] * 3, axis=-1)
            
            # Create caption
            labels = " | ".join([
                f"{self.input_modalities[0]}", f"{self.input_modalities[1]}", f"{self.input_modalities[2]}",
                f"{self.target_modality}_GT", f"{self.target_modality}_PRED",
                "SEG_GT", "SEG_PRED"
            ])
            
            caption = f"{stage_info} | {case_name} | {labels}"
            
            wandb.log({
                f"samples/multitask_{self.target_modality.lower()}": wandb.Image(comparison_rgb, caption=caption),
                f"samples/count": self.samples_logged
            }, step=self.step)
            
            self.samples_logged += 1
            self.step += 1
            
        except Exception as e:
            print(f"Error creating multi-task sample: {e}")
    
    def _convert_seg_for_viz(self, seg_tensor, slice_idx):
        """Convert segmentation tensor to visualization format"""
        # seg_tensor shape: [3, H, W, D] (TC, WT, ET)
        seg_viz = np.zeros((seg_tensor.shape[1], seg_tensor.shape[2]))
        seg_slice = seg_tensor[:, :, :, slice_idx]
        
        # Convert to BraTS format for visualization
        seg_viz[seg_slice[1] == 1] = 2  # WT
        seg_viz[seg_slice[0] == 1] = 1  # TC
        seg_viz[seg_slice[2] == 1] = 4  # ET
        
        return seg_viz
    
    def log_epoch_summary(self, epoch, train_losses, val_metrics, epoch_time):
        """Log epoch summary"""
        wandb.log({
            "epoch": epoch + 1,
            "summary/train_total_loss": train_losses["total"],
            "summary/train_synthesis_loss": train_losses["synthesis"],
            "summary/train_segmentation_loss": train_losses["segmentation"],
            "summary/val_synthesis_l1": val_metrics["synthesis_l1"],
            "summary/val_synthesis_psnr": val_metrics.get("synthesis_psnr", 0.0),
            "summary/val_seg_dice_avg": val_metrics["seg_dice_avg"],
            "summary/val_seg_dice_tc": val_metrics["seg_dice_tc"],
            "summary/val_seg_dice_wt": val_metrics["seg_dice_wt"],
            "summary/val_seg_dice_et": val_metrics["seg_dice_et"],
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

    for item in os.listdir(data_dir):
        if 'BraTS' in item and os.path.isdir(os.path.join(data_dir, item)):
            case_path = os.path.join(data_dir, item)
            
            # Build file paths
            files = {}
            for modality, suffix in modality_files.items():
                files[modality] = os.path.join(case_path, f"{item}-{suffix}")
            
            # Add segmentation file
            seg_file = os.path.join(case_path, f"{item}-seg.nii.gz")
            
            # Check if all files exist
            all_files = list(files.values()) + [seg_file]
            if all(os.path.exists(f) for f in all_files):
                # Get input modalities (exclude target)
                input_modalities = [mod for mod in modality_files.keys() if mod != target_modality]
                input_images = [files[mod] for mod in input_modalities]
                
                case_data = {
                    "input_image": input_images,  # 3 input modalities
                    "target_synthesis": files[target_modality],  # Missing modality
                    "target_segmentation": seg_file,  # Segmentation
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
    
    return {
        "total": run_total_loss.avg,
        "synthesis": run_synth_loss.avg,
        "segmentation": run_seg_loss.avg
    }


def multitask_val_epoch(model, loader, epoch, max_epochs, target_modality, logger):
    """Validation epoch for multi-task learning"""
    model.eval()
    
    # Synthesis metrics
    run_synth_l1 = AverageMeter()
    run_synth_psnr = AverageMeter()
    
    # Segmentation metrics
    dice_metric = DiceMetric(
        include_background=True,
        reduction=MetricReduction.MEAN_BATCH,
        get_not_nans=True,
        ignore_empty=False
    )
    
    # Collect samples
    sample_inputs, sample_targets_synth, sample_targets_seg = [], [], []
    sample_preds_synth, sample_preds_seg, sample_names = [], [], []
    
    roi = (128, 128, 128)
    post_sigmoid = Activations(sigmoid=True)
    post_pred = AsDiscrete(argmax=False, threshold=0.5)
    
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            try:
                input_data = batch_data["input_image"].cuda()
                target_synthesis = batch_data["target_synthesis"].cuda()
                target_segmentation = batch_data["target_segmentation"].cuda()
                
                # Use sliding window for variable sizes
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
                pred_segmentation = predicted[:, 1:4, ...]
                
                # Synthesis metrics
                synth_l1 = F.l1_loss(pred_synthesis, target_synthesis)
                run_synth_l1.update(synth_l1.item(), n=input_data.shape[0])
                
                # PSNR for synthesis
                try:
                    psnr_metric = PSNRMetric(max_val=1.0)
                    psnr_val = psnr_metric(pred_synthesis, target_synthesis).item()
                    run_synth_psnr.update(psnr_val, n=input_data.shape[0])
                except:
                    run_synth_psnr.update(0.0, n=input_data.shape[0])
                
                # Segmentation metrics
                try:
                    pred_seg_sigmoid = post_sigmoid(pred_segmentation)
                    pred_seg_discrete = post_pred(pred_seg_sigmoid)
                    
                    dice_metric.reset()
                    dice_metric(y_pred=[pred_seg_discrete[0]], y=[target_segmentation[0]])
                    dice_scores, _ = dice_metric.aggregate()
                    
                    # Store dice scores for later averaging
                    if not hasattr(run_synth_l1, 'dice_scores'):
                        run_synth_l1.dice_scores = []
                    run_synth_l1.dice_scores.append(dice_scores.cpu().numpy())
                    
                except Exception as seg_error:
                    print(f"Segmentation metric error: {seg_error}")
                
                # Collect samples
                if len(sample_inputs) < 8:
                    sample_inputs.append(input_data[0].cpu().numpy())
                    sample_targets_synth.append(target_synthesis[0].cpu().numpy())
                    sample_targets_seg.append(target_segmentation[0].cpu().numpy())
                    sample_preds_synth.append(pred_synthesis[0].cpu().numpy())
                    sample_preds_seg.append(pred_segmentation[0].cpu().numpy())
                    sample_names.append(batch_data.get("case_id", [f"val_case_{idx}"])[0])
                
                if (idx + 1) % 10 == 0:
                    print(f"Val [{idx+1}/{len(loader)}] Synth L1: {run_synth_l1.avg:.6f}")
                    
            except Exception as e:
                print(f"Error in validation step {idx}: {e}")
                continue
    
    # Calculate average segmentation dice
    if hasattr(run_synth_l1, 'dice_scores') and run_synth_l1.dice_scores:
        all_dice = np.array(run_synth_l1.dice_scores)
        dice_tc = np.mean(all_dice[:, 0]) if all_dice.shape[1] > 0 else 0.0
        dice_wt = np.mean(all_dice[:, 1]) if all_dice.shape[1] > 1 else 0.0
        dice_et = np.mean(all_dice[:, 2]) if all_dice.shape[1] > 2 else 0.0
        dice_avg = np.mean([dice_tc, dice_wt, dice_et])
    else:
        dice_tc = dice_wt = dice_et = dice_avg = 0.0
    
    # Log validation samples
    logger.log_validation_samples(
        sample_inputs, sample_targets_synth, sample_targets_seg,
        sample_preds_synth, sample_preds_seg, sample_names, epoch
    )
    
    return {
        "synthesis_l1": run_synth_l1.avg,
        "synthesis_psnr": run_synth_psnr.avg,
        "seg_dice_tc": dice_tc,
        "seg_dice_wt": dice_wt,
        "seg_dice_et": dice_et,
        "seg_dice_avg": dice_avg
    }


def get_multitask_transforms(roi):
    """Get transforms for multi-task learning"""
    train_transform = transforms.Compose([
        transforms.LoadImaged(keys=["input_image", "target_synthesis", "target_segmentation"]),
        transforms.EnsureChannelFirstd(keys=["target_synthesis"]),
        transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="target_segmentation"),
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
        # Augmentation
        transforms.RandFlipd(keys=["input_image", "target_synthesis", "target_segmentation"], prob=0.5, spatial_axis=0),
        transforms.RandFlipd(keys=["input_image", "target_synthesis", "target_segmentation"], prob=0.5, spatial_axis=1),
        transforms.RandFlipd(keys=["input_image", "target_synthesis", "target_segmentation"], prob=0.5, spatial_axis=2),
        transforms.RandRotate90d(keys=["input_image", "target_synthesis", "target_segmentation"], prob=0.3, spatial_axes=(0, 1)),
        transforms.RandScaleIntensityd(keys=["input_image", "target_synthesis"], factors=0.1, prob=0.5),
        transforms.RandShiftIntensityd(keys=["input_image", "target_synthesis"], offsets=0.1, prob=0.5),
    ])
    
    val_transform = transforms.Compose([
        transforms.LoadImaged(keys=["input_image", "target_synthesis", "target_segmentation"]),
        transforms.EnsureChannelFirstd(keys=["target_synthesis"]),
        transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="target_segmentation"),
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


def train_single_multitask_model(target_modality, save_path, max_epochs=50):
    """Train a single multi-task model for one missing modality"""
    
    print(f"\n=== TRAINING MULTI-TASK MODEL FOR {target_modality} ===")
    
    # Initialize W&B
    roi = (128, 128, 128)
    wandb.init(
        project="BraTS2025-MultiTask",
        name=f"multitask_{target_modality.lower()}_synth_seg_from_scratch",
        config={
            "target_modality": target_modality,
            "max_epochs": max_epochs,
            "save_path": save_path,
            "batch_size": 2,
            "roi": roi,
            "task": "synthesis_and_segmentation",
            "input_channels": 3,
            "output_channels": 4,
            "synthesis_weight": 1.0,
            "segmentation_weight": 1.0,
            "training_from_scratch": True,
        }
    )
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Find data
    base_dir = "/app/UNETR-BraTS-Synthesis"
    data_dir = os.path.join(base_dir, "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData")
    
    all_cases = find_multitask_cases(data_dir, target_modality=target_modality)
    print(f"Total cases found: {len(all_cases)}")
    
    # Split data
    np.random.seed(42)
    np.random.shuffle(all_cases)
    split_idx = int(0.8 * len(all_cases))
    train_cases = all_cases[:split_idx]
    val_cases = all_cases[split_idx:]
    
    print(f"Training cases: {len(train_cases)}")
    print(f"Validation cases: {len(val_cases)}")
    
    # Get transforms
    train_transform, val_transform = get_multitask_transforms(roi)
    
    # Data loaders
    batch_size = 2
    train_ds = Dataset(data=train_cases, transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)
    
    val_ds = Dataset(data=val_cases, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    
    # Create model (no pretrained weights)
    model = MultiTaskSwinUNETR().cuda()
    
    # Loss function
    loss_func = MultiTaskLoss(synthesis_weight=1.0, segmentation_weight=1.0)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
    
    # Logger
    logger = MultiTaskLogger(target_modality)
    
    print(f"Training multi-task model: {target_modality}")
    print(f"Input: 3 modalities, Output: 1 synthesis + 3 segmentation")
    print(f"Training from scratch without pretrained weights")
    
    best_combined_score = 0.0
    
    for epoch in range(max_epochs):
        print(f"\n=== EPOCH {epoch+1}/{max_epochs} ===")
        epoch_start = time.time()
        
        # Training
        train_losses = multitask_train_epoch(
            model, train_loader, optimizer, epoch,
            loss_func, max_epochs, target_modality, logger
        )
        
        print(f"Training complete: Total: {train_losses['total']:.4f}")
        
        # Validation
        val_metrics = multitask_val_epoch(
            model, val_loader, epoch, max_epochs, target_modality, logger
        )
        
        epoch_time = time.time() - epoch_start
        
        # Log epoch summary
        logger.log_epoch_summary(epoch, train_losses, val_metrics, epoch_time)
        
        print(f"Validation: Synth L1: {val_metrics['synthesis_l1']:.6f}, "
              f"Seg Dice: {val_metrics['seg_dice_avg']:.6f}")
        
        # Combined score (you can adjust weights)
        combined_score = (1.0 - val_metrics['synthesis_l1']) + val_metrics['seg_dice_avg']
        
        # Save best model
        if combined_score > best_combined_score:
            print(f"NEW BEST COMBINED SCORE! ({best_combined_score:.6f} --> {combined_score:.6f})")
            best_combined_score = combined_score
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_combined_score': best_combined_score,
                'val_metrics': val_metrics,
                'target_modality': target_modality,
                'task': 'multitask_synthesis_segmentation',
                'input_channels': 3,
                'output_channels': 4,
                'training_from_scratch': True,
            }, save_path)
            print(f"✓ Best model saved to: {save_path}")
        
        scheduler.step()
    
    print(f"\n🎉 {target_modality} TRAINING COMPLETE!")
    print(f"🏆 Best combined score: {best_combined_score:.6f}")
    print(f"✓ Model saved to: {save_path}")
    
    wandb.finish()
    
    return best_combined_score


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Multi-task BraTS: Synthesis + Segmentation (Training from scratch)')
    parser.add_argument('--save_dir', type=str, default='/data/multitask_models',
                        help='Directory to save the 4 trained models')
    parser.add_argument('--max_epochs', type=int, default=50,
                        help='Maximum number of training epochs per model')
    parser.add_argument('--target_modality', type=str, default='all',
                        choices=['FLAIR', 'T1CE', 'T1', 'T2', 'all'],
                        help='Which modality to train (or all for all 4 models)')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Define modalities
    modalities = ['FLAIR', 'T1CE', 'T1', 'T2']
    
    if args.target_modality != 'all':
        modalities = [args.target_modality]
    
    print(f"🚀 MULTI-TASK TRAINING: Synthesis + Segmentation")
    print(f"📊 Training {len(modalities)} model(s) from scratch")
    print(f"💾 Models will be saved to: {args.save_dir}")
    print(f"🔧 No pretrained weights - training from scratch")
    
    results = {}
    
    for modality in modalities:
        save_path = os.path.join(args.save_dir, f"multitask_{modality.lower()}_from_scratch_best.pt")
        
        print(f"\n{'='*60}")
        print(f"🎯 STARTING {modality} MULTI-TASK TRAINING")
        print(f"{'='*60}")
        
        try:
            score = train_single_multitask_model(
                target_modality=modality,
                save_path=save_path,
                max_epochs=args.max_epochs
            )
            results[modality] = score
            print(f"✅ {modality} completed with score: {score:.6f}")
            
        except Exception as e:
            print(f"❌ Error training {modality}: {e}")
            results[modality] = 0.0
    
    # Summary
    print(f"\n{'='*60}")
    print(f"🏁 ALL MULTI-TASK TRAINING COMPLETE!")
    print(f"{'='*60}")
    
    for modality, score in results.items():
        print(f"🎯 {modality}: {score:.6f}")
    
    avg_score = np.mean(list(results.values()))
    print(f"\n🏆 Average score across all models: {avg_score:.6f}")
    print(f"📁 All models saved in: {args.save_dir}")
    print(f"\nNow you can use these models for inference that simultaneously:")
    print(f"  • Synthesizes missing modalities")
    print(f"  • Performs segmentation")
    print(f"  • Each model takes 3 modalities → outputs 1 synthesis + 3 segmentation channels")
    print(f"  • All models trained from scratch without pretrained weights")


if __name__ == "__main__":
    main()