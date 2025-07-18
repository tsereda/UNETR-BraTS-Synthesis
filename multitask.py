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
        # 3 input channels (3 available modalities)
        # 1 for synthesis + N for segmentation
        self.num_segmentation_classes = num_segmentation_classes
        self.backbone = SwinUNETR(
            in_channels=3,
            out_channels=1 + num_segmentation_classes,  # 1 synthesis + N segmentation
            feature_size=48,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=True,
        )
        print(f"✓ Multi-task model initialized: 3 input → 1 synthesis + {num_segmentation_classes} segmentation channels")

    def forward(self, x):
        return self.backbone(x)


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
        # Segmentation loss (DiceFocal for multi-class)
        self.seg_loss = DiceFocalLoss(
            to_onehot_y=True, # Output is multi-class, need one-hot conversion
            softmax=True,     # Apply softmax to prediction before loss calculation
            gamma=2.0,
            lambda_dice=1.0,
            lambda_focal=1.0,
        )
    def forward(self, pred, target_synthesis, target_segmentation):
        # Split predictions
        pred_synthesis = pred[:, 0:1, ...]  # First channel
        pred_segmentation = pred[:, 1:1+self.num_segmentation_classes, ...]  # Next N channels
        # Synthesis loss
        synthesis_loss = self.l1_loss(pred_synthesis, target_synthesis) + \
                         0.5 * self.mse_loss(pred_synthesis, target_synthesis)
        # Segmentation loss
        segmentation_loss = self.seg_loss(pred_segmentation, target_segmentation.long())
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
                
                # Apply sigmoid and threshold for segmentation prediction visualization
                pred_seg = torch.sigmoid(predicted[:, 1:2, ...])
                pred_seg = (pred_seg > 0.5).float() # Binarize for visualization
                
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
            for i in range(min(5, len(inputs))): # Log up to 5 samples
                self._log_multitask_sample(
                    inputs[i], targets_synth[i], targets_seg[i],
                    predictions_synth[i], predictions_seg[i], case_names[i],
                    f"VALIDATION | Epoch {epoch+1}"
                )
        except Exception as e:
            print(f"Error logging validation samples: {e}")
    
    def _log_multitask_sample(self, input_vol, target_synth, target_seg, 
                               pred_synth, pred_seg, case_name, stage_info):
        """Log detailed multi-task sample with color overlays for multi-class segmentation"""
        try:
            # Squeeze channel dimension if it exists and is 1
            if input_vol.shape[0] == 1:
                input_vol = input_vol[0]
            target_synth = np.squeeze(target_synth)
            pred_synth = np.squeeze(pred_synth)
            target_seg = np.squeeze(target_seg)
            pred_seg = np.squeeze(pred_seg)

            slice_idx = input_vol.shape[-1] // 2
            input1_slice = input_vol[0, :, :, slice_idx]
            input2_slice = input_vol[1, :, :, slice_idx]
            input3_slice = input_vol[2, :, :, slice_idx]
            target_synth_slice = target_synth[:, :, slice_idx]
            pred_synth_slice = pred_synth[:, :, slice_idx]

            # For multi-class: get argmax for mask visualization
            target_seg_slice = target_seg[:, :, slice_idx] if target_seg.ndim == 3 else np.argmax(target_seg[:, :, :, slice_idx], axis=0)
            pred_seg_slice = pred_seg[:, :, slice_idx] if pred_seg.ndim == 3 else np.argmax(pred_seg[:, :, :, slice_idx], axis=0)

            # Normalize images (inputs and synth)
            def norm_img(img):
                return (img - img.min()) / (img.max() - img.min() + 1e-8)
            input1_slice = norm_img(input1_slice)
            input2_slice = norm_img(input2_slice)
            input3_slice = norm_img(input3_slice)
            target_synth_slice = norm_img(target_synth_slice)
            pred_synth_slice = norm_img(pred_synth_slice)

            # Color map for 4 classes (BraTS): background, edema, non-enhancing, enhancing
            class_colors = np.array([
                [0, 0, 0],        # 0: background - black
                [0, 255, 0],      # 1: edema - green
                [0, 0, 255],      # 2: non-enhancing/core - blue
                [255, 0, 0],      # 3: enhancing - red
            ], dtype=np.uint8)

            def colorize_mask(mask):
                mask = mask.astype(np.int32)
                rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
                for c in range(class_colors.shape[0]):
                    rgb[mask == c] = class_colors[c]
                return rgb

            # Colorize segmentation masks
            target_seg_rgb = colorize_mask(target_seg_slice)
            pred_seg_rgb = colorize_mask(pred_seg_slice)

            # Stack grayscale images to RGB for layout
            def gray2rgb(img):
                img = (img * 255).astype(np.uint8)
                return np.stack([img]*3, axis=-1)

            layout = np.concatenate([
                gray2rgb(input1_slice),
                gray2rgb(input2_slice),
                gray2rgb(input3_slice),
                gray2rgb(target_synth_slice),
                gray2rgb(pred_synth_slice),
                target_seg_rgb,
                pred_seg_rgb
            ], axis=1)

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
    
    def log_epoch_summary(self, epoch, train_losses, val_metrics, epoch_time):
        """Log epoch summary"""
        wandb.log({
            "epoch": epoch + 1,
            "summary/train_total_loss": train_losses["total"],
            "summary/train_synthesis_loss": train_losses["synthesis"],
            "summary/train_segmentation_loss": train_losses["segmentation"],
            "summary/val_synthesis_l1": val_metrics["synthesis_l1"],
            "summary/val_synthesis_psnr": val_metrics["synthesis_psnr"], # Ensure this is always calculated
            "summary/val_synthesis_ssim": val_metrics["synthesis_ssim"], # Add SSIM here
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

    # Use glob for the flat structure: each subject is a direct subdirectory
    for subject_dir in glob.glob(os.path.join(data_dir, 'BraTS*')):
        item = os.path.basename(subject_dir) # Get the actual case ID from the directory name

        # Build file paths
        files = {}
        for modality, suffix in modality_files.items():
            files[modality] = os.path.join(subject_dir, f"{item}-{suffix}")

        # Add segmentation file
        seg_file = os.path.join(subject_dir, f"{item}-seg.nii.gz")

        # Check if all files exist
        all_files_exist = True
        for f in list(files.values()) + [seg_file]:
            if not os.path.exists(f):
                # print(f"Missing file: {f}") # For debugging
                all_files_exist = False
                break

        if all_files_exist:
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
        # Ensure target_segmentation is float for loss calculation
        target_segmentation = batch_data["target_segmentation"].cuda().float() 
        
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
    run_synth_ssim = AverageMeter()
    
    # Segmentation metrics (multi-class)
    dice_metric = DiceMetric(
        include_background=True,
        reduction=MetricReduction.MEAN_BATCH,
        get_not_nans=True,
        ignore_empty=False
    )

    # Collect samples
    sample_inputs, sample_targets_synth, sample_targets_seg = [], [], []
    sample_preds_synth, sample_preds_seg, sample_names = [], [], []

    roi = (128, 128, 128) # For sliding window inference
    post_softmax = Activations(softmax=True)
    post_pred_seg = AsDiscrete(argmax=True)  # Use argmax for multi-class

    # Instantiate PSNR and SSIM metrics for validation
    psnr_calculator = PSNRMetric(max_val=1.0)
    ssim_calculator = SSIMMetric(data_range=1.0) # Assuming normalized input 0-1

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            try:
                input_data = batch_data["input_image"].cuda()
                target_synthesis = batch_data["target_synthesis"].cuda()
                target_segmentation = batch_data["target_segmentation"].cuda().long()  # Ensure long for multi-class

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
                pred_segmentation_raw = predicted[:, 1:1+4, ...]  # 4 segmentation channels

                # Synthesis metrics
                synth_l1 = F.l1_loss(pred_synthesis, target_synthesis)
                run_synth_l1.update(synth_l1.item(), n=input_data.shape[0])

                # PSNR for synthesis
                psnr_val = psnr_calculator(y_pred=pred_synthesis, y=target_synthesis).mean().item()
                run_synth_psnr.update(psnr_val, n=input_data.shape[0])

                # SSIM for synthesis
                ssim_val = ssim_calculator(y_pred=pred_synthesis, y=target_synthesis).mean().item()
                run_synth_ssim.update(ssim_val, n=input_data.shape[0])

                # Segmentation metrics (multi-class, per-class dice)
                pred_seg_softmax = post_softmax(pred_segmentation_raw)
                pred_seg_discrete = post_pred_seg(pred_seg_softmax)  # Argmax for multi-class

                dice_metric.reset()
                dice_metric(y_pred=[pred_seg_discrete], y=[target_segmentation])
                dice_scores, _ = dice_metric.aggregate()

                # dice_scores: (num_classes,) tensor
                dice_scores_np = dice_scores.cpu().numpy() if hasattr(dice_scores, 'cpu') else np.array(dice_scores)
                dice_avg_batch = np.nanmean(dice_scores_np)

                # Store per-class dice for later analysis if needed
                if not hasattr(run_synth_l1, 'overall_dice_scores'):
                    run_synth_l1.overall_dice_scores = []
                run_synth_l1.overall_dice_scores.append(dice_scores_np)

                # Collect samples for logging
                if len(sample_inputs) < 8:
                    sample_inputs.append(input_data[0].cpu().numpy())
                    sample_targets_synth.append(target_synthesis[0].cpu().numpy())
                    sample_targets_seg.append(target_segmentation[0].cpu().numpy())
                    sample_preds_synth.append(pred_synthesis[0].cpu().numpy())
                    sample_preds_seg.append(pred_seg_discrete[0].cpu().numpy())
                    sample_names.append(batch_data.get("case_id", [f"val_case_{idx}"])[0])

                if (idx + 1) % 10 == 0:
                    print(f"Val [{idx+1}/{len(loader)}] Synth L1: {run_synth_l1.avg:.6f} PSNR: {run_synth_psnr.avg:.2f} SSIM: {run_synth_ssim.avg:.4f} Seg Dice: {dice_avg_batch:.4f}")

            except Exception as e:
                print(f"Error in validation step {idx}: {e}")
                continue

    # Calculate overall average segmentation dice from collected batch averages
    if hasattr(run_synth_l1, 'overall_dice_scores') and run_synth_l1.overall_dice_scores:
        all_dice = np.stack(run_synth_l1.overall_dice_scores, axis=0)  # (num_batches, num_classes)
        overall_dice_avg = np.nanmean(all_dice)
    else:
        overall_dice_avg = 0.0

    # Log validation samples
    logger.log_validation_samples(
        sample_inputs, sample_targets_synth, sample_targets_seg,
        sample_preds_synth, sample_preds_seg, sample_names, epoch
    )

    return {
        "synthesis_l1": run_synth_l1.avg,
        "synthesis_psnr": run_synth_psnr.avg,
        "synthesis_ssim": run_synth_ssim.avg,
        "seg_dice": overall_dice_avg
    }


def get_multitask_transforms(roi, num_segmentation_classes=4):
    """Get transforms for multi-task learning (multi-class segmentation)"""
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
        # Convert target_segmentation to one-hot for multi-class segmentation
        transforms.EnsureTyped(keys=["target_segmentation"], dtype=np.int64),
        transforms.EnsureTyped(keys=["target_segmentation"], dtype=int),
        transforms.Lambdad(keys=["target_segmentation"], func=lambda x: x.astype(np.int64)),
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
        transforms.EnsureTyped(keys=["target_segmentation"], dtype=np.int64),
        transforms.EnsureTyped(keys=["target_segmentation"], dtype=int),
        transforms.Lambdad(keys=["target_segmentation"], func=lambda x: x.astype(np.int64)),
    ])
    return train_transform, val_transform


def train_single_multitask_model(target_modality, save_dir, max_epochs=50, batch_size=2, num_segmentation_classes=4):
    """Train a single multi-task model for one missing modality (multi-class segmentation)"""
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
            "input_channels": 3,
            "output_channels": 1 + num_segmentation_classes,
            "synthesis_weight": 1.0,
            "segmentation_weight": 1.0,
            "num_segmentation_classes": num_segmentation_classes,
        }
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using batch size: {batch_size}")
    base_dir = "/app/UNETR-BraTS-Synthesis"
    data_dir = os.path.join(base_dir, "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData")
    all_cases = find_multitask_cases(data_dir, target_modality=target_modality)
    print(f"Total cases found: {len(all_cases)}")
    if not all_cases:
        print("No cases found! Exiting training for this modality.")
        wandb.finish()
        return 0.0
    np.random.seed(42)
    np.random.shuffle(all_cases)
    split_idx = int(0.8 * len(all_cases))
    train_cases = all_cases[:split_idx]
    val_cases = all_cases[split_idx:]
    print(f"Training cases: {len(train_cases)}")
    print(f"Validation cases: {len(val_cases)}")
    train_transform, val_transform = get_multitask_transforms(roi, num_segmentation_classes=num_segmentation_classes)
    train_ds = Dataset(data=train_cases, transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)
    val_ds = Dataset(data=val_cases, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    model = MultiTaskSwinUNETR(num_segmentation_classes=num_segmentation_classes).to(device)
    loss_func = MultiTaskLoss(synthesis_weight=1.0, segmentation_weight=1.0, num_segmentation_classes=num_segmentation_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
    logger = MultiTaskLogger(target_modality)
    print(f"Training multi-task model: {target_modality}")
    print(f"Input: 3 modalities, Output: 1 synthesis + {num_segmentation_classes} segmentation")
    best_combined_score = -float('inf')
    for epoch in range(max_epochs):
        print(f"\n=== EPOCH {epoch+1}/{max_epochs} ===")
        epoch_start = time.time()
        train_losses = multitask_train_epoch(
            model, train_loader, optimizer, epoch,
            loss_func, max_epochs, target_modality, logger
        )
        print(f"Training complete: Total: {train_losses['total']:.4f}")
        val_metrics = multitask_val_epoch(
            model, val_loader, epoch, max_epochs, target_modality, logger
        )
        epoch_time = time.time() - epoch_start
        logger.log_epoch_summary(epoch, train_losses, val_metrics, epoch_time)
        print(f"Validation: Synth L1: {val_metrics['synthesis_l1']:.6f}, "
              f"Synth PSNR: {val_metrics['synthesis_psnr']:.2f}, "
              f"Synth SSIM: {val_metrics['synthesis_ssim']:.4f}, "
              f"Seg Dice: {val_metrics['seg_dice']:.6f}")
        combined_score = (1.0 - val_metrics['synthesis_l1']) + val_metrics['synthesis_psnr']/100.0 + \
                         val_metrics['synthesis_ssim'] + val_metrics['seg_dice']
        if combined_score > best_combined_score:
            print(f"NEW BEST COMBINED SCORE! ({best_combined_score:.6f} --> {combined_score:.6f})")
            best_combined_score = combined_score
            filename = (f"multitask_{target_modality.lower()}_"
                        f"L1{val_metrics['synthesis_l1']:.4f}_"
                        f"PSNR{val_metrics['synthesis_psnr']:.2f}_"
                        f"SSIM{val_metrics['synthesis_ssim']:.4f}_"
                        f"Dice{val_metrics['seg_dice']:.4f}_best.pt")
            save_path = os.path.join(save_dir, filename)
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
                'output_channels': 1 + num_segmentation_classes,
            }, save_path)
            print(f"✓ Best model saved to: {save_path}")
        scheduler.step()
    print(f"\n🎉 {target_modality} TRAINING COMPLETE!")
    print(f"🏆 Best combined score: {best_combined_score:.6f}")
    print(f"✓ Models for {target_modality} saved in: {save_dir}")
    wandb.finish()
    return best_combined_score


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Multi-task BraTS: Synthesis + Segmentation')
    parser.add_argument('--save_dir', type=str, default='/data/multitask_models',
                        help='Directory to save the 4 trained models')
    parser.add_argument('--max_epochs', type=int, default=50,
                        help='Maximum number of training epochs per model')
    parser.add_argument('--target_modality', type=str, default='all',
                        choices=['FLAIR', 'T1CE', 'T1', 'T2', 'all'],
                        help='Which modality to train (or all for all 4 models)')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for training (default: 2)')
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
    print(f"📦 Batch size: {args.batch_size}")
    print(f"🔢 Max epochs: {args.max_epochs}")
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
            results[modality] = 0.0
    print(f"\n{'='*60}")
    print(f"🏁 ALL MULTI-TASK TRAINING COMPLETE!")
    print(f"{'='*60}")
    for modality, score in results.items():
        print(f"🎯 {modality}: {score:.6f}")
    avg_score = np.mean(list(results.values()))
    print(f"\n🏆 Average score across all models: {avg_score:.6f}")
    print(f"📁 All models saved in: {args.save_dir}")
    print(f"\nNow you can use these models for inference that simultaneously:")
    print(f"  • Synthesizes missing modalities")
    print(f"  • Performs segmentation")
    print(f"  • Each model takes 3 modalities → outputs 1 synthesis + 4 segmentation channels")

if __name__ == "__main__":
    main()