#!/usr/bin/env python3
"""
BraTS Segmentation - Enhanced Version with Proper Validation Dataset and Batch Logging
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import time
import warnings

import torch
import wandb

from monai.losses import DiceFocalLoss
from monai.inferers import sliding_window_inference
from monai import transforms
from monai.transforms import AsDiscrete, Activations
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from monai.networks.nets import SwinUNETR
from monai.data import Dataset, DataLoader
from monai.data import decollate_batch

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
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


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


def log_segmentation_samples(images, labels, predictions, case_names, epoch=None, batch_idx=None):
    """Log multiple segmentation samples to W&B in one figure"""
    try:
        num_samples = len(images)
        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
        
        # Handle single sample case
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            slice_idx = images[i].shape[-1] // 2
            
            axes[i, 0].imshow(images[i][1, :, :, slice_idx], cmap='gray')
            axes[i, 0].set_title(f'{case_names[i][:15]}... - T1CE')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(images[i][0, :, :, slice_idx], cmap='gray')
            axes[i, 1].set_title(f'{case_names[i][:15]}... - FLAIR')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(labels[i][:, :, slice_idx])
            axes[i, 2].set_title('Ground Truth')
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow(predictions[i][:, :, slice_idx])
            axes[i, 3].set_title('Prediction')
            axes[i, 3].axis('off')
        
        plt.tight_layout()
        
        title = "segmentation_samples"
        if epoch is not None:
            title += f"_epoch_{epoch}"
        if batch_idx is not None:
            title += f"_batch_{batch_idx}"
            
        wandb.log({f"segmentation/{title}": wandb.Image(fig)})
        plt.close(fig)
    except Exception as e:
        print(f"Error logging segmentation samples: {e}")


def train_epoch(model, loader, optimizer, epoch, loss_func, max_epochs, model_inferer=None, post_sigmoid=None, post_pred=None, val_cases=None):
    """Training epoch with batch-level logging"""
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    batch_log_freq = 20  # Log every 20 batches
    
    for idx, batch_data in enumerate(loader):
        data, target = batch_data["image"].cuda(), batch_data["label"].cuda()
        
        optimizer.zero_grad()
        logits = model(data)
        loss = loss_func(logits, target)
        loss.backward()
        optimizer.step()
        
        run_loss.update(loss.item(), n=data.shape[0])
        
        print(
            "Epoch {}/{} Batch {}/{} (batch_size=2)".format(epoch + 1, max_epochs, idx + 1, len(loader)),
            "loss: {:.4f}".format(run_loss.avg),
            "time {:.2f}s".format(time.time() - start_time),
        )
        
        # Log to W&B every batch_log_freq batches
        if (idx + 1) % batch_log_freq == 0:
            wandb.log({
                "batch_step": epoch * len(loader) + idx,
                "batch_loss": loss.item(),
                "batch_loss_avg": run_loss.avg,
                "epoch": epoch + 1,
                "batch": idx + 1
            })
            
            # Log sample predictions every 20 batches
            if model_inferer is not None and post_sigmoid is not None and post_pred is not None and val_cases is not None:
                log_batch_samples(model, data, target, batch_data, model_inferer, post_sigmoid, post_pred, epoch, idx)
        
        start_time = time.time()
    
    return run_loss.avg


def log_batch_samples(model, data, target, batch_data, model_inferer, post_sigmoid, post_pred, epoch, batch_idx):
    """Log a quick sample during training"""
    try:
        model.eval()
        with torch.no_grad():
            # Just use the current batch for quick sampling
            logits = model(data[:1])  # Take only first sample from batch
            pred_sigmoid = post_sigmoid(logits)
            pred_discrete = post_pred(pred_sigmoid)
            
            pred = pred_discrete[0].cpu().numpy()
            pred_viz = np.zeros((pred.shape[1], pred.shape[2], pred.shape[3]))
            pred_viz[pred[1] == 1] = 2  # ED
            pred_viz[pred[0] == 1] = 1  # TC  
            pred_viz[pred[2] == 1] = 4  # ET
            
            # Get original label
            label_viz = target[0].cpu().numpy()
            label_orig = np.zeros((label_viz.shape[1], label_viz.shape[2], label_viz.shape[3]))
            label_orig[label_viz[1] == 1] = 2
            label_orig[label_viz[0] == 1] = 1
            label_orig[label_viz[2] == 1] = 4
            
            # Log single sample
            case_name = f"batch_sample_{batch_idx}"
            log_segmentation_samples(
                [data[0].cpu().numpy()], 
                [label_orig], 
                [pred_viz], 
                [case_name], 
                epoch,
                batch_idx
            )
        model.train()
    except Exception as e:
        print(f"Error logging batch sample: {e}")


def safe_dice_computation(pred, target, smooth=1e-6):
    """Safely compute dice score to avoid division by zero"""
    intersection = (pred * target).sum()
    pred_sum = pred.sum()
    target_sum = target.sum()
    
    if pred_sum == 0 and target_sum == 0:
        return 1.0  # Perfect score when both are empty
    elif pred_sum == 0 or target_sum == 0:
        return 0.0  # No overlap when one is empty
    else:
        return (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)


def val_epoch(model, loader, epoch, acc_func, model_inferer, post_sigmoid, post_pred, max_epochs):
    """Validation epoch with improved error handling"""
    model.eval()
    start_time = time.time()
    run_acc = AverageMeter()

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data, target = batch_data["image"].cuda(), batch_data["label"].cuda()
            
            try:
                logits = model_inferer(data)
                
                val_labels_list = decollate_batch(target)
                val_outputs_list = decollate_batch(logits)
                val_output_convert = [post_pred(post_sigmoid(val_pred_tensor)) for val_pred_tensor in val_outputs_list]
                
                acc_func.reset()
                acc_func(y_pred=val_output_convert, y=val_labels_list)
                acc, not_nans = acc_func.aggregate()
                
                # Handle NaN values safely
                acc_safe = torch.nan_to_num(acc, nan=0.0)
                not_nans_safe = torch.clamp(not_nans, min=1)  # Avoid division by zero
                
                run_acc.update(acc_safe.cpu().numpy(), n=not_nans_safe.cpu().numpy())
                
                dice_tc = run_acc.avg[0] if len(run_acc.avg) > 0 else 0.0
                dice_wt = run_acc.avg[1] if len(run_acc.avg) > 1 else 0.0
                dice_et = run_acc.avg[2] if len(run_acc.avg) > 2 else 0.0
                
                print(
                    "Val Epoch {}/{} Batch {}/{} (batch_size=1)".format(epoch + 1, max_epochs, idx + 1, len(loader)),
                    ", dice_tc: {:.6f}".format(dice_tc),
                    ", dice_wt: {:.6f}".format(dice_wt),
                    ", dice_et: {:.6f}".format(dice_et),
                    ", time {:.2f}s".format(time.time() - start_time),
                )
                start_time = time.time()
                
            except Exception as e:
                print(f"Error in validation step {idx}: {e}")
                continue

    return run_acc.avg


def log_training_samples(model, val_loader, val_cases, model_inferer, post_sigmoid, post_pred, epoch, num_samples=3):
    """Log segmentation samples during training - now in one figure"""
    model.eval()
    
    images, labels, predictions, case_names = [], [], [], []
    
    with torch.no_grad():
        for idx, batch_data in enumerate(val_loader):
            if len(images) >= num_samples:
                break
                
            try:
                data, target = batch_data["image"].cuda(), batch_data["label"].cuda()
                logits = model_inferer(data)
                
                # Convert prediction for visualization
                pred_sigmoid = post_sigmoid(logits)
                pred_discrete = post_pred(pred_sigmoid)
                
                pred = pred_discrete[0].cpu().numpy()
                pred_viz = np.zeros((pred.shape[1], pred.shape[2], pred.shape[3]))
                pred_viz[pred[1] == 1] = 2  # ED
                pred_viz[pred[0] == 1] = 1  # TC  
                pred_viz[pred[2] == 1] = 4  # ET
                
                # Get original label
                label_viz = target[0].cpu().numpy()
                label_orig = np.zeros((label_viz.shape[1], label_viz.shape[2], label_viz.shape[3]))
                label_orig[label_viz[1] == 1] = 2
                label_orig[label_viz[0] == 1] = 1
                label_orig[label_viz[2] == 1] = 4
                
                # Collect data
                images.append(data[0].cpu().numpy())
                labels.append(label_orig)
                predictions.append(pred_viz)
                case_names.append(val_cases[idx]["case_id"])
                
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue
    
    # Log all samples in one figure
    if images:
        log_segmentation_samples(images, labels, predictions, case_names, epoch)


def main():
    # Initialize W&B
    wandb.init(project="BraTS-Enhanced-Seg", name="enhanced_focal_warmrestart_proper_val")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Find training and validation cases from separate directories
    print("Looking for BraTS data...")
    base_dir = "/app/UNETR-BraTS-Synthesis"
    training_dir = os.path.join(base_dir, "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData")
    validation_dir = os.path.join(base_dir, "ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData")
    
    print(f"Training directory: {training_dir}")
    print(f"Validation directory: {validation_dir}")
    print(f"Training dir exists: {os.path.exists(training_dir)}")
    print(f"Validation dir exists: {os.path.exists(validation_dir)}")
    
    # Load datasets
    train_cases = find_brats_cases(training_dir, "train")
    val_cases = find_brats_cases(validation_dir, "val")
    
    print(f"\n=== DATASET SUMMARY ===")
    print(f"Training cases: {len(train_cases)}")
    print(f"Validation cases: {len(val_cases)}")
    print(f"Training batch size: 2")
    print(f"Validation batch size: 1")
    print(f"Training batches per epoch: {len(train_cases) // 2}")
    print(f"Validation batches per epoch: {len(val_cases)}")
    
    if not train_cases:
        print("No training cases found!")
        return
    
    if not val_cases:
        print("No validation cases found!")
        return
    
    # Log dataset info to W&B
    wandb.log({
        "dataset/train_cases": len(train_cases),
        "dataset/val_cases": len(val_cases),
        "dataset/train_batch_size": 2,
        "dataset/val_batch_size": 1,
        "dataset/train_batches_per_epoch": len(train_cases) // 2,
        "dataset/val_batches_per_epoch": len(val_cases)
    })
    
    # Transforms
    roi = (128, 128, 128)
    
    train_transform = transforms.Compose([
        transforms.LoadImaged(keys=["image", "label"]),
        transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        transforms.CropForegroundd(
            keys=["image", "label"],
            source_key="image",
            k_divisible=[roi[0], roi[1], roi[2]],
            allow_smaller=True,
        ),
        transforms.RandSpatialCropd(
            keys=["image", "label"],
            roi_size=[roi[0], roi[1], roi[2]],
            random_size=False,
        ),
        transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
    ])
    
    val_transform = transforms.Compose([
        transforms.LoadImaged(keys=["image", "label"]),
        transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ])
    
    # Data loaders
    train_ds = Dataset(data=train_cases, transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=8, pin_memory=True)
    
    val_ds = Dataset(data=val_cases, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    
    print(f"Actual training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
    
    # Model
    model = SwinUNETR(
        in_channels=4,
        out_channels=3,
        feature_size=48,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=True,
    ).cuda()
    
    # Enhanced Loss Function with Focal Loss and Class Weighting
    torch.backends.cudnn.benchmark = True
    dice_focal_loss = DiceFocalLoss(
        to_onehot_y=False, 
        sigmoid=True,
        weight=[1.0, 2.0, 3.0],  # Weight TC, WT, ET (enhancing tumor gets 3x weight)
        gamma=2.0,  # Focal loss gamma parameter
        lambda_dice=1.0,  # Balance between dice and focal loss
        lambda_focal=1.0,
        alpha=None  # Can be set to balance positive/negative examples
    )
    
    post_sigmoid = Activations(sigmoid=True)
    post_pred = AsDiscrete(argmax=False, threshold=0.5)
    dice_acc = DiceMetric(
        include_background=True, 
        reduction=MetricReduction.MEAN_BATCH, 
        get_not_nans=True,
        ignore_empty=False  # Handle empty predictions gracefully
    )
    
    model_inferer = partial(
        sliding_window_inference,
        roi_size=[roi[0], roi[1], roi[2]],
        sw_batch_size=1,
        predictor=model,
        overlap=0.5,
    )
    
    # Training setup with enhanced learning rate scheduling
    max_epochs = 50
    val_every = 2  # Validate every 2 epochs
    sample_log_every = 1  # Log samples every epoch
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # Enhanced Learning Rate Scheduler with Warm Restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,        # Initial restart period
        T_mult=2,      # Multiply restart period by this factor after each restart
        eta_min=1e-6   # Minimum learning rate
    )
    
    print(f"\n=== TRAINING CONFIGURATION ===")
    print(f"Max epochs: {max_epochs}")
    print(f"Validation every {val_every} epochs")
    print(f"Sample logging every {sample_log_every} epochs")
    print(f"Batch logging every 20 batches")
    print(f"Loss function: DiceFocalLoss with class weights [1.0, 2.0, 3.0]")
    print(f"Scheduler: CosineAnnealingWarmRestarts")
    print(f"ROI size: {roi}")
    
    val_acc_max = 0.0
    
    for epoch in range(max_epochs):
        print(f"\n=== EPOCH {epoch+1}/{max_epochs} ===")
        epoch_time = time.time()
        
        # Training with batch logging
        train_loss = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            loss_func=dice_focal_loss,
            max_epochs=max_epochs,
            model_inferer=model_inferer,
            post_sigmoid=post_sigmoid,
            post_pred=post_pred,
            val_cases=val_cases
        )
        
        print(
            "EPOCH {}/{} COMPLETE".format(epoch + 1, max_epochs),
            "avg_loss: {:.4f}".format(train_loss),
            "epoch_time: {:.2f}s".format(time.time() - epoch_time),
        )
        
        # Log epoch-level training metrics
        wandb.log({
            "epoch": epoch + 1,
            "train_loss_epoch": train_loss,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "epoch_time": time.time() - epoch_time
        })
        
        # Log segmentation samples every epoch
        if (epoch + 1) % sample_log_every == 0 or epoch == 0:
            print("Logging segmentation samples...")
            log_training_samples(
                model=model,
                val_loader=val_loader,
                val_cases=val_cases,
                model_inferer=model_inferer,
                post_sigmoid=post_sigmoid,
                post_pred=post_pred,
                epoch=epoch,
                num_samples=3  # Log 3 samples in one figure
            )
        
        # Validation every 2 epochs (but not on epoch 0 since we did it above)
        if (epoch + 1) % val_every == 0:
            print(f"Starting validation for epoch {epoch + 1}...")
            epoch_time = time.time()
            val_acc = val_epoch(
                model=model,
                loader=val_loader,
                epoch=epoch,
                acc_func=dice_acc,
                model_inferer=model_inferer,
                post_sigmoid=post_sigmoid,
                post_pred=post_pred,
                max_epochs=max_epochs
            )
            
            # Safe extraction of dice scores
            dice_tc = val_acc[0] if len(val_acc) > 0 and not np.isnan(val_acc[0]) else 0.0
            dice_wt = val_acc[1] if len(val_acc) > 1 and not np.isnan(val_acc[1]) else 0.0
            dice_et = val_acc[2] if len(val_acc) > 2 and not np.isnan(val_acc[2]) else 0.0
            val_avg_acc = np.nanmean(val_acc) if len(val_acc) > 0 else 0.0
            
            print(
                "VALIDATION COMPLETE {}/{}".format(epoch + 1, max_epochs),
                ", dice_tc: {:.6f}".format(dice_tc),
                ", dice_wt: {:.6f}".format(dice_wt),
                ", dice_et: {:.6f}".format(dice_et),
                ", dice_avg: {:.6f}".format(val_avg_acc),
                ", val_time: {:.2f}s".format(time.time() - epoch_time),
            )
            
            # Log validation metrics to W&B
            wandb.log({
                "val_dice_tc": dice_tc,
                "val_dice_wt": dice_wt,
                "val_dice_et": dice_et,
                "val_dice_avg": val_avg_acc,
                "val_time": time.time() - epoch_time
            })
            
            if val_avg_acc > val_acc_max:
                print("NEW BEST VALIDATION SCORE! ({:.6f} --> {:.6f})".format(val_acc_max, val_avg_acc))
                val_acc_max = val_avg_acc
                wandb.log({"best_val_dice_avg": val_acc_max})
                
        scheduler.step()
    
    print(f"\n✓ TRAINING COMPLETED!")
    print(f"✓ Best average dice score: {val_acc_max:.6f}")
    print(f"✓ Check your W&B project for detailed logs and segmentation samples!")
    print(f"✓ Training used proper validation dataset from ValidationData folder")
    wandb.finish()


if __name__ == "__main__":
    main()