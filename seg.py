#!/usr/bin/env python3
"""
Simple BraTS Segmentation with Swin UNETR
Logs segmentation samples to W&B
"""

import os
import json
import time
import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from functools import partial

import torch
import wandb

from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai import transforms
from monai.transforms import AsDiscrete, Activations
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from monai.networks.nets import SwinUNETR
from monai import data
from monai.data import decollate_batch


class AverageMeter:
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


def datafold_read(datalist, basedir, fold=0, key="training"):
    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    for d in json_data:
        for k in d:
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    tr = []
    val = []
    for d in json_data:
        if "fold" in d and d["fold"] == fold:
            val.append(d)
        else:
            tr.append(d)

    return tr, val


def get_transforms(roi):
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
    
    return train_transform, val_transform


def get_loader(batch_size, data_dir, json_list, fold, roi):
    train_files, validation_files = datafold_read(datalist=json_list, basedir=data_dir, fold=fold)
    train_transform, val_transform = get_transforms(roi)

    train_ds = data.Dataset(data=train_files, transform=train_transform)
    train_loader = data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
    )
    
    val_ds = data.Dataset(data=validation_files, transform=val_transform)
    val_loader = data.DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True,
    )

    return train_loader, val_loader


def log_segmentation_sample(image, label, prediction, case_name, epoch=None):
    """Log a segmentation sample to W&B"""
    # Get middle slice for visualization
    slice_idx = image.shape[-1] // 2
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Plot T1CE image (channel 1)
    axes[0].imshow(image[1, :, :, slice_idx], cmap='gray')
    axes[0].set_title('T1CE Image')
    axes[0].axis('off')
    
    # Plot FLAIR image (channel 0) 
    axes[1].imshow(image[0, :, :, slice_idx], cmap='gray')
    axes[1].set_title('FLAIR Image')
    axes[1].axis('off')
    
    # Plot ground truth
    axes[2].imshow(label[:, :, slice_idx])
    axes[2].set_title('Ground Truth')
    axes[2].axis('off')
    
    # Plot prediction
    axes[3].imshow(prediction[:, :, slice_idx])
    axes[3].set_title('Prediction')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    # Log to W&B
    title = f"Segmentation_{case_name}_slice_{slice_idx}"
    if epoch is not None:
        title += f"_epoch_{epoch}"
        
    wandb.log({f"segmentation_samples/{title}": wandb.Image(fig)})
    plt.close(fig)


def train_epoch(model, loader, optimizer, epoch, loss_func, device):
    model.train()
    run_loss = AverageMeter()
    
    for idx, batch_data in enumerate(loader):
        data, target = batch_data["image"].to(device), batch_data["label"].to(device)
        
        optimizer.zero_grad()
        logits = model(data)
        loss = loss_func(logits, target)
        loss.backward()
        optimizer.step()
        
        run_loss.update(loss.item(), n=data.shape[0])
        
        if idx % 10 == 0:
            print(f"Epoch {epoch} [{idx}/{len(loader)}] Loss: {run_loss.avg:.4f}")
    
    return run_loss.avg


def val_epoch(model, loader, epoch, acc_func, model_inferer, post_sigmoid, post_pred, device, log_samples=True):
    model.eval()
    run_acc = AverageMeter()
    
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data, target = batch_data["image"].to(device), batch_data["label"].to(device)
            logits = model_inferer(data)
            
            val_labels_list = decollate_batch(target)
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(post_sigmoid(val_pred_tensor)) for val_pred_tensor in val_outputs_list]
            
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_list)
            acc, not_nans = acc_func.aggregate()
            run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())
            
            # Log segmentation samples for first few cases
            if log_samples and idx < 3:
                # Convert prediction back to original format for visualization
                pred = val_output_convert[0].cpu().numpy()
                pred_viz = np.zeros((pred.shape[1], pred.shape[2], pred.shape[3]))
                pred_viz[pred[1] == 1] = 2  # ED
                pred_viz[pred[0] == 1] = 1  # TC  
                pred_viz[pred[2] == 1] = 4  # ET
                
                # Get original label
                label_viz = val_labels_list[0].cpu().numpy()
                label_orig = np.zeros((label_viz.shape[1], label_viz.shape[2], label_viz.shape[3]))
                label_orig[label_viz[1] == 1] = 2
                label_orig[label_viz[0] == 1] = 1
                label_orig[label_viz[2] == 1] = 4
                
                # Log sample
                case_name = f"val_case_{idx}"
                log_segmentation_sample(
                    data[0].cpu().numpy(), 
                    label_orig, 
                    pred_viz, 
                    case_name, 
                    epoch
                )
    
    dice_tc, dice_wt, dice_et = run_acc.avg[0], run_acc.avg[1], run_acc.avg[2]
    dice_avg = np.mean(run_acc.avg)
    
    return dice_avg, dice_tc, dice_wt, dice_et


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/data/brats2021challenge', help='Data directory')
    parser.add_argument('--json_list', default='./brats21_folds.json', help='JSON file with data splits')
    parser.add_argument('--fold', type=int, default=1, help='Fold for cross-validation')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=10, help='Maximum epochs')
    parser.add_argument('--val_every', type=int, default=2, help='Validation frequency')
    parser.add_argument('--use_wandb', action='store_true', help='Use W&B logging')
    parser.add_argument('--project_name', default='BraTS-Segmentation', help='W&B project name')
    args = parser.parse_args()

    # Initialize W&B
    if args.use_wandb:
        wandb.init(project=args.project_name, name=f"brats_fold_{args.fold}")
        wandb.config.update(args)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    roi = (128, 128, 128)
    sw_batch_size = 2 if args.batch_size > 1 else 1
    
    # Data
    train_loader, val_loader = get_loader(args.batch_size, args.data_dir, args.json_list, args.fold, roi)
    print(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
    
    # Model
    model = SwinUNETR(
        in_channels=4,
        out_channels=3,
        feature_size=48,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=True,
    ).to(device)
    
    # Loss and metrics
    dice_loss = DiceLoss(to_onehot_y=False, sigmoid=True)
    post_sigmoid = Activations(sigmoid=True)
    post_pred = AsDiscrete(argmax=False, threshold=0.5)
    dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)
    
    model_inferer = partial(
        sliding_window_inference,
        roi_size=roi,
        sw_batch_size=sw_batch_size,
        predictor=model,
        overlap=0.5,
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
    
    # Training loop
    best_dice = 0.0
    
    for epoch in range(args.max_epochs):
        print(f"\n--- Epoch {epoch+1}/{args.max_epochs} ---")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, epoch, dice_loss, device)
        print(f"Training loss: {train_loss:.4f}")
        
        # Validation
        if (epoch + 1) % args.val_every == 0 or epoch == 0:
            dice_avg, dice_tc, dice_wt, dice_et = val_epoch(
                model, val_loader, epoch, dice_acc, model_inferer, 
                post_sigmoid, post_pred, device, log_samples=args.use_wandb
            )
            
            print(f"Validation - Avg: {dice_avg:.4f}, TC: {dice_tc:.4f}, WT: {dice_wt:.4f}, ET: {dice_et:.4f}")
            
            if args.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_dice_avg": dice_avg,
                    "val_dice_tc": dice_tc,
                    "val_dice_wt": dice_wt,
                    "val_dice_et": dice_et,
                })
            
            if dice_avg > best_dice:
                best_dice = dice_avg
                torch.save(model.state_dict(), "best_model.pth")
                print(f"New best model saved! Dice: {best_dice:.4f}")
        
        scheduler.step()
    
    print(f"\nTraining completed! Best Dice: {best_dice:.4f}")
    
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()