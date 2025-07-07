#!/usr/bin/env python3
"""
Super Simple BraTS Segmentation - No JSON needed!
Just scans your data and runs segmentation with W&B logging.
"""

import os
import glob
import numpy as np
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


def find_brats_cases(data_dir, max_cases=10):
    """Find BraTS cases directly from directory structure"""
    cases = []
    
    # Look for BraTS directories
    for item in os.listdir(data_dir):
        if 'BraTS' in item and os.path.isdir(os.path.join(data_dir, item)):
            case_path = os.path.join(data_dir, item)
            files = os.listdir(case_path)
            
            # Find the modality files
            flair = next((f for f in files if 'flair' in f and f.endswith('.nii.gz')), None)
            t1ce = next((f for f in files if 't1ce' in f and f.endswith('.nii.gz')), None)
            t1 = next((f for f in files if f.endswith('_t1.nii.gz')), None)
            t2 = next((f for f in files if 't2' in f and f.endswith('.nii.gz')), None)
            seg = next((f for f in files if 'seg' in f and f.endswith('.nii.gz')), None)
            
            if all([flair, t1ce, t1, t2, seg]):
                case_data = {
                    "image": [
                        os.path.join(case_path, flair),
                        os.path.join(case_path, t1ce), 
                        os.path.join(case_path, t1),
                        os.path.join(case_path, t2)
                    ],
                    "label": os.path.join(case_path, seg),
                    "case_id": item
                }
                cases.append(case_data)
                print(f"Found case: {item}")
                
                if len(cases) >= max_cases:
                    break
    
    return cases


def log_segmentation_sample(image, label, prediction, case_name, epoch=None):
    """Log a segmentation sample to W&B"""
    slice_idx = image.shape[-1] // 2
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Plot images
    axes[0].imshow(image[1, :, :, slice_idx], cmap='gray')
    axes[0].set_title('T1CE')
    axes[0].axis('off')
    
    axes[1].imshow(image[0, :, :, slice_idx], cmap='gray')
    axes[1].set_title('FLAIR')
    axes[1].axis('off')
    
    axes[2].imshow(label[:, :, slice_idx])
    axes[2].set_title('Ground Truth')
    axes[2].axis('off')
    
    axes[3].imshow(prediction[:, :, slice_idx])
    axes[3].set_title('Prediction')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    title = f"seg_{case_name}_slice_{slice_idx}"
    if epoch is not None:
        title += f"_epoch_{epoch}"
        
    wandb.log({f"segmentation/{title}": wandb.Image(fig)})
    plt.close(fig)


def main():
    # Initialize W&B
    wandb.init(project="BraTS-Simple-Seg", name="quick_test")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Find data
    print("Looking for BraTS data...")
    base_dir = "/app/UNETR-BraTS-Synthesis"
    training_dir = os.path.join(base_dir, "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData")
    
    cases = find_brats_cases(training_dir, max_cases=5)  # Just 5 cases for quick test
    print(f"Found {len(cases)} cases")
    
    if not cases:
        print("No BraTS cases found!")
        return
    
    # Split into train/val (simple split)
    train_cases = cases[:3]
    val_cases = cases[3:]
    
    # Transforms
    roi = (128, 128, 128)
    
    train_transform = transforms.Compose([
        transforms.LoadImaged(keys=["image", "label"]),
        transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
        transforms.RandSpatialCropd(keys=["image", "label"], roi_size=roi, random_size=False),
        transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ])
    
    val_transform = transforms.Compose([
        transforms.LoadImaged(keys=["image", "label"]),
        transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ])
    
    # Data loaders
    train_ds = data.Dataset(data=train_cases, transform=train_transform)
    train_loader = data.DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=2)
    
    val_ds = data.Dataset(data=val_cases, transform=val_transform)
    val_loader = data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)
    
    print(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
    
    # Model
    model = SwinUNETR(
        in_channels=4,
        out_channels=3,
        feature_size=48,
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
        sw_batch_size=1,
        predictor=model,
        overlap=0.5,
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # Quick training loop
    max_epochs = 3
    
    for epoch in range(max_epochs):
        print(f"\n--- Epoch {epoch+1}/{max_epochs} ---")
        
        # Quick training
        model.train()
        train_loss = 0
        for batch_data in train_loader:
            data, target = batch_data["image"].to(device), batch_data["label"].to(device)
            
            optimizer.zero_grad()
            logits = model(data)
            loss = dice_loss(logits, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        print(f"Training loss: {train_loss:.4f}")
        
        # Validation with segmentation logging
        model.eval()
        with torch.no_grad():
            for idx, batch_data in enumerate(val_loader):
                data, target = batch_data["image"].to(device), batch_data["label"].to(device)
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
                
                # Log to W&B
                case_name = val_cases[idx]["case_id"]
                log_segmentation_sample(
                    data[0].cpu().numpy(), 
                    label_orig, 
                    pred_viz, 
                    case_name, 
                    epoch
                )
                
                print(f"Logged segmentation for {case_name}")
        
        wandb.log({"epoch": epoch, "train_loss": train_loss})
    
    print("\n✓ Done! Check your W&B project for segmentation samples!")
    wandb.finish()


if __name__ == "__main__":
    main()