#!/usr/bin/env python3
"""
3D Brain Tumor Modality Synthesis with Swin UNETR
Based on MONAI BraTS tutorial, modified for image-to-image translation.

This script trains a Swin UNETR model to synthesize one MRI modality from others.
Default: t1c + t1n + t2w → t2f (FLAIR synthesis)
"""

import os
import json
import shutil
import tempfile
import time
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib

from monai.losses import SSIMLoss
from monai.inferers import sliding_window_inference
from monai import transforms
from monai.transforms import (
    AsDiscrete,
    Activations,
)

from monai.config import print_config
from monai.metrics import MSEMetric
from monai.utils.enums import MetricReduction
from monai.networks.nets import SwinUNETR
from monai import data
from monai.data import decollate_batch
from functools import partial

import torch
import torch.nn as nn

print_config()

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


def datafold_read(datalist, basedir, fold=0, key="training"):
    """Read data from JSON file and create fold splits"""
    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    for d in json_data:
        for k in d:
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    # For simplicity, we'll do a simple split based on index
    # In practice, you might want to implement actual k-fold CV
    num_cases = len(json_data)
    val_size = max(1, num_cases // 5)  # 20% for validation
    
    if fold == 0:
        val_indices = list(range(0, val_size))
    else:
        start_idx = (fold * val_size) % num_cases
        val_indices = list(range(start_idx, min(start_idx + val_size, num_cases)))
    
    tr = []
    val = []
    for i, d in enumerate(json_data):
        if i in val_indices:
            val.append(d)
        else:
            tr.append(d)

    return tr, val


def save_checkpoint(model, epoch, filename="model.pt", best_acc=0, dir_add=None):
    """Save model checkpoint"""
    if dir_add is None:
        dir_add = "./"
    state_dict = model.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    filename = os.path.join(dir_add, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


class CombinedLoss(nn.Module):
    """Combined loss for image synthesis"""
    def __init__(self, l1_weight=1.0, ssim_weight=0.1):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss(spatial_dims=3)
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
    
    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        return self.l1_weight * l1 + self.ssim_weight * ssim


def get_loader(batch_size, data_dir, json_list, fold, roi):
    """Create data loaders for training and validation"""
    data_dir = data_dir
    datalist_json = json_list
    train_files, validation_files = datafold_read(datalist=datalist_json, basedir=data_dir, fold=fold)
    
    train_transform = transforms.Compose([
        transforms.LoadImaged(keys=["image", "target"]),
        transforms.EnsureChannelFirstd(keys=["image", "target"]),
        transforms.CropForegroundd(
            keys=["image", "target"],
            source_key="image",
            k_divisible=[roi[0], roi[1], roi[2]],
            allow_smaller=True,
        ),
        transforms.RandSpatialCropd(
            keys=["image", "target"],
            roi_size=[roi[0], roi[1], roi[2]],
            random_size=False,
        ),
        transforms.RandFlipd(keys=["image", "target"], prob=0.5, spatial_axis=0),
        transforms.RandFlipd(keys=["image", "target"], prob=0.5, spatial_axis=1),
        transforms.RandFlipd(keys=["image", "target"], prob=0.5, spatial_axis=2),
        transforms.NormalizeIntensityd(keys=["image", "target"], nonzero=True, channel_wise=True),
        transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
    ])
    
    val_transform = transforms.Compose([
        transforms.LoadImaged(keys=["image", "target"]),
        transforms.EnsureChannelFirstd(keys=["image", "target"]),
        transforms.NormalizeIntensityd(keys=["image", "target"], nonzero=True, channel_wise=True),
    ])

    train_ds = data.Dataset(data=train_files, transform=train_transform)
    train_loader = data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    val_ds = data.Dataset(data=validation_files, transform=val_transform)
    val_loader = data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader


def train_epoch(model, loader, optimizer, epoch, loss_func, max_epochs):
    """Training epoch"""
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    
    for idx, batch_data in enumerate(loader):
        data = batch_data["image"].to(device)
        target = batch_data["target"].to(device)
        
        optimizer.zero_grad()
        logits = model(data)
        loss = loss_func(logits, target)
        loss.backward()
        optimizer.step()
        
        run_loss.update(loss.item(), n=data.size(0))
        
        if idx % 10 == 0:
            print(
                "Epoch {}/{} {}/{}".format(epoch, max_epochs, idx, len(loader)),
                "loss: {:.4f}".format(run_loss.avg),
                "time {:.2f}s".format(time.time() - start_time),
            )
            start_time = time.time()
    
    return run_loss.avg


def val_epoch(model, loader, epoch, max_epochs, model_inferer=None):
    """Validation epoch"""
    model.eval()
    start_time = time.time()
    run_mse = AverageMeter()
    run_mae = AverageMeter()

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data = batch_data["image"].to(device)
            target = batch_data["target"].to(device)
            
            logits = model_inferer(data)
            
            # Calculate metrics
            mse = torch.mean((logits - target) ** 2)
            mae = torch.mean(torch.abs(logits - target))
            
            run_mse.update(mse.cpu().numpy())
            run_mae.update(mae.cpu().numpy())
            
            if idx % 5 == 0:
                print(
                    "Val {}/{} {}/{}".format(epoch, max_epochs, idx, len(loader)),
                    ", MSE:", run_mse.avg,
                    ", MAE:", run_mae.avg,
                    ", time {:.2f}s".format(time.time() - start_time),
                )
                start_time = time.time()

    return run_mse.avg, run_mae.avg


def trainer(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    scheduler,
    model_inferer=None,
    start_epoch=0,
    max_epochs=100,
    val_every=10,
    save_dir="./",
):
    """Main training loop"""
    best_mse = float('inf')
    mse_values = []
    mae_values = []
    loss_epochs = []
    train_epochs = []
    
    for epoch in range(start_epoch, max_epochs):
        print(time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            epoch=epoch,
            loss_func=loss_func,
            max_epochs=max_epochs,
        )
        
        print(
            "Final training  {}/{}".format(epoch, max_epochs - 1),
            "loss: {:.4f}".format(train_loss),
            "time {:.2f}s".format(time.time() - epoch_time),
        )

        if (epoch + 1) % val_every == 0 or epoch == 0:
            loss_epochs.append(train_loss)
            train_epochs.append(int(epoch))
            epoch_time = time.time()
            
            val_mse, val_mae = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                max_epochs=max_epochs,
                model_inferer=model_inferer,
            )
            
            print(
                "Final validation stats {}/{}".format(epoch, max_epochs - 1),
                ", MSE:", val_mse,
                ", MAE:", val_mae,
                ", time {:.2f}s".format(time.time() - epoch_time),
            )
            
            mse_values.append(val_mse)
            mae_values.append(val_mae)
            
            if val_mse < best_mse:
                print("new best MSE ({:.6f} --> {:.6f}). ".format(best_mse, val_mse))
                best_mse = val_mse
                save_checkpoint(
                    model,
                    epoch,
                    best_acc=best_mse,
                    dir_add=save_dir,
                )
            
            scheduler.step()
    
    print("Training Finished!, Best MSE: ", best_mse)
    return best_mse, mse_values, mae_values, loss_epochs, train_epochs


# Configuration
if __name__ == "__main__":
    # Set up data directory
    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    if directory is not None:
        os.makedirs(directory, exist_ok=True)
    root_dir = tempfile.mkdtemp() if directory is None else directory
    print("Working directory:", root_dir)

    # Dataset configuration - UPDATE THESE PATHS
    data_dir = "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"  # Update this path
    json_list = "brats_synthesis_data.json"  # Update this path
    
    # Training configuration
    roi = (128, 128, 128)
    batch_size = 2
    sw_batch_size = 4
    fold = 0
    infer_overlap = 0.5
    max_epochs = 100
    val_every = 10

    # Check if paths exist
    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} does not exist!")
        print("Please update the data_dir variable with your actual data path.")
        exit(1)
        
    if not os.path.exists(json_list):
        print(f"Error: JSON file {json_list} does not exist!")
        print("Please run generate_data_json.py first to create the JSON file.")
        exit(1)

    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = get_loader(batch_size, data_dir, json_list, fold, roi)

    # Set up device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model for modality synthesis
    # Input: 3 modalities (t1c, t1n, t2w) -> Output: 1 modality (t2f)
    model = SwinUNETR(
        in_channels=3,      # 3 input modalities
        out_channels=1,     # 1 output modality
        feature_size=48,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=True,
    ).to(device)

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Set up training components
    torch.backends.cudnn.benchmark = True
    synthesis_loss = CombinedLoss(l1_weight=1.0, ssim_weight=0.1)
    
    model_inferer = partial(
        sliding_window_inference,
        roi_size=[roi[0], roi[1], roi[2]],
        sw_batch_size=sw_batch_size,
        predictor=model,
        overlap=infer_overlap,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    # Train the model
    print("Starting training...")
    start_epoch = 0

    (
        best_mse,
        mse_values,
        mae_values,
        loss_epochs,
        train_epochs,
    ) = trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_func=synthesis_loss,
        scheduler=scheduler,
        model_inferer=model_inferer,
        start_epoch=start_epoch,
        max_epochs=max_epochs,
        val_every=val_every,
        save_dir=root_dir,
    )

    print(f"Training completed, best MSE: {best_mse:.6f}")

    # Plot training results
    plt.figure("training", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    plt.xlabel("epoch")
    plt.plot(train_epochs, loss_epochs, color="red")
    plt.subplot(1, 2, 2)
    plt.title("Validation MSE")
    plt.xlabel("epoch")
    plt.plot(train_epochs, mse_values, color="blue")
    plt.tight_layout()
    plt.savefig(os.path.join(root_dir, "training_curves.png"))
    plt.show()

    print(f"Training curves saved to: {os.path.join(root_dir, 'training_curves.png')}")
    print(f"Model checkpoint saved to: {os.path.join(root_dir, 'model.pt')}")