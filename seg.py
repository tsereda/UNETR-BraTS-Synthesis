#!/usr/bin/env python3
"""
BraTS Segmentation
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import time

import torch
import wandb

from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai import transforms
from monai.transforms import AsDiscrete, Activations
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from monai.networks.nets import SwinUNETR
from monai.data import Dataset, DataLoader
from monai.data import decollate_batch


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


def find_brats_cases(data_dir):
    """Simple BraTS case finder for 2023 GLI format"""
    cases = []
    
    print(f"Scanning {data_dir} for BraTS cases...")
    
    for item in os.listdir(data_dir):
        if 'BraTS' in item and os.path.isdir(os.path.join(data_dir, item)):
            case_path = os.path.join(data_dir, item)
            
            # Simple pattern: {case_name}-{modality}.nii.gz
            flair_file = f"{item}-t2f.nii.gz"
            t1ce_file = f"{item}-t1c.nii.gz"
            t1_file = f"{item}-t1n.nii.gz"
            t2_file = f"{item}-t2w.nii.gz"
            seg_file = f"{item}-seg.nii.gz"
            
            # Check if all files exist
            if all(os.path.exists(os.path.join(case_path, f)) for f in [flair_file, t1ce_file, t1_file, t2_file, seg_file]):
                case_data = {
                    "image": [
                        os.path.join(case_path, flair_file),
                        os.path.join(case_path, t1ce_file), 
                        os.path.join(case_path, t1_file),
                        os.path.join(case_path, t2_file)
                    ],
                    "label": os.path.join(case_path, seg_file),
                    "case_id": item
                }
                cases.append(case_data)
                
                # Progress update every 50 cases instead of 100
                if len(cases) % 50 == 0:
                    print(f"Found {len(cases)} valid cases so far...")
    
    print(f"Finished scanning. Total cases found: {len(cases)}")
    return cases


def log_segmentation_sample(image, label, prediction, case_name, epoch=None):
    """Log a segmentation sample to W&B"""
    try:
        slice_idx = image.shape[-1] // 2
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
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
    except Exception as e:
        print(f"Error logging segmentation sample: {e}")


def train_epoch(model, loader, optimizer, epoch, loss_func, max_epochs):
    """Training epoch"""
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    
    for idx, batch_data in enumerate(loader):
        data, target = batch_data["image"].cuda(), batch_data["label"].cuda()
        
        optimizer.zero_grad()
        logits = model(data)
        loss = loss_func(logits, target)
        loss.backward()
        optimizer.step()
        
        run_loss.update(loss.item(), n=data.shape[0])
        print(
            "Epoch {}/{} {}/{}".format(epoch, max_epochs, idx, len(loader)),
            "loss: {:.4f}".format(run_loss.avg),
            "time {:.2f}s".format(time.time() - start_time),
        )
        start_time = time.time()
    
    return run_loss.avg


def val_epoch(model, loader, epoch, acc_func, model_inferer, post_sigmoid, post_pred, max_epochs):
    """Validation epoch"""
    model.eval()
    start_time = time.time()
    run_acc = AverageMeter()

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data, target = batch_data["image"].cuda(), batch_data["label"].cuda()
            logits = model_inferer(data)
            
            val_labels_list = decollate_batch(target)
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(post_sigmoid(val_pred_tensor)) for val_pred_tensor in val_outputs_list]
            
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_list)
            acc, not_nans = acc_func.aggregate()
            run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())
            
            dice_tc = run_acc.avg[0]
            dice_wt = run_acc.avg[1]
            dice_et = run_acc.avg[2]
            print(
                "Val {}/{} {}/{}".format(epoch, max_epochs, idx, len(loader)),
                ", dice_tc:",
                dice_tc,
                ", dice_wt:",
                dice_wt,
                ", dice_et:",
                dice_et,
                ", time {:.2f}s".format(time.time() - start_time),
            )
            start_time = time.time()

    return run_acc.avg


def log_training_samples(model, val_loader, val_cases, model_inferer, post_sigmoid, post_pred, epoch, num_samples=3):
    """Log segmentation samples during training"""
    model.eval()
    with torch.no_grad():
        for idx, batch_data in enumerate(val_loader):
            if idx >= num_samples:  # Log more samples for better monitoring
                break
                
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
            
            # Log to W&B
            case_name = val_cases[idx]["case_id"]
            log_segmentation_sample(
                data[0].cpu().numpy(), 
                label_orig, 
                pred_viz, 
                case_name, 
                epoch
            )


def main():
    # Initialize W&B
    wandb.init(project="BraTS-Simple-Seg", name="simple_full_training")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Find all BraTS cases - NO ARTIFICIAL LIMITS
    print("Looking for BraTS data...")
    base_dir = "/app/UNETR-BraTS-Synthesis"
    training_dir = os.path.join(base_dir, "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData")
    
    print(f"Scanning directory: {training_dir}")
    print(f"Directory exists: {os.path.exists(training_dir)}")
    
    cases = find_brats_cases(training_dir)
    print(f"\n=== SUMMARY ===")
    print(f"Found {len(cases)} valid cases")
    
    if not cases:
        print("No BraTS cases found!")
        return
    
    # Split into train/val (80/20)
    split_idx = max(1, int(len(cases) * 0.8))
    train_cases = cases[:split_idx]
    val_cases = cases[split_idx:]
    
    print(f"Training cases: {len(train_cases)}")
    print(f"Validation cases: {len(val_cases)}")
    
    # Log dataset info to W&B
    wandb.log({
        "dataset/total_cases": len(cases),
        "dataset/train_cases": len(train_cases),
        "dataset/val_cases": len(val_cases)
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
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
    
    val_ds = Dataset(data=val_cases, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    
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
    ).cuda()
    
    # Loss and metrics
    torch.backends.cudnn.benchmark = True
    dice_loss = DiceLoss(to_onehot_y=False, sigmoid=True)
    post_sigmoid = Activations(sigmoid=True)
    post_pred = AsDiscrete(argmax=False, threshold=0.5)
    dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)
    
    model_inferer = partial(
        sliding_window_inference,
        roi_size=[roi[0], roi[1], roi[2]],
        sw_batch_size=1,
        predictor=model,
        overlap=0.5,
    )
    
    # Training setup
    max_epochs = 50
    val_every = 2  # Validate every 2 epochs instead of 5
    sample_log_every = 1  # Log samples every epoch
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    
    print(f"Training for {max_epochs} epochs")
    print(f"Validation every {val_every} epochs")
    print(f"Sample logging every {sample_log_every} epochs")
    
    val_acc_max = 0.0
    
    for epoch in range(max_epochs):
        print(f"\n--- Epoch {epoch+1}/{max_epochs} ---")
        epoch_time = time.time()
        
        # Training
        train_loss = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            loss_func=dice_loss,
            max_epochs=max_epochs
        )
        
        print(
            "Final training  {}/{}".format(epoch, max_epochs - 1),
            "loss: {:.4f}".format(train_loss),
            "time {:.2f}s".format(time.time() - epoch_time),
        )
        
        # Log training loss every epoch
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        # Log segmentation samples more frequently
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
                num_samples=3  # Log 3 samples each time
            )
        
        # Validation
        if (epoch + 1) % val_every == 0 or epoch == 0:
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
            
            dice_tc = val_acc[0]
            dice_wt = val_acc[1]
            dice_et = val_acc[2]
            val_avg_acc = np.mean(val_acc)
            
            print(
                "Final validation stats {}/{}".format(epoch, max_epochs - 1),
                ", dice_tc:",
                dice_tc,
                ", dice_wt:",
                dice_wt,
                ", dice_et:",
                dice_et,
                ", Dice_Avg:",
                val_avg_acc,
                ", time {:.2f}s".format(time.time() - epoch_time),
            )
            
            # Log validation metrics to W&B
            wandb.log({
                "val_dice_tc": dice_tc,
                "val_dice_wt": dice_wt,
                "val_dice_et": dice_et,
                "val_dice_avg": val_avg_acc
            })
            
            if val_avg_acc > val_acc_max:
                print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                val_acc_max = val_avg_acc
                wandb.log({"best_val_dice_avg": val_acc_max})
                
        scheduler.step()
    
    print(f"\n✓ Training completed! Best average dice: {val_acc_max:.4f}")
    print("Check your W&B project for detailed logs and segmentation samples!")
    wandb.finish()


if __name__ == "__main__":
    main()