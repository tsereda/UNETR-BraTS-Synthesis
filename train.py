#!/usr/bin/env python3
"""
Training script for UNETR-based BraTS synthesis and segmentation models.

This script handles training with support for transfer learning, mixed precision,
and various optimization strategies. It includes a base `Trainer` for image synthesis
and a `SegmentationTrainer` subclass for a segmentation sanity check.
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import wandb
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from models.unetr_synthesis import create_model
    from models.loss_functions import create_loss
    from data.brats_dataset import create_dataloader
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)


class Trainer:
    """Trainer class for UNETR synthesis model."""

    def __init__(self, config: Dict[str, Any], exp_name: str):
        self.config = config
        self.exp_name = exp_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup directories
        self.exp_dir = Path('experiments') / exp_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.exp_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Initialize components
        self.model = self._create_model()
        self.criterion = self._create_loss()
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Initialize data loaders
        self.train_loader = create_dataloader(config, phase='train')
        self.val_loader = create_dataloader(config, phase='val')
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.global_step = 0  # Monotonically increasing global step for wandb
        
        # Setup logging
        self.use_wandb = False
        self._setup_logging()
        
        print(f"Trainer initialized. Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def _create_model(self) -> nn.Module:
        """Create and initialize the synthesis model."""
        model = create_model(self.config)
        model = model.to(self.device)
        
        # Load pretrained weights if specified
        transfer_config = self.config.get('transfer_learning', {})
        if transfer_config.get('enabled', False):
            pretrained_path = transfer_config.get('pretrained_path')
            if pretrained_path and os.path.exists(pretrained_path):
                freeze_layers = transfer_config.get('freeze_layers', 0)
                model.load_pretrained_weights(
                    pretrained_path, 
                    freeze_layers=freeze_layers
                )
                print(f"Loaded pretrained weights from: {pretrained_path}")
            else:
                print(f"Warning: Pretrained path not found: {pretrained_path}")
        
        return model

    def _create_loss(self) -> nn.Module:
        """Create loss function for synthesis."""
        loss = create_loss(self.config)
        return loss.to(self.device)

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        training_config = self.config.get('training', {})
        lr = float(training_config.get('learning_rate', 1e-4))
        weight_decay = float(training_config.get('weight_decay', 1e-4))
        optimizer_name = training_config.get('optimizer', 'AdamW')
        
        # Get parameter groups for transfer learning
        transfer_config = self.config.get('transfer_learning', {})
        params = self.model.get_parameter_groups(lr) if transfer_config.get('enabled', False) else self.model.parameters()
        
        if optimizer_name.lower() == 'adamw':
            optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'adam':
            optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'sgd':
            optimizer = torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        return optimizer

    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        training_config = self.config.get('training', {})
        scheduler_name = training_config.get('scheduler', 'cosine')
        
        if scheduler_name == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=training_config.get('epochs', 200)
            )
        elif scheduler_name == 'step':
            scheduler_params = training_config.get('scheduler_params', {})
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_params.get('step_size', 30),
                gamma=scheduler_params.get('gamma', 0.5)
            )
        elif scheduler_name == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
            )
        else:
            scheduler = None
        
        return scheduler

    def _setup_logging(self):
        """Setup logging with wandb if enabled."""
        logging_config = self.config.get('logging', {})
        if logging_config.get('use_wandb', False):
            self.use_wandb = True
            project_name = os.getenv('WANDB_PROJECT', logging_config.get('project_name', 'unetr-brats-synthesis'))
            entity_name = os.getenv('WANDB_ENTITY', None)
            
            wandb_config = {
                'project': project_name,
                'name': self.exp_name,
                'config': self.config,
                'tags': logging_config.get('tags', [])
            }
            if entity_name:
                wandb_config['entity'] = entity_name
                
            wandb.init(**wandb_config)
            
            # Log static, step-independent info using wandb.summary
            wandb.summary["system/gpu_count"] = torch.cuda.device_count()
            wandb.summary["system/model_parameters"] = sum(p.numel() for p in self.model.parameters())
            wandb.summary["system/trainable_parameters"] = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            wandb.summary["system/device"] = str(self.device)
            
            print(f"W&B logging enabled. Project: {project_name}")

    def train(self):
        """Main training loop."""
        epochs = self.config.get('training', {}).get('epochs', 200)
        val_frequency = self.config.get('validation', {}).get('frequency', 10)
        early_stopping_patience = self.config.get('validation', {}).get('early_stopping_patience', 50)
        
        patience_counter = 0
        print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            
            # Train for one epoch
            train_metrics = self.train_epoch()
            print(f"Epoch {epoch}/{epochs} - Train Loss: {train_metrics['total_loss']:.6f}")
            
            # Log training metrics
            if self.use_wandb:
                log_data = {
                    'epoch': epoch,
                    'train/epoch_loss': train_metrics['total_loss'],
                    'train/learning_rate': self.optimizer.param_groups[0]['lr']
                }
                for k, v in train_metrics.items():
                    if k != 'total_loss':
                        log_data[f'train/{k}'] = v
                wandb.log(log_data, step=self.global_step)

            # Validate the model
            if epoch % val_frequency == 0:
                val_metrics = self.validate()
                val_loss = val_metrics['total_loss']
                is_best = val_loss < self.best_val_loss
                
                if is_best:
                    self.best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += val_frequency
                
                self.save_checkpoint(is_best)
                
                print(f"  Validation Loss: {val_loss:.6f} (Best: {self.best_val_loss:.6f})")
                
                # Log validation metrics and sample predictions
                if self.use_wandb:
                    val_log_data = {
                        'val/total_loss': val_loss,
                        'val/best_loss': self.best_val_loss,
                        'val/is_best': is_best,
                        'patience_counter': patience_counter
                    }
                    for k, v in val_metrics.items():
                        if k != 'total_loss':
                            val_log_data[f'val/{k}'] = v
                    wandb.log(val_log_data, step=self.global_step)

                    if epoch % 20 == 0:  # Log images more frequently
                        self.log_sample_predictions()

                # Check for early stopping
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # Step scheduler on validation loss if available
                    if 'val_loss' in locals():
                        self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
        
        print("Training completed!")
        if self.use_wandb:
            wandb.finish()

    def train_epoch(self) -> Dict[str, float]:
        """Train model for one epoch (synthesis task)."""
        self.model.train()
        total_loss = 0.0
        loss_components = {}
        
        for batch_idx, batch in enumerate(self.train_loader):
            inputs = batch['input'].to(self.device)
            targets = batch['target'].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            losses = self.criterion(outputs, targets)
            total_loss_batch = losses['total']
            total_loss_batch.backward()
            self.optimizer.step()

            # Increment global step once per batch
            self.global_step += 1

            # Accumulate losses for epoch average
            total_loss += total_loss_batch.item()
            for key, value in losses.items():
                loss_components.setdefault(key, 0.0)
                loss_components[key] += value.item()

            # Log batch metrics periodically to console and wandb
            if batch_idx % self.config.get('logging', {}).get('log_frequency', 100) == 0:
                print(f"  Batch {batch_idx}/{len(self.train_loader)}, Loss: {total_loss_batch.item():.6f}")
                if self.use_wandb:
                    wandb.log({
                        'batch/train_loss': total_loss_batch.item(),
                        'batch/epoch': self.current_epoch,
                        'batch/learning_rate': self.optimizer.param_groups[0]['lr'],
                    }, step=self.global_step)

        # Average losses over all batches in the epoch
        num_batches = len(self.train_loader)
        avg_loss = total_loss / num_batches
        for key in loss_components:
            loss_components[key] /= num_batches
            
        return {'total_loss': avg_loss, **loss_components}

    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        loss_components = {}
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in self.val_loader:
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                outputs = self.model(inputs)
                losses = self.criterion(outputs, targets)
                
                total_loss += losses['total'].item()
                for key, value in losses.items():
                    loss_components.setdefault(key, 0.0)
                    loss_components[key] += value.item()
        
        avg_loss = total_loss / num_batches
        for key in loss_components:
            loss_components[key] /= num_batches
        
        return {'total_loss': avg_loss, **loss_components}

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'global_step': self.global_step,
            'config': self.config
        }
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, self.exp_dir / 'last_checkpoint.pth')
        if is_best:
            torch.save(checkpoint, self.exp_dir / 'best_model.pth')
            print(f"Saved new best model at epoch {self.current_epoch}")

    def visualize_batch(self, num_samples: int = 2, phase: str = 'train'):
        """Visualize a batch of data for a sanity check before training."""
        loader = self.train_loader if phase == 'train' else self.val_loader
        batch = next(iter(loader))
        inputs = batch['input'][:num_samples]
        targets = batch['target'][:num_samples]
        subject_names = batch.get('subject_name', [f'sample_{i}' for i in range(num_samples)])

        for i in range(num_samples):
            fig, axs = plt.subplots(1, inputs.shape[1] + 1, figsize=(16, 4))
            for c in range(inputs.shape[1]):
                img = inputs[i, c].cpu().numpy()
                mid = img.shape[-1] // 2
                axs[c].imshow(img[..., mid], cmap='gray')
                axs[c].set_title(f'Input ch{c}')
                axs[c].axis('off')
            
            tgt = targets[i, 0].cpu().numpy()
            mid = tgt.shape[-1] // 2
            axs[-1].imshow(tgt[..., mid], cmap='hot')
            axs[-1].set_title('Target')
            axs[-1].axis('off')
            
            plt.suptitle(f'Subject: {subject_names[i]}')
            plt.show()

    def log_sample_predictions(self, num_samples: int = 2):
        """Log sample predictions to W&B for visualization."""
        if not self.use_wandb:
            return

        self.model.eval()
        try:
            batch = next(iter(self.val_loader))
        except StopIteration:
            return
        
        inputs = batch['input'][:num_samples].to(self.device)
        targets = batch['target'][:num_samples]
        outputs = self.model(inputs).cpu()
        
        def norm255(x):
            x = x.astype(np.float32)
            x_min, x_max = x.min(), x.max()
            if x_max - x_min < 1e-8:
                return (x * 0).astype(np.uint8)
            return ((x - x_min) / (x_max - x_min) * 255).astype(np.uint8)

        images = []
        for i in range(min(num_samples, inputs.shape[0])):
            subject_name = batch.get('subject_name', ['N/A']*num_samples)[i]
            
            input_slice = inputs[i, 0].cpu().numpy()
            target_slice = targets[i, 0].numpy()
            output_slice = outputs[i, 0].detach().numpy()
            
            mid_slice = input_slice.shape[-1] // 2
            
            stacked = np.concatenate([
                norm255(input_slice[..., mid_slice]),
                norm255(output_slice[..., mid_slice]),
                norm255(target_slice[..., mid_slice])
            ], axis=1)
            
            caption = f"{subject_name} (Input | Prediction | Target)"
            images.append(wandb.Image(stacked, caption=caption))
        
        # Log using the current global_step without incrementing it
        wandb.log({"sample_predictions": images}, step=self.global_step)
        self.model.train()


class SegmentationTrainer(Trainer):
    """Trainer for segmentation sanity check, inheriting from the base Trainer."""
    
    def _create_model(self) -> nn.Module:
        """Override to create a UNETR model for segmentation."""
        seg_config = self.config.copy()
        seg_config['model'] = seg_config.get('model', {}).copy()
        seg_config['model']['out_channels'] = 1  # Binary segmentation for a quick test
        
        from models.unetr_synthesis import create_model
        model = create_model(seg_config)
        return model.to(self.device)

    def _create_loss(self) -> nn.Module:
        """Override to use Dice loss for segmentation."""
        try:
            from monai.losses import DiceLoss
        except ImportError:
            raise ImportError("MONAI is required for SegmentationTrainer. Please install it.")
        return DiceLoss(sigmoid=True, reduction='mean').to(self.device)

    def train_epoch(self) -> Dict[str, float]:
        """Override training epoch for segmentation sanity check."""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(self.train_loader):
            inputs = batch['input'].to(self.device)
            # Use channel 0 as image and channel 1 as the mask ground truth
            # This is a specific setup for a sanity check
            if inputs.shape[1] > 1:
                image, mask = inputs[:, 0:1], inputs[:, 1:2]
            else: # Fallback if only one channel is provided
                image, mask = inputs, batch['target'].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(image)
            loss = self.criterion(outputs, mask)
            loss.backward()
            self.optimizer.step()
            
            self.global_step += 1
            total_loss += loss.item()

            # Visualize prediction vs. mask for the first batch of the epoch
            if batch_idx == 0:
                print("Visualizing segmentation sanity check prediction...")
                pred = torch.sigmoid(outputs[0, 0]).detach().cpu().numpy()
                msk = mask[0, 0].detach().cpu().numpy()
                mid = pred.shape[-1] // 2
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
                ax1.imshow(pred[..., mid], cmap='Blues')
                ax1.set_title('Predicted Mask')
                ax1.axis('off')
                ax2.imshow(msk[..., mid], cmap='Reds')
                ax2.set_title('GT Mask')
                ax2.axis('off')
                plt.show()

        avg_loss = total_loss / len(self.train_loader)
        return {'total_loss': avg_loss}


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Train UNETR models for synthesis or segmentation.')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--data_root', type=str, required=True, help='Root directory of BraTS data')
    parser.add_argument('--exp_name', type=str, required=True, help='Experiment name')
    parser.add_argument('--pretrained_path', type=str, default=None, help='Path to pretrained model')
    parser.add_argument('--freeze_layers', type=int, default=None, help='Number of layers to freeze')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--sanity_check', action='store_true', help='Run segmentation sanity check instead of synthesis training')
    
    args = parser.parse_args()
    
    # Load and override configuration
    config = load_config(args.config)
    config['data_root'] = args.data_root
    
    if args.pretrained_path:
        config.setdefault('transfer_learning', {})['enabled'] = True
        config['transfer_learning']['pretrained_path'] = args.pretrained_path
    
    if args.freeze_layers is not None:
        config.setdefault('transfer_learning', {})['freeze_layers'] = args.freeze_layers
    
    if args.use_wandb:
        config.setdefault('logging', {})['use_wandb'] = True
    
    # Instantiate the appropriate trainer
    if args.sanity_check:
        print("Initializing SegmentationTrainer for sanity check.")
        trainer = SegmentationTrainer(config, args.exp_name)
    else:
        print("Initializing standard Trainer for synthesis.")
        trainer = Trainer(config, args.exp_name)
    
    # Visualize a batch before training starts
    print("Visualizing a sample batch for review...")
    trainer.visualize_batch(num_samples=2, phase='train')
    
    # Start the training process
    trainer.train()


if __name__ == "__main__":
    main()