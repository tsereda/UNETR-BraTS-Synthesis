import matplotlib.pyplot as plt
import random
#!/usr/bin/env python3
"""
Training script for UNETR-based BraTS synthesis model.

This script handles training with support for transfer learning, mixed precision,
and various optimization strategies.
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
from typing import Dict, Any, Optional

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from models.unetr_synthesis import create_model
    from models.loss_functions import create_loss
    from data.brats_dataset import create_dataloader
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure to install dependencies: pip install -r requirements.txt")
    sys.exit(1)


class Trainer:

    def visualize_batch(self, num_samples: int = 2, phase: str = 'train'):
        """Visualize a batch of data (inputs and targets) for sanity check."""
        loader = self.train_loader if phase == 'train' else self.val_loader
        batch = next(iter(loader))
        inputs = batch['input'][:num_samples]
        targets = batch['target'][:num_samples]
        subject_names = batch.get('subject_name', [f'sample_{i}' for i in range(num_samples)])
        # Plot middle slice for each sample and channel
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

class SegmentationTrainer(Trainer):
    """Trainer for segmentation sanity check."""
    def _create_model(self) -> nn.Module:
        # Use UNETR backbone for segmentation (out_channels = num_classes)
        seg_config = self.config.copy()
        seg_config['model'] = seg_config.get('model', {}).copy()
        seg_config['model']['out_channels'] = 1  # binary seg for quick test
        from models.unetr_synthesis import create_model
        model = create_model(seg_config)
        return model.to(self.device)

    def _create_loss(self) -> nn.Module:
        # Use Dice loss for segmentation
        from monai.losses import DiceLoss
        return DiceLoss(sigmoid=True, reduction='mean').to(self.device)

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        for batch_idx, batch in enumerate(self.train_loader):
            inputs = batch['input'].to(self.device)
            # For seg sanity, use one input channel as image, one as mask
            # Here, use channel 0 as image, channel 1 as mask (if available)
            # If not, use target as mask
            if inputs.shape[1] > 1:
                image = inputs[:, 0:1]
                mask = inputs[:, 1:2]
            else:
                image = inputs
                mask = batch['target'].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(image)
            loss = self.criterion(outputs, mask)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            num_batches += 1
            if batch_idx == 0:
                # Visualize prediction vs mask
                pred = torch.sigmoid(outputs[0, 0]).detach().cpu().numpy()
                msk = mask[0, 0].detach().cpu().numpy()
                mid = pred.shape[-1] // 2
                plt.figure(figsize=(8, 4))
                plt.subplot(1, 2, 1)
                plt.imshow(pred[..., mid], cmap='Blues')
                plt.title('Predicted mask')
                plt.axis('off')
                plt.subplot(1, 2, 2)
                plt.imshow(msk[..., mid], cmap='Reds')
                plt.title('GT mask')
                plt.axis('off')
                plt.show()
        avg_loss = total_loss / num_batches
        return {'total_loss': avg_loss}

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
        self._setup_logging()
        
        print(f"Trainer initialized. Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _create_model(self) -> nn.Module:
        """Create and initialize the model."""
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
        """Create loss function."""
        loss = create_loss(self.config)
        return loss.to(self.device)
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        training_config = self.config.get('training', {})
        lr = training_config.get('learning_rate', 1e-4)
        weight_decay = training_config.get('weight_decay', 1e-4)
        optimizer_name = training_config.get('optimizer', 'AdamW')
        
        # Ensure lr and weight_decay are floats
        lr = float(lr)
        weight_decay = float(weight_decay)
        
        # Get parameter groups for transfer learning
        transfer_config = self.config.get('transfer_learning', {})
        if transfer_config.get('enabled', False):
            param_groups = self.model.get_parameter_groups(lr)
            params = param_groups
        else:
            params = self.model.parameters()
        
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
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True
            )
        else:
            scheduler = None
        
        return scheduler
    
    def _setup_logging(self):
        """Setup logging with wandb if enabled."""
        logging_config = self.config.get('logging', {})
        if logging_config.get('use_wandb', False):
            # Use environment variables if available, otherwise fall back to config
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
            # Log additional system info at the first step (step=1, not 0)
            # Log additional system info at the first step (step=1, not 0)
            # But do NOT log any metrics to step=0 or step=1, only use wandb.summary for static info
            wandb.summary["system/gpu_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
            wandb.summary["system/model_parameters"] = sum(p.numel() for p in self.model.parameters())
            wandb.summary["system/trainable_parameters"] = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            wandb.summary["system/device"] = str(self.device)
            self._wandb_first_log = True
            self.use_wandb = True
            print(f"W&B logging enabled. Project: {project_name}")
        else:
            self.use_wandb = False
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        loss_components = {}
        num_batches = 0
        for batch_idx, batch in enumerate(self.train_loader):
            # Move data to device
            inputs = batch['input'].to(self.device)
            targets = batch['target'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            # Compute loss
            losses = self.criterion(outputs, targets)
            total_loss_batch = losses['total']

            # Backward pass
            total_loss_batch.backward()
            self.optimizer.step()

            # Accumulate losses
            total_loss += total_loss_batch.item()
            for key, value in losses.items():
                if key not in loss_components:
                    loss_components[key] = 0.0
                loss_components[key] += value.item()

            num_batches += 1
            self.global_step += 1  # Increment global step once per batch

            # Log batch metrics
            if batch_idx % self.config.get('logging', {}).get('log_frequency', 100) == 0:
                print(f"Batch {batch_idx}/{len(self.train_loader)}, "
                      f"Loss: {total_loss_batch.item():.6f}")

                # Log to W&B during training for more frequent updates
                if self.use_wandb:
                    wandb.log({
                        'batch/train_loss': total_loss_batch.item(),
                        'batch/epoch': self.current_epoch,
                        'batch/learning_rate': self.optimizer.param_groups[0]['lr'],
                        'batch/global_step': self.global_step
                    }, step=self.global_step)

            # Log sample predictions to W&B every 100 batches
            if self.use_wandb and batch_idx % 100 == 0:
                if hasattr(self, 'log_sample_predictions'):
                    self.log_sample_predictions()
        # Average losses
        avg_loss = total_loss / num_batches
        for key in loss_components:
            loss_components[key] /= num_batches
        return {'total_loss': avg_loss, **loss_components}
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        loss_components = {}
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move data to device
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Compute loss
                losses = self.criterion(outputs, targets)
                
                # Accumulate losses
                total_loss += losses['total'].item()
                for key, value in losses.items():
                    if key not in loss_components:
                        loss_components[key] = 0.0
                    loss_components[key] += value.item()
                
                num_batches += 1
        
        # Average losses
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
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save current checkpoint
        torch.save(checkpoint, self.exp_dir / 'last_checkpoint.pth')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.exp_dir / 'best_model.pth')
            print(f"Saved best model at epoch {self.current_epoch}")
    
    def train(self):
        """Main training loop."""
        epochs = self.config.get('training', {}).get('epochs', 200)
        val_frequency = self.config.get('validation', {}).get('frequency', 10)
        early_stopping_patience = self.config.get('validation', {}).get('early_stopping_patience', 50)
        
        patience_counter = 0
        
        print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Log training metrics every epoch
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train/epoch_loss': train_metrics['total_loss'],
                    'train/learning_rate': self.optimizer.param_groups[0]['lr']
                }, step=self.global_step)
            
            print(f"Epoch {epoch}/{epochs} - Train Loss: {train_metrics['total_loss']:.6f}")
            
            # Validate
            if epoch % val_frequency == 0:
                val_metrics = self.validate()
                
                # Check for improvement
                val_loss = val_metrics['total_loss']
                is_best = val_loss < self.best_val_loss
                
                if is_best:
                    self.best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += val_frequency
                
                # Save checkpoint
                self.save_checkpoint(is_best)
                
                # Log metrics
                print(f"Epoch {epoch}/{epochs}")
                print(f"  Train Loss: {train_metrics['total_loss']:.6f}")
                print(f"  Val Loss: {val_loss:.6f} (Best: {self.best_val_loss:.6f})")
                
                if self.use_wandb:
                    # Log basic metrics
                    self.global_step += 1
                    log_dict = {
                        'epoch': epoch,
                        'train/total_loss': train_metrics['total_loss'],
                        'val/total_loss': val_loss,
                        'val/best_loss': self.best_val_loss,
                        'learning_rate': self.optimizer.param_groups[0]['lr'],
                        'val/is_best': is_best,
                        'patience_counter': patience_counter
                    }
                    # Log detailed loss components
                    for k, v in train_metrics.items():
                        if k != 'total_loss':
                            log_dict[f'train/{k}'] = v
                    for k, v in val_metrics.items():
                        if k != 'total_loss':
                            log_dict[f'val/{k}'] = v
                    wandb.log(log_dict, step=self.global_step)
                
                # Log sample predictions every 20 epochs (more frequent)
                if self.use_wandb and epoch % 20 == 0:
                    self.log_sample_predictions()
                
                # Early stopping
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['total_loss'] if epoch % val_frequency == 0 else train_metrics['total_loss'])
                else:
                    self.scheduler.step()
        
        print("Training completed!")
        
        if self.use_wandb:
            wandb.finish()
    
    def log_sample_predictions(self, num_samples: int = 2):
        """Log sample predictions to W&B for visualization."""
        if not getattr(self, 'use_wandb', False):
            return
        import wandb
        import numpy as np
        self.model.eval()
        # Get a batch from validation loader (or train loader if val not available)
        loader = self.val_loader if hasattr(self, 'val_loader') and self.val_loader is not None else self.train_loader
        try:
            batch = next(iter(loader))
        except Exception:
            return
        inputs = batch['input'][:num_samples].to(self.device)
        targets = batch['target'][:num_samples].to(self.device)
        subject_names = batch.get('subject_name', [f'sample_{i}' for i in range(num_samples)])
        with torch.no_grad():
            outputs = self.model(inputs)
        # Convert tensors to numpy for visualization
        inputs_np = inputs.cpu().numpy()
        targets_np = targets.cpu().numpy()
        outputs_np = outputs.cpu().numpy()
        # For each sample, log a middle slice of the volume
        images = []
        for i in range(min(num_samples, inputs_np.shape[0])):
            # Take the middle slice along the last axis (axial view)
            input_img = inputs_np[i, 0]  # first channel
            target_img = targets_np[i, 0]
            output_img = outputs_np[i, 0]
            mid_slice = input_img.shape[-1] // 2
            input_slice = input_img[..., mid_slice]
            target_slice = target_img[..., mid_slice]
            output_slice = output_img[..., mid_slice]
            # Normalize to [0, 255] for wandb.Image (uint8)
            def norm255(x):
                x = x.astype(np.float32)
                x = (x - x.min()) / (x.max() - x.min() + 1e-8)
                return (x * 255).astype(np.uint8)
            input_slice = norm255(input_slice)
            output_slice = norm255(output_slice)
            target_slice = norm255(target_slice)
            # Stack input, output, target for comparison
            stacked = np.stack([input_slice, output_slice, target_slice], axis=-1)
            caption = f"{subject_names[i] if isinstance(subject_names, list) else i} (input/output/target)"
            images.append(wandb.Image(stacked, caption=caption))
        # Log at a strictly increasing global step
        self.global_step += 1
        wandb.log({"sample_predictions": images}, step=self.global_step)
        self.model.train()

    # ...existing code...
def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Train UNETR synthesis model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory of BraTS data')
    parser.add_argument('--exp_name', type=str, required=True,
                       help='Experiment name')
    parser.add_argument('--pretrained_path', type=str, default=None,
                       help='Path to pretrained model')
    parser.add_argument('--freeze_layers', type=int, default=None,
                       help='Number of layers to freeze')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases logging')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    config['data_root'] = args.data_root
    
    if args.pretrained_path:
        config.setdefault('transfer_learning', {})
        config['transfer_learning']['enabled'] = True
        config['transfer_learning']['pretrained_path'] = args.pretrained_path
    
    if args.freeze_layers is not None:
        config.setdefault('transfer_learning', {})
        config['transfer_learning']['freeze_layers'] = args.freeze_layers
    
    if args.use_wandb:
        config.setdefault('logging', {})
        config['logging']['use_wandb'] = True
    
    # For sanity check, use SegmentationTrainer instead of Trainer
    trainer = SegmentationTrainer(config, args.exp_name)
    # Visualize a batch before training
    print("Visualizing a batch for sanity check...")
    trainer.visualize_batch(num_samples=2, phase='train')
    trainer.train()


if __name__ == "__main__":
    main()
