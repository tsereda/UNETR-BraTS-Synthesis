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
            
            # Log additional system info
            wandb.log({
                "system/gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "system/model_parameters": sum(p.numel() for p in self.model.parameters()),
                "system/trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                "system/device": str(self.device)
            }, step=0)
            
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

            # Log batch metrics
            if batch_idx % self.config.get('logging', {}).get('log_frequency', 100) == 0:
                print(f"Batch {batch_idx}/{len(self.train_loader)}, "
                      f"Loss: {total_loss_batch.item():.6f}")

                # Log to W&B during training for more frequent updates
                if self.use_wandb:
                    global_step = self.current_epoch * len(self.train_loader) + batch_idx
                    wandb.log({
                        'batch/train_loss': total_loss_batch.item(),
                        'batch/epoch': self.current_epoch,
                        'batch/learning_rate': self.optimizer.param_groups[0]['lr'],
                        'batch/global_step': global_step
                    }, step=global_step)

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
                epoch_step = (epoch + 1) * len(self.train_loader)
                wandb.log({
                    'epoch': epoch,
                    'train/epoch_loss': train_metrics['total_loss'],
                    'train/learning_rate': self.optimizer.param_groups[0]['lr']
                }, step=epoch_step)
            
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
                    validation_step = (epoch + 1) * len(self.train_loader)
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
                    
                    wandb.log(log_dict, step=validation_step)
                
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
        if not self.use_wandb:
            return
        
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                if i >= num_samples:
                    break
                
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                outputs = self.model(inputs)
                
                # Take the first sample from the batch
                input_sample = inputs[0].cpu().numpy()
                target_sample = targets[0].cpu().numpy()
                output_sample = outputs[0].cpu().numpy()
                
                # Log each modality and the prediction
                images = []
                
                # Log input modalities (middle slice)
                modality_names = self.config.get('data', {}).get('modalities', ['t1n', 't1c', 't2w', 't2f'])
                mid_slice = input_sample.shape[-1] // 2
                
                for mod_idx, mod_name in enumerate(modality_names):
                    if mod_idx < input_sample.shape[0]:
                        images.append(wandb.Image(
                            input_sample[mod_idx, :, :, mid_slice],
                            caption=f"Input {mod_name} - Sample {i+1}"
                        ))
                
                # Log target and prediction
                images.append(wandb.Image(
                    target_sample[0, :, :, mid_slice],
                    caption=f"Target - Sample {i+1}"
                ))
                images.append(wandb.Image(
                    output_sample[0, :, :, mid_slice],
                    caption=f"Prediction - Sample {i+1}"
                ))
                
                prediction_step = (self.current_epoch + 1) * len(self.train_loader)
                wandb.log({f"sample_predictions_epoch_{self.current_epoch}": images}, step=prediction_step)
        
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
    
    # Create trainer and start training
    trainer = Trainer(config, args.exp_name)
    trainer.train()


if __name__ == "__main__":
    main()
