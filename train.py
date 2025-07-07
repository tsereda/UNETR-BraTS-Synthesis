#!/usr/bin/env python3
"""
Simplified working training script for UNETR-based BraTS synthesis and segmentation models.
Fixed wandb sample logging, removed matplotlib dependencies.
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
    """Simplified trainer class for UNETR synthesis model."""

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
        self.global_step = 0
        
        # Setup wandb logging
        self.use_wandb = False
        self._setup_logging()
        
        print(f"Trainer initialized. Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Using wandb: {self.use_wandb}")

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
            try:
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
                print(f"W&B logging enabled. Project: {project_name}")
            except Exception as e:
                print(f"Warning: Failed to initialize wandb: {e}")
                self.use_wandb = False

    def train(self):
        """Main training loop."""
        epochs = self.config.get('training', {}).get('epochs', 200)
        val_frequency = self.config.get('validation', {}).get('frequency', 10)
        early_stopping_patience = self.config.get('validation', {}).get('early_stopping_patience', 50)
        
        patience_counter = 0
        print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            print(f"\n=== Epoch {epoch}/{epochs} ===")
            
            # Train for one epoch
            train_metrics = self.train_epoch()
            print(f"Train Loss: {train_metrics['total_loss']:.6f}")
            
            # Log training metrics
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train/loss': train_metrics['total_loss'],
                    'train/learning_rate': self.optimizer.param_groups[0]['lr']
                }, step=self.global_step)

            # Validate and log samples
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
                
                print(f"Validation Loss: {val_loss:.6f} (Best: {self.best_val_loss:.6f})")
                
                # Log validation metrics
                if self.use_wandb:
                    wandb.log({
                        'val/loss': val_loss,
                        'val/best_loss': self.best_val_loss,
                        'val/is_best': is_best
                    }, step=self.global_step)

                # Log sample predictions during validation
                if self.use_wandb:
                    print("Logging sample predictions to wandb...")
                    self.log_sample_predictions()

                # Check for early stopping
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if 'val_loss' in locals():
                        self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
        
        print("Training completed!")
        if self.use_wandb:
            wandb.finish()

    def train_epoch(self) -> Dict[str, float]:
        """Train model for one epoch."""
        self.model.train()
        total_loss = 0.0
        loss_components = {}
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            inputs = batch['input'].to(self.device)
            targets = batch['target'].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            losses = self.criterion(outputs, targets)
            total_loss_batch = losses['total']
            total_loss_batch.backward()
            self.optimizer.step()

            self.global_step += 1
            total_loss += total_loss_batch.item()
            
            # Accumulate loss components
            for key, value in losses.items():
                loss_components.setdefault(key, 0.0)
                loss_components[key] += value.item()

            # Print progress every 25 batches
            if batch_idx % 25 == 0:
                print(f"  Batch {batch_idx}/{num_batches}, Loss: {total_loss_batch.item():.6f}")

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

    def log_sample_predictions(self, num_samples: int = 2):
        """Log sample predictions to W&B."""
        if not self.use_wandb:
            return
            
        print(f"Logging {num_samples} sample predictions to wandb...")
        
        self.model.eval()
        
        # Get a batch from validation loader
        try:
            val_iter = iter(self.val_loader)
            batch = next(val_iter)
        except Exception as e:
            print(f"Error loading validation batch: {e}")
            return

        # Process inputs and get predictions
        with torch.no_grad():
            inputs = batch['input'][:num_samples].to(self.device)
            targets = batch['target'][:num_samples].to(self.device)
            outputs = self.model(inputs)
            
            # Move to CPU and convert to numpy
            inputs_np = inputs.cpu().numpy()
            targets_np = targets.cpu().numpy()
            outputs_np = outputs.cpu().numpy()
            
        print(f"Generated predictions for {inputs_np.shape[0]} samples")
        print(f"Input shape: {inputs_np.shape}, Target shape: {targets_np.shape}, Output shape: {outputs_np.shape}")

        # Create wandb images
        try:
            images = []
            for i in range(min(num_samples, inputs_np.shape[0])):
                # Get middle slice for visualization
                mid_slice = inputs_np.shape[-1] // 2
                
                # Extract slices (assuming first channel for visualization)
                input_slice = inputs_np[i, 0, ..., mid_slice]
                target_slice = targets_np[i, 0, ..., mid_slice]
                output_slice = outputs_np[i, 0, ..., mid_slice]
                
                # Normalize slices to [0, 1] range for wandb
                def normalize_slice(x):
                    x = x.astype(np.float32)
                    x_min, x_max = x.min(), x.max()
                    if x_max - x_min < 1e-8:
                        return np.zeros_like(x)
                    return (x - x_min) / (x_max - x_min)
                
                input_norm = normalize_slice(input_slice)
                target_norm = normalize_slice(target_slice)
                output_norm = normalize_slice(output_slice)
                
                # Create side-by-side comparison
                h, w = input_norm.shape
                combined = np.zeros((h, w * 3))
                combined[:, :w] = input_norm
                combined[:, w:2*w] = output_norm
                combined[:, 2*w:] = target_norm
                
                # Get subject name if available
                subject_names = batch.get('subject_name', [f'sample_{j}' for j in range(num_samples)])
                subject_name = subject_names[i] if i < len(subject_names) else f'sample_{i}'
                
                caption = f"Epoch {self.current_epoch} - {subject_name}: Input | Prediction | Target"
                
                # Create wandb image
                wandb_img = wandb.Image(
                    combined, 
                    caption=caption,
                    mode="F"  # Float mode for grayscale
                )
                images.append(wandb_img)
                
            # Log to wandb
            if images:
                log_dict = {"predictions/samples": images}
                wandb.log(log_dict, step=self.global_step)
                print(f"Successfully logged {len(images)} images to wandb at step {self.global_step}")
            else:
                print("No images to log")
                
        except Exception as e:
            print(f"Error creating wandb images: {e}")
            import traceback
            traceback.print_exc()
        
        self.model.train()

    def print_data_summary(self, num_samples: int = 2, phase: str = 'train'):
        """Print summary of data batch for debugging (no visualization)."""
        loader = self.train_loader if phase == 'train' else self.val_loader
        batch = next(iter(loader))
        inputs = batch['input'][:num_samples]
        targets = batch['target'][:num_samples]
        subject_names = batch.get('subject_name', [f'sample_{i}' for i in range(num_samples)])

        print(f"\n=== {phase.upper()} Data Summary ===")
        for i in range(num_samples):
            inp = inputs[i]
            tgt = targets[i]
            subj = subject_names[i] if i < len(subject_names) else f'sample_{i}'
            
            print(f"Subject {i}: {subj}")
            print(f"  Input shape: {inp.shape}, dtype: {inp.dtype}")
            print(f"  Input range: [{inp.min():.3f}, {inp.max():.3f}]")
            print(f"  Target shape: {tgt.shape}, dtype: {tgt.dtype}")
            print(f"  Target range: [{tgt.min():.3f}, {tgt.max():.3f}]")


class SegmentationTrainer(Trainer):
    """Trainer for segmentation sanity check."""
    
    def _create_model(self) -> nn.Module:
        """Override to create a UNETR model for segmentation."""
        seg_config = self.config.copy()
        seg_config['model'] = seg_config.get('model', {}).copy()
        seg_config['model']['out_channels'] = 1  # Binary segmentation
        
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
        """Override training epoch for segmentation."""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(self.train_loader):
            inputs = batch['input'].to(self.device)
            
            # Use first channel as image and second as mask (if available)
            if inputs.shape[1] > 1:
                image, mask = inputs[:, 0:1], inputs[:, 1:2]
            else:
                image, mask = inputs, batch['target'].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(image)
            loss = self.criterion(outputs, mask)
            loss.backward()
            self.optimizer.step()
            
            self.global_step += 1
            total_loss += loss.item()

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
    parser.add_argument('--sanity_check', action='store_true', help='Run segmentation sanity check')
    
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
    
    # Instantiate trainer
    if args.sanity_check:
        print("Initializing SegmentationTrainer for sanity check.")
        trainer = SegmentationTrainer(config, args.exp_name)
    else:
        print("Initializing standard Trainer for synthesis.")
        trainer = Trainer(config, args.exp_name)
    
    # Print data summary instead of visualization
    print("Checking sample data...")
    trainer.print_data_summary(num_samples=2, phase='train')
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()