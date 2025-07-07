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
import traceback
import matplotlib.pyplot as plt
import random
from typing import Dict, Any, Optional, Tuple, Union
import logging

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


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StepTracker:
    """Manages consistent step tracking for logging and monitoring."""
    
    def __init__(self):
        self.global_step = 0
        self.epoch_step = 0
        self.current_epoch = 0
    
    def start_epoch(self, epoch: int):
        """Start a new epoch."""
        self.current_epoch = epoch
        self.epoch_step = 0
    
    def step(self) -> int:
        """Increment and return the global step."""
        self.global_step += 1
        self.epoch_step += 1
        return self.global_step
    
    def get_global_step(self) -> int:
        """Get current global step without incrementing."""
        return self.global_step
    
    def get_epoch_step(self) -> int:
        """Get current epoch step."""
        return self.epoch_step


class InputValidator:
    """Validates inputs and configurations."""
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        """Validate configuration dictionary."""
        required_keys = ['data_root']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
        
        # Validate data config
        data_config = config.get('data', {})
        if 'volume_size' in data_config:
            vol_size = data_config['volume_size']
            if not isinstance(vol_size, (list, tuple)) or len(vol_size) != 3:
                raise ValueError("volume_size must be a list/tuple of 3 integers")
            if not all(isinstance(x, int) and x > 0 for x in vol_size):
                raise ValueError("volume_size values must be positive integers")
        
        # Validate model config
        model_config = config.get('model', {})
        if 'in_channels' in model_config:
            if not isinstance(model_config['in_channels'], int) or model_config['in_channels'] <= 0:
                raise ValueError("in_channels must be a positive integer")
        
        if 'out_channels' in model_config:
            if not isinstance(model_config['out_channels'], int) or model_config['out_channels'] <= 0:
                raise ValueError("out_channels must be a positive integer")
        
        # Validate training config
        training_config = config.get('training', {})
        if 'learning_rate' in training_config:
            lr = training_config['learning_rate']
            if not isinstance(lr, (int, float)) or lr <= 0:
                raise ValueError("learning_rate must be a positive number")
        
        if 'epochs' in training_config:
            epochs = training_config['epochs']
            if not isinstance(epochs, int) or epochs <= 0:
                raise ValueError("epochs must be a positive integer")
    
    @staticmethod
    def validate_tensors(input_tensor: torch.Tensor, target_tensor: torch.Tensor, 
                        expected_input_shape: Tuple[int, ...], 
                        expected_target_shape: Tuple[int, ...]) -> None:
        """Validate tensor shapes and properties."""
        if not isinstance(input_tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor for input, got {type(input_tensor)}")
        
        if not isinstance(target_tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor for target, got {type(target_tensor)}")
        
        # Check shapes (ignore batch dimension)
        if len(input_tensor.shape) != len(expected_input_shape):
            raise ValueError(f"Expected input tensor with {len(expected_input_shape)} dimensions, "
                           f"got {len(input_tensor.shape)}")
        
        if len(target_tensor.shape) != len(expected_target_shape):
            raise ValueError(f"Expected target tensor with {len(expected_target_shape)} dimensions, "
                           f"got {len(target_tensor.shape)}")
        
        # Check for NaN or infinite values
        if torch.isnan(input_tensor).any():
            raise ValueError("Input tensor contains NaN values")
        
        if torch.isnan(target_tensor).any():
            raise ValueError("Target tensor contains NaN values")
        
        if torch.isinf(input_tensor).any():
            raise ValueError("Input tensor contains infinite values")
        
        if torch.isinf(target_tensor).any():
            raise ValueError("Target tensor contains infinite values")


class BaseTrainer:
    """Base trainer class with common functionality."""
    
    def __init__(self, config: Dict[str, Any], exp_name: str):
        self.config = config
        self.exp_name = exp_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Validate configuration
        try:
            InputValidator.validate_config(config)
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
        
        # Initialize step tracker
        self.step_tracker = StepTracker()
        
        # Setup directories
        self.exp_dir = Path('experiments') / exp_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        try:
            with open(self.exp_dir / 'config.yaml', 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            raise
        
        # Training state
        self.best_val_loss = float('inf')
        
        # Initialize components
        self._initialize_components()
        
        # Setup logging
        self._setup_logging()
        
        logger.info(f"Trainer initialized. Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _initialize_components(self):
        """Initialize model, loss, optimizer, scheduler, and data loaders."""
        try:
            self.model = self._create_model()
            self.criterion = self._create_loss()
            self.optimizer = self._create_optimizer()
            self.scheduler = self._create_scheduler()
            
            # Initialize data loaders
            self.train_loader = create_dataloader(self.config, phase='train')
            self.val_loader = create_dataloader(self.config, phase='val')
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _create_model(self) -> nn.Module:
        """Create and initialize the model."""
        try:
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
                    logger.info(f"Loaded pretrained weights from: {pretrained_path}")
                else:
                    logger.warning(f"Pretrained path not found: {pretrained_path}")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            raise
    
    def _create_loss(self) -> nn.Module:
        """Create loss function."""
        try:
            loss = create_loss(self.config)
            return loss.to(self.device)
        except Exception as e:
            logger.error(f"Failed to create loss function: {e}")
            raise
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        try:
            training_config = self.config.get('training', {})
            lr = float(training_config.get('learning_rate', 1e-4))
            weight_decay = float(training_config.get('weight_decay', 1e-4))
            optimizer_name = training_config.get('optimizer', 'AdamW')
            
            # Get parameter groups for transfer learning
            transfer_config = self.config.get('transfer_learning', {})
            if transfer_config.get('enabled', False):
                params = self.model.get_parameter_groups(lr)
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
            
        except Exception as e:
            logger.error(f"Failed to create optimizer: {e}")
            raise
    
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        try:
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
            
        except Exception as e:
            logger.error(f"Failed to create scheduler: {e}")
            raise
    
    def _setup_logging(self):
        """Setup logging with wandb if enabled."""
        try:
            logging_config = self.config.get('logging', {})
            if logging_config.get('use_wandb', False):
                project_name = os.getenv('WANDB_PROJECT', 
                                       logging_config.get('project_name', 'unetr-brats-synthesis'))
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
                
                # Log system info as summary (not step-based metrics)
                wandb.summary["system/gpu_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
                wandb.summary["system/model_parameters"] = sum(p.numel() for p in self.model.parameters())
                wandb.summary["system/trainable_parameters"] = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                wandb.summary["system/device"] = str(self.device)
                
                self.use_wandb = True
                logger.info(f"W&B logging enabled. Project: {project_name}")
            else:
                self.use_wandb = False
                
        except Exception as e:
            logger.error(f"Failed to setup logging: {e}")
            self.use_wandb = False
    
    def visualize_batch(self, num_samples: int = 2, phase: str = 'train'):
        """Visualize a batch of data (inputs and targets) for sanity check."""
        try:
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
                
        except Exception as e:
            logger.error(f"Failed to visualize batch: {e}")
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        try:
            checkpoint = {
                'epoch': self.step_tracker.current_epoch,
                'global_step': self.step_tracker.get_global_step(),
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
                logger.info(f"Saved best model at epoch {self.step_tracker.current_epoch}")
                
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def log_sample_predictions(self, num_samples: int = 2):
        """Log sample predictions to W&B for visualization."""
        if not self.use_wandb:
            return

        try:
            import numpy as np

            self.model.eval()
            loader = self.val_loader if hasattr(self, 'val_loader') and self.val_loader is not None else self.train_loader
            batch = next(iter(loader))
            inputs = batch['input'][:num_samples].to(self.device)
            targets = batch['target'][:num_samples].to(self.device)
            subject_names = batch.get('subject_name', [f'sample_{i}' for i in range(num_samples)])

            with torch.no_grad():
                outputs = self.model(inputs)

            # Convert tensors to numpy for visualization
            inputs_np = inputs.cpu().numpy()
            targets_np = targets.cpu().numpy()
            outputs_np = outputs.cpu().numpy()

            images = []
            for i in range(min(num_samples, inputs_np.shape[0])):
                input_img = inputs_np[i, 0]  # first channel
                target_img = targets_np[i, 0]
                output_img = outputs_np[i, 0]
                mid_slice = input_img.shape[-1] // 2

                input_slice = input_img[..., mid_slice]
                target_slice = target_img[..., mid_slice]
                output_slice = output_img[..., mid_slice]

                # Normalize to [0, 255] for wandb.Image
                def norm255(x):
                    x = x.astype(np.float32)
                    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
                    return (x * 255).astype(np.uint8)

                input_slice_norm = norm255(input_slice)
                output_slice_norm = norm255(output_slice)
                target_slice_norm = norm255(target_slice)

                # Log each as a separate grayscale image
                caption_base = subject_names[i] if isinstance(subject_names, list) else str(i)
                images.append(wandb.Image(input_slice_norm, caption=f"{caption_base} - input"))
                images.append(wandb.Image(output_slice_norm, caption=f"{caption_base} - output"))
                images.append(wandb.Image(target_slice_norm, caption=f"{caption_base} - ground truth"))

                # Also log a side-by-side comparison
                comparison = np.concatenate([input_slice_norm, output_slice_norm, target_slice_norm], axis=1)
                images.append(wandb.Image(comparison, caption=f"{caption_base} - input | output | ground truth"))

            # Log with consistent step tracking
            current_step = self.step_tracker.step()
            wandb.log({"sample_predictions": images}, step=current_step)
            self.model.train()

        except Exception as e:
            logger.error(f"Failed to log sample predictions: {e}")


class SynthesisTrainer(BaseTrainer):
    """Main trainer for UNETR synthesis model."""
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        loss_components = {}
        num_batches = 0
        
        try:
            for batch_idx, batch in enumerate(self.train_loader):
                try:
                    # Move data to device
                    inputs = batch['input'].to(self.device)
                    targets = batch['target'].to(self.device)
                    
                    # Validate tensor shapes
                    expected_input_shape = (inputs.shape[0], self.config.get('model', {}).get('in_channels', 4), *inputs.shape[2:])
                    expected_target_shape = (targets.shape[0], self.config.get('model', {}).get('out_channels', 1), *targets.shape[2:])
                    
                    InputValidator.validate_tensors(inputs, targets, expected_input_shape, expected_target_shape)
                    
                    # Forward pass
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    
                    # Validate outputs
                    if torch.isnan(outputs).any():
                        logger.warning(f"NaN detected in model outputs at batch {batch_idx}")
                        continue
                    
                    # Compute loss
                    losses = self.criterion(outputs, targets)
                    total_loss_batch = losses['total']
                    
                    # Check for NaN loss
                    if torch.isnan(total_loss_batch):
                        logger.warning(f"NaN loss detected at batch {batch_idx}, skipping")
                        continue
                    
                    # Backward pass
                    total_loss_batch.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
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
                        logger.info(f"Epoch {self.step_tracker.current_epoch}, "
                                  f"Batch {batch_idx}/{len(self.train_loader)}, "
                                  f"Loss: {total_loss_batch.item():.6f}")
                        
                        # Log to W&B during training
                        if self.use_wandb:
                            current_step = self.step_tracker.step()
                            wandb.log({
                                'batch/train_loss': total_loss_batch.item(),
                                'batch/epoch': self.step_tracker.current_epoch,
                                'batch/learning_rate': self.optimizer.param_groups[0]['lr'],
                            }, step=current_step)
                
                except Exception as e:
                    logger.error(f"Error in batch {batch_idx}: {e}")
                    continue
            
            # Average losses
            if num_batches > 0:
                avg_loss = total_loss / num_batches
                for key in loss_components:
                    loss_components[key] /= num_batches
            else:
                avg_loss = float('inf')
                logger.warning("No valid batches processed in epoch")
            
            return {'total_loss': avg_loss, **loss_components}
            
        except Exception as e:
            logger.error(f"Fatal error in train_epoch: {e}")
            logger.error(traceback.format_exc())
            return {'total_loss': float('inf')}
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        loss_components = {}
        num_batches = 0
        
        try:
            with torch.no_grad():
                for batch_idx, batch in enumerate(self.val_loader):
                    try:
                        # Move data to device
                        inputs = batch['input'].to(self.device)
                        targets = batch['target'].to(self.device)
                        
                        # Validate tensor shapes
                        expected_input_shape = (inputs.shape[0], self.config.get('model', {}).get('in_channels', 4), *inputs.shape[2:])
                        expected_target_shape = (targets.shape[0], self.config.get('model', {}).get('out_channels', 1), *targets.shape[2:])
                        
                        InputValidator.validate_tensors(inputs, targets, expected_input_shape, expected_target_shape)
                        
                        # Forward pass
                        outputs = self.model(inputs)
                        
                        # Validate outputs
                        if torch.isnan(outputs).any():
                            logger.warning(f"NaN detected in validation outputs at batch {batch_idx}")
                            continue
                        
                        # Compute loss
                        losses = self.criterion(outputs, targets)
                        
                        # Check for NaN loss
                        if torch.isnan(losses['total']):
                            logger.warning(f"NaN validation loss detected at batch {batch_idx}")
                            continue
                        
                        # Accumulate losses
                        total_loss += losses['total'].item()
                        for key, value in losses.items():
                            if key not in loss_components:
                                loss_components[key] = 0.0
                            loss_components[key] += value.item()
                        
                        num_batches += 1
                        
                    except Exception as e:
                        logger.error(f"Error in validation batch {batch_idx}: {e}")
                        continue
            
            # Average losses
            if num_batches > 0:
                avg_loss = total_loss / num_batches
                for key in loss_components:
                    loss_components[key] /= num_batches
            else:
                avg_loss = float('inf')
                logger.warning("No valid validation batches processed")
            
            return {'total_loss': avg_loss, **loss_components}
            
        except Exception as e:
            logger.error(f"Fatal error in validate: {e}")
            logger.error(traceback.format_exc())
            return {'total_loss': float('inf')}
    
    def train(self):
        """Main training loop."""
        try:
            epochs = self.config.get('training', {}).get('epochs', 200)
            val_frequency = self.config.get('validation', {}).get('frequency', 10)
            early_stopping_patience = self.config.get('validation', {}).get('early_stopping_patience', 50)
            
            patience_counter = 0
            
            logger.info(f"Starting training for {epochs} epochs...")
            
            for epoch in range(epochs):
                try:
                    self.step_tracker.start_epoch(epoch)
                    
                    # Train
                    train_metrics = self.train_epoch()
                    
                    # Log training metrics every epoch
                    if self.use_wandb:
                        current_step = self.step_tracker.step()
                        wandb.log({
                            'epoch': epoch,
                            'train/epoch_loss': train_metrics['total_loss'],
                            'train/learning_rate': self.optimizer.param_groups[0]['lr']
                        }, step=current_step)
                    
                    logger.info(f"Epoch {epoch}/{epochs} - Train Loss: {train_metrics['total_loss']:.6f}")
                    
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
                        logger.info(f"Epoch {epoch}/{epochs}")
                        logger.info(f"  Train Loss: {train_metrics['total_loss']:.6f}")
                        logger.info(f"  Val Loss: {val_loss:.6f} (Best: {self.best_val_loss:.6f})")
                        
                        if self.use_wandb:
                            current_step = self.step_tracker.step()
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
                            
                            wandb.log(log_dict, step=current_step)
                        
                        # Log sample predictions every 20 epochs
                        if self.use_wandb and epoch % 20 == 0:
                            self.log_sample_predictions()
                        
                        # Early stopping
                        if patience_counter >= early_stopping_patience:
                            logger.info(f"Early stopping at epoch {epoch}")
                            break
                    
                    # Update scheduler
                    if self.scheduler is not None:
                        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            if epoch % val_frequency == 0:
                                self.scheduler.step(val_metrics['total_loss'])
                            else:
                                self.scheduler.step(train_metrics['total_loss'])
                        else:
                            self.scheduler.step()
                
                except Exception as e:
                    logger.error(f"Error in epoch {epoch}: {e}")
                    logger.error(traceback.format_exc())
                    continue
            
            logger.info("Training completed!")
            
        except Exception as e:
            logger.error(f"Fatal error in training loop: {e}")
            logger.error(traceback.format_exc())
            raise
        finally:
            if self.use_wandb:
                wandb.finish()


class SegmentationTrainer(BaseTrainer):
    """Trainer for segmentation sanity check."""
    
    def _create_model(self) -> nn.Module:
        """Create UNETR model for segmentation."""
        try:
            # Use UNETR backbone for segmentation (out_channels = num_classes)
            seg_config = self.config.copy()
            seg_config['model'] = seg_config.get('model', {}).copy()
            seg_config['model']['out_channels'] = 1  # binary seg for quick test
            
            from models.unetr_synthesis import create_model
            model = create_model(seg_config)
            return model.to(self.device)
            
        except Exception as e:
            logger.error(f"Failed to create segmentation model: {e}")
            raise
    
    def _create_loss(self) -> nn.Module:
        """Create Dice loss for segmentation."""
        try:
            from monai.losses import DiceLoss
            return DiceLoss(sigmoid=True, reduction='mean').to(self.device)
        except ImportError:
            logger.warning("MONAI not available, using BCEWithLogitsLoss")
            return nn.BCEWithLogitsLoss().to(self.device)
        except Exception as e:
            logger.error(f"Failed to create segmentation loss: {e}")
            raise
    
    def train_epoch(self) -> Dict[str, float]:
        """Train segmentation model for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        try:
            for batch_idx, batch in enumerate(self.train_loader):
                try:
                    inputs = batch['input'].to(self.device)
                    
                    # For seg sanity, use one input channel as image, one as mask
                    if inputs.shape[1] > 1:
                        image = inputs[:, 0:1]
                        mask = inputs[:, 1:2]
                    else:
                        image = inputs
                        mask = batch['target'].to(self.device)
                    
                    self.optimizer.zero_grad()
                    outputs = self.model(image)
                    loss = self.criterion(outputs, mask)
                    
                    if torch.isnan(loss):
                        logger.warning(f"NaN loss at batch {batch_idx}, skipping")
                        continue
                    
                    loss.backward()
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    if batch_idx == 0:
                        # Visualize prediction vs mask
                        try:
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
                        except Exception as e:
                            logger.warning(f"Failed to visualize predictions: {e}")
                
                except Exception as e:
                    logger.error(f"Error in segmentation batch {batch_idx}: {e}")
                    continue
            
            avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
            return {'total_loss': avg_loss}
            
        except Exception as e:
            logger.error(f"Fatal error in segmentation train_epoch: {e}")
            return {'total_loss': float('inf')}


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        raise


def main():
    try:
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
        parser.add_argument('--sanity_check', action='store_true',
                           help='Run segmentation sanity check instead of synthesis training')
        
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
        
        # Choose trainer based on arguments
        if args.sanity_check:
            trainer = SegmentationTrainer(config, args.exp_name)
            logger.info("Running segmentation sanity check...")
        else:
            trainer = SynthesisTrainer(config, args.exp_name)
            logger.info("Running synthesis training...")
        
        # Visualize a batch before training
        logger.info("Visualizing a batch for sanity check...")
        trainer.visualize_batch(num_samples=2, phase='train')
        
        # Start training
        trainer.train()
        
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()