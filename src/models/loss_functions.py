"""
Loss Functions for Brain MRI Synthesis

This module implements various loss functions optimized for medical image synthesis,
including competition-specific metrics for BraTS evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import numpy as np
from monai.losses import SSIMLoss, DiceLoss
from monai.metrics import SSIMMetric


class CombinedSynthesisLoss(nn.Module):
    """
    Combined loss function for brain MRI synthesis that optimizes multiple metrics
    relevant to the BraTS synthesis challenge.
    """
    
    def __init__(
        self,
        mse_weight: float = 1.0,
        ssim_weight: float = 0.2,
        perceptual_weight: float = 0.1,
        l1_weight: float = 0.1,
        spatial_dims: int = 3,
        data_range: float = 2.0,  # [-1, 1] range
    ):
        super().__init__()
        
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
        self.perceptual_weight = perceptual_weight
        self.l1_weight = l1_weight
        
        # MSE Loss
        self.mse_loss = nn.MSELoss()
        
        # L1 Loss
        self.l1_loss = nn.L1Loss()
        
        # SSIM Loss
        self.ssim_loss = SSIMLoss(
            spatial_dims=spatial_dims,
            data_range=data_range,
            win_size=11,
            k1=0.01,
            k2=0.03,
        )
        
        # Perceptual loss using simple feature maps
        self.perceptual_loss = PerceptualLoss3D()
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            pred: Predicted tensor of shape (B, 1, H, W, D)
            target: Target tensor of shape (B, 1, H, W, D)
            
        Returns:
            Dictionary containing individual losses and total loss
        """
        losses = {}
        
        # MSE Loss
        if self.mse_weight > 0:
            losses['mse'] = self.mse_loss(pred, target)
        
        # L1 Loss
        if self.l1_weight > 0:
            losses['l1'] = self.l1_loss(pred, target)
        
        # SSIM Loss
        if self.ssim_weight > 0:
            losses['ssim'] = self.ssim_loss(pred, target)
        
        # Perceptual Loss
        if self.perceptual_weight > 0:
            losses['perceptual'] = self.perceptual_loss(pred, target)
        
        # Total loss
        total_loss = 0
        if 'mse' in losses:
            total_loss += self.mse_weight * losses['mse']
        if 'l1' in losses:
            total_loss += self.l1_weight * losses['l1']
        if 'ssim' in losses:
            total_loss += self.ssim_weight * losses['ssim']
        if 'perceptual' in losses:
            total_loss += self.perceptual_weight * losses['perceptual']
        
        losses['total'] = total_loss
        
        return losses


class PerceptualLoss3D(nn.Module):
    """
    3D Perceptual loss using simple feature extraction network.
    Inspired by VGG-based perceptual loss but adapted for 3D medical images.
    """
    
    def __init__(self, feature_layers: list = [2, 4, 6]):
        super().__init__()
        
        self.feature_layers = feature_layers
        
        # Simple 3D feature extractor
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),  # Layer 2
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),  # Layer 4
            nn.Conv3d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),  # Layer 6
        )
        
        # Freeze feature extractor
        for param in self.features.parameters():
            param.requires_grad = False
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss.
        
        Args:
            pred: Predicted tensor
            target: Target tensor
            
        Returns:
            Perceptual loss value
        """
        pred_features = self._extract_features(pred)
        target_features = self._extract_features(target)
        
        loss = 0
        for pred_feat, target_feat in zip(pred_features, target_features):
            loss += F.mse_loss(pred_feat, target_feat)
        
        return loss / len(pred_features)
    
    def _extract_features(self, x: torch.Tensor) -> list:
        """Extract features from specified layers."""
        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.feature_layers:
                features.append(x)
        return features


class GradientLoss(nn.Module):
    """
    Gradient loss to preserve edge information in synthesized images.
    """
    
    def __init__(self, penalty: str = 'l1'):
        super().__init__()
        self.penalty = penalty
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient loss.
        
        Args:
            pred: Predicted tensor
            target: Target tensor
            
        Returns:
            Gradient loss value
        """
        # Compute gradients
        pred_grad = self._compute_gradient(pred)
        target_grad = self._compute_gradient(target)
        
        # Compute loss
        if self.penalty == 'l1':
            return F.l1_loss(pred_grad, target_grad)
        elif self.penalty == 'l2':
            return F.mse_loss(pred_grad, target_grad)
        else:
            raise ValueError(f"Unknown penalty: {self.penalty}")
    
    def _compute_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """Compute 3D gradient magnitude."""
        # Sobel-like filters for 3D
        grad_x = F.conv3d(x, self._sobel_kernel_x(), padding=1)
        grad_y = F.conv3d(x, self._sobel_kernel_y(), padding=1)
        grad_z = F.conv3d(x, self._sobel_kernel_z(), padding=1)
        
        # Gradient magnitude
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2 + 1e-8)
        return grad_mag
    
    def _sobel_kernel_x(self) -> torch.Tensor:
        """3D Sobel kernel for x-direction."""
        kernel = torch.tensor([
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        ], dtype=torch.float32)
        return kernel.unsqueeze(0).unsqueeze(0)
    
    def _sobel_kernel_y(self) -> torch.Tensor:
        """3D Sobel kernel for y-direction."""
        kernel = torch.tensor([
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            [[-2, -4, -2], [0, 0, 0], [2, 4, 2]],
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        ], dtype=torch.float32)
        return kernel.unsqueeze(0).unsqueeze(0)
    
    def _sobel_kernel_z(self) -> torch.Tensor:
        """3D Sobel kernel for z-direction."""
        kernel = torch.tensor([
            [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
        ], dtype=torch.float32)
        return kernel.unsqueeze(0).unsqueeze(0)


# Registry for loss functions
LOSS_REGISTRY = {
    'combined': CombinedSynthesisLoss,
    'mse': nn.MSELoss,
    'l1': nn.L1Loss,
    'ssim': SSIMLoss,
    'perceptual': PerceptualLoss3D,
    'gradient': GradientLoss,
}


def register_loss(name: str):
    """Decorator to register custom loss functions."""
    def decorator(loss_class):
        LOSS_REGISTRY[name] = loss_class
        return loss_class
    return decorator


def create_loss(config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to create loss function from config.
    
    Args:
        config: Loss configuration dictionary
        
    Returns:
        Initialized loss function
    """
    loss_config = config.get('loss', {})
    loss_name = loss_config.get('name', 'combined')
    
    if loss_name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss function: {loss_name}")
    
    loss_class = LOSS_REGISTRY[loss_name]
    
    # Remove 'name' from config before passing to loss class
    loss_params = {k: v for k, v in loss_config.items() if k != 'name'}
    
    return loss_class(**loss_params)


if __name__ == "__main__":
    # Test loss functions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test combined loss
    loss_fn = CombinedSynthesisLoss()
    loss_fn = loss_fn.to(device)
    
    # Create test tensors
    pred = torch.randn(2, 1, 32, 32, 32, device=device)
    target = torch.randn(2, 1, 32, 32, 32, device=device)
    
    # Compute loss
    losses = loss_fn(pred, target)
    
    print("Loss components:")
    for name, value in losses.items():
        print(f"  {name}: {value.item():.6f}")
