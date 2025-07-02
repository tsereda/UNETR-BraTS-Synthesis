"""
UNETR-based Brain MRI Synthesis Model

This module implements a UNETR-based architecture for brain MRI modality synthesis,
specifically designed for the BraTS dataset. It leverages MONAI's UNETR implementation
with custom modifications for synthesis tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from monai.networks.nets import UNETR
from monai.networks.layers import Norm


class UNETR_Synthesis(nn.Module):
    """
    UNETR-based model for brain MRI modality synthesis.
    
    This model adapts MONAI's UNETR architecture for the task of synthesizing
    missing brain MRI modalities from available ones. The architecture consists of:
    1. Vision Transformer encoder for global feature extraction
    2. U-Net decoder with skip connections for fine-grained details
    3. Custom synthesis head for final modality generation
    
    Args:
        config: Configuration dictionary containing model parameters
        img_size: Input image size (H, W, D)
        in_channels: Number of input channels (4 for BraTS modalities)
        out_channels: Number of output channels (1 for target modality)
        hidden_size: Hidden dimension size for transformer
        mlp_dim: MLP dimension in transformer blocks
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        feature_size: Feature size for decoder
        dropout_rate: Dropout rate
        norm_name: Normalization layer name
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        img_size: Tuple[int, int, int] = (96, 96, 96),
        in_channels: int = 4,
        out_channels: int = 1,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        num_layers: int = 12,  # This parameter will be ignored for MONAI UNETR
        feature_size: int = 16,
        dropout_rate: float = 0.1,
        norm_name: str = "instance",
    ):
        super().__init__()
        
        self.config = config
        self.img_size = img_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Core UNETR model from MONAI
        # Note: MONAI UNETR has fixed 12 transformer layers
        self.unetr = UNETR(
            in_channels=in_channels,
            out_channels=feature_size,  # Use feature_size for intermediate features
            img_size=img_size,
            feature_size=feature_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            # num_layers parameter is not supported in MONAI UNETR
            dropout_rate=dropout_rate,
            norm_name=norm_name,
        )
        
        # Custom synthesis head for final modality generation
        self.synthesis_head = nn.Sequential(
            nn.Conv3d(feature_size, feature_size // 2, kernel_size=3, padding=1),
            Norm[norm_name, 3](feature_size // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(feature_size // 2, feature_size // 4, kernel_size=3, padding=1),
            Norm[norm_name, 3](feature_size // 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(feature_size // 4, out_channels, kernel_size=1),
            nn.Tanh()  # Output in [-1, 1] range for medical images
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize model weights."""
        for module in self.synthesis_head.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the synthesis model.
        
        Args:
            x: Input tensor of shape (B, 4, H, W, D) where the target modality
               channel is zeroed out.
        
        Returns:
            Synthetic modality tensor of shape (B, 1, H, W, D)
        """
        # Extract features using UNETR
        features = self.unetr(x)
        
        # Generate synthetic modality
        synthetic = self.synthesis_head(features)
        
        return synthetic
    
    def load_pretrained_weights(
        self, 
        pretrained_path: str, 
        freeze_layers: Optional[int] = None,
        strict: bool = False
    ) -> None:
        """
        Load pre-trained weights and optionally freeze layers.
        
        Args:
            pretrained_path: Path to pre-trained model weights
            freeze_layers: Number of encoder layers to freeze (None for no freezing)
            strict: Whether to strictly enforce key matching
        """
        print(f"Loading pre-trained weights from: {pretrained_path}")
        
        # Load pretrained state dict
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Filter out synthesis head weights (only load UNETR backbone)
        unetr_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('unetr.'):
                unetr_key = key[6:]  # Remove 'unetr.' prefix
                unetr_state_dict[unetr_key] = value
            elif not key.startswith('synthesis_head'):
                # Direct UNETR weights without prefix
                unetr_state_dict[key] = value
        
        # Load UNETR weights
        missing_keys, unexpected_keys = self.unetr.load_state_dict(
            unetr_state_dict, strict=strict
        )
        
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
        
        # Freeze encoder layers if specified
        if freeze_layers is not None:
            self._freeze_layers(freeze_layers)
    
    def _freeze_layers(self, num_layers: int) -> None:
        """
        Freeze the first num_layers of the transformer encoder.
        
        Args:
            num_layers: Number of transformer layers to freeze
        """
        print(f"Freezing first {num_layers} transformer layers")
        
        # Freeze patch embedding
        for param in self.unetr.vit.patch_embedding.parameters():
            param.requires_grad = False
        
        # Freeze specified transformer blocks
        for i in range(min(num_layers, len(self.unetr.vit.blocks))):
            for param in self.unetr.vit.blocks[i].parameters():
                param.requires_grad = False
    
    def get_parameter_groups(self, base_lr: float = 1e-4) -> list:
        """
        Get parameter groups with different learning rates for transfer learning.
        
        Args:
            base_lr: Base learning rate
            
        Returns:
            List of parameter groups for optimizer
        """
        # Define learning rate multipliers
        lr_multipliers = self.config.get('transfer_learning', {}).get('lr_multipliers', {})
        
        encoder_lr = base_lr * lr_multipliers.get('encoder', 0.1)
        decoder_lr = base_lr * lr_multipliers.get('decoder', 1.0)
        head_lr = base_lr * lr_multipliers.get('synthesis_head', 1.0)
        
        parameter_groups = [
            {
                'params': self.unetr.vit.parameters(),
                'lr': encoder_lr,
                'name': 'encoder'
            },
            {
                'params': [
                    p for name, p in self.unetr.named_parameters() 
                    if not name.startswith('vit')
                ],
                'lr': decoder_lr,
                'name': 'decoder'
            },
            {
                'params': self.synthesis_head.parameters(),
                'lr': head_lr,
                'name': 'synthesis_head'
            }
        ]
        
        return parameter_groups


def create_model(config: Dict[str, Any]) -> UNETR_Synthesis:
    """
    Factory function to create UNETR synthesis model from config.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Initialized UNETR_Synthesis model
    """
    model_config = config.get('model', {})
    
    return UNETR_Synthesis(
        config=config,
        img_size=model_config.get('img_size', [96, 96, 96]),
        in_channels=model_config.get('in_channels', 4),
        out_channels=model_config.get('out_channels', 1),
        hidden_size=model_config.get('hidden_size', 768),
        mlp_dim=model_config.get('mlp_dim', 3072),
        num_heads=model_config.get('num_heads', 12),
        num_layers=model_config.get('num_layers', 12),
        feature_size=model_config.get('feature_size', 16),
        dropout_rate=model_config.get('dropout_rate', 0.1),
        norm_name=model_config.get('norm_name', 'instance'),
    )


if __name__ == "__main__":
    # Test model creation
    config = {
        'model': {
            'img_size': [96, 96, 96],
            'in_channels': 4,
            'out_channels': 1,
            'hidden_size': 768,
            'mlp_dim': 3072,
            'num_heads': 12,
            'num_layers': 12,
            'feature_size': 16,
            'dropout_rate': 0.1,
            'norm_name': 'instance'
        }
    }
    
    model = create_model(config)
    
    # Test forward pass
    x = torch.randn(1, 4, 96, 96, 96)
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
