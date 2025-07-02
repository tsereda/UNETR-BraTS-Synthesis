"""
BraTS Dataset Implementation

This module implements the dataset class for loading and preprocessing BraTS data
for brain MRI modality synthesis.
"""

import os
import glob
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Callable
import random
from pathlib import Path

try:
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, Orientationd, 
        Spacingd, ScaleIntensityRanged, RandSpatialCropd,
        RandFlipd, RandRotated, RandGaussianNoised,
        RandAdjustContrastd, ToTensord, EnsureTyped
    )
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False
    print("MONAI not available. Install with: pip install monai[all]")


class BraTSDataset(Dataset):
    """
    BraTS dataset for brain MRI modality synthesis.
    
    This dataset loads BraTS data and prepares it for synthesis tasks where
    one modality is used as target and the others as input (with target zeroed).
    """
    
    def __init__(
        self,
        data_root: str,
        modalities: List[str] = ['t1n', 't1c', 't2w', 't2f'],
        target_modality: Optional[str] = None,
        transforms: Optional[Callable] = None,
        cache_rate: float = 0.0,
        num_workers: int = 0,
        phase: str = 'train',
        volume_size: Tuple[int, int, int] = (96, 96, 96),
    ):
        """
        Initialize BraTS dataset.
        
        Args:
            data_root: Root directory containing BraTS data
            modalities: List of modality names to load
            target_modality: Target modality for synthesis (None for random)
            transforms: Data transforms to apply
            cache_rate: Fraction of data to cache in memory
            num_workers: Number of workers for data loading
            phase: Dataset phase ('train', 'val', 'test')
            volume_size: Target volume size for cropping/padding
        """
        self.data_root = Path(data_root)
        self.modalities = modalities
        self.target_modality = target_modality
        self.phase = phase
        self.volume_size = volume_size
        
        # Find all subject directories
        self.subjects = self._find_subjects()
        print(f"Found {len(self.subjects)} subjects in {phase} phase")
        
        # Setup transforms
        if transforms is None and MONAI_AVAILABLE:
            self.transforms = self._get_default_transforms()
        else:
            self.transforms = transforms
        
        # Validate data
        self._validate_data()
    
    def _find_subjects(self) -> List[str]:
        """Find all subject directories in the data root."""
        pattern = os.path.join(self.data_root, "BraTS-*")
        subject_dirs = glob.glob(pattern)
        subject_dirs = [d for d in subject_dirs if os.path.isdir(d)]
        subject_dirs.sort()
        
        # Split data based on phase
        if self.phase == 'train':
            return subject_dirs[:int(0.8 * len(subject_dirs))]
        elif self.phase == 'val':
            return subject_dirs[int(0.8 * len(subject_dirs)):int(0.9 * len(subject_dirs))]
        else:  # test
            return subject_dirs[int(0.9 * len(subject_dirs)):]
    
    def _get_default_transforms(self):
        """Get default MONAI transforms for preprocessing."""
        if not MONAI_AVAILABLE:
            return None
        
        # Keys for each modality
        keys = [f"{mod}_image" for mod in self.modalities]
        
        transforms = [
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            Orientationd(keys=keys, axcodes="RAS"),
            Spacingd(keys=keys, pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            ScaleIntensityRanged(
                keys=keys,
                a_min=-1000, a_max=1000,
                b_min=-1.0, b_max=1.0,
                clip=True
            ),
        ]
        
        # Add augmentations for training
        if self.phase == 'train':
            transforms.extend([
                RandSpatialCropd(keys=keys, roi_size=self.volume_size, random_size=False),
                RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
                RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
                RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
                RandRotated(keys=keys, prob=0.2, range_x=0.2, range_y=0.2, range_z=0.2),
                RandGaussianNoised(keys=keys, prob=0.1, std=0.01),
                RandAdjustContrastd(keys=keys, prob=0.2, gamma=(0.8, 1.2)),
            ])
        
        transforms.extend([
            ToTensord(keys=keys),
            EnsureTyped(keys=keys),
        ])
        
        return Compose(transforms)
    
    def _validate_data(self):
        """Validate that all required files exist."""
        missing_files = []
        
        for subject_dir in self.subjects[:5]:  # Check first 5 subjects
            subject_name = os.path.basename(subject_dir)
            
            for modality in self.modalities:
                file_path = os.path.join(
                    subject_dir, 
                    f"{subject_name}-{modality}.nii.gz"
                )
                if not os.path.exists(file_path):
                    missing_files.append(file_path)
        
        if missing_files:
            print(f"Warning: Missing files detected:")
            for file_path in missing_files[:10]:  # Show first 10
                print(f"  {file_path}")
            if len(missing_files) > 10:
                print(f"  ... and {len(missing_files) - 10} more")
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.subjects)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a data sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing input and target tensors
        """
        subject_dir = self.subjects[idx]
        subject_name = os.path.basename(subject_dir)
        
        # Choose target modality
        if self.target_modality is None:
            target_mod = random.choice(self.modalities)
        else:
            target_mod = self.target_modality
        
        # Load all modalities
        data_dict = {}
        for modality in self.modalities:
            file_path = os.path.join(
                subject_dir,
                f"{subject_name}-{modality}.nii.gz"
            )
            
            if os.path.exists(file_path):
                data_dict[f"{modality}_image"] = file_path
            else:
                # Handle missing files by creating dummy path
                print(f"Warning: Missing file {file_path}")
                data_dict[f"{modality}_image"] = None
        
        # Apply transforms if available
        if self.transforms is not None:
            # Filter out None values
            data_dict = {k: v for k, v in data_dict.items() if v is not None}
            try:
                data_dict = self.transforms(data_dict)
            except Exception as e:
                print(f"Transform error for {subject_name}: {e}")
                return self._get_dummy_sample(target_mod)
        else:
            # Load data manually without MONAI
            data_dict = self._load_data_manually(data_dict)
        
        # Prepare input and target
        input_channels = []
        target = None
        
        for modality in self.modalities:
            key = f"{modality}_image"
            if key in data_dict:
                volume = data_dict[key]
                if isinstance(volume, torch.Tensor):
                    volume = volume.squeeze(0)  # Remove channel dim if present
                
                if modality == target_mod:
                    target = volume.unsqueeze(0)  # Add channel dim
                    input_channels.append(torch.zeros_like(volume))  # Zero out target
                else:
                    input_channels.append(volume)
            else:
                # Handle missing modality
                dummy_volume = torch.zeros(self.volume_size)
                input_channels.append(dummy_volume)
                if modality == target_mod:
                    target = dummy_volume.unsqueeze(0)
        
        # Stack input channels
        input_tensor = torch.stack(input_channels, dim=0)  # (4, H, W, D)
        
        if target is None:
            target = torch.zeros(1, *self.volume_size)
        
        return {
            'input': input_tensor,
            'target': target,
            'target_modality': target_mod,
            'subject_name': subject_name
        }
    
    def _load_data_manually(self, data_dict: Dict[str, str]) -> Dict[str, torch.Tensor]:
        """Load data manually without MONAI transforms."""
        loaded_data = {}
        
        for key, file_path in data_dict.items():
            if file_path is None:
                continue
                
            try:
                # Load NIfTI file
                img = nib.load(file_path)
                data = img.get_fdata()
                
                # Basic preprocessing
                data = self._preprocess_volume(data)
                
                # Convert to tensor
                loaded_data[key] = torch.from_numpy(data).float()
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        return loaded_data
    
    def _preprocess_volume(self, volume: np.ndarray) -> np.ndarray:
        """Basic preprocessing without MONAI."""
        # Normalize to [-1, 1]
        volume = np.clip(volume, -1000, 1000)
        volume = 2.0 * (volume - (-1000)) / (1000 - (-1000)) - 1.0
        
        # Resize/crop to target size
        volume = self._resize_volume(volume, self.volume_size)
        
        return volume
    
    def _resize_volume(self, volume: np.ndarray, target_size: Tuple[int, int, int]) -> np.ndarray:
        """Resize volume to target size using center crop/pad."""
        current_size = volume.shape
        
        # Calculate crop/pad amounts
        crops = []
        pads = []
        
        for i in range(3):
            if current_size[i] > target_size[i]:
                # Need to crop
                crop_amount = current_size[i] - target_size[i]
                crop_start = crop_amount // 2
                crop_end = crop_start + target_size[i]
                crops.append((crop_start, crop_end))
                pads.append((0, 0))
            else:
                # Need to pad
                pad_amount = target_size[i] - current_size[i]
                pad_before = pad_amount // 2
                pad_after = pad_amount - pad_before
                pads.append((pad_before, pad_after))
                crops.append((0, current_size[i]))
        
        # Apply crops
        volume = volume[crops[0][0]:crops[0][1], 
                       crops[1][0]:crops[1][1], 
                       crops[2][0]:crops[2][1]]
        
        # Apply pads
        volume = np.pad(volume, pads, mode='constant', constant_values=0)
        
        return volume
    
    def _get_dummy_sample(self, target_modality: str) -> Dict[str, torch.Tensor]:
        """Get a dummy sample in case of errors."""
        return {
            'input': torch.zeros(4, *self.volume_size),
            'target': torch.zeros(1, *self.volume_size),
            'target_modality': target_modality,
            'subject_name': 'dummy'
        }


def create_dataloader(
    config: Dict,
    phase: str = 'train'
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for BraTS dataset.
    
    Args:
        config: Configuration dictionary
        phase: Dataset phase ('train', 'val', 'test')
        
    Returns:
        DataLoader instance
    """
    data_config = config.get('data', {})
    
    dataset = BraTSDataset(
        data_root=config['data_root'],
        modalities=data_config.get('modalities', ['t1n', 't1c', 't2w', 't2f']),
        target_modality=data_config.get('target_modality', None),
        phase=phase,
        volume_size=data_config.get('volume_size', (96, 96, 96)),
    )
    
    # DataLoader parameters
    batch_size = data_config.get('batch_size', 2)
    num_workers = data_config.get('num_workers', 4)
    
    # Adjust for different phases
    if phase == 'train':
        shuffle = True
        drop_last = True
    else:
        shuffle = False
        drop_last = False
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available(),
    )
    
    return dataloader


if __name__ == "__main__":
    # Test dataset
    config = {
        'data_root': '/path/to/brats/data',
        'data': {
            'modalities': ['t1n', 't1c', 't2w', 't2f'],
            'volume_size': (96, 96, 96),
            'batch_size': 2,
            'num_workers': 0,
        }
    }
    
    # Create dataset
    dataset = BraTSDataset(
        data_root=config['data_root'],
        phase='train',
        volume_size=config['data']['volume_size']
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading a sample
    if len(dataset) > 0:
        sample = dataset[1]
        print(f"Input shape: {sample['input'].shape}")
        print(f"Target shape: {sample['target'].shape}")
        print(f"Target modality: {sample['target_modality']}")
        print(f"Subject name: {sample['subject_name']}")
