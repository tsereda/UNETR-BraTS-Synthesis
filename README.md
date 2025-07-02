# UNETR-BraTS-Synthesis

**Vision Transformer-based Medical Image Synthesis with Transfer Learning for BraTS 2025**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![MONAI](https://img.shields.io/badge/MONAI-1.3+-green.svg)](https://monai.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A state-of-the-art approach for brain MRI modality synthesis using UNETR (UNEt TRansformer) with advanced transfer learning strategies. Designed for the BraTS Syn 2025 challenge and clinical deployment.

## 🧠 Overview

This project implements a deterministic reconstruction approach for medical image synthesis, leveraging:

- **UNETR Architecture**: Vision Transformer encoder + U-Net decoder for global-local feature fusion
- **Transfer Learning**: Medical imaging pre-trained models for immediate performance boost
- **Multi-Loss Optimization**: Direct optimization of competition metrics (SSIM, MSE, perceptual)
- **Clinical Focus**: Designed for downstream segmentation task performance

### Key Features

- ✅ **4→1 Unified Architecture**: Single model synthesizes any missing BraTS modality
- ✅ **Transfer Learning Ready**: Supports multiple pre-training strategies
- ✅ **Direct Metric Optimization**: Loss functions aligned with BraTS evaluation
- ✅ **Clinical Deployment**: Fast inference, deterministic outputs
- ✅ **Ensemble Support**: Framework for combining multiple models

## 🏗️ Architecture

```
Input: [B, 4, H, W, D] → UNETR Encoder → Global Features → UNETR Decoder → Synthesis Head → Output: [B, 1, H, W, D]
        ↑                     ↓                            ↑                    ↓
    4 BraTS Modalities    Vision Transformer         U-Net Decoder       Target Modality
    (target zeroed)       (12 layers, 768d)         (Skip Connections)   (synthesized)
```

### Model Components

1. **Vision Transformer Encoder**
   - 12 transformer blocks with 768 hidden dimensions
   - 12 attention heads, 3072 MLP dimension
   - Learns global anatomical relationships

2. **U-Net Decoder**
   - Multi-scale feature fusion
   - Skip connections for fine-grained details
   - Progressive upsampling to full resolution

3. **Synthesis Head**
   - Lightweight CNN for final refinement
   - Tanh activation for [-1, 1] output range
   - Optimized for medical image characteristics

## 📋 Requirements

### System Requirements
- **GPU**: 16GB+ VRAM recommended (12GB minimum)
- **RAM**: 32GB+ system memory
- **Storage**: 100GB+ for BraTS dataset and experiments
- **OS**: Linux (Ubuntu 20.04+), macOS, Windows 10+

### Dependencies

```bash
# Core dependencies
torch>=2.0.0
torchvision>=0.15.0
monai[all]>=1.3.0
nibabel>=5.1.0
numpy>=1.24.0
scipy>=1.11.0

# Training and evaluation
wandb>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
PyYAML>=6.0.1

# Medical imaging
SimpleITK>=2.3.0
scikit-image>=0.21.0
medpy>=0.4.0
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/yourusername/UNETR-BraTS-Synthesis.git
cd UNETR-BraTS-Synthesis

# Create conda environment
conda create -n unetr-brats python=3.9
conda activate unetr-brats

# Install dependencies
pip install -r requirements.txt

# Install MONAI with all extras
pip install 'monai[all]'
```

### 2. Data Preparation

```bash
# Download BraTS 2023/2024 data
# Organize as follows:
data/
├── BraTS-Training/
│   ├── BraTS-GLI-00000-000/
│   │   ├── BraTS-GLI-00000-000-t1n.nii.gz
│   │   ├── BraTS-GLI-00000-000-t1c.nii.gz
│   │   ├── BraTS-GLI-00000-000-t2w.nii.gz
│   │   ├── BraTS-GLI-00000-000-t2f.nii.gz
│   │   └── BraTS-GLI-00000-000-seg.nii.gz
│   └── ...
└── BraTS-Validation/
    └── ...

# Validate data structure
python scripts/validate_data.py --data_root data/BraTS-Training
```

### 3. Download Pre-trained Models

```bash
# Download BraTS segmentation pre-trained UNETR
python scripts/download_pretrained.py --model brats_segmentation

# Or download from MONAI Model Zoo
python -c "
from monai.bundle import download
download(name='brats_mri_segmentation', bundle_dir='./pretrained/')
"
```

### 4. Training

```bash
# Quick training (default config)
python train.py \
    --data_root data/BraTS-Training \
    --config configs/unetr_base.yaml \
    --exp_name brats_synthesis_v1

# Advanced training with transfer learning
python train.py \
    --data_root data/BraTS-Training \
    --config configs/unetr_transfer.yaml \
    --pretrained_path pretrained/brats_segmentation.pth \
    --freeze_layers 6 \
    --use_wandb \
    --exp_name brats_transfer_v1
```

### 5. Inference

```bash
# Synthesize missing modalities
python inference.py \
    --checkpoint experiments/brats_synthesis_v1/best_model.pth \
    --input_dir data/BraTS-Test \
    --output_dir results/ \
    --target_modality t1c

# Batch inference for competition
python scripts/competition_inference.py \
    --checkpoint experiments/brats_synthesis_v1/best_model.pth \
    --test_dir data/BraTS-Competition \
    --output_dir submission/
```

## 📊 Configuration

### Base Configuration (`configs/unetr_base.yaml`)

```yaml
model:
  name: "UNETR_Synthesis"
  img_size: [96, 96, 96]
  in_channels: 4
  out_channels: 1
  hidden_size: 768
  mlp_dim: 3072
  num_heads: 12
  num_layers: 12
  feature_size: 16
  norm_name: "instance"
  dropout_rate: 0.1

data:
  volume_size: [96, 96, 96]
  num_workers: 4
  cache_rate: 0.1
  
training:
  batch_size: 2
  learning_rate: 1e-4
  epochs: 200
  optimizer: "AdamW"
  weight_decay: 1e-4
  scheduler: "cosine"
  
loss:
  mse_weight: 1.0
  ssim_weight: 0.2
  perceptual_weight: 0.1
  
transfer_learning:
  enabled: true
  freeze_layers: 6
  lr_multipliers:
    encoder: 0.1
    decoder: 1.0
    head: 10.0
```

### Transfer Learning Configuration (`configs/unetr_transfer.yaml`)

```yaml
# Inherits from base config
extends: "unetr_base.yaml"

transfer_learning:
  enabled: true
  pretrained_path: "pretrained/brats_segmentation.pth"
  freeze_strategy: "gradual"  # "none", "partial", "gradual"
  freeze_layers: 8
  unfreeze_schedule:
    epoch_10: 6
    epoch_20: 4  
    epoch_30: 0
  
  # Different learning rates for different components
  lr_multipliers:
    frozen_layers: 0.0
    encoder: 0.01
    decoder: 0.1
    synthesis_head: 1.0

validation:
  frequency: 10
  metrics: ["mse", "ssim", "psnr", "lpips"]
  save_best: true
  early_stopping_patience: 50
```

## 🧪 Training Strategies

### 1. Baseline Training

```bash
# Train from scratch
python train.py \
    --config configs/unetr_base.yaml \
    --data_root /path/to/brats \
    --exp_name baseline_scratch
```

### 2. Transfer Learning

```bash
# Transfer from segmentation model
python train.py \
    --config configs/unetr_transfer.yaml \
    --pretrained_path pretrained/brats_segmentation.pth \
    --exp_name transfer_seg

# Transfer from ImageNet ViT
python train.py \
    --config configs/unetr_imagenet.yaml \
    --pretrained_path pretrained/vit_base_patch16_224.pth \
    --exp_name transfer_imagenet
```

### 3. Multi-Loss Optimization

```bash
# Optimize directly for competition metrics
python train.py \
    --config configs/unetr_competition.yaml \
    --loss_config "mse:1.0,ssim:0.5,perceptual:0.2" \
    --exp_name competition_metrics
```

### 4. Ensemble Training

```bash
# Train multiple models for ensemble
for seed in 42 123 456; do
    python train.py \
        --config configs/unetr_ensemble.yaml \
        --seed $seed \
        --exp_name ensemble_$seed
done

# Combine ensemble predictions
python scripts/ensemble_inference.py \
    --models experiments/ensemble_*/best_model.pth \
    --test_dir data/BraTS-Test \
    --output_dir ensemble_results/
```

## 📈 Evaluation

### Standard Metrics

```bash
# Evaluate single model
python evaluate.py \
    --checkpoint experiments/brats_synthesis_v1/best_model.pth \
    --test_dir data/BraTS-Test \
    --metrics mse ssim psnr lpips

# Comprehensive evaluation
python scripts/comprehensive_eval.py \
    --checkpoint experiments/brats_synthesis_v1/best_model.pth \
    --test_dir data/BraTS-Test \
    --output_dir evaluation/ \
    --include_downstream_task
```

### Competition Evaluation

```bash
# BraTS Syn 2025 format
python scripts/brats_evaluation.py \
    --predictions results/ \
    --ground_truth data/BraTS-Test \
    --output_file competition_scores.json
```

### Downstream Task Evaluation

```bash
# Test synthesis quality for segmentation
python scripts/downstream_segmentation.py \
    --synthetic_dir results/ \
    --ground_truth data/BraTS-Test \
    --segmentation_model pretrained/segmentation_model.pth
```

## 🏆 Advanced Features

### 1. Progressive Training

```bash
# Start with small volumes, gradually increase
python train_progressive.py \
    --start_size 64 64 64 \
    --end_size 128 128 128 \
    --growth_epochs 20
```

### 2. Uncertainty Quantification

```bash
# Train with Monte Carlo Dropout
python train.py \
    --config configs/unetr_uncertainty.yaml \
    --enable_mc_dropout \
    --dropout_samples 10
```

### 3. Domain Adaptation

```bash
# Adapt to different scanner/protocol
python domain_adapt.py \
    --source_data data/BraTS-Training \
    --target_data data/External-Dataset \
    --adaptation_method "coral"
```

## 🔧 Development

### Project Structure

```
UNETR-BraTS-Synthesis/
├── configs/                 # Configuration files
│   ├── unetr_base.yaml
│   ├── unetr_transfer.yaml
│   └── unetr_competition.yaml
├── src/
│   ├── models/              # Model architectures
│   │   ├── unetr_synthesis.py
│   │   ├── loss_functions.py
│   │   └── transfer_learning.py
│   ├── data/                # Data handling
│   │   ├── brats_dataset.py
│   │   ├── transforms.py
│   │   └── utils.py
│   ├── training/            # Training utilities
│   │   ├── trainer.py
│   │   ├── validator.py
│   │   └── callbacks.py
│   └── evaluation/          # Evaluation scripts
│       ├── metrics.py
│       ├── visualization.py
│       └── statistical_tests.py
├── scripts/                 # Standalone scripts
│   ├── train.py
│   ├── inference.py
│   ├── evaluate.py
│   └── competition_tools/
├── experiments/             # Training outputs
├── pretrained/             # Pre-trained models
├── data/                   # Datasets
└── requirements.txt
```

### Custom Model Development

```python
# Create custom UNETR variant
from src.models.unetr_synthesis import UNETR_Synthesis

class CustomUNETR(UNETR_Synthesis):
    def __init__(self, config):
        super().__init__(config)
        # Add custom modifications
        self.custom_attention = MultiScaleAttention()
    
    def forward(self, x):
        # Custom forward pass
        features = self.encoder(x)
        refined_features = self.custom_attention(features)
        return self.decoder(refined_features)
```

### Adding New Loss Functions

```python
# Register custom loss
from src.models.loss_functions import register_loss

@register_loss("custom_perceptual")
class CustomPerceptualLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Initialize perceptual network
    
    def forward(self, pred, target):
        # Implement custom loss
        return loss_value
```

## 📊 Benchmarks and Results

### Expected Performance

| Configuration | SSIM ↑ | PSNR ↑ | MSE ↓ | Training Time |
|--------------|--------|--------|-------|---------------|
| Baseline (scratch) | 0.85 | 28.5 | 0.12 | 12 hours |
| Transfer Learning | 0.89 | 31.2 | 0.08 | 8 hours |
| Multi-loss Optimized | 0.91 | 32.8 | 0.07 | 10 hours |
| Ensemble (5 models) | 0.93 | 34.1 | 0.06 | 50 hours |

### Computational Requirements

| Volume Size | Batch Size | GPU Memory | Training Speed |
|-------------|------------|------------|----------------|
| 64³ | 4 | 8GB | 2 min/epoch |
| 96³ | 2 | 12GB | 4 min/epoch |
| 128³ | 1 | 16GB | 8 min/epoch |

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Code formatting
black src/ scripts/
isort src/ scripts/

# Type checking
mypy src/
```

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@article{unetr_brats_synthesis,
    title={UNETR-based Medical Image Synthesis with Transfer Learning for Brain MRI},
    author={Your Name},
    journal={arXiv preprint arXiv:XXXX.XXXXX},
    year={2025}
}
```

## 🙏 Acknowledgments

- [MONAI](https://monai.io/) for medical imaging framework
- [BraTS Challenge](http://braintumorsegmentation.org/) for dataset and evaluation
- Original UNETR paper: [Hatamizadeh et al., 2022](https://arxiv.org/abs/2103.10504)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size or volume size
python train.py --config configs/unetr_small.yaml --batch_size 1
```

**2. Transfer Learning Errors**
```bash
# Check pretrained model compatibility
python scripts/check_pretrained.py --model_path pretrained/model.pth
```

**3. Data Loading Issues**
```bash
# Validate data format
python scripts/validate_data.py --data_root /path/to/data --verbose
```

### Performance Optimization

**1. Training Speedup**
- Use mixed precision: `--mixed_precision`
- Increase num_workers: `--num_workers 8`
- Enable data caching: `--cache_rate 0.5`

**2. Memory Optimization**
- Gradient checkpointing: `--gradient_checkpointing`
- Smaller patch sizes: `--patch_size 64 64 64`
- Reduce model size: `--hidden_size 384`

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/UNETR-BraTS-Synthesis/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/UNETR-BraTS-Synthesis/discussions)
- **Email**: your.email@institution.edu

---

**Happy Synthesizing! 🧠✨**