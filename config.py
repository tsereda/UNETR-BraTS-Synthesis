#!/usr/bin/env python3
"""
Configuration file for BraTS modality synthesis project.
Update the paths and parameters here before running the scripts.
"""

import os

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

# Path to your BRATS data directory (containing training/ and validation/ folders)
DATA_ROOT = "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"

# Path where the JSON file will be created/loaded from
JSON_FILE = os.path.join(DATA_ROOT, "brats_synthesis_data.json")

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Input modalities (3 modalities as input)
INPUT_MODALITIES = ["t1c", "t1n", "t2w"]

# Target modality to synthesize
TARGET_MODALITY = "t2f"  # FLAIR

# Model architecture parameters
MODEL_CONFIG = {
    "in_channels": len(INPUT_MODALITIES),  # 3
    "out_channels": 1,                     # 1 target modality
    "feature_size": 48,
    "drop_rate": 0.0,
    "attn_drop_rate": 0.0,
    "dropout_path_rate": 0.0,
    "use_checkpoint": True,
}

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

# Training parameters
TRAINING_CONFIG = {
    "roi": (128, 128, 128),        # Patch size for training
    "batch_size": 2,               # Reduce if GPU memory is limited
    "sw_batch_size": 4,            # Sliding window batch size for inference
    "max_epochs": 100,             # Total training epochs
    "val_every": 10,               # Validation frequency
    "fold": 0,                     # Cross-validation fold
    "infer_overlap": 0.5,          # Overlap for sliding window inference
}

# Optimizer parameters
OPTIMIZER_CONFIG = {
    "lr": 1e-4,                    # Learning rate
    "weight_decay": 1e-5,          # Weight decay
}

# Loss function weights
LOSS_CONFIG = {
    "l1_weight": 1.0,              # L1 loss weight
    "ssim_weight": 0.1,            # SSIM loss weight
}

# =============================================================================
# INFERENCE CONFIGURATION
# =============================================================================

# Inference parameters
INFERENCE_CONFIG = {
    "roi": TRAINING_CONFIG["roi"],
    "sw_batch_size": 1,            # Reduced for inference
    "overlap": 0.6,                # Higher overlap for better quality
}

# =============================================================================
# PATHS
# =============================================================================

# Output directories
OUTPUT_DIR = "./outputs"
MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, "models")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
INFERENCE_DIR = os.path.join(OUTPUT_DIR, "inference")

# Model file paths
MODEL_CHECKPOINT = os.path.join(MODEL_SAVE_DIR, "model.pt")
BEST_MODEL = os.path.join(MODEL_SAVE_DIR, "best_model.pt")

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_directories():
    """Create necessary output directories"""
    dirs = [OUTPUT_DIR, MODEL_SAVE_DIR, RESULTS_DIR, INFERENCE_DIR]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    print("Created output directories:")
    for dir_path in dirs:
        print(f"  - {dir_path}")


def validate_paths():
    """Validate that required paths exist"""
    issues = []
    
    if not os.path.exists(DATA_ROOT):
        issues.append(f"Data root directory not found: {DATA_ROOT}")
    else:
        training_dir = os.path.join(DATA_ROOT, "training")
        validation_dir = os.path.join(DATA_ROOT, "validation")
        
        if not os.path.exists(training_dir):
            issues.append(f"Training directory not found: {training_dir}")
        if not os.path.exists(validation_dir):
            issues.append(f"Validation directory not found: {validation_dir}")
    
    if issues:
        print("❌ Configuration issues found:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nPlease update the paths in config.py")
        return False
    else:
        print("✅ Configuration looks good!")
        return True


def print_config():
    """Print current configuration"""
    print("=" * 60)
    print("BRATS MODALITY SYNTHESIS CONFIGURATION")
    print("=" * 60)
    print(f"Data root: {DATA_ROOT}")
    print(f"JSON file: {JSON_FILE}")
    print(f"Input modalities: {INPUT_MODALITIES}")
    print(f"Target modality: {TARGET_MODALITY}")
    print(f"Model channels: {MODEL_CONFIG['in_channels']} → {MODEL_CONFIG['out_channels']}")
    print(f"Training ROI: {TRAINING_CONFIG['roi']}")
    print(f"Batch size: {TRAINING_CONFIG['batch_size']}")
    print(f"Max epochs: {TRAINING_CONFIG['max_epochs']}")
    print(f"Learning rate: {OPTIMIZER_CONFIG['lr']}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
    create_directories()
    validate_paths()