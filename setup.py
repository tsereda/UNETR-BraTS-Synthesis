#!/usr/bin/env python3
"""
Setup script for development environment.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Command: {command}")
        print(f"Error: {e.stderr}")
        return False


def check_python_version():
    """Check Python version compatibility."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3 or version.minor < 8:
        print("❌ Python 3.8+ is required")
        return False
    
    print("✅ Python version is compatible")
    return True


def setup_environment():
    """Setup the development environment."""
    print("🚀 Setting up UNETR-BraTS-Synthesis development environment")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create directories
    directories = [
        "data",
        "experiments", 
        "pretrained",
        "results",
        "logs"
    ]
    
    print(f"\nCreating project directories...")
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"  ✅ {directory}/")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        print("⚠️  Some dependencies failed to install. You may need to install them manually:")
        print("   pip install torch torchvision")  
        print("   pip install monai[all]")
        print("   pip install nibabel")
    
    # Create __init__.py files for proper imports
    init_files = [
        "src/__init__.py",
        "src/models/__init__.py", 
        "src/data/__init__.py"
    ]
    
    print(f"\nCreating __init__.py files...")
    for init_file in init_files:
        Path(init_file).touch()
        print(f"  ✅ {init_file}")
    
    # Make scripts executable
    script_files = [
        "scripts/download_pretrained.py",
        "scripts/validate_data.py",
        "train.py",
        "inference.py"
    ]
    
    print(f"\nMaking scripts executable...")
    for script in script_files:
        if Path(script).exists():
            os.chmod(script, 0o755)
            print(f"  ✅ {script}")
    
    print("\n" + "=" * 60)
    print("🎉 Setup completed!")
    print("\nNext steps:")
    print("1. Prepare your BraTS data:")
    print("   python scripts/validate_data.py --data_root /path/to/brats/data")
    print("\n2. Download pre-trained models (optional):")
    print("   python scripts/download_pretrained.py --model brats_segmentation")
    print("\n3. Start training:")
    print("   python train.py --config configs/unetr_base.yaml --data_root /path/to/data --exp_name my_experiment")
    print("\n4. Run inference:")
    print("   python inference.py --checkpoint experiments/my_experiment/best_model.pth --input_dir /path/to/test --output_dir results --target_modality t1c")
    
    return True


if __name__ == "__main__":
    success = setup_environment()
    sys.exit(0 if success else 1)
