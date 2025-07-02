#!/usr/bin/env python3
"""
Debug script to examine the actual data structure.
"""

import os
import glob
from pathlib import Path

def examine_data_structure(data_root):
    """Examine the actual data structure."""
    data_path = Path(data_root)
    
    print(f"Examining data root: {data_path}")
    print(f"Data root exists: {data_path.exists()}")
    
    if not data_path.exists():
        print("Data root does not exist!")
        return
    
    print("\nTop-level contents:")
    for item in data_path.iterdir():
        if item.is_dir():
            print(f"  DIR:  {item.name}")
            # Look inside this directory
            subdirs = list(item.iterdir())[:5]  # First 5 items
            for subdir in subdirs:
                if subdir.is_dir():
                    print(f"    SUBDIR: {subdir.name}")
                    # Look for .nii.gz files
                    nii_files = list(subdir.glob("*.nii.gz"))
                    if nii_files:
                        print(f"      Found {len(nii_files)} .nii.gz files:")
                        for nii_file in nii_files[:3]:  # First 3 files
                            print(f"        {nii_file.name}")
                        if len(nii_files) > 3:
                            print(f"        ... and {len(nii_files) - 3} more")
                else:
                    print(f"    FILE: {subdir.name}")
        else:
            print(f"  FILE: {item.name}")
    
    # Try different patterns
    print("\nTrying different subject patterns:")
    
    patterns = [
        "BraTS-*",
        "BraTS2023-*", 
        "BraTS-GLI-*",
        "BraTS-*-*",
        "*BraTS*",
        "*"
    ]
    
    for pattern in patterns:
        full_pattern = str(data_path / pattern)
        matches = glob.glob(full_pattern)
        directories = [d for d in matches if os.path.isdir(d)]
        print(f"  Pattern '{pattern}': {len(directories)} directories")
        if directories:
            for i, d in enumerate(directories[:3]):  # Show first 3
                print(f"    {os.path.basename(d)}")
            if len(directories) > 3:
                print(f"    ... and {len(directories) - 3} more")

if __name__ == "__main__":
    import sys
    data_root = sys.argv[1] if len(sys.argv) > 1 else "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    examine_data_structure(data_root)
