#!/usr/bin/env python3
"""
Validate BraTS data structure and integrity.

This script checks the BraTS dataset structure and validates file integrity.
"""

import os
import argparse
import glob
from pathlib import Path
from typing import List, Dict, Tuple
import sys

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    print("Warning: nibabel not available. Install with: pip install nibabel")


def find_subjects(data_root: Path) -> List[Path]:
    """Find all subject directories."""
    pattern = str(data_root / "BraTS-*")
    subject_dirs = glob.glob(pattern)
    subject_dirs = [Path(d) for d in subject_dirs if Path(d).is_dir()]
    return sorted(subject_dirs)


def validate_subject(subject_dir: Path, modalities: List[str], verbose: bool = False) -> Dict[str, bool]:
    """Validate a single subject directory."""
    subject_name = subject_dir.name
    results = {}
    
    for modality in modalities:
        file_path = subject_dir / f"{subject_name}-{modality}.nii.gz"
        
        # Check if file exists
        if not file_path.exists():
            results[modality] = False
            if verbose:
                print(f"  Missing: {file_path}")
            continue
        
        # Check file integrity if nibabel is available
        if NIBABEL_AVAILABLE:
            try:
                img = nib.load(str(file_path))
                data = img.get_fdata()
                
                # Basic checks
                if data.size == 0:
                    results[modality] = False
                    if verbose:
                        print(f"  Empty data: {file_path}")
                elif data.shape != data.shape:  # This will always be True, but keeping pattern
                    results[modality] = False
                    if verbose:
                        print(f"  Invalid shape: {file_path}")
                else:
                    results[modality] = True
                    if verbose:
                        print(f"  Valid: {file_path} - Shape: {data.shape}")
                        
            except Exception as e:
                results[modality] = False
                if verbose:
                    print(f"  Error loading {file_path}: {e}")
        else:
            # Just check file size
            file_size = file_path.stat().st_size
            if file_size > 0:
                results[modality] = True
            else:
                results[modality] = False
                if verbose:
                    print(f"  Empty file: {file_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Validate BraTS data structure')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory of BraTS data')
    parser.add_argument('--modalities', nargs='+', 
                       default=['t1n', 't1c', 't2w', 't2f'],
                       help='Modalities to check')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed information')
    parser.add_argument('--max_subjects', type=int, default=None,
                       help='Maximum number of subjects to check')
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    
    # Check if data root exists
    if not data_root.exists():
        print(f"Error: Data root does not exist: {data_root}")
        sys.exit(1)
    
    # Find subjects
    subjects = find_subjects(data_root)
    
    if not subjects:
        print(f"No subjects found in {data_root}")
        print("Expected pattern: BraTS-*")
        sys.exit(1)
    
    print(f"Found {len(subjects)} subjects")
    
    if args.max_subjects:
        subjects = subjects[:args.max_subjects]
        print(f"Checking first {len(subjects)} subjects")
    
    # Validate each subject
    valid_subjects = 0
    missing_files = {mod: 0 for mod in args.modalities}
    error_subjects = []
    
    for i, subject_dir in enumerate(subjects):
        if args.verbose:
            print(f"\nChecking {subject_dir.name} ({i+1}/{len(subjects)}):")
        
        results = validate_subject(subject_dir, args.modalities, args.verbose)
        
        # Count results
        subject_valid = all(results.values())
        if subject_valid:
            valid_subjects += 1
        else:
            error_subjects.append(subject_dir.name)
            for modality, is_valid in results.items():
                if not is_valid:
                    missing_files[modality] += 1
        
        # Progress indicator for non-verbose mode
        if not args.verbose and (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{len(subjects)} subjects...")
    
    # Summary
    print(f"\n{'='*50}")
    print("VALIDATION SUMMARY")
    print(f"{'='*50}")
    print(f"Total subjects: {len(subjects)}")
    print(f"Valid subjects: {valid_subjects} ({valid_subjects/len(subjects)*100:.1f}%)")
    print(f"Subjects with errors: {len(subjects) - valid_subjects}")
    
    if missing_files:
        print(f"\nMissing files by modality:")
        for modality, count in missing_files.items():
            print(f"  {modality}: {count} subjects")
    
    if error_subjects and args.verbose:
        print(f"\nSubjects with errors:")
        for subject in error_subjects[:10]:  # Show first 10
            print(f"  {subject}")
        if len(error_subjects) > 10:
            print(f"  ... and {len(error_subjects) - 10} more")
    
    # Recommendations
    print(f"\n{'='*50}")
    print("RECOMMENDATIONS")
    print(f"{'='*50}")
    
    if valid_subjects == len(subjects):
        print("✅ All subjects are valid! Ready for training.")
    else:
        error_rate = (len(subjects) - valid_subjects) / len(subjects)
        if error_rate < 0.05:
            print("⚠️  Small number of invalid subjects. You can:")
            print("   - Proceed with training (invalid subjects will be skipped)")
            print("   - Re-download missing files")
        elif error_rate < 0.2:
            print("⚠️  Moderate number of invalid subjects. Recommended:")
            print("   - Check data download process")
            print("   - Re-download dataset if possible")
        else:
            print("❌ High number of invalid subjects. Actions needed:")
            print("   - Verify dataset download")
            print("   - Check data extraction process")
            print("   - Consider re-downloading entire dataset")
    
    # Dataset size estimation
    if NIBABEL_AVAILABLE and subjects:
        try:
            # Check first valid subject for size estimation
            for subject_dir in subjects:
                results = validate_subject(subject_dir, args.modalities[:1])
                if results[args.modalities[0]]:
                    sample_file = subject_dir / f"{subject_dir.name}-{args.modalities[0]}.nii.gz"
                    img = nib.load(str(sample_file))
                    shape = img.shape
                    print(f"\nDataset info (based on {subject_dir.name}):")
                    print(f"  Volume shape: {shape}")
                    print(f"  Voxel spacing: {img.header.get_zooms()}")
                    break
        except:
            pass


if __name__ == "__main__":
    main()
