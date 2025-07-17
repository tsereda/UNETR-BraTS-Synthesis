#!/usr/bin/env python3
"""
Create pseudo validation set by randomly dropping one modality per case.
Improved version for BraSyn validation pipeline.

Usage:
python scripts/dropout_modality.py --input_dir ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData --output_dir pseudo_validation
"""

import os
import argparse
import numpy as np
import shutil
from pathlib import Path


def create_pseudo_validation(val_set_folder, val_set_missing, seed=123456):
    """
    Create a pseudo validation set by randomly dropping one modality per case.
    Based on BraSyn tutorial but with better error handling and marker files.
    """
    
    print(f"Creating pseudo validation set...")
    print(f"Source: {val_set_folder}")
    print(f"Output: {val_set_missing}")
    
    if not os.path.exists(val_set_folder):
        print(f"ERROR: Source directory does not exist: {val_set_folder}")
        return False
    
    if not os.path.exists(val_set_missing):
        os.makedirs(val_set_missing)
        print(f"Created output directory: {val_set_missing}")
    
    # Fix random seed for reproducibility
    np.random.seed(seed)
    modality_list = ['t1c', 't1n', 't2f', 't2w']  # T1CE, T1, FLAIR, T2
    modality_names = ['T1CE', 'T1', 'FLAIR', 'T2']
    
    # Find all BraTS case directories
    folder_list = []
    for item in os.listdir(val_set_folder):
        case_path = os.path.join(val_set_folder, item)
        if os.path.isdir(case_path) and 'BraTS' in item:
            folder_list.append(item)
    
    if not folder_list:
        print(f"ERROR: No BraTS case directories found in {val_set_folder}")
        return False
    
    folder_list.sort()
    
    # Randomly assign which modality to drop for each case
    drop_index = np.random.randint(0, 4, size=len(folder_list))
    
    print(f"Processing {len(folder_list)} cases...")
    
    # Track statistics
    modality_counts = {name: 0 for name in modality_names}
    successful_cases = 0
    failed_cases = 0
    
    for count, case_folder in enumerate(folder_list):
        try:
            case_src = os.path.join(val_set_folder, case_folder)
            case_dst = os.path.join(val_set_missing, case_folder)
            
            if not os.path.exists(case_dst):
                os.makedirs(case_dst)
            
            file_list = os.listdir(case_src)
            dropped_modality_suffix = modality_list[drop_index[count]]
            dropped_modality_name = modality_names[drop_index[count]]
            
            modality_counts[dropped_modality_name] += 1
            
            print(f"Case {count+1:3d}/{len(folder_list)}: {case_folder} -> dropping {dropped_modality_name}")
            
            # Copy all files except the dropped modality
            copied_files = 0
            skipped_files = 0
            
            for filename in file_list:
                src_file = os.path.join(case_src, filename)
                dst_file = os.path.join(case_dst, filename)
                
                # Skip the dropped modality file
                if dropped_modality_suffix in filename and filename.endswith('.nii.gz'):
                    skipped_files += 1
                    print(f"    Skipping: {filename}")
                else:
                    # Copy all other files (including segmentation if present)
                    shutil.copyfile(src_file, dst_file)
                    copied_files += 1
            
            # Create marker file to indicate which modality is missing
            # This helps the synthesis inference script detect what to synthesize
            marker_file = os.path.join(case_dst, f"missing_{dropped_modality_suffix}.txt")
            with open(marker_file, 'w') as f:
                f.write(f"Missing modality: {dropped_modality_name} ({dropped_modality_suffix})\n")
                f.write(f"Case: {case_folder}\n")
                f.write(f"Seed: {seed}\n")
                f.write(f"Drop index: {drop_index[count]}\n")
            
            print(f"    Copied: {copied_files} files, Skipped: {skipped_files} files")
            successful_cases += 1
            
        except Exception as e:
            print(f"ERROR processing case {case_folder}: {e}")
            failed_cases += 1
            continue
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"PSEUDO VALIDATION SET CREATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total cases: {len(folder_list)}")
    print(f"Successful: {successful_cases}")
    print(f"Failed: {failed_cases}")
    print(f"Output directory: {val_set_missing}")
    
    print(f"\nMissing modality distribution:")
    for modality, count in modality_counts.items():
        percentage = (count / len(folder_list)) * 100 if len(folder_list) > 0 else 0
        print(f"  {modality}: {count} cases ({percentage:.1f}%)")
    
    print(f"\n✅ Ready for synthesis inference!")
    print(f"Next step: Run synthesis on {val_set_missing}")
    
    return successful_cases > 0


def verify_pseudo_validation(val_set_missing):
    """Verify the created pseudo validation set"""
    
    print(f"\nVerifying pseudo validation set: {val_set_missing}")
    
    if not os.path.exists(val_set_missing):
        print(f"❌ Directory does not exist: {val_set_missing}")
        return False
    
    case_dirs = [d for d in os.listdir(val_set_missing) 
                if os.path.isdir(os.path.join(val_set_missing, d)) and 'BraTS' in d]
    
    print(f"Found {len(case_dirs)} cases")
    
    # Check first few cases
    modality_files = ['t1c.nii.gz', 't1n.nii.gz', 't2f.nii.gz', 't2w.nii.gz']
    modality_names = ['T1CE', 'T1', 'FLAIR', 'T2']
    
    verification_summary = {name: 0 for name in modality_names}
    
    for case_dir in case_dirs:
        case_path = os.path.join(val_set_missing, case_dir)
        case_files = os.listdir(case_path)
        
        # Count present/missing modalities
        missing_count = 0
        for i, modality_file in enumerate(modality_files):
            expected_file = f"{case_dir}-{modality_file}"
            if expected_file not in case_files:
                verification_summary[modality_names[i]] += 1
                missing_count += 1
        
        # Should have exactly 1 missing modality
        if missing_count != 1:
            print(f"⚠️  Case {case_dir}: {missing_count} missing modalities (expected 1)")
    
    print(f"\nMissing modality verification:")
    for modality, count in verification_summary.items():
        print(f"  {modality}: {count} cases")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Create pseudo validation set with dropped modalities")
    parser.add_argument("--input_dir", type=str, 
                       default="ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData",
                       help="Source validation data directory")
    parser.add_argument("--output_dir", type=str, default="pseudo_validation",
                       help="Output directory for pseudo validation with missing modalities")
    parser.add_argument("--seed", type=int, default=123456,
                       help="Random seed for reproducibility (BraSyn tutorial uses 123456)")
    parser.add_argument("--verify", action="store_true",
                       help="Verify an existing pseudo validation set")
    
    args = parser.parse_args()
    
    # Handle both relative and absolute paths
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    
    print(f"BraSyn Pseudo Validation Set Creator")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Seed: {args.seed}")
    
    if args.verify:
        # Verify existing pseudo validation set
        verify_pseudo_validation(output_dir)
    else:
        # Create new pseudo validation set
        success = create_pseudo_validation(
            val_set_folder=input_dir,
            val_set_missing=output_dir,
            seed=args.seed
        )
        
        if success:
            print(f"\n🎯 Next steps:")
            print(f"1. Run synthesis inference on {output_dir}")
            print(f"2. Convert to FeTS format")
            print(f"3. Run FeTS segmentation")
            print(f"4. Create BraSyn submission")
        else:
            print(f"\n❌ Failed to create pseudo validation set")


if __name__ == "__main__":
    main()