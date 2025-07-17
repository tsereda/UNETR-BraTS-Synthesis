#!/bin/bash

# Script to convert BraTS format to FeTS format
# Creates proper directory structure and file naming for FeTS

echo "Converting to FeTS format..."

# Create output directory
mkdir -p fets_formatted

# Counter for patient numbering
patient_num=1

# Process each case directory
for case_dir in completed_cases/BraTS-GLI-*/; do
    if [ -d "$case_dir" ]; then
        case_name=$(basename "$case_dir")
        
        # Create patient directory with zero-padded number
        patient_dir=$(printf "Patient_%03d" $patient_num)
        mkdir -p "fets_formatted/$patient_dir"
        
        echo "Processing $case_name → $patient_dir"
        
        # Check if all required files exist
        if [ -f "$case_dir/t1.nii.gz" ] && [ -f "$case_dir/t1ce.nii.gz" ] && [ -f "$case_dir/t2.nii.gz" ] && [ -f "$case_dir/flair.nii.gz" ]; then
            # Copy and rename files to FeTS format
            cp "$case_dir/t1.nii.gz" "fets_formatted/$patient_dir/${patient_dir}_brain_t1.nii.gz"
            cp "$case_dir/t1ce.nii.gz" "fets_formatted/$patient_dir/${patient_dir}_brain_t1ce.nii.gz"
            cp "$case_dir/t2.nii.gz" "fets_formatted/$patient_dir/${patient_dir}_brain_t2.nii.gz"
            cp "$case_dir/flair.nii.gz" "fets_formatted/$patient_dir/${patient_dir}_brain_flair.nii.gz"
            
            echo "  ✅ Converted $case_name to $patient_dir"
            ((patient_num++))
        else
            echo "  ❌ Missing files in $case_name, skipping"
        fi
    fi
done

echo "Conversion complete! Created $((patient_num-1)) patient directories."
echo "Now you can run FeTS segmentation:"
echo "./squashfs-root/usr/bin/fets_cli_segment -d fets_formatted/ -a fets_singlet,fets_triplet -lF STAPLE,ITKVoting,SIMPLE,MajorityVoting -g 1 -t 0"