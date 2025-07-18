#!/bin/bash

# Script to convert BraTS format to FeTS format
# FIXED: Uses correct BraTS file naming convention

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
        
        # FIXED: Check for actual BraTS file names
        t1_file="$case_dir/${case_name}-t1n.nii.gz"     # T1 native
        t1ce_file="$case_dir/${case_name}-t1c.nii.gz"   # T1 contrast enhanced
        t2_file="$case_dir/${case_name}-t2w.nii.gz"     # T2 weighted  
        flair_file="$case_dir/${case_name}-t2f.nii.gz"  # FLAIR
        
        if [ -f "$t1_file" ] && [ -f "$t1ce_file" ] && [ -f "$t2_file" ] && [ -f "$flair_file" ]; then
            # Copy and rename files to FeTS format
            cp "$t1_file" "fets_formatted/$patient_dir/${patient_dir}_brain_t1.nii.gz"
            cp "$t1ce_file" "fets_formatted/$patient_dir/${patient_dir}_brain_t1ce.nii.gz"
            cp "$t2_file" "fets_formatted/$patient_dir/${patient_dir}_brain_t2.nii.gz"
            cp "$flair_file" "fets_formatted/$patient_dir/${patient_dir}_brain_flair.nii.gz"
            
            echo "  ✅ Converted $case_name to $patient_dir"
            ((patient_num++))
        else
            echo "  ❌ Missing files in $case_name, skipping"
            echo "     Looking for:"
            echo "       T1:    ${case_name}-t1n.nii.gz $([ -f "$t1_file" ] && echo "✅" || echo "❌")"
            echo "       T1CE:  ${case_name}-t1c.nii.gz $([ -f "$t1ce_file" ] && echo "✅" || echo "❌")"
            echo "       T2:    ${case_name}-t2w.nii.gz $([ -f "$t2_file" ] && echo "✅" || echo "❌")"
            echo "       FLAIR: ${case_name}-t2f.nii.gz $([ -f "$flair_file" ] && echo "✅" || echo "❌")"
            echo "     Actually found:"
            ls -1 "$case_dir"/*.nii.gz 2>/dev/null | head -5
        fi
    fi
done

echo "Conversion complete! Created $((patient_num-1)) patient directories."
echo "Now you can run FeTS segmentation:"
echo "./squashfs-root/usr/bin/fets_cli_segment -d fets_formatted/ -a deepMedic -g 0 -t 0"