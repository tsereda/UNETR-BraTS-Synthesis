#!/bin/bash
# Script to convert BraTS format to FeTS format
# FIXED: Uses correct directory and improved error handling

echo "Converting to FeTS format..."

# Determine source directory
if [ -d "completed_cases_fixed" ]; then
    SOURCE_DIR="completed_cases_fixed"
    echo "Using source directory: completed_cases_fixed"
elif [ -d "completed_cases" ]; then
    SOURCE_DIR="completed_cases"
    echo "Using source directory: completed_cases"
else
    echo "❌ Error: Neither completed_cases_fixed nor completed_cases directory found!"
    echo "Available directories:"
    ls -la | grep "completed"
    exit 1
fi

# Create output directory
mkdir -p fets_formatted

# Counter for patient numbering
patient_num=1
total_cases=0
successful_cases=0

# Count total cases first
for case_dir in $SOURCE_DIR/BraTS-GLI-*/; do
    if [ -d "$case_dir" ]; then
        ((total_cases++))
    fi
done

echo "Found $total_cases cases to convert"

# Process each case directory
for case_dir in $SOURCE_DIR/BraTS-GLI-*/; do
    if [ -d "$case_dir" ]; then
        case_name=$(basename "$case_dir")
        
        # Create patient directory with zero-padded number
        patient_dir=$(printf "Patient_%03d" $patient_num)
        mkdir -p "fets_formatted/$patient_dir"
        
        echo "[$patient_num/$total_cases] Processing $case_name → $patient_dir"
        
        # Check for actual BraTS file names
        t1_file="$case_dir/${case_name}-t1n.nii.gz"     # T1 native
        t1ce_file="$case_dir/${case_name}-t1c.nii.gz"   # T1 contrast enhanced
        t2_file="$case_dir/${case_name}-t2w.nii.gz"     # T2 weighted  
        flair_file="$case_dir/${case_name}-t2f.nii.gz"  # FLAIR
        
        # Check file existence and sizes
        all_files_ok=true
        file_info=""
        
        for file_pair in "$t1_file:T1" "$t1ce_file:T1CE" "$t2_file:T2" "$flair_file:FLAIR"; do
            file_path="${file_pair%:*}"
            modality="${file_pair#*:}"
            
            if [ -f "$file_path" ]; then
                file_size=$(stat -c%s "$file_path" 2>/dev/null || echo "0")
                file_size_mb=$((file_size / 1024 / 1024))
                
                if [ $file_size_mb -lt 1 ]; then
                    file_info="$file_info\n    ⚠️  $modality: ${file_size_mb}MB (suspiciously small)"
                    # Still proceed but warn
                elif [ $file_size_mb -gt 15 ]; then
                    file_info="$file_info\n    ⚠️  $modality: ${file_size_mb}MB (unusually large)"
                else
                    file_info="$file_info\n    ✅ $modality: ${file_size_mb}MB"
                fi
            else
                file_info="$file_info\n    ❌ $modality: MISSING"
                all_files_ok=false
            fi
        done
        
        if [ "$all_files_ok" = true ]; then
            # Copy and rename files to FeTS format
            cp "$t1_file" "fets_formatted/$patient_dir/${patient_dir}_brain_t1.nii.gz"
            cp "$t1ce_file" "fets_formatted/$patient_dir/${patient_dir}_brain_t1ce.nii.gz"
            cp "$t2_file" "fets_formatted/$patient_dir/${patient_dir}_brain_t2.nii.gz"
            cp "$flair_file" "fets_formatted/$patient_dir/${patient_dir}_brain_flair.nii.gz"
            
            echo "  ✅ Converted $case_name to $patient_dir"
            echo -e "$file_info"
            ((successful_cases++))
        else
            echo "  ❌ Missing files in $case_name, skipping"
            echo -e "$file_info"
            
            # Remove the empty patient directory
            rmdir "fets_formatted/$patient_dir" 2>/dev/null
            continue
        fi
        
        ((patient_num++))
    fi
done

echo ""
echo "============================================================"
echo "CONVERSION SUMMARY"
echo "============================================================"
echo "✅ Successfully converted: $successful_cases/$total_cases cases"
echo "📁 Created patient directories: Patient_001 to Patient_$(printf "%03d" $successful_cases)"
echo "📍 Output location: $(pwd)/fets_formatted/"

if [ $successful_cases -gt 0 ]; then
    echo ""
    echo "🎯 Ready for FeTS segmentation!"
    echo "Run the following command:"
    echo "./squashfs-root/usr/bin/fets_cli_segment -d fets_formatted/ -a fets_singlet,fets_triplet -lF STAPLE,ITKVoting,SIMPLE,MajorityVoting -g 1 -t 0"
    
    # Show file size distribution
    echo ""
    echo "File size check:"
    find fets_formatted -name "*.nii.gz" -exec stat -c "%s %n" {} \; | \
    awk '{size=$1/1024/1024; if(size<1) print "⚠️  Small: " $2 " (" size "MB)"; else if(size>10) print "⚠️  Large: " $2 " (" size "MB)"; else print "✅ Normal: " $2 " (" size "MB)"}' | \
    head -20
    
    if [ $(find fets_formatted -name "*.nii.gz" | wc -l) -gt 20 ]; then
        echo "... (showing first 20 files)"
    fi
else
    echo ""
    echo "❌ No cases were successfully converted."
    echo "Check the synthesis output and file structure."
fi