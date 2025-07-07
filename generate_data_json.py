#!/usr/bin/env python3
"""
Generate JSON file for BraTS modality synthesis dataset.
This script creates the data splits for training and validation.
"""

import os
import json
import glob
from pathlib import Path

def generate_brats_synthesis_json(data_root, output_file="brats_synthesis_data.json"):
    """
    Generate JSON file for BraTS modality synthesis.
    
    Args:
        data_root: Path to BRATS folder containing training/validation
        output_file: Output JSON filename
    """
    
    data_root = Path(data_root)
    training_dir = data_root / "training"
    validation_dir = data_root / "validation"
    
    # Modalities to use as input (3 modalities -> 1 target)
    input_modalities = ["t1c", "t1n", "t2w"]  # Input modalities
    target_modality = "t2f"  # Target modality (FLAIR)
    
    def process_cases(cases_dir, split_name):
        """Process cases in a directory"""
        cases = []
        
        if not cases_dir.exists():
            print(f"Warning: {cases_dir} does not exist")
            return cases
            
        case_dirs = sorted([d for d in cases_dir.iterdir() if d.is_dir()])
        
        for case_dir in case_dirs:
            case_name = case_dir.name
            
            # Find all modality files
            input_files = []
            target_file = None
            
            # Check if all required input modalities exist
            missing_modalities = []
            for modality in input_modalities:
                modality_file = case_dir / f"{case_name}-{modality}.nii.gz"
                if modality_file.exists():
                    input_files.append(str(modality_file.relative_to(data_root)))
                else:
                    missing_modalities.append(modality)
            
            # Check target modality
            target_file_path = case_dir / f"{case_name}-{target_modality}.nii.gz"
            if target_file_path.exists():
                target_file = str(target_file_path.relative_to(data_root))
            else:
                missing_modalities.append(target_modality)
            
            if missing_modalities:
                print(f"Skipping {case_name}: missing {missing_modalities}")
                continue
                
            case_data = {
                "image": input_files,
                "target": target_file,
                "case_name": case_name
            }
            
            cases.append(case_data)
            
        print(f"Found {len(cases)} valid cases in {split_name}")
        return cases
    
    # Process training and validation cases
    training_cases = process_cases(training_dir, "training")
    validation_cases = process_cases(validation_dir, "validation")
    
    # Create final data structure
    data = {
        "description": "BraTS Modality Synthesis Dataset",
        "labels": {
            "0": "background",
            "1": "synthesized_modality"
        },
        "licence": "BraTS Challenge",
        "modality": {
            "0": "t1c",
            "1": "t1n", 
            "2": "t2w"
        },
        "target_modality": target_modality,
        "name": "BraTS_Modality_Synthesis",
        "numTraining": len(training_cases),
        "numValidation": len(validation_cases),
        "training": training_cases,
        "validation": validation_cases
    }
    
    # Save JSON file
    output_path = data_root / output_file
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nDataset JSON saved to: {output_path}")
    print(f"Training cases: {len(training_cases)}")
    print(f"Validation cases: {len(validation_cases)}")
    print(f"Input modalities: {input_modalities}")
    print(f"Target modality: {target_modality}")
    
    return output_path

if __name__ == "__main__":
    # Update this path to your data directory
    data_root = "/path/to/your/data/BRATS"
    
    # Generate the JSON file
    json_file = generate_brats_synthesis_json(data_root)
    
    print(f"\nTo use this dataset:")
    print(f"1. Update data_dir in modality_synthesis.py to: {data_root}")
    print(f"2. Update json_list in modality_synthesis.py to: {json_file}")