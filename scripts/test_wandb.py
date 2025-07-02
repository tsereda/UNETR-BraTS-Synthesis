#!/usr/bin/env python3
"""
Script to test W&B integration.
"""

import os
import wandb

def test_wandb_connection():
    """Test if W&B is properly configured."""
    print("Testing W&B integration...")
    
    # Check environment variables
    api_key = os.getenv('WANDB_API_KEY')
    project = os.getenv('WANDB_PROJECT')
    entity = os.getenv('WANDB_ENTITY')
    mode = os.getenv('WANDB_MODE')
    
    print(f"WANDB_API_KEY: {'Set' if api_key else 'Not set'}")
    print(f"WANDB_PROJECT: {project}")
    print(f"WANDB_ENTITY: {entity}")
    print(f"WANDB_MODE: {mode}")
    
    if not api_key:
        print("ERROR: WANDB_API_KEY is not set!")
        return False
    
    try:
        # Initialize a test run
        wandb.init(
            project=project or "test-project",
            entity=entity,
            name="wandb-test-run",
            mode=mode or "online",
            tags=["test"]
        )
        
        # Log some test data
        wandb.log({
            "test_metric": 0.5,
            "epoch": 0
        })
        
        print("✓ W&B connection successful!")
        wandb.finish()
        return True
        
    except Exception as e:
        print(f"ERROR: W&B connection failed: {e}")
        return False

if __name__ == "__main__":
    success = test_wandb_connection()
    exit(0 if success else 1)
