#!/usr/bin/env python3
"""
Script to manually log some test metrics to W&B to verify logging is working.
"""

import os
import wandb
import numpy as np
import torch

def test_manual_logging():
    """Test manual logging to W&B."""
    print("Testing manual W&B logging...")
    
    try:
        # Initialize wandb
        wandb.init(
            project=os.getenv('WANDB_PROJECT', 'UNETR-BraTS-Synthesis'),
            entity=os.getenv('WANDB_ENTITY'),
            name="manual-test-logging",
            tags=["test", "manual"]
        )
        
        # Log some test metrics
        for step in range(10):
            wandb.log({
                'test/loss': np.random.random() * 0.5 + 0.1,
                'test/accuracy': np.random.random() * 0.3 + 0.7,
                'test/learning_rate': 0.001 * (0.9 ** step),
                'test/epoch': step
            })
        
        # Log a test image
        test_image = np.random.rand(64, 64)
        wandb.log({
            "test/sample_image": wandb.Image(test_image, caption="Test image")
        })
        
        print("✓ Manual W&B logging successful!")
        print("Check your W&B dashboard to see the test metrics.")
        
        wandb.finish()
        return True
        
    except Exception as e:
        print(f"ERROR: Manual W&B logging failed: {e}")
        return False

if __name__ == "__main__":
    success = test_manual_logging()
    exit(0 if success else 1)
