#!/usr/bin/env python3
"""Real-time training monitor for Phase 3 multi-disease training."""

import json
import os
import time
from pathlib import Path
from datetime import datetime

def monitor_training():
    """Monitor training progress in real-time."""
    
    checkpoint_dir = Path("phase3_multi_disease/checkpoints")
    
    # Find the latest training directory
    training_dirs = sorted([d for d in checkpoint_dir.glob("multidisease_*") if d.is_dir()])
    
    if not training_dirs:
        print("❌ No training directory found")
        return
    
    train_dir = training_dirs[-1]
    print(f"\n{'='*70}")
    print(f"📊 MONITORING PHASE 3 TRAINING")
    print(f"{'='*70}")
    print(f"📁 Directory: {train_dir.name}")
    
    # Load config
    config_file = train_dir / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
        print(f"\n⚙️  Configuration:")
        print(f"   Model: {config['model']}")
        print(f"   Epochs: {config['epochs']}")
        print(f"   Batch size: {config['batch_size']}")
        print(f"   Device: {config['device']}")
    
    # Monitor files
    print(f"\n📁 Output files being created:")
    
    start_time = datetime.now()
    max_wait = 3600  # 1 hour timeout
    
    while (datetime.now() - start_time).total_seconds() < max_wait:
        files = list(train_dir.glob("*"))
        
        print(f"\r   Files: {len(files):2d}", end="", flush=True)
        
        # Check for best model
        best_model = train_dir / "best_model.pt"
        if best_model.exists():
            size_mb = best_model.stat().st_size / (1024 * 1024)
            print(f"\n   ✅ best_model.pt ({size_mb:.1f} MB) - Model is being trained!")
            
            # Check for intermediate files
            for f in files:
                if f.is_file():
                    print(f"      - {f.name}")
            
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"\n⏱️  Elapsed: {elapsed:.0f} seconds ({elapsed/60:.1f} minutes)")
            print(f"   Status: 🟢 TRAINING IN PROGRESS")
            break
        
        time.sleep(2)  # Check every 2 seconds
    else:
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"\n⏱️  Elapsed: {elapsed:.0f} seconds ({elapsed/60:.1f} minutes)")
        print(f"   Files created: {len(files)}")
        print(f"   Status: 🟡 STILL LOADING DATA or TRAINING STARTING...")

if __name__ == "__main__":
    monitor_training()
