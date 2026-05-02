"""
Quick script to check training progress
"""
import os
import glob

# Check if training has started
runs_dir = "RetailEye_Runs/v1_mosaic_strategy"
if os.path.exists(runs_dir):
    print(f"✅ Training directory exists: {runs_dir}")
    
    # Check for weights
    weights_dir = os.path.join(runs_dir, "weights")
    if os.path.exists(weights_dir):
        weights = glob.glob(os.path.join(weights_dir, "*.pt"))
        print(f"📊 Weights found: {len(weights)}")
        for w in weights:
            size_mb = os.path.getsize(w) / (1024 * 1024)
            print(f"   - {os.path.basename(w)} ({size_mb:.1f} MB)")
    
    # Check results file
    results_file = os.path.join(runs_dir, "results.csv")
    if os.path.exists(results_file):
        with open(results_file) as f:
            lines = f.readlines()
            print(f"\n📈 Training Progress: {len(lines)-1} epochs completed")
            if len(lines) > 1:
                print(f"   Latest: {lines[-1].strip()}")
else:
    print("⚠️  Training hasn't created output directory yet")
    print("   Check terminal for progress...")
