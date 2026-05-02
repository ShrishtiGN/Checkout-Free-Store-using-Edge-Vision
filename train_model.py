from ultralytics import YOLO
import os
import glob
import yaml
import sys
import torch

def validate_dataset(data_yaml_path):
    """Validate dataset before training"""
    print("\n" + "="*60)
    print("🔍 PRE-TRAINING VALIDATION")
    print("="*60)
    
    # Check if data.yaml exists
    if not os.path.exists(data_yaml_path):
        print(f"❌ ERROR: {data_yaml_path} not found!")
        return False
    
    # Load yaml
    with open(data_yaml_path) as f:
        data_config = yaml.safe_load(f)
    
    # Get paths - resolve relative to the yaml file's directory
    yaml_dir = os.path.dirname(os.path.abspath(data_yaml_path))
    base_path = data_config.get('path', '.')
    
    # If base_path is relative, resolve it relative to the yaml file
    if not os.path.isabs(base_path):
        base_path = os.path.join(yaml_dir, base_path)
    
    train_path = os.path.join(base_path, data_config.get('train', 'images/train'))
    val_path = os.path.join(base_path, data_config.get('val', 'images/val'))
    
    # Count images and labels
    train_images = len(glob.glob(os.path.join(train_path, '*.jpg')))
    val_images = len(glob.glob(os.path.join(val_path, '*.jpg')))
    
    train_labels_dir = train_path.replace('images', 'labels')
    val_labels_dir = val_path.replace('images', 'labels')
    
    train_labels = len(glob.glob(os.path.join(train_labels_dir, '*.txt')))
    val_labels = len(glob.glob(os.path.join(val_labels_dir, '*.txt')))
    
    num_classes = len(data_config.get('names', {}))
    
    print(f"📊 Dataset Overview:")
    print(f"  Training: {train_images} images, {train_labels} labels")
    print(f"  Validation: {val_images} images, {val_labels} labels")
    print(f"  Classes: {num_classes}")
    
    # Check GPU
    print(f"\n🖥️  Hardware:")
    if torch.cuda.is_available():
        print(f"  ✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"  ✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print(f"  ⚠️  No GPU detected! Training will be VERY slow.")
    
    # Validation checks
    issues = []
    warnings = []
    
    if train_images == 0:
        issues.append("No training images found")
    elif train_images < 50:
        warnings.append(f"Only {train_images} training images (need 50+ minimum, 500+ recommended)")
    elif train_images < 500:
        warnings.append(f"Only {train_images} training images (recommended: 500+)")
    
    if val_images == 0:
        issues.append("No validation images found")
    
    if train_labels != train_images:
        issues.append(f"Mismatch: {train_images} images but {train_labels} labels")
    
    if val_labels != val_images:
        warnings.append(f"Validation: {val_images} images but {val_labels} labels")
    
    # Check if validation labels are empty
    empty_val_labels = 0
    for label_file in glob.glob(os.path.join(val_labels_dir, '*.txt')):
        if os.path.getsize(label_file) == 0:
            empty_val_labels += 1
    
    if empty_val_labels == val_labels and val_labels > 0:
        warnings.append("All validation labels are EMPTY! Metrics will be meaningless.")
    
    print(f"\n⚠️  Issues & Warnings:")
    if not issues and not warnings:
        print("  ✅ No issues found!")
    else:
        for issue in issues:
            print(f"  ❌ CRITICAL: {issue}")
        for warning in warnings:
            print(f"  ⚠️  WARNING: {warning}")
    
    if issues:
        print("\n" + "="*60)
        print("❌ TRAINING BLOCKED: Fix critical issues above!")
        print("="*60)
        return False
    
    if warnings:
        print("\n" + "="*60)
        print("⚠️  Warnings detected - proceeding with training anyway...")
        print("="*60)
    
    print("="*60 + "\n")
    return True

def train():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_yaml = os.path.join(script_dir, 'data', 'vista.yaml')
    
    # Validate dataset BEFORE training
    if not validate_dataset(data_yaml):
        print("\n❌ Training aborted due to validation failures.")
        print("Fix the issues above and try again.")
        sys.exit(1)
    
    # Load the "Small" model (Optimal for RTX 3050)
    model = YOLO('yolov8s.pt') 

    # Determine device (GPU or CPU)
    if torch.cuda.is_available():
        device = 0
        device_name = torch.cuda.get_device_name(0)
    else:
        device = 'cpu'
        device_name = 'CPU'
    
    print(f"🚀 Starting Training on {device_name}...")
    
    # Create absolute path for results
    results_dir = os.path.join(script_dir, 'results', 'training')
    os.makedirs(results_dir, exist_ok=True)
    
    results = model.train(
        data=data_yaml,
        
        # --- HARDWARE TUNING (6GB VRAM) ---
        epochs=40,            # 40 epochs is usually enough for transfer learning
        imgsz=640,            # Do not go higher than 640 or VRAM will crash
        batch=8,              # Reduced to prevent memory issues
        device=device,        # Auto-detect GPU or use CPU
        workers=0,            # 0 = No multiprocessing (fixes MemoryError)
        
        # --- THE WINNING STRATEGY (Augmentation) ---
        # "Mosaic" stitches 4 images together -> Simulates clutter
        mosaic=1.0,           
        # "Mixup" blends images -> Simulates transparency/shadows
        mixup=0.1,            
        
        project=results_dir,
        name='v1_training',
        exist_ok=True,
        
        # Enable validation and plotting
        val=True,
        plots=True,
        save=True,
        save_period=-1  # Only save best and last (avoid I/O issues with frequent saves)
    )
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETED!")
    print("="*60)
    
    # Load best model and run validation
    print("\n📊 Evaluating Best Model...")
    best_model_path = os.path.join(results_dir, 'v1_training', 'weights', 'best.pt')
    
    if os.path.exists(best_model_path):
        best_model = YOLO(best_model_path)
        metrics = best_model.val(data=data_yaml)
        
        print("\n📈 FINAL METRICS:")
        print(f"  Precision: {metrics.box.mp:.3f}")
        print(f"  Recall: {metrics.box.mr:.3f}")
        print(f"  mAP@50: {metrics.box.map50:.3f}")
        print(f"  mAP@50-95: {metrics.box.map:.3f}")
        
        # Interpret results
        print("\n🎯 PERFORMANCE ASSESSMENT:")
        if metrics.box.map50 == 0:
            print("  ❌ CRITICAL: Model has ZERO accuracy!")
            print("  Possible causes:")
            print("    - Dataset too small (need 100+ images minimum)")
            print("    - Labels incorrect or empty")
            print("    - Training configuration issue")
            print("  ⚠️  DO NOT use this model for inference!")
        elif metrics.box.map50 < 0.3:
            print("  ⚠️  Poor performance (mAP@50 < 0.3)")
            print("  Consider: More data, longer training, or different augmentations")
        elif metrics.box.map50 < 0.6:
            print("  👍 Moderate performance (mAP@50: 0.3-0.6)")
            print("  Can be improved with more data or tuning")
        else:
            print("  ✅ Good performance (mAP@50 > 0.6)")
            print("  Ready for inference!")
        
        print("\n📁 Model saved to:", best_model_path)
    else:
        print("⚠️  Best model not found. Check training logs for errors.")
    
    print("\n📋 NEXT STEPS:")
    print(f"1. Review training plots in {os.path.join(results_dir, 'v1_training')}")
    print("2. If mAP > 0, run: python inference.py")
    print("3. If mAP = 0, acquire more training data and retrain")
    print("="*60)

if __name__ == '__main__':
    train()
