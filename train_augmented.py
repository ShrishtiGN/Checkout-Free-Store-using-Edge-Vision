"""
Optimized Training Script for Expanded Dataset (99 train + 24 val images)
"""

from ultralytics import YOLO
import torch
import os


def train_with_augmented_data():
    print("\n" + "=" * 70)
    print("🚀 RETAILEYE TRAINING - OPTIMIZED FOR AUGMENTED DATASET")
    print("=" * 70)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"\n✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        device = 0
    else:
        print("\n⚠️  No GPU detected - training will be slower")
        device = 'cpu'
    
    # Dataset info
    print("\n📊 Dataset:")
    print("   Training: 99 images (9 original + 90 augmented)")
    print("   Validation: 24 images (4 original + 20 augmented)")
    print("   Total: 123 annotated images")
    
    # Load model
    model_name = 'yolov8s.pt'
    print(f"\n📦 Loading model: {model_name}")
    model = YOLO(model_name)
    
    # Training configuration for ~100 images
    print("\n⚙️  Training Configuration:")
    config = {
        'data': 'data/vista.yaml',
        'epochs': 80,              # Good for augmented dataset
        'imgsz': 640,              # Standard size
        'batch': 8,                # Fits in 6GB VRAM
        'device': device,
        'workers': 0,              # Prevents memory issues
        
        # Moderate augmentation (already have augmented data)
        'mosaic': 0.8,             # Some mosaic for variety
        'mixup': 0.05,             # Light mixup
        'augment': True,           # Enable base augmentations
        'hsv_h': 0.015,            # Hue variation
        'hsv_s': 0.7,              # Saturation variation
        'hsv_v': 0.4,              # Value variation
        'degrees': 5.0,            # Slight rotation
        'translate': 0.1,          # Translation
        'scale': 0.5,              # Scaling
        'flipud': 0.1,             # Vertical flip chance
        'fliplr': 0.5,             # Horizontal flip chance
        
        # Optimization
        'patience': 20,            # Early stopping patience
        'optimizer': 'AdamW',      # Modern optimizer
        'lr0': 0.001,              # Initial learning rate
        'lrf': 0.01,               # Final learning rate
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        
        # Output
        'project': 'runs/detect',
        'name': 'RetailEye_Runs/augmented_v1',
        'exist_ok': True,
        'save': True,
        'save_period': -1,         # Only save best and last
        'val': True,
        'plots': True,
        'verbose': True,
    }
    
    print(f"   Epochs: {config['epochs']}")
    print(f"   Batch Size: {config['batch']}")
    print(f"   Image Size: {config['imgsz']}")
    print(f"   Optimizer: {config['optimizer']}")
    print(f"   Mosaic: {config['mosaic']}, Mixup: {config['mixup']}")
    
    # Start training
    print("\n" + "=" * 70)
    print("🔥 TRAINING STARTED")
    print("=" * 70)
    print("\nThis will take approximately 20-40 minutes on RTX 3050...")
    print("Watch the mAP@50 metric - should improve over epochs\n")
    
    results = model.train(**config)
    
    # Training complete
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETED")
    print("=" * 70)
    
    # Evaluate best model
    best_model_path = 'runs/detect/RetailEye_Runs/augmented_v1/weights/best.pt'
    
    if os.path.exists(best_model_path):
        print("\n📈 Evaluating Best Model...")
        best_model = YOLO(best_model_path)
        metrics = best_model.val(data='data/vista.yaml', workers=0)
        
        print("\n" + "=" * 70)
        print("📊 FINAL RESULTS")
        print("=" * 70)
        print(f"\n💯 Metrics:")
        print(f"   Precision:   {metrics.box.mp:.3f}")
        print(f"   Recall:      {metrics.box.mr:.3f}")
        print(f"   mAP@50:      {metrics.box.map50:.3f}")
        print(f"   mAP@50-95:   {metrics.box.map:.3f}")
        
        # Performance assessment
        print("\n🎯 Performance Assessment:")
        map50 = metrics.box.map50
        
        if map50 == 0:
            print("   ❌ CRITICAL: Model failed to learn (mAP = 0)")
            print("\n   🔧 Solutions:")
            print("      1. Verify labels are correct")
            print("      2. Train for more epochs (100+)")
            print("      3. Try different model (yolov8n.pt)")
            print("      4. Check if images match labels")
        elif map50 < 0.2:
            print(f"   ⚠️  Very Poor (mAP@50 = {map50:.3f})")
            print("\n   🔧 Improvements:")
            print("      1. Create more diverse augmentations")
            print("      2. Train for 100+ epochs")
            print("      3. Review label quality")
        elif map50 < 0.4:
            print(f"   ⚠️  Poor (mAP@50 = {map50:.3f})")
            print("\n   💡 Tips:")
            print("      1. Augment dataset more (multiplier=15)")
            print("      2. Train longer (100 epochs)")
            print("      3. Usable for testing but needs improvement")
        elif map50 < 0.6:
            print(f"   👍 Moderate (mAP@50 = {map50:.3f})")
            print("\n   💡 Tips:")
            print("      1. Model is functional!")
            print("      2. Can be improved with more real images")
            print("      3. Good for initial testing")
        else:
            print(f"   ✅ Excellent (mAP@50 = {map50:.3f})")
            print("\n   🎉 Success!")
            print("      1. Model is ready for production")
            print("      2. Run inference with confidence")
            print("      3. Consider fine-tuning for specific scenarios")
        
        print(f"\n💾 Best Model Saved:")
        print(f"   {best_model_path}")
        
        print("\n📋 Next Steps:")
        print("   1. Review training plots:")
        print("      runs/detect/RetailEye_Runs/augmented_v1/")
        print("   2. Run inference:")
        print("      python inference.py")
        print("   3. If mAP < 0.4, augment more:")
        print("      python augment_dataset.py --multiplier 15")
        
        print("\n" + "=" * 70)
        
        return metrics.box.map50
    else:
        print("\n❌ Error: Best model not found!")
        print("   Check training logs for errors")
        return 0


if __name__ == "__main__":
    try:
        map50 = train_with_augmented_data()
        
        if map50 == 0:
            print("\n⚠️  Training completed but model did not learn")
            print("Review the solutions above and try again")
        elif map50 < 0.4:
            print("\n⚠️  Model needs improvement - consider more augmentation")
        else:
            print("\n✅ Training successful! Model is ready to use")
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
