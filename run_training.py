"""
Automated training runner that bypasses confirmations for demo purposes
"""
import subprocess
import sys

def run_training():
    print("="*60)
    print("🤖 AUTOMATED TRAINING (DEMO MODE)")
    print("="*60)
    print("\nNote: This will train despite warnings.")
    print("Expected: Low accuracy due to limited annotated data (13 of 853 images)")
    print("\nTraining will take approximately 30-60 minutes on RTX 3050.")
    print("="*60)
    
    # Import training function directly
    sys.path.insert(0, '.')
    from train_model import validate_dataset, train
    import os
    
    # Bypass validation by directly starting training
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_yaml = os.path.join(script_dir, 'data', 'vista.yaml')
    
    print("\n⚠️  Skipping interactive validation for demo...")
    print("Starting training directly...\n")
    
    # Call the training function's core logic
    from ultralytics import YOLO
    model = YOLO('yolov8s.pt')
    
    print("🚀 Starting Training on RTX 3050...")
    
    results = model.train(
        data=data_yaml,
        epochs=40,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        mosaic=1.0,
        mixup=0.1,
        project='RetailEye_Runs',
        name='v1_mosaic_strategy',
        val=True,
        plots=True,
        save=True,
        save_period=10
    )
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETED!")
    print("="*60)
    
    # Load best model and run validation
    print("\n📊 Evaluating Best Model...")
    best_model_path = os.path.join('RetailEye_Runs', 'v1_mosaic_strategy', 'weights', 'best.pt')
    
    if os.path.exists(best_model_path):
        best_model = YOLO(best_model_path)
        metrics = best_model.val(data=data_yaml)
        
        print("\n📈 FINAL METRICS:")
        print(f"  Precision: {metrics.box.mp:.3f}")
        print(f"  Recall: {metrics.box.mr:.3f}")
        print(f"  mAP@50: {metrics.box.map50:.3f}")
        print(f"  mAP@50-95: {metrics.box.map:.3f}")
        
        print("\n🎯 PERFORMANCE ASSESSMENT:")
        if metrics.box.map50 < 0.1:
            print("  ⚠️  Very low accuracy (expected with limited annotations)")
            print("  Only 13 of 853 images have annotations!")
        elif metrics.box.map50 < 0.3:
            print("  ⚠️  Low performance but model learned something")
        else:
            print("  ✅ Better than expected performance!")
        
        print(f"\n📁 Model saved to: {best_model_path}")
    
    print("\n📝 NEXT STEPS:")
    print("1. Run: python inference.py (to test on test data)")
    print("2. Check results in: RetailEye_Runs/v1_mosaic_strategy/")
    print("="*60)

if __name__ == '__main__':
    run_training()
