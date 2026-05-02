"""
Automated Dataset Expansion and Training Pipeline
Expands dataset → Validates → Trains model with optimal settings
"""

import os
import sys
import subprocess
from pathlib import Path
import yaml


def run_command(command, description):
    """Run a command and show progress"""
    print(f"\n{'=' * 60}")
    print(f"🔧 {description}")
    print(f"{'=' * 60}")
    
    result = subprocess.run(command, shell=True)
    
    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed!")
        return False
    
    print(f"✅ {description} completed successfully!")
    return True


def count_dataset_files(split="train"):
    """Count images and labels in a split"""
    images_dir = Path("data") / "images" / split
    labels_dir = Path("data") / "labels" / split
    
    img_count = len(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))) if images_dir.exists() else 0
    label_count = len(list(labels_dir.glob("*.txt"))) if labels_dir.exists() else 0
    
    return img_count, label_count


def get_recommended_training_params(train_images):
    """Get optimal training parameters based on dataset size"""
    if train_images < 20:
        return {
            'epochs': 100,
            'batch': 4,
            'imgsz': 640,
            'patience': 30,
            'mosaic': 1.0,
            'mixup': 0.2,
            'augment': True,
            'comment': 'Very small dataset - high augmentation needed'
        }
    elif train_images < 50:
        return {
            'epochs': 80,
            'batch': 8,
            'imgsz': 640,
            'patience': 25,
            'mosaic': 1.0,
            'mixup': 0.15,
            'augment': True,
            'comment': 'Small dataset - strong augmentation'
        }
    elif train_images < 100:
        return {
            'epochs': 60,
            'batch': 8,
            'imgsz': 640,
            'patience': 20,
            'mosaic': 1.0,
            'mixup': 0.1,
            'augment': True,
            'comment': 'Medium dataset - moderate augmentation'
        }
    else:
        return {
            'epochs': 50,
            'batch': 16,
            'imgsz': 640,
            'patience': 15,
            'mosaic': 0.8,
            'mixup': 0.05,
            'augment': True,
            'comment': 'Large dataset - standard augmentation'
        }


def main():
    print("\n" + "=" * 60)
    print("🚀 RETAILEYE - AUTOMATED EXPANSION & TRAINING")
    print("=" * 60)
    
    # Step 1: Check current dataset status
    print("\n[STEP 1] Checking current dataset...")
    train_imgs, train_lbls = count_dataset_files("train")
    val_imgs, val_lbls = count_dataset_files("val")
    
    print(f"  Current Training: {train_imgs} images, {train_lbls} labels")
    print(f"  Current Validation: {val_imgs} images, {val_lbls} labels")
    print(f"  Total Annotated: {train_imgs + val_imgs}")
    
    # Step 2: Determine target size
    current_total = train_imgs + val_imgs
    
    if current_total < 20:
        recommended_target = 50
        print(f"\n⚠️  WARNING: Very small dataset ({current_total} images)")
    elif current_total < 50:
        recommended_target = 100
        print(f"\n⚠️  Dataset is small ({current_total} images)")
    elif current_total < 100:
        recommended_target = 150
        print(f"\n✓ Dataset is moderate ({current_total} images)")
    else:
        recommended_target = current_total + 50
        print(f"\n✅ Dataset is good ({current_total} images)")
    
    print(f"  Recommended target: {recommended_target} images")
    
    # Get user input
    print("\n" + "=" * 60)
    target_input = input(f"Enter target number of images (press Enter for {recommended_target}): ").strip()
    
    if target_input:
        try:
            target = int(target_input)
        except ValueError:
            print("Invalid input, using recommended target")
            target = recommended_target
    else:
        target = recommended_target
    
    # Step 3: Expand dataset if needed
    if target > current_total:
        needed = target - current_total
        print(f"\n[STEP 2] Expanding dataset by {needed} images...")
        
        if not run_command(
            f"python expand_dataset.py --target {target}",
            f"Expand dataset to {target} images"
        ):
            print("\n❌ Dataset expansion failed!")
            return False
        
        # Recount after expansion
        train_imgs, train_lbls = count_dataset_files("train")
        val_imgs, val_lbls = count_dataset_files("val")
        print(f"\n✅ New Training: {train_imgs} images, {train_lbls} labels")
        print(f"✅ New Validation: {val_imgs} images, {val_lbls} labels")
    else:
        print(f"\n[STEP 2] Dataset already has {current_total} images (target: {target})")
        print("  Skipping expansion...")
    
    # Step 4: Get optimal training parameters
    print(f"\n[STEP 3] Calculating optimal training parameters...")
    params = get_recommended_training_params(train_imgs)
    
    print(f"\n📊 Recommended Training Configuration:")
    print(f"  Epochs: {params['epochs']}")
    print(f"  Batch Size: {params['batch']}")
    print(f"  Image Size: {params['imgsz']}")
    print(f"  Mosaic Aug: {params['mosaic']}")
    print(f"  Mixup Aug: {params['mixup']}")
    print(f"  Note: {params['comment']}")
    
    # Step 5: Ask to proceed with training
    print("\n" + "=" * 60)
    proceed = input("Proceed with training? (yes/no): ").strip().lower()
    
    if proceed != 'yes':
        print("\nTraining cancelled. Run 'python train_model.py' when ready.")
        return True
    
    # Step 6: Create updated training script
    print(f"\n[STEP 4] Starting optimized training...")
    
    from ultralytics import YOLO
    import torch
    
    # Hardware check
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        device = 0
    else:
        print("⚠️  No GPU - using CPU (will be slower)")
        device = 'cpu'
    
    # Load model
    model_path = 'yolov8s.pt'
    if not os.path.exists(model_path):
        print(f"⚠️  {model_path} not found, downloading...")
    
    model = YOLO(model_path)
    
    # Train with optimized parameters
    print("\n🚀 Training started...")
    
    results = model.train(
        data='data/vista.yaml',
        epochs=params['epochs'],
        imgsz=params['imgsz'],
        batch=params['batch'],
        device=device,
        workers=0,
        
        # Augmentation
        mosaic=params['mosaic'],
        mixup=params['mixup'],
        augment=params['augment'],
        
        # Optimization
        patience=params['patience'],
        save=True,
        save_period=-1,  # Only save last and best
        
        # Output
        project='runs/detect',
        name=f'RetailEye_Runs/expanded_{train_imgs}imgs',
        exist_ok=True,
        
        val=True,
        plots=True,
    )
    
    # Step 7: Evaluate results
    print("\n[STEP 5] Evaluating trained model...")
    
    best_model_path = f'runs/detect/RetailEye_Runs/expanded_{train_imgs}imgs/weights/best.pt'
    
    if os.path.exists(best_model_path):
        best_model = YOLO(best_model_path)
        metrics = best_model.val(data='data/vista.yaml')
        
        print("\n" + "=" * 60)
        print("📈 TRAINING RESULTS")
        print("=" * 60)
        print(f"Dataset Size: {train_imgs} training, {val_imgs} validation")
        print(f"Precision: {metrics.box.mp:.3f}")
        print(f"Recall: {metrics.box.mr:.3f}")
        print(f"mAP@50: {metrics.box.map50:.3f}")
        print(f"mAP@50-95: {metrics.box.map:.3f}")
        
        # Performance assessment
        print("\n🎯 ASSESSMENT:")
        if metrics.box.map50 == 0:
            print("  ❌ Model failed to learn (mAP = 0)")
            print("  Solutions:")
            print("    1. Need MORE training images (current: {train_imgs})")
            print("    2. Verify labels are correct")
            print("    3. Train longer with more augmentation")
            return False
        elif metrics.box.map50 < 0.3:
            print("  ⚠️  Poor performance (mAP < 0.3)")
            print("  Recommendations:")
            print(f"    1. Add more images (currently {train_imgs}, aim for 100+)")
            print("    2. Train for more epochs")
            print("    3. Check data quality")
        elif metrics.box.map50 < 0.5:
            print("  👍 Moderate performance (mAP: 0.3-0.5)")
            print("  Can be improved with more data")
        else:
            print("  ✅ Good performance (mAP > 0.5)")
            print("  Model is ready for inference!")
        
        print(f"\n💾 Best model saved: {best_model_path}")
        print("\n📋 NEXT STEPS:")
        print("  1. Review plots in the training folder")
        print("  2. Run: python inference.py")
        print("  3. To improve further, expand dataset and retrain")
        print("=" * 60)
        
        return True
    else:
        print("\n❌ Training completed but best model not found!")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
