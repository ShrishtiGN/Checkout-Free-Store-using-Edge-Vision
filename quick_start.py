"""
Quick Training Launcher - Interactive script to help you train
"""

import os
import sys
from pathlib import Path


def count_files(directory, extension):
    """Count files with specific extension"""
    path = Path(directory)
    if not path.exists():
        return 0
    return len(list(path.glob(f"*.{extension}")))


def display_status():
    """Display current dataset status"""
    train_imgs = count_files("data/images/train", "jpg") + count_files("data/images/train", "png")
    train_lbls = count_files("data/labels/train", "txt")
    val_imgs = count_files("data/images/val", "jpg") + count_files("data/images/val", "png")
    val_lbls = count_files("data/labels/val", "txt")
    
    print("\n" + "=" * 60)
    print("📊 CURRENT DATASET STATUS")
    print("=" * 60)
    print(f"Training:   {train_imgs} images, {train_lbls} labels")
    print(f"Validation: {val_imgs} images, {val_lbls} labels")
    print(f"Total:      {train_imgs + val_imgs} annotated images")
    
    if train_imgs == train_lbls and val_imgs == val_lbls:
        print("✅ All images have matching labels!")
    else:
        print("⚠️  WARNING: Image-label mismatch detected!")
    
    return train_imgs, val_imgs


def show_menu():
    """Show interactive menu"""
    print("\n" + "=" * 60)
    print("🚀 RETAILEYE TRAINING LAUNCHER")
    print("=" * 60)
    print("\nWhat would you like to do?\n")
    print("1. 📊 Check dataset status")
    print("2. 🔄 Augment dataset (create more images)")
    print("3. 🎯 Train model with current dataset")
    print("4. 📈 View training results")
    print("5. 🔍 Run inference on test images")
    print("6. ❓ Show help and tips")
    print("7. 🚪 Exit")
    print("\n" + "=" * 60)
    
    choice = input("\nEnter your choice (1-7): ").strip()
    return choice


def augment_menu():
    """Augmentation submenu"""
    print("\n" + "=" * 60)
    print("🔄 DATASET AUGMENTATION")
    print("=" * 60)
    print("\n1. Light augmentation (5x multiplier)")
    print("2. Moderate augmentation (10x multiplier) [RECOMMENDED]")
    print("3. Heavy augmentation (15x multiplier)")
    print("4. Custom multiplier")
    print("5. Back to main menu")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == "1":
        multiplier = 5
    elif choice == "2":
        multiplier = 10
    elif choice == "3":
        multiplier = 15
    elif choice == "4":
        try:
            multiplier = int(input("Enter multiplier (1-20): "))
            multiplier = max(1, min(20, multiplier))
        except:
            print("Invalid input, using default: 10")
            multiplier = 10
    else:
        return
    
    print(f"\n🔄 Augmenting training data with {multiplier}x multiplier...")
    os.system(f"python augment_dataset.py --split train --multiplier {multiplier}")
    
    input("\nPress Enter to continue...")


def train_menu():
    """Training submenu"""
    train_imgs, val_imgs = display_status()
    
    print("\n" + "=" * 60)
    print("🎯 TRAINING OPTIONS")
    print("=" * 60)
    
    if train_imgs < 20:
        print("\n⚠️  WARNING: Very few images!")
        print("   Recommend: Augment dataset first (Option 2 from main menu)")
        print("   Or training will likely fail")
        
        proceed = input("\nProceed anyway? (yes/no): ").strip().lower()
        if proceed != 'yes':
            return
    elif train_imgs < 50:
        print("\n⚠️  Small dataset detected")
        print("   Recommend: Augment more for better results")
        print("   Current training images: " + str(train_imgs))
    
    print("\n1. Train with augmented dataset script (RECOMMENDED)")
    print("2. Train with original script")
    print("3. Back to main menu")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        print("\n🚀 Starting training with augmented dataset configuration...")
        print("This will take 20-40 minutes on RTX 3050...\n")
        os.system("python train_augmented.py")
    elif choice == "2":
        print("\n🚀 Starting training with original configuration...\n")
        os.system("python train_model.py")
    else:
        return
    
    input("\nPress Enter to continue...")


def view_results():
    """View training results"""
    print("\n" + "=" * 60)
    print("📈 TRAINING RESULTS")
    print("=" * 60)
    
    # Find recent training runs
    runs_dir = Path("runs/detect/RetailEye_Runs")
    
    if not runs_dir.exists():
        print("\n❌ No training runs found!")
        print("   Train a model first (Option 3)")
        input("\nPress Enter to continue...")
        return
    
    # List all runs
    runs = [d for d in runs_dir.iterdir() if d.is_dir()]
    
    if not runs:
        print("\n❌ No training runs found!")
        input("\nPress Enter to continue...")
        return
    
    print(f"\nFound {len(runs)} training run(s):\n")
    
    for i, run in enumerate(runs, 1):
        print(f"{i}. {run.name}")
        
        # Check for results
        results_csv = run / "results.csv"
        best_pt = run / "weights" / "best.pt"
        
        if results_csv.exists():
            print(f"   ✅ Results: {results_csv}")
        if best_pt.exists():
            print(f"   ✅ Best model: {best_pt}")
    
    print("\n💡 To view detailed plots:")
    print(f"   Navigate to: {runs_dir}")
    print("   Open: results.png, confusion_matrix.png, etc.")
    
    input("\nPress Enter to continue...")


def show_help():
    """Show help and tips"""
    print("\n" + "=" * 60)
    print("❓ HELP & TIPS")
    print("=" * 60)
    
    print("\n📚 Quick Guide:")
    print("\n1. FIRST TIME USERS:")
    print("   → Check dataset status (Option 1)")
    print("   → Augment dataset to 100+ images (Option 2)")
    print("   → Train model (Option 3)")
    print("   → Wait 20-40 minutes")
    print("   → Run inference (Option 5)")
    
    print("\n2. IF mAP IS LOW (< 0.4):")
    print("   → Augment more (Option 2, use 15x multiplier)")
    print("   → Or annotate real images from train_unannotated/")
    print("   → Retrain (Option 3)")
    
    print("\n3. IF mAP IS GOOD (> 0.4):")
    print("   → Congratulations! 🎉")
    print("   → Run inference to see detections")
    print("   → Fine-tune with more real images if needed")
    
    print("\n4. RECOMMENDED WORKFLOW:")
    print("   Step 1: python augment_dataset.py --multiplier 10")
    print("   Step 2: python train_augmented.py")
    print("   Step 3: python inference.py")
    
    print("\n5. FILES TO CHECK:")
    print("   - TRAINING_SUMMARY.md (Complete guide)")
    print("   - EXPANSION_GUIDE.md (Dataset expansion)")
    print("   - runs/detect/RetailEye_Runs/ (Training results)")
    
    print("\n6. COMMON ISSUES:")
    print("   • Out of memory → Reduce batch size in script")
    print("   • mAP = 0 → Need more data or longer training")
    print("   • Slow training → Normal on CPU, use GPU if possible")
    
    input("\nPress Enter to continue...")


def main():
    """Main interactive loop"""
    while True:
        try:
            choice = show_menu()
            
            if choice == "1":
                display_status()
                input("\nPress Enter to continue...")
            
            elif choice == "2":
                augment_menu()
            
            elif choice == "3":
                train_menu()
            
            elif choice == "4":
                view_results()
            
            elif choice == "5":
                print("\n🔍 Running inference on test images...")
                os.system("python inference.py")
                input("\nPress Enter to continue...")
            
            elif choice == "6":
                show_help()
            
            elif choice == "7":
                print("\n👋 Goodbye! Happy training!")
                break
            
            else:
                print("\n❌ Invalid choice. Please enter 1-7.")
                input("Press Enter to continue...")
        
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            input("Press Enter to continue...")


if __name__ == "__main__":
    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
