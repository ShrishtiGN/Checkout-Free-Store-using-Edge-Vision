"""
RetailEye Setup Verification Script
Checks if your environment is ready for training
"""
import sys
import os

def check_gpu():
    print("\n🔍 Checking GPU Configuration...")
    try:
        import torch
        print(f"   ✅ PyTorch Version: {torch.__version__}")
        print(f"   ✅ CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   ✅ GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"   ✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            print("   ⚠️  WARNING: CUDA not available! Training will be SLOW on CPU.")
            return False
    except ImportError:
        print("   ❌ PyTorch not installed!")
        return False
    return True

def check_packages():
    print("\n📦 Checking Required Packages...")
    required = ['ultralytics', 'pandas', 'cv2', 'tqdm', 'numpy']
    all_good = True
    
    for package in required:
        try:
            if package == 'cv2':
                import cv2
                print(f"   ✅ opencv-python: {cv2.__version__}")
            else:
                mod = __import__(package)
                version = getattr(mod, '__version__', 'unknown')
                print(f"   ✅ {package}: {version}")
        except ImportError:
            print(f"   ❌ {package} not installed!")
            all_good = False
    
    return all_good

def check_directories():
    print("\n📁 Checking Directory Structure...")
    required_dirs = [
        'data/images/train',
        'data/images/val',
        'data/images/test',
        'data/labels/train',
        'data/labels/val',
        'data/raw_annotations',
        'submissions'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"   ✅ {dir_path}")
        else:
            print(f"   ❌ {dir_path} missing!")
            all_exist = False
    
    return all_exist

def check_data_files():
    print("\n📊 Checking Data Files...")
    
    # Check annotations
    json_file = 'data/raw_annotations/train_annotations.json'
    if os.path.exists(json_file):
        print(f"   ✅ {json_file} found")
        
        # Load and check annotation quality
        try:
            import json
            with open(json_file) as f:
                data = json.load(f)
            
            num_images = len(data.get('images', []))
            num_annotations = len(data.get('annotations', []))
            num_categories = len(data.get('categories', []))
            
            print(f"      Images: {num_images}")
            print(f"      Annotations: {num_annotations}")
            print(f"      Categories: {num_categories}")
            
            if num_images < 100:
                print(f"   ⚠️  WARNING: Only {num_images} images (need 100+ minimum)")
            if num_images < 500:
                print(f"   ⚠️  RECOMMENDATION: Add more data (current: {num_images}, recommended: 500+)")
        except Exception as e:
            print(f"   ⚠️  Could not parse JSON: {e}")
    else:
        print(f"   ⚠️  {json_file} NOT FOUND - You need to add this before running convert_data.py")
    
    # Check training images
    train_imgs = len(os.listdir('data/images/train')) if os.path.exists('data/images/train') else 0
    val_imgs = len(os.listdir('data/images/val')) if os.path.exists('data/images/val') else 0
    test_imgs = len(os.listdir('data/images/test')) if os.path.exists('data/images/test') else 0
    
    # Check labels
    train_labels = len(os.listdir('data/labels/train')) if os.path.exists('data/labels/train') else 0
    val_labels = len(os.listdir('data/labels/val')) if os.path.exists('data/labels/val') else 0
    
    print(f"   📷 Training images: {train_imgs} | Labels: {train_labels}")
    print(f"   📷 Validation images: {val_imgs} | Labels: {val_labels}")
    print(f"   📷 Test images: {test_imgs}")
    
    if train_imgs == 0:
        print("   ⚠️  No training images found - Add images to data/images/train/")
        return False
    if val_imgs == 0:
        print("   ⚠️  No validation images - Move 20% of training images to data/images/val/")
        return False
    if test_imgs == 0:
        print("   ⚠️  No test images found - Add images to data/images/test/")
    
    # Check if labels match images
    if train_imgs != train_labels:
        print(f"   ⚠️  WARNING: Image/label mismatch - {train_imgs} images but {train_labels} labels")
        print(f"      Run: python convert_data.py")
    
    # Check for empty validation labels
    if val_labels > 0:
        empty_val_labels = 0
        import glob
        for label_file in glob.glob('data/labels/val/*.txt'):
            if os.path.getsize(label_file) == 0:
                empty_val_labels += 1
        
        if empty_val_labels == val_labels:
            print(f"   ⚠️  CRITICAL: All validation labels are EMPTY!")
            print(f"      Validation metrics will be meaningless (all zeros)")
            print(f"      Fix: Split annotated data properly into train/val")
            return False
    
    return train_imgs > 0 and val_imgs > 0

def main():
    print("="*60)
    print("🔧 RetailEye Environment Verification")
    print("="*60)
    
    gpu_ok = check_gpu()
    packages_ok = check_packages()
    dirs_ok = check_directories()
    data_ok = check_data_files()
    
    print("\n" + "="*60)
    print("📋 SUMMARY")
    print("="*60)
    
    if gpu_ok and packages_ok and dirs_ok:
        print("✅ Environment setup is COMPLETE!")
        if data_ok:
            print("✅ Data files are ready!")
            print("\n🚀 Next Steps:")
            print("   1. python convert_data.py")
            print("   2. python train_model.py")
            print("   3. python inference.py")
        else:
            print("\n⚠️  You need to add data files:")
            print("   1. Place train_annotations.json in data/raw_annotations/")
            print("   2. Add training images to data/images/train/")
            print("   3. Move 20% to data/images/val/")
            print("   4. Add test images to data/images/test/")
    else:
        print("❌ Setup incomplete! Fix the errors above.")
    
    print("="*60)

if __name__ == '__main__':
    main()
