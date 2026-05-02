"""
FIX VALIDATION SET - Create proper train/val split from annotated images
"""

import os
import glob
import shutil
import json
from pathlib import Path

# CONFIGURATION
JSON_PATH = 'data/raw_annotations/train_annotations.json'
TRAIN_IMG_DIR = 'data/images/train/'
VAL_IMG_DIR = 'data/images/val/'
TRAIN_LABEL_DIR = 'data/labels/train/'
VAL_LABEL_DIR = 'data/labels/val/'
VAL_SPLIT = 0.2  # 20% for validation

def fix_validation_split():
    print("="*70)
    print("🔧 FIXING VALIDATION SET")
    print("="*70)
    
    # Load JSON to get annotated images
    with open(JSON_PATH) as f:
        data = json.load(f)
    
    annotated_images = {img['file_name'] for img in data['images']}
    
    # Get current train images that have annotations
    train_imgs = [os.path.basename(f) for f in glob.glob(os.path.join(TRAIN_IMG_DIR, '*.jpg'))]
    annotated_in_train = [img for img in train_imgs if img in annotated_images]
    
    print(f"\n📊 CURRENT STATUS:")
    print(f"  Total annotated images: {len(annotated_images)}")
    print(f"  Annotated images in train: {len(annotated_in_train)}")
    
    # Calculate split
    num_val = max(1, int(len(annotated_in_train) * VAL_SPLIT))
    num_train = len(annotated_in_train) - num_val
    
    print(f"\n📋 PROPOSED SPLIT ({int(VAL_SPLIT*100)}% validation):")
    print(f"  Training: {num_train} images")
    print(f"  Validation: {num_val} images")
    
    if num_val < 2:
        print(f"\n⚠️  WARNING: Only {num_val} validation image(s)!")
        print(f"   With only {len(annotated_in_train)} total annotated images,")
        print(f"   this split may not be ideal.")
    
    # Ask for confirmation
    response = input(f"\n🔧 Proceed with creating {num_val} image validation set? (yes/no): ")
    
    if response.lower() != 'yes':
        print("Operation cancelled.")
        return
    
    # Clear current val folder
    print(f"\n🧹 Clearing current validation folder...")
    for img in glob.glob(os.path.join(VAL_IMG_DIR, '*.jpg')):
        os.remove(img)
    for label in glob.glob(os.path.join(VAL_LABEL_DIR, '*.txt')):
        os.remove(label)
    
    # Select images for validation (take last N images)
    # Sorted to ensure consistency
    sorted_annotated = sorted(annotated_in_train)
    val_images = sorted_annotated[-num_val:]
    
    print(f"\n📦 Moving {num_val} images to validation set...")
    moved_count = 0
    
    for img_filename in val_images:
        # Move image
        img_src = os.path.join(TRAIN_IMG_DIR, img_filename)
        img_dst = os.path.join(VAL_IMG_DIR, img_filename)
        
        if os.path.exists(img_src):
            shutil.move(img_src, img_dst)
            
            # Move corresponding label
            label_filename = img_filename.replace('.jpg', '.txt')
            label_src = os.path.join(TRAIN_LABEL_DIR, label_filename)
            label_dst = os.path.join(VAL_LABEL_DIR, label_filename)
            
            if os.path.exists(label_src):
                shutil.move(label_src, label_dst)
                moved_count += 1
                print(f"  ✓ Moved {img_filename}")
            else:
                print(f"  ⚠️  Label not found for {img_filename}")
        else:
            print(f"  ⚠️  Image not found: {img_filename}")
    
    print(f"\n✅ VALIDATION SET CREATED!")
    print(f"  Moved {moved_count} image-label pairs to validation")
    print(f"  Training images remaining: {len(annotated_in_train) - moved_count}")
    
    # Verify
    print(f"\n🔍 VERIFICATION:")
    train_imgs_after = len(glob.glob(os.path.join(TRAIN_IMG_DIR, '*.jpg')))
    val_imgs_after = len(glob.glob(os.path.join(VAL_IMG_DIR, '*.jpg')))
    train_labels_after = len(glob.glob(os.path.join(TRAIN_LABEL_DIR, '*.txt')))
    val_labels_after = len(glob.glob(os.path.join(VAL_LABEL_DIR, '*.txt')))
    
    print(f"  Train: {train_imgs_after} images, {train_labels_after} labels")
    print(f"  Val: {val_imgs_after} images, {val_labels_after} labels")
    
    if val_imgs_after == val_labels_after and val_imgs_after == num_val:
        print(f"\n✅ SUCCESS! Validation set properly configured.")
    else:
        print(f"\n⚠️  WARNING: Image-label count mismatch detected!")
    
    print(f"\n📋 NEXT STEPS:")
    print(f"  1. Verify your vista.yaml file points to correct directories")
    print(f"  2. Run: python verify_setup.py")
    print(f"  3. Run: python train_model.py")
    print("="*70)

if __name__ == '__main__':
    fix_validation_split()
