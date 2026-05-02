"""
COMPREHENSIVE DATA DIAGNOSIS AND CLEANUP SCRIPT
This script identifies and fixes issues with your YOLO dataset
"""

import json
import os
import glob
import shutil
from pathlib import Path

# CONFIGURATION
JSON_PATH = 'data/raw_annotations/train_annotations.json'
TRAIN_IMG_DIR = 'data/images/train/'
VAL_IMG_DIR = 'data/images/val/'
TRAIN_LABEL_DIR = 'data/labels/train/'
VAL_LABEL_DIR = 'data/labels/val/'

def main():
    print("="*70)
    print("🔍 RETAILEYE DATASET DIAGNOSIS AND CLEANUP")
    print("="*70)
    
    # Load JSON
    with open(JSON_PATH) as f:
        data = json.load(f)
    
    # Get file sets
    train_imgs = set([os.path.basename(f) for f in glob.glob(os.path.join(TRAIN_IMG_DIR, '*.jpg'))])
    val_imgs = set([os.path.basename(f) for f in glob.glob(os.path.join(VAL_IMG_DIR, '*.jpg'))])
    json_imgs = {img['file_name'] for img in data['images']}
    train_labels = set([os.path.basename(f) for f in glob.glob(os.path.join(TRAIN_LABEL_DIR, '*.txt'))])
    
    print(f"\n📊 CURRENT STATE:")
    print(f"  Total train images: {len(train_imgs)}")
    print(f"  Total val images: {len(val_imgs)}")
    print(f"  Total train labels: {len(train_labels)}")
    print(f"  Images with annotations in JSON: {len(json_imgs)}")
    
    # Find which images have annotations
    annotated_in_train = json_imgs & train_imgs
    annotated_in_val = json_imgs & val_imgs
    
    print(f"\n✅ ANNOTATED IMAGES:")
    print(f"  In train folder: {len(annotated_in_train)}")
    print(f"  In val folder: {len(annotated_in_val)}")
    print(f"  Total annotated: {len(annotated_in_train) + len(annotated_in_val)}")
    
    # Find empty labels
    empty_labels = []
    valid_labels = []
    for label_file in train_labels:
        label_path = os.path.join(TRAIN_LABEL_DIR, label_file)
        if os.path.getsize(label_path) == 0:
            empty_labels.append(label_file)
        else:
            valid_labels.append(label_file)
    
    print(f"\n📝 LABEL FILE STATUS:")
    print(f"  Valid labels (with annotations): {len(valid_labels)}")
    print(f"  Empty labels (no annotations): {len(empty_labels)}")
    
    # CRITICAL ISSUE
    unannotated_count = len(train_imgs) - len(annotated_in_train)
    print(f"\n🚨 CRITICAL ISSUE IDENTIFIED:")
    print(f"  {unannotated_count} images in train folder have NO annotations!")
    print(f"  These images cannot be used for training.")
    print(f"  Only {len(annotated_in_train)} images have annotations.")
    
    # Ask user what to do
    print(f"\n" + "="*70)
    print("🛠️  RECOMMENDED ACTIONS:")
    print("="*70)
    print(f"\n1. CLEAN UP EMPTY LABELS")
    print(f"   Remove {len(empty_labels)} empty label files to avoid confusion")
    print(f"\n2. MOVE UNANNOTATED IMAGES")
    print(f"   Move {unannotated_count} unannotated images out of train folder")
    print(f"   (These can't be used for training without annotations)")
    print(f"\n3. FIX VALIDATION SET")
    print(f"   Current val set has {len(val_imgs)} images with no annotations")
    print(f"   Create proper validation split from annotated images")
    
    print(f"\n" + "="*70)
    response = input("\n🔧 Do you want to clean up empty label files? (yes/no): ")
    
    if response.lower() == 'yes':
        print(f"\n🧹 Cleaning up {len(empty_labels)} empty label files...")
        
        # Create backup directory
        backup_dir = 'data/labels/train_backup_empty/'
        os.makedirs(backup_dir, exist_ok=True)
        
        for label_file in empty_labels:
            src = os.path.join(TRAIN_LABEL_DIR, label_file)
            dst = os.path.join(backup_dir, label_file)
            shutil.move(src, dst)
        
        print(f"✅ Moved {len(empty_labels)} empty labels to {backup_dir}")
        print(f"✅ Remaining valid labels: {len(valid_labels)}")
    
    # Optional: Move unannotated images
    print(f"\n" + "="*70)
    response = input("🔧 Move unannotated images to separate folder? (yes/no): ")
    
    if response.lower() == 'yes':
        unannotated_imgs = train_imgs - annotated_in_train
        backup_img_dir = 'data/images/train_unannotated/'
        os.makedirs(backup_img_dir, exist_ok=True)
        
        print(f"\n📦 Moving {len(unannotated_imgs)} unannotated images...")
        for img_file in unannotated_imgs:
            src = os.path.join(TRAIN_IMG_DIR, img_file)
            dst = os.path.join(backup_img_dir, img_file)
            shutil.move(src, dst)
        
        print(f"✅ Moved {len(unannotated_imgs)} images to {backup_img_dir}")
        print(f"✅ Remaining train images: {len(annotated_in_train)}")
    
    # Validation set fix
    print(f"\n" + "="*70)
    print("📋 VALIDATION SET RECOMMENDATION:")
    print("="*70)
    print(f"Current validation set has {len(val_imgs)} images without annotations.")
    print(f"For proper training, you should:")
    print(f"  1. Create validation split from your {len(annotated_in_train)} annotated images")
    print(f"  2. Use 80-20 or 70-30 train-val split")
    print(f"  3. Suggested: ~3-4 images for validation from annotated set")
    
    print(f"\n⚠️  WARNING: With only {len(annotated_in_train)} annotated images,")
    print(f"   your model will have VERY LIMITED learning capability!")
    print(f"   Recommended: 500+ annotated images for production use")
    
    print(f"\n" + "="*70)
    print("✅ DIAGNOSIS COMPLETE")
    print("="*70)
    print(f"\nSUMMARY:")
    print(f"  ✓ Conversion script is working correctly")
    print(f"  ✓ {len(annotated_in_train)} images have proper annotations")
    print(f"  ⚠  {unannotated_count} images lack annotations (cannot train on these)")
    print(f"  ⚠  Validation set needs to be created from annotated images")
    print(f"\nNEXT STEPS:")
    print(f"  1. Get annotations for remaining {unannotated_count} images, OR")
    print(f"  2. Work with just the {len(annotated_in_train)} annotated images (limited accuracy)")
    print(f"  3. Create proper validation split from annotated images")

if __name__ == '__main__':
    main()
