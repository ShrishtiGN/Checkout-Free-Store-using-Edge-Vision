import json
import os
import glob
from tqdm import tqdm
import sys

# CONFIGURATION
JSON_FILE = 'data/raw_annotations/train_annotations.json' 
TRAIN_OUTPUT_DIR = 'data/labels/train/'
VAL_OUTPUT_DIR = 'data/labels/val/'
TRAIN_IMG_DIR = 'data/images/train/'
VAL_IMG_DIR = 'data/images/val/'

# MINIMUM DATASET REQUIREMENTS
MIN_TRAIN_IMAGES = 100  # Absolute minimum for deep learning
RECOMMENDED_TRAIN_IMAGES = 500  # Recommended for good performance
MIN_ANNOTATIONS_PER_CLASS = 50  # Minimum annotations per class

def convert_coco_to_yolo():
    print(f"Loading {JSON_FILE}...")
    if not os.path.exists(JSON_FILE):
        print(f"❌ ERROR: {JSON_FILE} not found!")
        sys.exit(1)
    
    with open(JSON_FILE) as f:
        data = json.load(f)

    # 1. Map Image IDs to Filenames (with type safety)
    # Convert IDs to strings to ensure consistent type matching
    images = {str(img['id']): img for img in data['images']}
    annotated_files = {img['file_name'] for img in data['images']}
    
    # Debug: Show ID types
    if data['images']:
        first_img_id = data['images'][0]['id']
        print(f"\nDebug - Image ID type: {type(first_img_id)} (value: {first_img_id})")
    if data['annotations']:
        first_ann_id = data['annotations'][0]['image_id']
        print(f"Debug - Annotation image_id type: {type(first_ann_id)} (value: {first_ann_id})")
    
    # DATASET VALIDATION
    num_images = len(data['images'])
    num_annotations = len(data['annotations'])
    num_categories = len(data.get('categories', []))
    
    print("\n" + "="*60)
    print("📊 DATASET STATISTICS")
    print("="*60)
    print(f"Images: {num_images}")
    print(f"Annotations: {num_annotations}")
    print(f"Categories: {num_categories}")
    print(f"Avg annotations per image: {num_annotations/num_images:.1f}")
    
    # Check annotations per class
    if 'categories' in data:
        category_counts = {cat['id']: 0 for cat in data['categories']}
        for ann in data['annotations']:
            category_counts[ann['category_id']] += 1
        
        print("\nAnnotations per class:")
        for cat in data['categories']:
            count = category_counts[cat['id']]
            status = "✅" if count >= MIN_ANNOTATIONS_PER_CLASS else "⚠️"
            print(f"  {status} {cat['name']}: {count}")
    
    # WARNINGS
    print("\n" + "="*60)
    print("⚠️  DATASET QUALITY ASSESSMENT")
    print("="*60)
    
    if num_images < MIN_TRAIN_IMAGES:
        print(f"❌ CRITICAL: Only {num_images} images (need {MIN_TRAIN_IMAGES}+ minimum)")
        print(f"   Deep learning requires MUCH more data!")
        print(f"   Recommended: {RECOMMENDED_TRAIN_IMAGES}+ images for production")
        print(f"   Current model will have ZERO accuracy with this data!")
    elif num_images < RECOMMENDED_TRAIN_IMAGES:
        print(f"⚠️  WARNING: {num_images} images (recommended: {RECOMMENDED_TRAIN_IMAGES}+)")
        print(f"   Model may underperform. Consider data augmentation.")
    else:
        print(f"✅ Good: {num_images} images")
    
    if num_annotations / num_images < 2:
        print(f"⚠️  WARNING: Low annotation density ({num_annotations/num_images:.1f} per image)")
    
    if 'categories' in data:
        min_class_count = min(category_counts.values())
        if min_class_count < MIN_ANNOTATIONS_PER_CLASS:
            print(f"❌ CRITICAL: Some classes have < {MIN_ANNOTATIONS_PER_CLASS} annotations")
            print(f"   Model cannot learn from insufficient examples!")
    
    print("="*60 + "\n")
    
    # User confirmation for small datasets
    if num_images < MIN_TRAIN_IMAGES:
        response = input("⚠️  Dataset too small! Continue anyway? (yes/no): ")
        if response.lower() != 'yes':
            print("Conversion cancelled. Please add more training data.")
            sys.exit(1)
    
    # 2. Process Training Annotations
    print("Converting training annotations...")
    annotations_written = 0
    missing_images = set()
    
    for ann in tqdm(data['annotations']):
        # Convert to string for consistent type matching
        img_id = str(ann['image_id'])
        
        # Check if image exists in our mapping
        if img_id not in images:
            missing_images.add(img_id)
            continue
            
        img_info = images[img_id]
        
        # Dimensions
        w_img = img_info['width']
        h_img = img_info['height']
        
        # COCO BBox: [top_left_x, top_left_y, width, height]
        x_min, y_min, bbox_w, bbox_h = ann['bbox']
        
        # YOLO BBox: [center_x, center_y, width, height] (Normalized 0-1)
        x_center = (x_min + bbox_w / 2) / w_img
        y_center = (y_min + bbox_h / 2) / h_img
        w_norm = bbox_w / w_img
        h_norm = bbox_h / h_img
        
        # Class ID (0-indexed as per JSON)
        class_id = ann['category_id'] 

        # Write to TXT file (Same name as image)
        file_name = img_info['file_name'].replace('.jpg', '.txt')
        save_path = os.path.join(TRAIN_OUTPUT_DIR, file_name)
        
        with open(save_path, 'a') as f:
            f.write(f"{class_id} {x_center} {y_center} {w_norm} {h_norm}\n")
        
        annotations_written += 1
    
    # Report any issues
    if missing_images:
        print(f"\n⚠️  WARNING: {len(missing_images)} annotations reference missing image IDs: {missing_images}")
    
    print(f"\n✅ Successfully wrote {annotations_written} annotations to label files")
    
    # 3. Handle validation images
    print("\nProcessing validation labels...")
    val_images = glob.glob(os.path.join(VAL_IMG_DIR, '*.jpg'))
    val_with_annotations = 0
    val_without_annotations = 0
    
    for img_path in val_images:
        img_filename = os.path.basename(img_path)
        label_filename = img_filename.replace('.jpg', '.txt')
        label_path = os.path.join(VAL_OUTPUT_DIR, label_filename)
        
        # Check if this image has annotations in the JSON
        if img_filename in annotated_files:
            print(f"⚠️  CRITICAL: {img_filename} found in val folder but has annotations!")
            print(f"   This image should be in train folder, not val!")
            print(f"   Moving its labels to validation anyway...")
            val_with_annotations += 1
            # Labels were already created in step 2, need to move them
            train_label_path = os.path.join(TRAIN_OUTPUT_DIR, label_filename)
            if os.path.exists(train_label_path):
                import shutil
                shutil.move(train_label_path, label_path)
        else:
            # Create empty label file for validation images without annotations
            # This is OK for test/validation images where we just need predictions
            with open(label_path, 'w') as f:
                pass  # Empty file means no objects in image
            val_without_annotations += 1
    
    print(f"✅ Validation labels: {val_with_annotations} with annotations, {val_without_annotations} without")
    
    # FINAL SUMMARY
    print("\n" + "="*60)
    print("📊 CONVERSION SUMMARY")
    print("="*60)
    print(f"Annotated images in JSON: {len(data['images'])}")
    print(f"Total annotations written: {annotations_written}")
    print(f"Validation images processed: {len(val_images)}")
    print("="*60)
    
    if val_without_annotations == len(val_images):
        print("\n⚠️  WARNING: All validation images have NO annotations!")
        print("   Validation metrics will be meaningless (all zeros).")
        print("   RECOMMENDATION: Split your annotated training data into train/val.")
        print("   Example: 80% train (with labels), 20% val (with labels)")

if __name__ == '__main__':
    # Create output directories
    os.makedirs(TRAIN_OUTPUT_DIR, exist_ok=True)
    os.makedirs(VAL_OUTPUT_DIR, exist_ok=True)
    
    convert_coco_to_yolo()
    
    print("\n" + "="*60)
    print("✅ CONVERSION COMPLETE!")
    print("="*60)
    print(f"Training labels: {TRAIN_OUTPUT_DIR}")
    print(f"Validation labels: {VAL_OUTPUT_DIR}")
    
    # Count created labels
    train_labels = len(glob.glob(os.path.join(TRAIN_OUTPUT_DIR, '*.txt')))
    val_labels = len(glob.glob(os.path.join(VAL_OUTPUT_DIR, '*.txt')))
    
    print(f"\nCreated {train_labels} training labels")
    print(f"Created {val_labels} validation labels")
    
    print("\n📋 NEXT STEPS:")
    print("1. Review the dataset warnings above")
    print("2. If dataset is too small, acquire more data before training")
    print("3. Run: python verify_setup.py")
    print("4. Run: python train_model.py")
    print("="*60)
