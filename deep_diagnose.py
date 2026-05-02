import json
import os
import glob

# ADJUST PATH TO YOUR JSON
JSON_PATH = 'data/raw_annotations/train_annotations.json'

with open(JSON_PATH) as f:
    data = json.load(f)

print(f"Total Images in JSON: {len(data['images'])}")
print(f"Total Annotations in JSON: {len(data['annotations'])}")

# Create mapping
images = {img['id']: img for img in data['images']}

print("\n--- Image ID to Filename Mapping ---")
for img_id, img_info in list(images.items())[:5]:
    print(f"ID {img_id} -> {img_info['file_name']}")

# Check which image IDs have annotations
image_ids_with_annotations = set()
annotations_per_image = {}
for ann in data['annotations']:
    img_id = ann['image_id']
    image_ids_with_annotations.add(img_id)
    annotations_per_image[img_id] = annotations_per_image.get(img_id, 0) + 1

print(f"\n--- Images with Annotations ---")
print(f"Total unique image IDs with annotations: {len(image_ids_with_annotations)}")
print(f"Images in dataset: {len(images)}")
print(f"Images WITHOUT annotations: {len(images) - len(image_ids_with_annotations)}")

if len(images) > len(image_ids_with_annotations):
    print("\n⚠️ WARNING: Some images in the JSON have NO annotations!")
    images_without_annotations = set(images.keys()) - image_ids_with_annotations
    print(f"Image IDs without annotations: {images_without_annotations}")
    for img_id in list(images_without_annotations)[:3]:
        print(f"  - {images[img_id]['file_name']}")

# Check annotation distribution
print(f"\n--- Annotation Distribution ---")
for img_id, count in list(annotations_per_image.items())[:10]:
    filename = images[img_id]['file_name']
    print(f"Image {img_id} ({filename}): {count} annotations")

# Check if the train label files exist
print("\n--- Checking Generated Label Files ---")
TRAIN_LABEL_DIR = 'data/labels/train/'
if os.path.exists(TRAIN_LABEL_DIR):
    label_files = glob.glob(os.path.join(TRAIN_LABEL_DIR, '*.txt'))
    print(f"Total label files in train/: {len(label_files)}")
    
    # Check first few label files for content
    empty_labels = 0
    for label_path in label_files[:10]:
        size = os.path.getsize(label_path)
        if size == 0:
            empty_labels += 1
            print(f"  EMPTY: {os.path.basename(label_path)}")
        else:
            with open(label_path) as f:
                lines = f.readlines()
            print(f"  ✓ {os.path.basename(label_path)}: {len(lines)} annotations ({size} bytes)")
    
    if empty_labels > 0:
        print(f"\n🚨 WARNING: {empty_labels} out of {len(label_files[:10])} checked label files are EMPTY!")
else:
    print(f"Label directory does not exist: {TRAIN_LABEL_DIR}")

# Check validation images
print("\n--- Checking Validation Set ---")
VAL_IMG_DIR = 'data/images/val/'
VAL_LABEL_DIR = 'data/labels/val/'
if os.path.exists(VAL_IMG_DIR):
    val_images = glob.glob(os.path.join(VAL_IMG_DIR, '*.jpg'))
    print(f"Total images in val/: {len(val_images)}")
    
    if os.path.exists(VAL_LABEL_DIR):
        val_labels = glob.glob(os.path.join(VAL_LABEL_DIR, '*.txt'))
        print(f"Total label files in val/: {len(val_labels)}")
        
        # Check if val labels are empty
        for label_path in val_labels:
            size = os.path.getsize(label_path)
            filename = os.path.basename(label_path)
            if size == 0:
                print(f"  EMPTY: {filename}")
            else:
                with open(label_path) as f:
                    lines = f.readlines()
                print(f"  ✓ {filename}: {len(lines)} annotations")
