import json
import os
import glob

# Load JSON
JSON_PATH = 'data/raw_annotations/train_annotations.json'
with open(JSON_PATH) as f:
    data = json.load(f)

# Get all train images
train_imgs = set([os.path.basename(f) for f in glob.glob('data/images/train/*.jpg')])
print(f"Total images in train folder: {len(train_imgs)}")

# Get all JSON images
json_imgs = {img['file_name'] for img in data['images']}
print(f"Total images in JSON: {len(json_imgs)}")

# Find overlap
found_in_train = json_imgs & train_imgs
not_found_in_train = json_imgs - train_imgs

print(f"\n✅ Annotated images found in train folder: {len(found_in_train)}")
for img in sorted(found_in_train):
    print(f"  {img}")

print(f"\n❌ Annotated images NOT in train folder: {len(not_found_in_train)}")
for img in sorted(not_found_in_train):
    print(f"  {img}")

# Check if they're in val folder
val_imgs = set([os.path.basename(f) for f in glob.glob('data/images/val/*.jpg')])
found_in_val = not_found_in_train & val_imgs
print(f"\n📁 Of the missing images, {len(found_in_val)} are in val folder:")
for img in sorted(found_in_val):
    print(f"  {img}")

# Check generated labels for the annotated images in train
print(f"\n--- Checking Label Files for Annotated Images in Train ---")
TRAIN_LABEL_DIR = 'data/labels/train/'
for img in sorted(found_in_train):
    label_file = img.replace('.jpg', '.txt')
    label_path = os.path.join(TRAIN_LABEL_DIR, label_file)
    if os.path.exists(label_path):
        size = os.path.getsize(label_path)
        if size > 0:
            with open(label_path) as f:
                lines = f.readlines()
            print(f"  ✓ {label_file}: {len(lines)} annotations")
        else:
            print(f"  ❌ {label_file}: EMPTY!")
    else:
        print(f"  ❌ {label_file}: MISSING!")
