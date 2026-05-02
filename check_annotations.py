"""
Check annotations availability for unannotated images
"""
import json
from pathlib import Path

# Load annotations
with open('data/raw_annotations/train_annotations.json', 'r') as f:
    data = json.load(f)

print("=" * 70)
print("ANNOTATION STATUS CHECK")
print("=" * 70)

print(f"\n📋 Annotation JSON contains:")
print(f"   - Total images: {len(data['images'])}")
print(f"   - Total annotations: {len(data['annotations'])}")
print(f"   - Categories: {len(data['categories'])}")

# Get image filenames from JSON
json_images = {img['file_name'] for img in data['images']}

# Check train directory
train_dir = Path('data/images/train')
train_images = {f.name for f in train_dir.glob('*.jpg') if not f.name.endswith('_aug*.jpg')}

# Check train_unannotated directory  
unannotated_dir = Path('data/images/train_unannotated')
unannotated_images = {f.name for f in unannotated_dir.glob('*.jpg')}

print(f"\n📁 Current dataset:")
print(f"   - Training (labeled): {len(train_images)} images")
print(f"   - Unannotated: {len(unannotated_images)} images")

# Check if unannotated images have annotations in JSON
unannotated_with_json = json_images & unannotated_images
unannotated_without_json = unannotated_images - json_images

print(f"\n🔍 Unannotated directory analysis:")
print(f"   - Images WITH annotations in JSON: {len(unannotated_with_json)}")
print(f"   - Images WITHOUT annotations in JSON: {len(unannotated_without_json)}")

if unannotated_with_json:
    print(f"\n✅ Good news! {len(unannotated_with_json)} images can be converted to training data!")
    print("   Run: python expand_dataset.py --target 100")
else:
    print("\n❌ None of the unannotated images have annotations in the JSON file.")
    print("   These images cannot be used for supervised training without manual annotation.")

print("=" * 70)
