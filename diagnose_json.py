import json

# ADJUST PATH TO YOUR JSON
JSON_PATH = 'data/raw_annotations/train_annotations.json'

with open(JSON_PATH) as f:
    data = json.load(f)

print(f"Total Images in JSON: {len(data['images'])}")
print(f"Total Annotations in JSON: {len(data['annotations'])}")

# Check ID types for the first item
first_img_id = data['images'][0]['id']
first_ann_img_id = data['annotations'][0]['image_id']

print(f"\nImage ID Type: {type(first_img_id)} (Value: {first_img_id})")
print(f"Annotation ID Type: {type(first_ann_img_id)} (Value: {first_ann_img_id})")

if type(first_img_id) != type(first_ann_img_id):
    print("\n🚨 MISMATCH DETECTED! One is int, one is string. Fix convert_data.py!")
else:
    print("\n✅ Types match. The issue might be filenames.")

# Check a few more samples to be sure
print("\n--- Checking first 5 image IDs ---")
for i in range(min(5, len(data['images']))):
    print(f"Image {i}: ID = {data['images'][i]['id']} (type: {type(data['images'][i]['id'])})")

print("\n--- Checking first 5 annotation image_ids ---")
for i in range(min(5, len(data['annotations']))):
    print(f"Annotation {i}: image_id = {data['annotations'][i]['image_id']} (type: {type(data['annotations'][i]['image_id'])})")
