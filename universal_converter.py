import json
import os
from tqdm import tqdm

# --- CONFIGURATION (VERIFY THESE PATHS!) ---
IMAGES_DIR = 'data/images/train/'       # Folder containing your 853 images
JSON_FILE  = 'data/raw_annotations/train_annotations.json' # Your uploaded JSON
OUTPUT_DIR = 'data/labels/train/'       # Where .txt files will be saved
# ---------------------------------------------

def fix_dataset():
    # 1. Load Annotations (which includes categories)
    print(f"📖 Loading Annotations from {JSON_FILE}...")
    with open(JSON_FILE) as f:
        data = json.load(f)
    
    # 2. Load Categories to create ID -> Index Map
    print(f"📖 Processing Categories...")
    cat_data = data
    
    # Map real category_id to YOLO index
    id_to_idx = {}
    
    # Handle if 'categories' key exists or if it's a direct list
    cats_list = cat_data.get('categories', []) if isinstance(cat_data, dict) else cat_data
    
    for entry in cats_list:
        if 'ind' in entry:
            # TRUST THE 'ind' FIELD
            id_to_idx[entry['id']] = entry['ind']
        else:
            # Use the category ID directly as the index (0-based)
            id_to_idx[entry['id']] = entry['id']

    print(f"✅ Mapped {len(id_to_idx)} categories.")

    # 3. Create Robust Image Map (Force String IDs)
    # This fixes the "Int vs String" bug
    img_map = {str(img['id']): img for img in data['images']}
    print(f"✅ Loaded metadata for {len(img_map)} images.")

    # 4. Filter: Only process images that actually exist on disk
    if not os.path.exists(IMAGES_DIR):
        print(f"❌ ERROR: Image directory not found at {IMAGES_DIR}")
        return
        
    existing_files = set(os.listdir(IMAGES_DIR))
    print(f"📂 Found {len(existing_files)} images in {IMAGES_DIR}.")

    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    # 5. Process Annotations
    matches = 0
    misses = 0
    
    print("🔄 Generating YOLO labels...")
    for ann in tqdm(data['annotations']):
        img_id = str(ann['image_id'])
        
        # Check 1: Does this image ID exist in JSON?
        if img_id not in img_map:
            continue
            
        img_info = img_map[img_id]
        file_name = img_info['file_name']
        
        # Check 2: Do you actually have this file on your laptop?
        if file_name not in existing_files:
            continue

        # Prepare Data
        w_img = img_info['width']
        h_img = img_info['height']
        x, y, w, h = ann['bbox']
        
        # Normalize for YOLO (Center X, Center Y, Width, Height)
        x_c = (x + w / 2) / w_img
        y_c = (y + h / 2) / h_img
        w_n = w / w_img
        h_n = h / h_img
        
        # Get Correct Class Index
        cat_id = ann['category_id']
        class_idx = id_to_idx.get(cat_id, -1)
        
        if class_idx == -1:
            continue # Skip unknown classes

        # Write to TXT
        txt_name = file_name.replace('.jpg', '.txt')
        with open(os.path.join(OUTPUT_DIR, txt_name), 'a') as f:
            f.write(f"{class_idx} {x_c} {y_c} {w_n} {h_n}\n")
        
        matches += 1

    print(f"\n🎉 DONE! Generated {matches} label lines.")
    
    # 6. Verification
    generated_files = len(os.listdir(OUTPUT_DIR))
    print(f"📊 Statistics: Created {generated_files} label files for {len(existing_files)} images.")

if __name__ == '__main__':
    fix_dataset()
