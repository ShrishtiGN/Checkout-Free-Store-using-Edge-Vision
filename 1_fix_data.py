import json
import os
from tqdm import tqdm

# --- CONFIGURATION ---
IMAGES_DIR = 'data/images/train/' 
JSON_FILE  = 'data/raw_annotations/train_annotations.json'
OUTPUT_DIR = 'data/labels/train/'

def fix_dataset():
    print("🚀 PHASE 1: Data Engineering Initiated...")
    
    # 1. Load Annotations and Categories
    print("📖 Loading Annotations and Categories...")
    with open(JSON_FILE) as f:
        data = json.load(f)
    
    # Extract categories from the same JSON file
    cat_map = {}
    cats_list = data.get('categories', [])
    
    for c in cats_list:
        # Use 'ind' if available, else use id directly (already 0-indexed)
        idx = c.get('ind', c['id']) 
        cat_map[c['id']] = {'name': c['name'], 'idx': idx}

    print(f"✅ Loaded {len(cat_map)} categories.")

    # 2. Create Image Lookup Map (ID -> Filename)
    # Force str(id) to fix "Integer vs String" bug
    img_map = {str(img['id']): img for img in data['images']}
    
    # 3. Check for existing files on disk
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    existing_files = set(os.listdir(IMAGES_DIR))
    
    print("🔄 Generating YOLO Labels...")
    count = 0
    skipped = 0
    
    for ann in tqdm(data['annotations']):
        img_id = str(ann['image_id'])
        
        # Validation checks
        if img_id not in img_map: continue
        fname = img_map[img_id]['file_name']
        if fname not in existing_files: continue
        
        # Geometry Normalization
        w_img, h_img = img_map[img_id]['width'], img_map[img_id]['height']
        x, y, w, h = ann['bbox']
        
        # Convert to YOLO (Center_X, Center_Y, Width, Height) - Normalized 0-1
        xc, yc = (x + w/2)/w_img, (y + h/2)/h_img
        wn, hn = w/w_img, h/h_img
        
        # Class Index Lookup
        cid = ann['category_id']
        if cid not in cat_map: 
            skipped += 1
            continue
            
        class_idx = cat_map[cid]['idx']
        
        # Write to .txt file
        txt_name = fname.replace('.jpg', '.txt')
        with open(os.path.join(OUTPUT_DIR, txt_name), 'a') as f:
            f.write(f"{class_idx} {xc} {yc} {wn} {hn}\n")
        count += 1
        
    print(f"✅ Generated {count} label lines (Skipped {skipped} unknown classes).")
    
    # 5. Generate YAML Config
    print("📝 Generaing 'data/vista.yaml'...")
    sorted_cats = sorted(cat_map.values(), key=lambda x: x['idx'])
    
    # Use absolute path to avoid "file not found" errors
    abs_path = os.path.abspath('data')
    
    with open('data/vista.yaml', 'w') as f:
        f.write(f"path: {abs_path}\n")
        f.write("train: images/train\n")
        f.write("val: images/train\n")  # Use train as val (since we have no labeled val data)
        f.write("test: images/test\n\n")
        f.write("names:\n")
        for c in sorted_cats:
            # Enforce quotes for names to handle weird characters
            f.write(f"  {c['idx']}: \"{c['name']}\"\n")
            
    print("✅ Configuration saved. Ready for training!")

if __name__ == '__main__':
    fix_dataset()
