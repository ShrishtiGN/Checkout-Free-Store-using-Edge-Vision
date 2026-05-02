from ultralytics import YOLO
import os
import glob
import shutil
from tqdm import tqdm

# --- CONFIGURATION ---
# 1. The Teacher: Your best model so far
MODEL_PATH = 'runs/detect/RetailEye_Runs/Mosaic_Model_v1/weights/best.pt'

# 2. The Unlabeled Data: Multiple sources
UNLABELED_DIRS = [
    'data/images/train_unannotated/',  # 1313 images
    'data/images/test/',                # 154 images
]

# 3. The Destination: Where training data lives
TRAIN_IMG_DIR = 'data/images/train/' 
TRAIN_LBL_DIR = 'data/labels/train/'

# 4. Safety Settings
# Very low threshold to maximize pseudo-label generation
CONF_THRESHOLD = 0.01  # 1% confidence - very aggressive

def auto_label():
    print("🚀 PHASE 4: Auto-Labeling (Teacher-Student) - FULL EXPANSION")
    print("=" * 60)
    
    # Validation
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Teacher model not found at {MODEL_PATH}")
        print("   -> Run '2_train.py' first!")
        return

    print(f"👨‍🏫 Loading Teacher Model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    # Collect all unlabeled images from multiple sources
    all_images = []
    for unlabeled_dir in UNLABELED_DIRS:
        if os.path.exists(unlabeled_dir):
            imgs = glob.glob(os.path.join(unlabeled_dir, "*.jpg"))
            all_images.extend(imgs)
            print(f"📂 Found {len(imgs)} images in {unlabeled_dir}")
    
    print(f"📊 Total unlabeled candidates: {len(all_images)}")
    print(f"🎯 Target: Generate labels for all {len(all_images)} images")
    print("=" * 60)

    count = 0
    copied_imgs = 0
    skipped = 0
    
    print("\n🔄 Generating Pseudo-Labels...")
    for img_path in tqdm(all_images, desc="Processing"):
        
        basename = os.path.basename(img_path)
        dest_img_path = os.path.join(TRAIN_IMG_DIR, basename)
        txt_name = basename.replace('.jpg', '.txt')
        txt_save_path = os.path.join(TRAIN_LBL_DIR, txt_name)
        
        # Skip if already processed
        if os.path.exists(txt_save_path) and os.path.exists(dest_img_path):
            skipped += 1
            continue
        
        # Run Inference
        results = model.predict(img_path, conf=CONF_THRESHOLD, verbose=False)
        result = results[0]
        
        # Generate label file even if empty (some images may have no objects)
        with open(txt_save_path, 'w') as f:
            if len(result.boxes) > 0:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    # YOLO Format: class x_center y_center width height
                    x, y, w, h = box.xywhn[0].tolist()
                    f.write(f"{cls_id} {x} {y} {w} {h}\n")
        
        # Copy the image to the training folder
        if not os.path.exists(dest_img_path):
            shutil.copy(img_path, dest_img_path)
            copied_imgs += 1
        
        count += 1

    print("\n" + "=" * 60)
    print(f"✅ SUCCESS: Full Expansion Complete!")
    print("=" * 60)
    print(f"   📝 Created: {count} new label files")
    print(f"   🖼️  Copied: {copied_imgs} new images to training")
    print(f"   ⏭️  Skipped: {skipped} already processed")
    print(f"   📊 Total Training Images: {len(glob.glob(os.path.join(TRAIN_IMG_DIR, '*.jpg')))}")
    print("=" * 60)
    print("👉 NEXT STEP: python 2_train.py → Train Student_Model_v2")
    print("   (Training will take significantly longer with 1400+ images!)")

if __name__ == '__main__':
    auto_label()
