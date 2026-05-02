import os
import shutil
import random
import glob

# Setup
TRAIN_IMG = 'data/images/train/'
TRAIN_LBL = 'data/labels/train/'
VAL_IMG   = 'data/images/val/'
VAL_LBL   = 'data/labels/val/'

def split():
    # Make val folders
    os.makedirs(VAL_IMG, exist_ok=True)
    os.makedirs(VAL_LBL, exist_ok=True)
    
    # Get all labeled images
    all_labels = glob.glob(os.path.join(TRAIN_LBL, "*.txt"))
    
    # Calculate 20%
    num_val = int(len(all_labels) * 0.2)
    val_files = random.sample(all_labels, num_val)
    
    print(f"Moving {num_val} items to Validation...")
    
    for lbl_path in val_files:
        basename = os.path.basename(lbl_path)
        img_name = basename.replace('.txt', '.jpg')
        
        src_img = os.path.join(TRAIN_IMG, img_name)
        
        # Move both
        if os.path.exists(src_img):
            shutil.move(src_img, os.path.join(VAL_IMG, img_name))
            shutil.move(lbl_path, os.path.join(VAL_LBL, basename))

    print("✅ Split Complete.")

if __name__ == '__main__':
    split()
