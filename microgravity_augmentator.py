import albumentations as A
import cv2
import os
import glob
from pathlib import Path

def get_microgravity_pipeline():
    """
    Creates an extreme data augmentation pipeline using Albumentations 
    to simulate microgravity conditions, e.g., tumbling boxes and light plumes.
    """
    return A.Compose([
        # Simulate extreme tumbling (floating upside down, spinning)
        A.Rotate(limit=180, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),

        # Simulate uneven, volatile lighting and "plumes" from station windows
        A.RandomBrightnessContrast(
            brightness_limit=0.5, 
            contrast_limit=0.5, 
            p=0.8
        ),
        A.RGBShift(
            r_shift_limit=40, 
            g_shift_limit=40, 
            b_shift_limit=40, 
            p=0.5
        ),
        A.RandomSunFlare(
            flare_roi=(0, 0, 1, 1), 
            angle_lower=0, 
            angle_upper=1,
            num_flare_circles_lower=1, 
            num_flare_circles_upper=3, 
            src_radius=150, 
            src_color=(255, 255, 255), 
            p=0.3
        ),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.MotionBlur(blur_limit=7, p=0.4) # Simulating fast drifting
    ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.3, label_fields=['class_labels']))

def process_dataset(input_img_dir, input_lbl_dir, out_img_dir, out_lbl_dir):
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)
    
    pipeline = get_microgravity_pipeline()
    image_files = glob.glob(os.path.join(input_img_dir, '*.jpg'))
    
    for img_path in image_files:
        filename = os.path.basename(img_path)
        lbl_path = os.path.join(input_lbl_dir, filename.replace('.jpg', '.txt'))
        
        if not os.path.exists(lbl_path):
            continue
            
        # Read Image
        image = cv2.imread(img_path)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Read YOLO Labels
        bboxes = []
        class_labels = []
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    x_c, y_c, w, h = map(float, parts[1:5])
                    bboxes.append([x_c, y_c, w, h])
                    class_labels.append(cls_id)
        
        if len(bboxes) == 0:
            continue
            
        # Apply Pipeline
        try:
            transformed = pipeline(image=image, bboxes=bboxes, class_labels=class_labels)
        except ValueError:
            # Albumentations can throw bounds error on extreme augmentations
            continue
            
        aug_img = transformed['image']
        aug_bboxes = transformed['bboxes']
        aug_labels = transformed['class_labels']
        
        # Save output image
        aug_img = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
        out_img_path = os.path.join(out_img_dir, filename)
        cv2.imwrite(out_img_path, aug_img)
        
        # Save output YOLO labels
        out_lbl_path = os.path.join(out_lbl_dir, filename.replace('.jpg', '.txt'))
        with open(out_lbl_path, 'w') as f:
            for bbox, cls_id in zip(aug_bboxes, aug_labels):
                f.write(f"{cls_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

if __name__ == '__main__':
    print("Initiating Microgravity Data Augmentation Pipeline...")
    
    # Process Train Split
    print("Processing Training Data...")
    process_dataset(
        input_img_dir='data/images/train',
        input_lbl_dir='data/labels/train',
        out_img_dir='data_microgravity/images/train',
        out_lbl_dir='data_microgravity/labels/train'
    )
    
    # Process Val Split
    print("Processing Validation Data...")
    process_dataset(
        input_img_dir='data/images/val',
        input_lbl_dir='data/labels/val',
        out_img_dir='data_microgravity/images/val',
        out_lbl_dir='data_microgravity/labels/val'
    )
    
    print("✅ Microgravity Augmentation Completed successfully!")
    print("Augmented dataset is written to 'data_microgravity/'.")
