"""
Data Augmentation Script - Artificially expand dataset
When you have few labeled images, augmentation creates variations
"""

import os
import cv2
import numpy as np
from pathlib import Path
import shutil
import random


class DataAugmenter:
    def __init__(self, base_path="data"):
        self.base_path = Path(base_path)
        self.images_dir = self.base_path / "images"
        self.labels_dir = self.base_path / "labels"
    
    def adjust_bbox(self, bbox_line, flip_horizontal=False, flip_vertical=False):
        """Adjust bounding box coordinates after transformation"""
        parts = bbox_line.strip().split()
        if len(parts) != 5:
            return bbox_line
        
        class_id = parts[0]
        x_center, y_center, width, height = map(float, parts[1:])
        
        if flip_horizontal:
            x_center = 1.0 - x_center
        
        if flip_vertical:
            y_center = 1.0 - y_center
        
        return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
    
    def augment_image(self, img, aug_type):
        """Apply augmentation to image"""
        if aug_type == "flip_h":
            return cv2.flip(img, 1)  # Horizontal flip
        elif aug_type == "flip_v":
            return cv2.flip(img, 0)  # Vertical flip
        elif aug_type == "brightness_up":
            return cv2.convertScaleAbs(img, alpha=1.2, beta=20)
        elif aug_type == "brightness_down":
            return cv2.convertScaleAbs(img, alpha=0.8, beta=-20)
        elif aug_type == "contrast":
            return cv2.convertScaleAbs(img, alpha=1.3, beta=0)
        elif aug_type == "blur":
            return cv2.GaussianBlur(img, (5, 5), 0)
        elif aug_type == "noise":
            noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
            return cv2.add(img, noise)
        elif aug_type == "rotate_90":
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif aug_type == "rotate_180":
            return cv2.rotate(img, cv2.ROTATE_180)
        elif aug_type == "rotate_270":
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            return img
    
    def adjust_bbox_for_rotation(self, bbox_line, angle):
        """Adjust bounding boxes after rotation"""
        parts = bbox_line.strip().split()
        if len(parts) != 5:
            return bbox_line
        
        class_id = parts[0]
        x, y, w, h = map(float, parts[1:])
        
        if angle == 90 or angle == 270:
            # Swap dimensions for 90/270 degree rotations
            if angle == 90:
                new_x = 1.0 - y
                new_y = x
            else:  # 270
                new_x = y
                new_y = 1.0 - x
            new_w = h
            new_h = w
            return f"{class_id} {new_x:.6f} {new_y:.6f} {new_w:.6f} {new_h:.6f}"
        elif angle == 180:
            new_x = 1.0 - x
            new_y = 1.0 - y
            return f"{class_id} {new_x:.6f} {new_y:.6f} {w:.6f} {h:.6f}"
        
        return bbox_line
    
    def augment_dataset(self, split="train", multiplier=5):
        """
        Augment existing dataset
        
        Args:
            split: 'train' or 'val'
            multiplier: How many augmented versions per original image
        """
        print("=" * 60)
        print("DATA AUGMENTATION SCRIPT")
        print("=" * 60)
        
        img_dir = self.images_dir / split
        label_dir = self.labels_dir / split
        
        if not img_dir.exists() or not label_dir.exists():
            print(f"❌ {split} directory not found!")
            return
        
        # Get all images
        img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        original_count = len(img_files)
        
        print(f"\n[1] Found {original_count} original {split} images")
        print(f"[2] Will create {multiplier}x augmented versions each")
        print(f"[3] Target total: {original_count * (multiplier + 1)} images")
        
        # Augmentation types
        augmentations = [
            ("flip_h", False, True, False),
            ("flip_v", False, False, True),
            ("brightness_up", False, False, False),
            ("brightness_down", False, False, False),
            ("contrast", False, False, False),
            ("blur", False, False, False),
            ("noise", False, False, False),
            ("rotate_90", 90, False, False),
            ("rotate_180", 180, False, False),
            ("rotate_270", 270, False, False),
        ]
        
        # Create augmented versions
        created = 0
        
        for img_path in img_files:
            img_stem = img_path.stem
            label_path = label_dir / f"{img_stem}.txt"
            
            if not label_path.exists():
                print(f"⚠️  No label for {img_path.name}, skipping...")
                continue
            
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"⚠️  Could not read {img_path.name}, skipping...")
                continue
            
            # Read labels
            with open(label_path, 'r') as f:
                labels = f.readlines()
            
            # Select random augmentations
            selected_augs = random.sample(augmentations, min(multiplier, len(augmentations)))
            
            for aug_name, *aug_params in selected_augs:
                # Generate augmented image
                aug_img = self.augment_image(img, aug_name)
                
                # Adjust labels
                if aug_name == "flip_h":
                    aug_labels = [self.adjust_bbox(l, flip_horizontal=True) for l in labels]
                elif aug_name == "flip_v":
                    aug_labels = [self.adjust_bbox(l, flip_vertical=True) for l in labels]
                elif aug_name.startswith("rotate"):
                    angle = aug_params[0]
                    aug_labels = [self.adjust_bbox_for_rotation(l, angle) for l in labels]
                else:
                    aug_labels = labels  # No bbox adjustment needed
                
                # Save augmented image and label
                new_img_name = f"{img_stem}_aug_{aug_name}{img_path.suffix}"
                new_label_name = f"{img_stem}_aug_{aug_name}.txt"
                
                new_img_path = img_dir / new_img_name
                new_label_path = label_dir / new_label_name
                
                cv2.imwrite(str(new_img_path), aug_img)
                
                with open(new_label_path, 'w') as f:
                    f.writelines(aug_labels)
                
                created += 1
                
                if created % 10 == 0:
                    print(f"   Created: {created} augmented images...")
        
        print(f"\n✅ Augmentation complete!")
        print(f"   Original: {original_count}")
        print(f"   Augmented: {created}")
        print(f"   Total: {original_count + created}")
        print("=" * 60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Augment RetailEye dataset")
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val'],
                        help='Which split to augment (default: train)')
    parser.add_argument('--multiplier', type=int, default=5,
                        help='How many augmented versions per image (default: 5)')
    
    args = parser.parse_args()
    
    augmenter = DataAugmenter()
    augmenter.augment_dataset(split=args.split, multiplier=args.multiplier)


if __name__ == "__main__":
    main()
