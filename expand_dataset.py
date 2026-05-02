"""
Dataset Expansion Script for RetailEye
Converts more annotations and prepares an expanded training dataset
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import random


class DatasetExpander:
    def __init__(self, base_path: str = "data"):
        self.base_path = Path(base_path)
        self.images_dir = self.base_path / "images"
        self.labels_dir = self.base_path / "labels"
        self.raw_annotations_path = self.base_path / "raw_annotations" / "train_annotations.json"
        
    def load_annotations(self) -> Dict:
        """Load raw annotations from JSON file"""
        if not self.raw_annotations_path.exists():
            raise FileNotFoundError(f"Annotations file not found: {self.raw_annotations_path}")
        
        with open(self.raw_annotations_path, 'r') as f:
            return json.load(f)
    
    def convert_bbox_to_yolo(self, bbox: List[int], img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """
        Convert COCO bbox [x, y, width, height] to YOLO format [x_center, y_center, width, height]
        Normalized to [0, 1]
        """
        x, y, w, h = bbox
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        width_norm = w / img_width
        height_norm = h / img_height
        return x_center, y_center, width_norm, height_norm
    
    def expand_dataset(self, target_count: int = 50, train_split: float = 0.8):
        """
        Expand dataset by converting more annotations
        
        Args:
            target_count: Total number of annotated images to aim for
            train_split: Ratio of training vs validation images
        """
        print("=" * 60)
        print("DATASET EXPANSION SCRIPT")
        print("=" * 60)
        
        # Load annotations
        print("\n[1] Loading annotations...")
        annotations_data = self.load_annotations()
        images_info = {img['id']: img for img in annotations_data['images']}
        
        # Group annotations by image
        annotations_by_image = {}
        for ann in annotations_data['annotations']:
            img_id = ann['image_id']
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)
        
        print(f"   - Total images in annotations: {len(images_info)}")
        print(f"   - Images with annotations: {len(annotations_by_image)}")
        
        # Check which images already have labels
        existing_train_labels = set(
            Path(f).stem for f in (self.labels_dir / "train").glob("*.txt")
        )
        existing_val_labels = set(
            Path(f).stem for f in (self.labels_dir / "val").glob("*.txt")
        )
        existing_labels = existing_train_labels | existing_val_labels
        
        print(f"   - Already converted: {len(existing_labels)}")
        
        # Find images that need conversion
        available_images = []
        for img_id, img_info in images_info.items():
            img_stem = Path(img_info['file_name']).stem
            if img_stem not in existing_labels and img_id in annotations_by_image:
                available_images.append((img_id, img_info))
        
        print(f"   - Available for conversion: {len(available_images)}")
        
        # Calculate how many to convert
        current_total = len(existing_labels)
        needed = min(target_count - current_total, len(available_images))
        
        if needed <= 0:
            print(f"\n✓ Already have {current_total} annotated images (target: {target_count})")
            return
        
        print(f"\n[2] Converting {needed} more annotations...")
        print(f"   - Current: {current_total} → Target: {current_total + needed}")
        
        # Randomly select images to convert
        random.shuffle(available_images)
        selected_images = available_images[:needed]
        
        # Split into train/val
        split_idx = int(len(selected_images) * train_split)
        train_images = selected_images[:split_idx]
        val_images = selected_images[split_idx:]
        
        print(f"   - New training images: {len(train_images)}")
        print(f"   - New validation images: {len(val_images)}")
        
        # Convert and save
        converted_count = 0
        
        def convert_and_save(images_list, split_name):
            nonlocal converted_count
            
            for img_id, img_info in images_list:
                img_name = img_info['file_name']
                img_stem = Path(img_name).stem
                img_width = img_info['width']
                img_height = img_info['height']
                
                # Check if image exists
                img_paths = [
                    self.images_dir / "train_unannotated" / img_name,
                    self.images_dir / "train" / img_name,
                    self.images_dir / "val" / img_name
                ]
                
                img_source = None
                for path in img_paths:
                    if path.exists():
                        img_source = path
                        break
                
                if not img_source:
                    print(f"   ⚠ Image not found: {img_name}")
                    continue
                
                # Convert annotations
                yolo_annotations = []
                for ann in annotations_by_image[img_id]:
                    class_id = ann['category_id']
                    bbox = ann['bbox']
                    x_center, y_center, width, height = self.convert_bbox_to_yolo(
                        bbox, img_width, img_height
                    )
                    yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                
                # Save label file
                label_path = self.labels_dir / split_name / f"{img_stem}.txt"
                label_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(label_path, 'w') as f:
                    f.write('\n'.join(yolo_annotations))
                
                # Copy image to appropriate folder
                img_dest = self.images_dir / split_name / img_name
                img_dest.parent.mkdir(parents=True, exist_ok=True)
                
                if img_source != img_dest:
                    shutil.copy2(img_source, img_dest)
                
                converted_count += 1
                if converted_count % 10 == 0:
                    print(f"   - Converted: {converted_count}/{needed}")
        
        # Process train and val splits
        convert_and_save(train_images, "train")
        convert_and_save(val_images, "val")
        
        print(f"\n✓ Successfully converted {converted_count} images")
        
        # Show final statistics
        self.show_statistics()
    
    def show_statistics(self):
        """Display current dataset statistics"""
        print("\n" + "=" * 60)
        print("CURRENT DATASET STATISTICS")
        print("=" * 60)
        
        splits = ['train', 'val', 'test']
        total_images = 0
        total_labels = 0
        
        for split in splits:
            img_dir = self.images_dir / split
            label_dir = self.labels_dir / split
            
            if img_dir.exists():
                img_count = len(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
            else:
                img_count = 0
            
            if label_dir.exists():
                label_count = len(list(label_dir.glob("*.txt")))
            else:
                label_count = 0
            
            print(f"\n{split.upper()}:")
            print(f"  Images: {img_count}")
            print(f"  Labels: {label_count}")
            
            if img_count > 0:
                match_rate = (label_count / img_count) * 100
                print(f"  Match Rate: {match_rate:.1f}%")
            
            total_images += img_count
            total_labels += label_count
        
        # Check unannotated
        unannotated_dir = self.images_dir / "train_unannotated"
        if unannotated_dir.exists():
            unannotated_count = len(list(unannotated_dir.glob("*.jpg")) + list(unannotated_dir.glob("*.png")))
            print(f"\nUNANNOTATED:")
            print(f"  Images: {unannotated_count}")
        
        print(f"\nTOTAL ANNOTATED:")
        print(f"  Images: {total_images}")
        print(f"  Labels: {total_labels}")
        print("=" * 60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Expand RetailEye dataset with more annotations")
    parser.add_argument('--target', type=int, default=50,
                        help='Target number of total annotated images (default: 50)')
    parser.add_argument('--split', type=float, default=0.8,
                        help='Train/val split ratio (default: 0.8)')
    parser.add_argument('--stats-only', action='store_true',
                        help='Only show statistics without converting')
    
    args = parser.parse_args()
    
    expander = DatasetExpander()
    
    if args.stats_only:
        expander.show_statistics()
    else:
        expander.expand_dataset(target_count=args.target, train_split=args.split)


if __name__ == "__main__":
    main()
