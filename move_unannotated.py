"""
Move images without corresponding labels to train_unannotated directory
"""
import os
import shutil
from pathlib import Path

def move_unannotated_images():
    # Define paths
    images_dir = Path('data/images/train')
    labels_dir = Path('data/labels/train')
    unannotated_dir = Path('data/images/train_unannotated')
    
    # Create unannotated directory if it doesn't exist
    unannotated_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all label files (without extension)
    label_basenames = set()
    for label_file in labels_dir.glob('*.txt'):
        label_basenames.add(label_file.stem)
    
    print(f"Found {len(label_basenames)} label files")
    
    # Get all image files
    image_files = list(images_dir.glob('*.jpg'))
    print(f"Found {len(image_files)} image files")
    
    # Find images without labels
    moved_count = 0
    kept_count = 0
    
    for image_file in image_files:
        image_basename = image_file.stem
        
        # Check if corresponding label exists
        if image_basename not in label_basenames:
            # Move to unannotated directory
            dest = unannotated_dir / image_file.name
            shutil.move(str(image_file), str(dest))
            moved_count += 1
            if moved_count <= 5:
                print(f"  Moved: {image_file.name}")
        else:
            kept_count += 1
    
    print(f"\n✅ Summary:")
    print(f"  Kept (with labels): {kept_count} images")
    print(f"  Moved (no labels): {moved_count} images")
    print(f"\n📁 Unannotated images moved to: {unannotated_dir}")

if __name__ == '__main__':
    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    move_unannotated_images()
