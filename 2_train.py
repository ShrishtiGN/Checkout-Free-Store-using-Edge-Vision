from ultralytics import YOLO
import multiprocessing

def train():
    print("🚀 PHASE 2: Training Initiated on RTX 3050...")
    
    # Load Small model (Best balance for 6GB VRAM)
    model = YOLO('yolov8s.pt') 

    model.train(
        data='data/vista.yaml',
        
        # --- HARDWARE OPTIMIZATION ---
        epochs=50,
        imgsz=640,
        batch=4,        # Reduced for large dataset (3313 images)
        device=0,       # Use NVIDIA GPU 0
        workers=0,      # Disable multiprocessing on Windows
        cache=False,    # Disable caching to save RAM
        
        # --- THE WINNING STRATEGY (Synthetic Clutter) ---
        # Mosaic: 1.0 -> Always stitch 4 images together. 
        # This teaches the model to see multiple objects at once.
        mosaic=1.0,     
        
        # Mixup: 0.15 -> Transparently blend images.
        # This teaches the model to handle overlapping objects.
        mixup=0.15,
        
        # Data Augmentation (Robustness)
        degrees=10.0,   # Rotation
        scale=0.5,      # Zoom Variation
        fliplr=0.5,     # Horizontal Flip
        
        # Project Metadata
        project='RetailEye_Runs',
        name='Student_Model_v2',
        exist_ok=True   # Overwrite old run if exists
    )
    print("✅ Training Complete. Best model saved in RetailEye_Runs/Student_Model_v2/weights/best.pt")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    train()
