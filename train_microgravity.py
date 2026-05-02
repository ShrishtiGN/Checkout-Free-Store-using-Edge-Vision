from ultralytics import YOLO
import os
import sys

def train_microgravity():
    """
    Retrains the YOLO model using the 'tumbled' versions of the dataset
    and strictly overrides YOLO's internal augmentation hyper-parameters to 
    stack additional real-time tumbling and lighting permutations.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_yaml = os.path.join(script_dir, 'data_microgravity.yaml')
    
    if not os.path.exists(data_yaml):
        print(f"❌ ERROR: YAML file not found at {data_yaml}")
        sys.exit(1)
        
    # Load YOLOv8/v11 small architecture
    model = YOLO('yolov8s.pt') 
    
    print("🚀 Starting Microgravity-Oriented Retraining...")
    results_dir = os.path.join(script_dir, 'results', 'training_microgravity')
    os.makedirs(results_dir, exist_ok=True)
    
    model.train(
        data=data_yaml,
        epochs=50,
        imgsz=640,
        batch=8,
        workers=0,  # Prevent memory issues if testing locally
        
        # --- EXTREME MICROGRAVITY ONLINE AUGMENTATIONS ---
        degrees=180.0,       # Random continuous 360-degree rotation (+-180)
        translate=0.2,       # Emulate shifting drift out of frame bounds
        scale=0.5,           # Emulate objects floating closer or further away
        perspective=0.001,   # 3D spatial skewing
        flipud=0.5,          # 50% chance of vertical flip
        fliplr=0.5,          # 50% chance of horizontal flip
        hsv_h=0.1,           # Extreme hue variance
        hsv_s=0.8,           # Saturation mimicking glare from varying modules
        hsv_v=0.8,           # Plumes of station lighting
        
        project=results_dir,
        name='v_microgravity',
        exist_ok=True,
        val=True,
        plots=True,
        save=True
    )
    
    print("\n✅ MICROGRAVITY RETRAINING COMPLETED!")

if __name__ == '__main__':
    train_microgravity()
