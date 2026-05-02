from ultralytics import YOLO
import pandas as pd
import glob
import os
import sys
import yaml

# 1. Auto-detect the latest trained model
def find_latest_model():
    """Find the most recently trained model"""
    # Check both possible training output directories
    search_paths = [
        'runs/detect/RetailEye_Runs/*/weights/best.pt',
        'RetailEye_Runs/*/weights/best.pt'
    ]
    
    all_models = []
    for pattern in search_paths:
        all_models.extend(glob.glob(pattern))
    
    if all_models:
        # Get the most recently modified model
        latest_model = max(all_models, key=os.path.getmtime)
        return latest_model
    return None

def validate_model(model, data_yaml_path='data/vista.yaml'):
    """Validate model before inference"""
    print("\n" + "="*60)
    print("🔍 MODEL VALIDATION")
    print("="*60)
    
    # Get model classes
    model_classes = model.names
    print(f"Model has {len(model_classes)} classes")
    
    # Load expected classes from data.yaml
    if os.path.exists(data_yaml_path):
        with open(data_yaml_path) as f:
            data_config = yaml.safe_load(f)
        expected_classes = data_config.get('names', {})
        
        print(f"Expected {len(expected_classes)} classes from training")
        
        # Check if classes match
        if len(model_classes) != len(expected_classes):
            print("\n⚠️  WARNING: Class count mismatch!")
            print(f"  Model: {len(model_classes)} classes")
            print(f"  Expected: {len(expected_classes)} classes")
            print("\n  This suggests the model is NOT your trained model!")
            print("  You may be using the pre-trained COCO model instead.")
            return False
        
        # Show class comparison
        print("\nClass mapping check:")
        mismatch = False
        for idx in range(min(5, len(expected_classes))):  # Show first 5
            model_name = model_classes.get(idx, 'MISSING')
            expected_name = expected_classes.get(idx, 'MISSING')
            match = "✅" if model_name == expected_name else "❌"
            print(f"  {match} Class {idx}: {model_name} {'==' if model_name == expected_name else '!='} {expected_name}")
            if model_name != expected_name:
                mismatch = True
        
        if mismatch:
            print("\n❌ CRITICAL: Class names don't match!")
            print("   Model was not trained on your data!")
            return False
        else:
            print("\n✅ Class names match! Model is correctly trained.")
    
    print("="*60 + "\n")
    return True

model_path = find_latest_model()

if model_path and os.path.exists(model_path):
    print(f"✅ Loading trained model from {model_path}")
    model = YOLO(model_path)
    
    # Validate the model
    if not validate_model(model):
        print("\n❌ ERROR: Model validation failed!")
        print("The loaded model doesn't match your training data.")
        print("Please check:")
        print("  1. Training completed successfully (check mAP > 0)")
        print("  2. Model path is correct")
        print("  3. data/vista.yaml classes match training")
        response = input("\nContinue anyway? Results will be WRONG! (yes/no): ")
        if response.lower() != 'yes':
            sys.exit(1)
else:
    print("\n❌ ERROR: No trained model found!")
    print("\nSearched for:")
    print("  - runs/detect/RetailEye_Runs/*/weights/best.pt")
    print("  - RetailEye_Runs/*/weights/best.pt")
    print("\nYou must train a model first: python train_model.py")
    print("\nUsing pre-trained YOLOv8s would give WRONG results (COCO classes, not your classes)")
    response = input("\nUse pre-trained model anyway for demo? (yes/no): ")
    if response.lower() != 'yes':
        sys.exit(1)
    print("⚠️  Loading pre-trained YOLOv8s (results will be incorrect!)...")
    model = YOLO('yolov8s.pt')

# 2. Settings
TEST_DIR = 'data/images/test/*.jpg'
CONF_THRES = 0.50  # STRICT! Only count if 50% sure.
IOU_THRES = 0.5    # NMS: Remove duplicate boxes for the same item.

submission_rows = []

print("🔍 Running Inference on Test Data...")
images = glob.glob(TEST_DIR)

if len(images) == 0:
    print(f"❌ ERROR: No test images found in {TEST_DIR}")
    sys.exit(1)

print(f"Found {len(images)} test images\n")

detection_stats = {'total': 0, 'empty': 0, 'with_objects': 0}
class_counts = {}

for img_file in images:
    # Predict
    results = model.predict(
        img_file, 
        conf=CONF_THRES, 
        iou=IOU_THRES, 
        verbose=False
    )
    
    result = results[0]
    
    # Extract Class Names
    detected = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        cls_name = result.names[cls_id]
        detected.append(cls_name)
        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
    
    # Format: "item1 item2 item3"
    prediction_str = " ".join(detected)
    
    # Statistics
    detection_stats['total'] += len(detected)
    if len(detected) == 0:
        detection_stats['empty'] += 1
    else:
        detection_stats['with_objects'] += 1
    
    # Add to list
    img_id = os.path.basename(img_file)
    submission_rows.append({'ImageID': img_id, 'Label': prediction_str})

# 3. Save CSV
os.makedirs('submissions', exist_ok=True)
df = pd.DataFrame(submission_rows)
df.to_csv('submissions/submission_v1.csv', index=False)

print("\n" + "="*60)
print("✅ INFERENCE COMPLETE!")
print("="*60)
print(f"Processed: {len(df)} images")
print(f"Total detections: {detection_stats['total']}")
print(f"Images with objects: {detection_stats['with_objects']}")
print(f"Images with no objects: {detection_stats['empty']}")

if class_counts:
    print("\nDetected classes:")
    for cls_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cls_name}: {count}")
else:
    print("\n⚠️  WARNING: NO objects detected in ANY image!")
    print("   This suggests:")
    print("   - Model was not trained properly (mAP = 0)")
    print("   - Confidence threshold too high")
    print("   - Test images very different from training images")

print(f"\n📁 Submission saved to: submissions/submission_v1.csv")
print("="*60)
