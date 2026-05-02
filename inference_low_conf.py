"""
Inference with LOW confidence threshold to see what the model detects
"""
from ultralytics import YOLO
import glob
import os

print("\n" + "="*70)
print("🔍 INFERENCE WITH LOW CONFIDENCE THRESHOLD")
print("="*70)

model_path = 'runs/detect/RetailEye_Runs/augmented_v1/weights/best.pt'

if not os.path.exists(model_path):
    print(f"❌ Model not found: {model_path}")
    exit(1)

print(f"\nLoading model: {model_path}")
model = YOLO(model_path)

# Get test images
test_images = glob.glob('data/images/test/*.jpg')
print(f"Found {len(test_images)} test images")

if len(test_images) == 0:
    print("❌ No test images found!")
    exit(1)

# Try multiple confidence thresholds
confidence_thresholds = [0.01, 0.05, 0.1, 0.25, 0.5]

print("\n" + "="*70)
print("Testing different confidence thresholds...")
print("="*70)

for conf_thresh in confidence_thresholds:
    print(f"\n📊 Confidence threshold: {conf_thresh}")
    
    total_detections = 0
    images_with_detections = 0
    
    # Run inference on first 10 images as a sample
    sample_images = test_images[:10]
    
    for img_path in sample_images:
        results = model.predict(
            source=img_path,
            conf=conf_thresh,
            iou=0.5,
            verbose=False
        )
        
        num_detections = len(results[0].boxes)
        total_detections += num_detections
        if num_detections > 0:
            images_with_detections += 1
    
    print(f"   Sample (10 images): {total_detections} total detections")
    print(f"   Images with objects: {images_with_detections}/10")

# Now run full inference with lowest threshold
print("\n" + "="*70)
print("Running FULL inference with conf=0.01...")
print("="*70)

total_detections = 0
images_with_detections = 0
detection_examples = []

for img_path in test_images:
    results = model.predict(
        source=img_path,
        conf=0.01,
        iou=0.5,
        verbose=False,
        save=False
    )
    
    num_detections = len(results[0].boxes)
    total_detections += num_detections
    
    if num_detections > 0:
        images_with_detections += 1
        
        # Store first 5 examples
        if len(detection_examples) < 5:
            img_name = os.path.basename(img_path)
            boxes = results[0].boxes
            detections_info = []
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = model.names[cls_id]
                detections_info.append(f"{cls_name} ({conf:.3f})")
            detection_examples.append((img_name, detections_info))

print(f"\n✅ Inference complete!")
print(f"   Total images: {len(test_images)}")
print(f"   Images with detections: {images_with_detections}")
print(f"   Total detections: {total_detections}")

if detection_examples:
    print("\n📋 Example detections:")
    for img_name, detections in detection_examples:
        print(f"\n   {img_name}:")
        for det in detections:
            print(f"      - {det}")
else:
    print("\n⚠️  NO DETECTIONS at any threshold!")
    print("   Model needs more/better training")

print("\n" + "="*70)
