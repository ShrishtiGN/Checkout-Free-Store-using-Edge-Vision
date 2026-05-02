from ultralytics import YOLO
import glob
import os

MODEL_PATH = 'runs/detect/RetailEye_Runs/Mosaic_Model_v1/weights/best.pt'
TEST_DIR = 'data/images/test/'

print("🔍 Testing Teacher Model...")
print(f"Loading: {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    print(f"❌ Model not found at {MODEL_PATH}")
    exit(1)

model = YOLO(MODEL_PATH)

# Test on first 5 images with very low confidence
images = glob.glob(os.path.join(TEST_DIR, "*.jpg"))[:5]
print(f"\nTesting on {len(images)} sample images...")

for img_path in images:
    print(f"\n📸 {os.path.basename(img_path)}")
    
    # Try with very low confidence to see what it detects
    results = model.predict(img_path, conf=0.01, verbose=False)
    result = results[0]
    
    if len(result.boxes) > 0:
        print(f"   Found {len(result.boxes)} objects:")
        for box in result.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = result.names[cls_id] if cls_id < len(result.names) else f"Class_{cls_id}"
            print(f"      - {cls_name}: {conf:.2%} confidence")
    else:
        print("   ❌ No objects detected (even at 1% threshold)")

print("\n" + "="*50)
print("If you see detections above, the model works.")
print("If not, the model may not be trained or corrupted.")
