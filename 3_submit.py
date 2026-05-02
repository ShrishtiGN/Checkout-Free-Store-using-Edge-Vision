from ultralytics import YOLO
import pandas as pd
import glob
import os

# --- CONFIGURATION ---
MODEL_PATH = 'runs/detect/RetailEye_Runs/Student_Model_v2/weights/best.pt'
TEST_DIR   = 'data/images/test/*.jpg'
OUTPUT_CSV = 'submissions/final_submission.csv'

# --- STRATEGY SETTINGS ---
# 0.50 Conf = We must be 50% sure an object exists.
# 0.60 IoU  = Remove duplicate boxes overlapping by 60%.
CONF_THRESHOLD = 0.50  
IOU_THRESHOLD  = 0.60  

def generate_submission():
    print("🚀 PHASE 3: Inference Initiated...")
    
    if not os.path.exists('submissions'): os.makedirs('submissions')
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}. Did you run training?")
        return

    print(f"🤖 Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    test_images = glob.glob(TEST_DIR)
    print(f"📂 Found {len(test_images)} test images. Processing...")
    
    submission_data = []
    
    # Run Inference in batches (faster) or loop
    for i, img_path in enumerate(test_images):
        if i % 100 == 0: print(f"   Processing image {i}/{len(test_images)}...")
        
        results = model.predict(
            img_path, 
            conf=CONF_THRESHOLD, 
            iou=IOU_THRESHOLD, 
            imgsz=640, 
            verbose=False
        )
        
        result = results[0]
        detected_names = []
        
        if result.boxes:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                # Safety check for index out of range
                if cls_id < len(result.names):
                    cls_name = result.names[cls_id]
                    detected_names.append(cls_name)
        
        # Format: "Item1 Item2 Item3"
        label_str = " ".join(detected_names)
        
        img_id = os.path.basename(img_path)
        submission_data.append({'ImageID': img_id, 'Label': label_str})

    # Save to CSV
    df = pd.DataFrame(submission_data)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n🏆 SUCCESS: Submission Saved to {OUTPUT_CSV}")
    print("👉 Upload this file to Kaggle/Unstop immediately.")

if __name__ == '__main__':
    generate_submission()
