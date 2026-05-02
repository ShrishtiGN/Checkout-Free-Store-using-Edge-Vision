"""
Quick validation script for the trained model
"""
from ultralytics import YOLO
import os

print("\n" + "=" * 70)
print("📈 VALIDATING TRAINED MODEL")
print("=" * 70)

best_model_path = 'runs/detect/RetailEye_Runs/augmented_v1/weights/best.pt'

if os.path.exists(best_model_path):
    print(f"\nLoading model: {best_model_path}")
    model = YOLO(best_model_path)
    
    print("Running validation with workers=0 to avoid memory issues...")
    metrics = model.val(data='data/vista.yaml', workers=0)
    
    print("\n" + "=" * 70)
    print("📊 VALIDATION RESULTS")
    print("=" * 70)
    print(f"\n💯 Metrics:")
    print(f"   Precision:   {metrics.box.mp:.3f}")
    print(f"   Recall:      {metrics.box.mr:.3f}")
    print(f"   mAP@50:      {metrics.box.map50:.3f}")
    print(f"   mAP@50-95:   {metrics.box.map:.3f}")
    
    # Performance assessment
    print("\n🎯 Performance Assessment:")
    map50 = metrics.box.map50
    
    if map50 == 0:
        print("   ⚠️  Model didn't learn (mAP=0) - Need MORE data!")
    elif map50 < 0.2:
        print("   ⚠️  POOR - Need more training data or longer training")
    elif map50 < 0.4:
        print("   ⚠️  FAIR - Model learning, but needs improvement")
    elif map50 < 0.6:
        print("   ✅  GOOD - Decent performance for small dataset")
    elif map50 < 0.8:
        print("   ✅  VERY GOOD - Strong performance!")
    else:
        print("   ✅  EXCELLENT - Outstanding performance!")
        
    print(f"\n📁 Model saved at: {best_model_path}")
    print("=" * 70)
else:
    print(f"❌ Model not found at: {best_model_path}")
    print("Please train the model first!")
