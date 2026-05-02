"""
Comprehensive Model Evaluation Script
Evaluates trained model and provides detailed performance metrics
"""
from ultralytics import YOLO
import glob
import os
import sys
import yaml
from pathlib import Path

def find_latest_model():
    """Find the most recently trained model"""
    search_paths = [
        'runs/detect/RetailEye_Runs/*/weights/best.pt',
        'RetailEye_Runs/*/weights/best.pt'
    ]
    
    all_models = []
    for pattern in search_paths:
        all_models.extend(glob.glob(pattern))
    
    if all_models:
        latest_model = max(all_models, key=os.path.getmtime)
        return latest_model
    return None

def evaluate_model(model_path, data_yaml='data/vista.yaml'):
    """Run comprehensive evaluation"""
    print("="*60)
    print("🔬 MODEL EVALUATION")
    print("="*60)
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return False
    
    if not os.path.exists(data_yaml):
        print(f"❌ Data config not found: {data_yaml}")
        return False
    
    print(f"📦 Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Load data config
    with open(data_yaml) as f:
        data_config = yaml.safe_load(f)
    
    print(f"📊 Dataset: {data_yaml}")
    print(f"📝 Classes: {len(data_config.get('names', {}))}")
    
    # Run validation
    print("\n🔄 Running validation...")
    metrics = model.val(data=data_yaml, plots=True, save_json=True)
    
    print("\n" + "="*60)
    print("📈 OVERALL METRICS")
    print("="*60)
    print(f"Precision (P):    {metrics.box.mp:.4f}")
    print(f"Recall (R):       {metrics.box.mr:.4f}")
    print(f"mAP@50:           {metrics.box.map50:.4f}")
    print(f"mAP@50-95:        {metrics.box.map:.4f}")
    
    # Per-class metrics
    if hasattr(metrics.box, 'maps') and metrics.box.maps is not None:
        print("\n" + "="*60)
        print("📊 PER-CLASS METRICS")
        print("="*60)
        
        class_names = data_config.get('names', {})
        
        # Get per-class metrics
        if hasattr(metrics.box, 'ap_class_index'):
            for idx in metrics.box.ap_class_index:
                class_name = class_names.get(idx, f"class_{idx}")
                # Get metrics for this class
                class_map50 = metrics.box.maps[idx] if idx < len(metrics.box.maps) else 0
                print(f"  {class_name:20s}: mAP@50 = {class_map50:.4f}")
    
    # Performance assessment
    print("\n" + "="*60)
    print("🎯 PERFORMANCE ASSESSMENT")
    print("="*60)
    
    map50 = metrics.box.map50
    
    if map50 == 0:
        print("❌ FAILED: Model has ZERO accuracy!")
        print("\n🔍 Possible causes:")
        print("  1. Dataset too small (< 100 images)")
        print("  2. All validation labels are empty")
        print("  3. Class labels incorrectly formatted")
        print("  4. Training didn't converge")
        print("\n⚠️  CRITICAL: DO NOT use this model for production!")
        print("  Action: Acquire more training data and retrain")
    elif map50 < 0.2:
        print("⚠️  POOR: Very low accuracy (mAP@50 < 0.2)")
        print("\n💡 Recommendations:")
        print("  - Acquire significantly more training data")
        print("  - Check if labels are correct")
        print("  - Increase training epochs")
        print("  - Try different augmentation strategies")
    elif map50 < 0.4:
        print("⚠️  BELOW AVERAGE: Low accuracy (mAP@50: 0.2-0.4)")
        print("\n💡 Recommendations:")
        print("  - Add more training data")
        print("  - Train for more epochs")
        print("  - Adjust confidence threshold during inference")
    elif map50 < 0.6:
        print("👍 MODERATE: Acceptable accuracy (mAP@50: 0.4-0.6)")
        print("\n💡 Can be improved:")
        print("  - More diverse training data")
        print("  - Fine-tune hyperparameters")
        print("  - Consider model ensembling")
    elif map50 < 0.8:
        print("✅ GOOD: Strong accuracy (mAP@50: 0.6-0.8)")
        print("\n✨ Model is production-ready!")
        print("  Optional improvements:")
        print("  - More data for edge cases")
        print("  - Test on diverse lighting/angles")
    else:
        print("🏆 EXCELLENT: Outstanding accuracy (mAP@50 > 0.8)")
        print("\n🎉 Model is production-ready with high confidence!")
    
    print("\n" + "="*60)
    print("📁 OUTPUTS")
    print("="*60)
    
    # Find results directory
    model_dir = Path(model_path).parent.parent
    print(f"Results saved to: {model_dir}")
    
    # Check for plots
    plots = []
    for plot_name in ['confusion_matrix.png', 'results.png', 'PR_curve.png', 'F1_curve.png']:
        plot_path = model_dir / plot_name
        if plot_path.exists():
            plots.append(plot_name)
    
    if plots:
        print("\n📊 Generated plots:")
        for plot in plots:
            print(f"  - {plot}")
    
    print("\n" + "="*60)
    
    return map50 > 0

def main():
    # Find latest model
    model_path = find_latest_model()
    
    if not model_path:
        print("❌ No trained model found!")
        print("\nSearched locations:")
        print("  - runs/detect/RetailEye_Runs/*/weights/best.pt")
        print("  - RetailEye_Runs/*/weights/best.pt")
        print("\n⚠️  Train a model first: python train_model.py")
        sys.exit(1)
    
    print(f"🔍 Found model: {model_path}\n")
    
    # Evaluate
    success = evaluate_model(model_path)
    
    if success:
        print("\n✅ Model evaluation complete!")
        print("\n📝 Next steps:")
        print("  1. Review plots in the model directory")
        print("  2. If mAP > 0.4, run: python inference.py")
        print("  3. If mAP < 0.4, acquire more data and retrain")
    else:
        print("\n❌ Model is not functional!")
        print("  DO NOT proceed to inference.")
        print("  Fix training issues first.")
    
    print("="*60)

if __name__ == '__main__':
    main()
