"""
Quick Test Script - Tests the entire pipeline with current data
WARNING: This will confirm that model has 0% accuracy with 13 images
"""
import os
import sys

def run_command(cmd, description):
    """Run a command and report results"""
    print("\n" + "="*60)
    print(f"🔄 {description}")
    print("="*60)
    print(f"Running: {cmd}\n")
    
    result = os.system(cmd)
    
    if result != 0:
        print(f"\n❌ {description} failed with exit code {result}")
        return False
    else:
        print(f"\n✅ {description} completed")
        return True

def main():
    print("="*60)
    print("🧪 RETAILEYE QUICK TEST")
    print("="*60)
    print("\nThis will test your pipeline with the current dataset.")
    print("⚠️  WARNING: With only 13 images, the model will have 0% accuracy!")
    print("This test is to verify the PIPELINE works, not to create a usable model.\n")
    
    response = input("Continue with pipeline test? (yes/no): ")
    if response.lower() != 'yes':
        print("Test cancelled.")
        return
    
    # Check virtual environment
    if 'venv' not in sys.executable and 'virtualenv' not in sys.executable:
        print("\n⚠️  Virtual environment not detected!")
        print("Activate it first: venv\\Scripts\\activate")
        response = input("Continue anyway? (yes/no): ")
        if response.lower() != 'yes':
            return
    
    # Test sequence
    tests = [
        ("python verify_setup.py", "Environment Verification"),
        ("python convert_data.py", "Data Conversion"),
    ]
    
    for cmd, desc in tests:
        if not run_command(cmd, desc):
            print("\n❌ Test sequence failed!")
            print("Fix the errors above before continuing.")
            return
    
    # Ask about training (it will take time)
    print("\n" + "="*60)
    print("⏱️  TRAINING TEST")
    print("="*60)
    print("Training will take 30-60 minutes on GPU.")
    print("⚠️  With 13 images, the model will learn NOTHING (expected!).")
    print("This is just to test if training runs without errors.\n")
    
    response = input("Proceed with training test? (yes/no): ")
    if response.lower() == 'yes':
        if run_command("python train_model.py", "Model Training"):
            run_command("python evaluate_model.py", "Model Evaluation")
            run_command("python inference.py", "Inference Test")
    
    print("\n" + "="*60)
    print("🏁 QUICK TEST COMPLETE")
    print("="*60)
    print("\n📋 NEXT STEPS:")
    print("1. Review the warnings and errors above")
    print("2. See DATA_ACQUISITION_GUIDE.md for how to get more data")
    print("3. Acquire 500+ images")
    print("4. Re-run: python convert_data.py")
    print("5. Re-run: python train_model.py")
    print("6. Check mAP > 0.4 in evaluation")
    print("7. Run inference for final submission")
    print("="*60)

if __name__ == '__main__':
    main()
