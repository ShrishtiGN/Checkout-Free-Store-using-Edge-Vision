"""
GPU Integration Test Script
Verifies PyTorch and CUDA are properly configured for YOLO training
"""
import torch
from ultralytics import YOLO
import sys

def test_gpu_integration():
    """Test GPU availability and YOLO compatibility"""
    print("="*70)
    print("🧪 GPU INTEGRATION TEST")
    print("="*70)
    
    # Test 1: PyTorch CUDA availability
    print("\n1️⃣  Testing PyTorch CUDA Integration:")
    print(f"   PyTorch Version: {torch.__version__}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("   ❌ CUDA is NOT available!")
        print("   PyTorch is installed without CUDA support.")
        print("\n   To fix, reinstall PyTorch with:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
        return False
    
    print(f"   ✅ CUDA Version: {torch.version.cuda}")
    print(f"   ✅ GPU Count: {torch.cuda.device_count()}")
    print(f"   ✅ GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"   ✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Test 2: Create a simple tensor on GPU
    print("\n2️⃣  Testing GPU Tensor Operations:")
    try:
        test_tensor = torch.randn(1000, 1000).cuda()
        result = torch.matmul(test_tensor, test_tensor)
        print(f"   ✅ Successfully created and computed tensor on GPU")
        print(f"   ✅ Tensor device: {test_tensor.device}")
        del test_tensor, result
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"   ❌ Failed to create tensor on GPU: {e}")
        return False
    
    # Test 3: YOLO model on GPU
    print("\n3️⃣  Testing YOLO Model GPU Integration:")
    try:
        model = YOLO('yolov8n.pt')  # Use nano model for quick test
        print(f"   ✅ Model loaded successfully")
        print(f"   ✅ Default device: {model.device}")
        
        # Explicitly move to GPU
        model.to('cuda:0')
        print(f"   ✅ Model moved to GPU: {model.device}")
        
        # Test inference on GPU
        print("\n4️⃣  Testing GPU Inference (creating dummy image):")
        import numpy as np
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        results = model(dummy_image, device='cuda:0', verbose=False)
        print(f"   ✅ Inference completed on GPU successfully")
        
    except Exception as e:
        print(f"   ❌ Failed to run YOLO on GPU: {e}")
        return False
    
    # Test 4: Check VRAM usage
    print("\n5️⃣  GPU Memory Status:")
    allocated = torch.cuda.memory_allocated(0) / 1024**2
    reserved = torch.cuda.memory_reserved(0) / 1024**2
    print(f"   Memory Allocated: {allocated:.2f} MB")
    print(f"   Memory Reserved: {reserved:.2f} MB")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
    print("\n🎉 Your GPU is properly configured for YOLO training!")
    print("\nYou can now run training with:")
    print("   python train_model.py")
    print("\nThe training will automatically use your GPU:")
    print(f"   {torch.cuda.get_device_name(0)}")
    print("="*70)
    
    return True

if __name__ == '__main__':
    success = test_gpu_integration()
    sys.exit(0 if success else 1)
