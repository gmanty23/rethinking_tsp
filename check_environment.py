import sys
import torch
import subprocess

def get_cuda_version():
    try:
        # Try nvcc first
        output = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
        for line in output.split('\n'):
            if "release" in line:
                return line.strip()
    except:
        return "nvcc not found"

def check_env():
    print("=== Environment Check ===")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}")
    
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_avail}")
    
    if cuda_avail:
        print(f"CUDA Version (PyTorch): {torch.version.cuda}")
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        # Check CuDNN
        print(f"CuDNN Version: {torch.backends.cudnn.version()}")
    else:
        print("WARNING: CUDA is not available to PyTorch.")

    print(f"System CUDA (nvcc): {get_cuda_version()}")
    print("=======================")

    # Test a small matrix multiplication (The crash trigger)
    if cuda_avail:
        print("\nRunning Mini-GEMM Test...")
        try:
            a = torch.randn(128, 20, 2).cuda()
            # Create a non-contiguous slice (similar to what caused the crash)
            b = torch.randn(128, 20, 7).cuda()
            slice_b = b[..., 0:2] 
            
            print(f"Tensor is contiguous? {slice_b.is_contiguous()}")
            
            layer = torch.nn.Linear(2, 128).cuda()
            _ = layer(slice_b)
            print("SUCCESS: Linear layer handled non-contiguous input.")
        except Exception as e:
            print(f"FAILURE: Crashed on simple Linear layer test!\nError: {e}")

if __name__ == "__main__":
    check_env()