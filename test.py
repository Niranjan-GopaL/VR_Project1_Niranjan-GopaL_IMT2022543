import torch
import time

def main():
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Make sure you have PyTorch with CUDA support.")
        return

    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print("Press Ctrl+C to stop the program")
    
    # Create large tensors on GPU
    size = 8000  # Adjust size if needed
    tensor_a = torch.randn(size, size, device="cuda")
    tensor_b = torch.randn(size, size, device="cuda")
    
    # Keep GPU busy with matrix multiplications
    iteration = 0
    while True:
        # Matrix multiplication is computationally intensive for GPUs
        result = torch.matmul(tensor_a, tensor_b)
        
        # Force synchronization to ensure computation completes
        torch.cuda.synchronize()
        
        # Get GPU memory stats
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        
        iteration += 1
        if iteration % 10 == 0:
            print(f"Iteration {iteration}, GPU Memory: {allocated:.1f}MB allocated, {reserved:.1f}MB reserved")
            
        # Small sleep to allow checking nvidia-smi and prevent the system from becoming unresponsive
        time.sleep(0.1)

if __name__ == "__main__":
    main()