import torch
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate random matrices
matrix_size = 10000
a = torch.randn(matrix_size, matrix_size, device=device)
b = torch.randn(matrix_size, matrix_size, device=device)

# Perform matrix multiplication
print("Starting matrix multiplication...")
result = torch.matmul(a, b)
print("Matrix multiplication completed.")

# Keep the result on GPU to ensure GPU utilization
del a, b  # Free up memory
result.sum().backward()  # Ensure some computation is done on the result

# You can check GPU utilization with nvidia-smi
