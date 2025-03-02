/* 
    To compile the code, use the following command:
    nvcc test.cu -o test
    To run the code, use the following command:
    ./test

=> Aim was  to make GPU's utilization in nvidia-smi to 20% or something ( but it didin't cross 0% ) 
=> Use test.py's pytorch and Utilization shot upto to 60% 

=> CONCLUSION :- Pytroch is better than CUDA C++ for GPU utilization ( but both gets the job done )

*/


//  Matrix Multiplication of 2048x2048 matrices to load the GPU

#include <stdio.h>
#include <cuda_runtime.h>
#include <unistd.h>

// Matrix multiplication kernel
__global__ void matrixMul(float *A, float *B, float *C, int width) {
    // Calculate row and column index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    
    // Do matrix multiplication if within dimensions
    if (row < width && col < width) {
        // Compute a single element of C
        for (int i = 0; i < width; i++) {
            sum += A[row * width + i] * B[i * width + col];
        }
        C[row * width + col] = sum;
    }
}

int main() {
    int matrixSize = 2048; // Size of the matrix (2048x2048)
    size_t bytes = matrixSize * matrixSize * sizeof(float);
    
    // Allocate host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    
    // Initialize matrices with some values
    for (int i = 0; i < matrixSize * matrixSize; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    
    // Copy host memory to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    
    // Define grid and block dimensions for matrix multiplication
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(
        (matrixSize + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (matrixSize + threadsPerBlock.y - 1) / threadsPerBlock.y
    );
    
    printf("Starting matrix multiplication to load GPU...\n");
    printf("Using matrix size: %d x %d\n", matrixSize, matrixSize);
    printf("Press Ctrl+C to stop\n");
    
    // Create CUDA streams for overlapping operations
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    
    // Run kernels in an infinite loop
    while (1) {
        // Launch kernel on stream 1
        matrixMul<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_A, d_B, d_C, matrixSize);
        
        // Launch kernel on stream 2 (overlapping execution)
        matrixMul<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(d_A, d_B, d_C, matrixSize);
        
        // Small sleep to allow checking nvidia-smi
        usleep(10000); // 10ms
    }
    
    // Clean up (this won't be reached due to infinite loop)
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}


//  ------------------------------ More aggresive code ------------------------------

// #include <stdio.h>
// #include <cuda_runtime.h>

// // Kernel that performs matrix operations to load the GPU
// __global__ void loadGPU(float *a, float *b, float *c, int n) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < n) {
//         // Do some heavy computation
//         float result = 0.0f;
//         for (int j = 0; j < 10000; j++) {
//             result = sinf(result + a[i]) * cosf(result + b[i]);
//         }
//         c[i] = result;
//     }
// }

// int main() {
//     // Size of arrays
//     int n = 10000000;
//     size_t size = n * sizeof(float);

//     // Allocate memory on the host
//     float *h_a = (float*)malloc(size);
//     float *h_b = (float*)malloc(size);
//     float *h_c = (float*)malloc(size);

//     // Initialize arrays
//     for (int i = 0; i < n; i++) {
//         h_a[i] = sinf(i) * 0.01f;
//         h_b[i] = cosf(i) * 0.01f;
//     }

//     // Allocate memory on the device
//     float *d_a, *d_b, *d_c;
//     cudaMalloc(&d_a, size);
//     cudaMalloc(&d_b, size);
//     cudaMalloc(&d_c, size);

//     // Copy data from host to device
//     cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

//     // Define grid and block dimensions
//     int blockSize = 256;
//     int numBlocks = (n + blockSize - 1) / blockSize;

//     printf("Starting GPU load test (press Ctrl+C to stop)...\n");

//     // Run the kernel in a loop to keep the GPU busy
//     while (1) {
//         loadGPU<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);
        
//         // Don't synchronize to let the GPU queue up work
//         // This will allow the GPU to reach higher utilization
        
//         // Uncommenting this will make the CPU wait for the GPU to finish
//         // cudaDeviceSynchronize();
//     }

//     // These statements won't be reached due to the infinite loop
//     cudaFree(d_a);
//     cudaFree(d_b);
//     cudaFree(d_c);
//     free(h_a);
//     free(h_b);
//     free(h_c);

//     return 0;
// }



// ------------------------------ Working primitive code ------------------------------

// #include <stdio.h>
// #include <cuda_runtime.h>

// #define LOAD 5000000 


// // Kernel that performs a simple calculation to load the GPU
// __global__ void loadGPU(float *a, int iterations) {
//     int i = threadIdx.x + blockIdx.x * blockDim.x;
//     float x = 0.0f;
    
//     // Simple loop to keep the GPU busy
//     for (int j = 0; j < iterations; j++) {
//         x = x + sinf(cosf(x + j));
//     }
    
//     if (i < LOAD) {
//         a[i] = x;
//     }
// }

// int main() {
//     // Allocate memory on the GPU
//     float *d_a;
//     cudaMalloc(&d_a, LOAD * sizeof(float));
    
//     // Define grid and block dimensions
//     int blockSize = 256;
//     int numBlocks = (LOAD + blockSize - 1) / blockSize;
    
//     printf("Starting GPU load test...\n");
//     printf("Press Ctrl+C to stop the program\n");
    
//     // Run the kernel in an infinite loop
//     while (1) {
//         // Call the kernel with more iterations to increase load
//         loadGPU<<<numBlocks, blockSize>>>(d_a, 10000);
        
//         // Synchronize to ensure the kernel has completed
//         cudaDeviceSynchronize();
//     }
    
//     // Free memory (this won't be reached because of the infinite loop)
//     cudaFree(d_a);
    
//     return 0;
// }