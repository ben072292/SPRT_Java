#include <cuda_runtime.h>
#include <iostream>

#define CHECK_CUDA_ERROR(err) (checkCudaError(err, __FILE__, __LINE__))

void checkCudaError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error at " << file << ":" << line << " - " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
}

// Device function for addition
__device__ void add(const float *A, const float *B, float *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        C[idx] = A[idx] + B[idx];
    }
}

// Device function for multiplication
__device__ void multiply(const float *A, const float *B, float *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        C[idx] = A[idx] * B[idx];
    }
}

// Kernel to call different device functions based on thread index
__global__ void differentFunctionsKernel(float *A, float *B, float *C_add, float *C_multiply, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    if (idx % 2 == 0) // Even threads perform addition
    {
        add(A, B, C_add, N);
    }
    else // Odd threads perform multiplication
    {
        multiply(A, B, C_multiply, N);
    }
}

int main()
{
    const int N = 16;  // Size of the arrays
    const int SIZE = N * sizeof(float);

    float h_A[N] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    float h_B[N] = {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    float h_C_add[N];      // Result array for addition
    float h_C_multiply[N]; // Result array for multiplication

    float *d_A, *d_B, *d_C_add, *d_C_multiply;

    // Allocate device memory
    CUDACHECK(cudaMalloc(&d_A, SIZE));
    CUDACHECK(cudaMalloc(&d_B, SIZE));
    CUDACHECK(cudaMalloc(&d_C_add, SIZE));
    CUDACHECK(cudaMalloc(&d_C_multiply, SIZE));

    // Copy arrays from host to device
    CUDACHECK(cudaMemcpy(d_A, h_A, SIZE, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_B, h_B, SIZE, cudaMemcpyHostToDevice));

    // Define block size and grid size
    int blockSize = 16;
    int gridSize = (N + blockSize - 1) / blockSize;

    // Launch the kernel
    differentFunctionsKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C_add, d_C_multiply, N);
    CUDACHECK(cudaDeviceSynchronize());

    // Copy result arrays from device to host
    CUDACHECK(cudaMemcpy(h_C_add, d_C_add, SIZE, cudaMemcpyDeviceToHost));
    CUDACHECK(cudaMemcpy(h_C_multiply, d_C_multiply, SIZE, cudaMemcpyDeviceToHost));

    // Print the results for addition
    std::cout << "Result of addition:" << std::endl;
    for (int i = 0; i < N; ++i)
    {
        std::cout << h_C_add[i] << " ";
    }
    std::cout << std::endl;

    // Print the results for multiplication
    std::cout << "Result of multiplication:" << std::endl;
    for (int i = 0; i < N; ++i)
    {
        std::cout << h_C_multiply[i] << " ";
    }
    std::cout << std::endl;

    // Clean up
    CUDACHECK(cudaFree(d_A));
    CUDACHECK(cudaFree(d_B));
    CUDACHECK(cudaFree(d_C_add));
    CUDACHECK(cudaFree(d_C_multiply));

    return 0;
}