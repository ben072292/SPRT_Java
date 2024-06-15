#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void checkCublasError(cublasStatus_t err, const char* msg) {
    if (err != CUBLAS_STATUS_SUCCESS) {
        std::cerr << msg << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    const int N = 3; // Size of the matrix (N x N)

    // Host memory
    float h_A[N*N] = {4, 7, 2, 3, 6, 1, 2, 5, 1}; // Input matrix
    float h_Ainv[N*N]; // Output matrix (inverse)

    // Device memory
    float* d_A;
    checkCudaError(cudaMalloc((void**)&d_A, N * N * sizeof(float)), "Failed to allocate device memory for A");
    checkCudaError(cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice), "Failed to copy A to device");

    int* d_P; // Pivot indices
    checkCudaError(cudaMalloc((void**)&d_P, N * sizeof(int)), "Failed to allocate device memory for pivot indices");

    int* d_info; // Info output
    checkCudaError(cudaMalloc((void**)&d_info, sizeof(int)), "Failed to allocate device memory for info");

    // cuBLAS handle
    cublasHandle_t handle;
    checkCublasError(cublasCreate(&handle), "Failed to create cuBLAS handle");

    // Perform LU decomposition
    checkCublasError(cublasSgetrfBatched(handle, N, &d_A, N, d_P, d_info, 1), "Failed to perform LU decomposition");

    // Check for successful LU decomposition
    int h_info;
    checkCudaError(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy info to host");
    if (h_info != 0) {
        std::cerr << "LU decomposition failed: Matrix is singular" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Perform matrix inversion
    checkCublasError(cublasSgetriBatched(handle, N, (const float**)&d_A, N, d_P, &d_A, N, d_info, 1), "Failed to perform matrix inversion");

    // Check for successful matrix inversion
    checkCudaError(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy info to host");
    if (h_info != 0) {
        std::cerr << "Matrix inversion failed" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Copy the result back to host
    checkCudaError(cudaMemcpy(h_Ainv, d_A, N * N * sizeof(float), cudaMemcpyDeviceToHost), "Failed to copy Ainv to host");

    // Print the result
    std::cout << "Inverse matrix:" << std::endl;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << h_Ainv[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    // Clean up
    cudaFree(d_A);
    cudaFree(d_P);
    cudaFree(d_info);
    cublasDestroy(handle);

    return 0;
}