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
    const int N = 1024;
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    float alpha = 1.0f, beta = 0.0f;

    h_A = new float[N * N];
    h_B = new float[N * N];
    h_C = new float[N * N];

    // Initialize matrices
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
        h_C[i] = 0.0f;
    }

    checkCudaError(cudaMalloc((void**)&d_A, N * N * sizeof(float)), "CUDA malloc A");
    checkCudaError(cudaMalloc((void**)&d_B, N * N * sizeof(float)), "CUDA malloc B");
    checkCudaError(cudaMalloc((void**)&d_C, N * N * sizeof(float)), "CUDA malloc C");

    checkCudaError(cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice), "CUDA memcpy A");
    checkCudaError(cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice), "CUDA memcpy B");

    cublasHandle_t handle;
    checkCublasError(cublasCreate(&handle), "CUBLAS initialization");

    // Warm up
    checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N), "CUBLAS SGEMM");

    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start), "Event create start");
    checkCudaError(cudaEventCreate(&stop), "Event create stop");

    checkCudaError(cudaEventRecord(start, 0), "Event record start");
    for(int i = 0; i < 100; i++)
        checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N), "CUBLAS SGEMM");
    checkCudaError(cudaEventRecord(stop, 0), "Event record stop");

    checkCudaError(cudaEventSynchronize(stop), "Event synchronize stop");

    float milliseconds = 0;
    checkCudaError(cudaEventElapsedTime(&milliseconds, start, stop), "Event elapsed time");

    std::cout << "cuBLAS GEMM time: " << milliseconds << " ms" << std::endl;

    // Clean up
    checkCublasError(cublasDestroy(handle), "CUBLAS destroy");
    checkCudaError(cudaFree(d_A), "CUDA free A");
    checkCudaError(cudaFree(d_B), "CUDA free B");
    checkCudaError(cudaFree(d_C), "CUDA free C");
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
