#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define BATCH_COUNT 100

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
    const int strideA = N * N;
    const int strideB = N * N;
    const int strideC = N * N;

    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    float alpha = 1.0f, beta = 0.0f;

    h_A = new float[BATCH_COUNT * N * N];
    h_B = new float[BATCH_COUNT * N * N];
    h_C = new float[BATCH_COUNT * N * N];

    // Initialize matrices
    for (int i = 0; i < BATCH_COUNT * N * N; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
        h_C[i] = 0.0f;
    }

    checkCudaError(cudaMalloc((void**)&d_A, BATCH_COUNT * N * N * sizeof(float)), "CUDA malloc A");
    checkCudaError(cudaMalloc((void**)&d_B, BATCH_COUNT * N * N * sizeof(float)), "CUDA malloc B");
    checkCudaError(cudaMalloc((void**)&d_C, BATCH_COUNT * N * N * sizeof(float)), "CUDA malloc C");

    checkCudaError(cudaMemcpy(d_A, h_A, BATCH_COUNT * N * N * sizeof(float), cudaMemcpyHostToDevice), "CUDA memcpy A");
    checkCudaError(cudaMemcpy(d_B, h_B, BATCH_COUNT * N * N * sizeof(float), cudaMemcpyHostToDevice), "CUDA memcpy B");

    cublasHandle_t handle;
    checkCublasError(cublasCreate(&handle), "CUBLAS initialization");

    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start), "Event create start");
    checkCudaError(cudaEventCreate(&stop), "Event create stop");

    checkCudaError(cudaEventRecord(start, 0), "Event record start");

    checkCublasError(
        cublasSgemmStridedBatched(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, N, N,
            &alpha,
            d_A, N, strideA,
            d_B, N, strideB,
            &beta,
            d_C, N, strideC,
            BATCH_COUNT
        ),
        "CUBLAS Strided Batched SGEMM"
    );

    checkCudaError(cudaEventRecord(stop, 0), "Event record stop");
    checkCudaError(cudaEventSynchronize(stop), "Event synchronize stop");

    float milliseconds = 0;
    checkCudaError(cudaEventElapsedTime(&milliseconds, start, stop), "Event elapsed time");

    std::cout << "cuBLAS Strided Batched GEMM time for " << BATCH_COUNT << " GEMMs: " << milliseconds << " ms" << std::endl;

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
