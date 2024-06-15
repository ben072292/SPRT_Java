#include <iostream>
#include <mkl.h>
#include <chrono>
#include <cstdlib>

int main() {

    // Set environment variables
    setenv("MKL_NUM_THREADS", "6", 1);
    setenv("MKL_DYNAMIC", "FALSE", 1);
    setenv("OMP_NUM_THREADS", "8", 1);

    const int N = 8192;
    float *A, *B, *C;
    float alpha = 1.0f, beta = 0.0f;

    A = new float[N * N];
    B = new float[N * N];
    C = new float[N * N];

    // Initialize matrices
    for (int i = 0; i < N * N; ++i) {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
        B[i] = static_cast<float>(rand()) / RAND_MAX;
        C[i] = 0.0f;
    }

    // Warm up
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, alpha, A, N, B, N, beta, C, N);

    auto start = std::chrono::high_resolution_clock::now();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, alpha, A, N, B, N, beta, C, N);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float, std::milli> duration = end - start;
    std::cout << "Intel MKL GEMM time: " << duration.count() << " ms" << std::endl;

    // Clean up
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
