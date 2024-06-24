#include "sprt_Algorithm.h"
#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <jni.h>
#include <stdio.h>

#define CUDACHECK(err) (checkCudaError(err, __FILE__, __LINE__))

static void checkCudaError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error at " << file << ":" << line << " - " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
}

static __device__ void my_mmul(const float *A, const float *B, float *C, int m, int k, int n)
{
    for (int row = 0; row < m; ++row)
    {
        for (int col = 0; col < n; ++col)
        {
            float sum = 0.0f;
            for (int i = 0; i < k; ++i)
            {
                sum += A[row * k + i] * B[i * n + col];
            }
            C[row * n + col] = sum;
        }
    }
}

static __global__ void compute_SPRT(float *sprt,
                                    const float *C, int c_n, int c_row, int c_col,
                                    const float *X, int X_row, int X_col,
                                    const float *Y, int Y_n, int Y_row, int Y_col,
                                    const float *XTXInverseXT, int XTXInverseXT_row, int XTXInverseXT_col,
                                    const float *XXTXInverse, int XXTXInverse_row, int XXTXInverse_col,
                                    const float *H, int H_row, int H_col,
                                    float *d_betaHat, float *d_XbetaHat, float *d_D, float *d_cXTXInverseXTD, float *d_cXTXInverseXTDXXTXInverse, float *d_cBetaHat)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= Y_n)
        return;

    // compute beta_hat XTXInverseXTY (8 * 238 X 238 * 1 = 8 * 1)
    my_mmul(XTXInverseXT, Y + tid * Y_row * Y_col, d_betaHat + tid * XTXInverseXT_row * Y_col, XTXInverseXT_row, Y_row, Y_col);

    // compute XbetaHat (238 * 8 x 8 * 1 = 238 * 1)
    my_mmul(X, d_betaHat + tid * XTXInverseXT_row * Y_col, d_XbetaHat + tid * X_row * Y_col, X_row, X_col, Y_col);

    for (int i = 0; i < Y_row; i++)
    {
        d_D[tid * Y_row + i] = pow(Y[tid * Y_row * Y_col + i] - d_XbetaHat[tid * X_row * Y_col + i], 2.0f) / (1.0f - H[i]);
    }
    for (int i = 0; i < c_n; ++i)
    {
        // compute c(XTXInverseXT)D (1 * 238)
        my_mmul(C + i * c_row * c_col, XTXInverseXT, d_cXTXInverseXTD + tid * i * X_row, c_row, c_col, XTXInverseXT_col);
        for (int j = 0; j < XTXInverseXT_col; ++j)
        {
            d_cXTXInverseXTD[tid * X_row + j] *= d_D[tid * Y_row + j];
        }

        // compute cXTXInverseXTDXXTXInverse
        my_mmul(d_cXTXInverseXTD + tid * X_row, XXTXInverse, d_cXTXInverseXTDXXTXInverse + tid * c_row * XXTXInverse_col, c_row, XTXInverseXT_col, XXTXInverse_col);

        // compute cBetaHat (1 * 1)
        my_mmul(C + i * c_row * c_col, d_betaHat + tid * XTXInverseXT_row * Y_col, d_cBetaHat + tid * c_row * Y_col, c_row, c_col, Y_col);

        float variance = 0.0f;
        for (int j = 0; j < c_col; j++)
        {
            variance += C[i * c_row * c_col + j] * d_cXTXInverseXTDXXTXInverse[tid * c_row * XXTXInverse_col + j];
        }
        sprt[tid * c_n + i] = (pow(d_cBetaHat[tid * c_row * Y_col] - 0.0f, 2) - pow(d_cBetaHat[tid * c_row * Y_col] - 1.0f, 2)) / (2 * variance);
    }
}

// Initialize cuBLAS
static int initialized = 0;
static jclass arrayListClass = nullptr;
static jclass matrixClass = nullptr;
static jmethodID arrayListGet = nullptr;
static jmethodID arrayListSize = nullptr;
static jmethodID matrixGetRow = nullptr;
static jmethodID matrixGetCol = nullptr;
static jmethodID matrixGetDirectBuffer = nullptr;
static jint C_n = 0, Y_n = 0;
static jint c_row = 0, c_col = 0;
static float **h_C_s = nullptr;
static float **h_Y_s = nullptr;
static float *d_C_s = nullptr;
static float *d_sprt = nullptr;
static float *h_sprt = nullptr;
static cudaStream_t s;
static float *d_Y_s = nullptr;

/*
 * Class:     sprt_Algorithm
 * Method:    computeSPRT_CUDA_v2
 * Signature: (IILjava/util/ArrayList;Lsprt/Matrix;Ljava/util/ArrayList;Lsprt/Matrix;Lsprt/Matrix;Lsprt/Matrix;)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_sprt_Algorithm_computeSPRT_1CUDA_1v2(JNIEnv *env, jclass Algorithm_Class, jint scan, jint total_scan, jobject C_Matrix_List, jobject X_Matrix, jobject Y_Matrix_List, jobject XTXInverseXT_Matrix, jobject XXTXInverse_Matrix, jobject H_Matrix)
{
    std::cout << "Native Space: Entering." << std::endl;
    auto native_space_start = std::chrono::high_resolution_clock::now();

    if (!initialized)
    {
        // Retrieve ArrayList class and methods
        arrayListClass = env->FindClass("java/util/ArrayList");
        arrayListGet = env->GetMethodID(arrayListClass, "get", "(I)Ljava/lang/Object;");
        arrayListSize = env->GetMethodID(arrayListClass, "size", "()I");

        // Retrieve the size of the ArrayList
        C_n = env->CallIntMethod(C_Matrix_List, arrayListSize);
        Y_n = env->CallIntMethod(Y_Matrix_List, arrayListSize);

        // Retrieve Matrix class and methods
        matrixClass = env->FindClass("sprt/Matrix");
        matrixGetRow = env->GetMethodID(matrixClass, "getRow", "()I");
        matrixGetCol = env->GetMethodID(matrixClass, "getCol", "()I");
        matrixGetDirectBuffer = env->GetMethodID(matrixClass, "getDirectBuffer", "()Ljava/nio/ByteBuffer;");

        h_C_s = (float **)malloc(C_n * sizeof(float *));
        h_Y_s = (float **)malloc(Y_n * sizeof(float *));
        CUDACHECK(cudaMalloc(&d_Y_s, Y_n * total_scan * sizeof(float)));

        // Iterate through ArrayList to get Matrix objects
        jobject temp_c_mat = env->CallObjectMethod(C_Matrix_List, arrayListGet, 0);
        c_row = env->CallIntMethod(temp_c_mat, matrixGetRow);
        c_col = env->CallIntMethod(temp_c_mat, matrixGetCol);
        h_C_s[0] = static_cast<float *>(env->GetDirectBufferAddress(env->CallObjectMethod(temp_c_mat, matrixGetDirectBuffer)));
        for (int i = 1; i < C_n; ++i)
        {
            temp_c_mat = env->CallObjectMethod(C_Matrix_List, arrayListGet, i);

            // Retrieve row, col, and buffer for cMatrix
            h_C_s[i] = static_cast<float *>(env->GetDirectBufferAddress(env->CallObjectMethod(temp_c_mat, matrixGetDirectBuffer)));
            if (h_C_s[i] == nullptr)
            {
                std::cerr << "Failed to get direct buffer address for cMatrix at index " << i << std::endl;
                return NULL;
            }
        }
        CUDACHECK(cudaStreamCreate(&s));
        CUDACHECK(cudaMalloc(&d_C_s, C_n * c_row * c_col * sizeof(float)));
        for (int i = 0; i < C_n; i++)
        {
            CUDACHECK(cudaMemcpyAsync(d_C_s + i * c_row * c_col, h_C_s[i], c_row * c_col * sizeof(float), cudaMemcpyHostToDevice, s));
        }
        h_sprt = (float *)malloc(C_n * Y_n * sizeof(float));
        CUDACHECK(cudaMalloc(&d_sprt, C_n * Y_n * sizeof(float)));
    } // end initialization

    jobject temp_Y_mat = env->CallObjectMethod(Y_Matrix_List, arrayListGet, 0);
    jint Y_row = env->CallIntMethod(temp_Y_mat, matrixGetRow);
    jint Y_col = env->CallIntMethod(temp_Y_mat, matrixGetCol);

    // Retrieve row, col, and buffer for YMatrix
    if (!initialized)
    {
        h_Y_s[0] = static_cast<float *>(env->GetDirectBufferAddress(env->CallObjectMethod(temp_Y_mat, matrixGetDirectBuffer)));
        for (int i = 1; i < Y_n; i++)
        {
            temp_Y_mat = env->CallObjectMethod(Y_Matrix_List, arrayListGet, i);
            // Retrieve row, col, and buffer for XMatrix
            h_Y_s[i] = static_cast<float *>(env->GetDirectBufferAddress(env->CallObjectMethod(temp_Y_mat, matrixGetDirectBuffer)));
            if (h_Y_s[i] == nullptr)
            {
                std::cerr << "Failed to get direct buffer address for XMatrix at index " << i << std::endl;
                return NULL;
            }
        }
    }

    // Retrieve dimensions and buffers for Y, XTXInverseXT, XXTXInverse, H
    int X_row = env->CallIntMethod(X_Matrix, matrixGetRow);
    int X_col = env->CallIntMethod(X_Matrix, matrixGetCol);
    float *h_X = static_cast<float *>(env->GetDirectBufferAddress(env->CallObjectMethod(X_Matrix, matrixGetDirectBuffer)));
    float *d_X = nullptr;

    int XTXInverseXT_row = env->CallIntMethod(XTXInverseXT_Matrix, matrixGetRow);
    int XTXInverseXT_col = env->CallIntMethod(XTXInverseXT_Matrix, matrixGetCol);
    float *h_XTXInverseXT = static_cast<float *>(env->GetDirectBufferAddress(env->CallObjectMethod(XTXInverseXT_Matrix, matrixGetDirectBuffer)));
    float *d_XTXInverseXT = nullptr;

    int XXTXInverse_row = env->CallIntMethod(XXTXInverse_Matrix, matrixGetRow);
    int XXTXInverse_col = env->CallIntMethod(XXTXInverse_Matrix, matrixGetCol);
    float *h_XXTXInverse = static_cast<float *>(env->GetDirectBufferAddress(env->CallObjectMethod(XXTXInverse_Matrix, matrixGetDirectBuffer)));
    float *d_XXTXInverse = nullptr;

    int H_row = env->CallIntMethod(H_Matrix, matrixGetRow);
    int H_col = env->CallIntMethod(H_Matrix, matrixGetCol);
    float *h_H = static_cast<float *>(env->GetDirectBufferAddress(env->CallObjectMethod(H_Matrix, matrixGetDirectBuffer)));
    float *d_H = nullptr;

    CUDACHECK(cudaMalloc(&d_X, X_row * X_col * sizeof(float)));
    CUDACHECK(cudaMemcpyAsync(d_X, h_X, X_row * X_col * sizeof(float), cudaMemcpyHostToDevice, s));

    CUDACHECK(cudaMalloc(&d_XTXInverseXT, XTXInverseXT_row * XTXInverseXT_col * sizeof(float)));
    CUDACHECK(cudaMemcpyAsync(d_XTXInverseXT, h_XTXInverseXT, XTXInverseXT_row * XTXInverseXT_col * sizeof(float), cudaMemcpyHostToDevice, s));

    CUDACHECK(cudaMalloc(&d_XXTXInverse, XXTXInverse_row * XXTXInverse_col * sizeof(float)));
    CUDACHECK(cudaMemcpyAsync(d_XXTXInverse, h_XXTXInverse, XXTXInverse_row * XXTXInverse_col * sizeof(float), cudaMemcpyHostToDevice, s));

    CUDACHECK(cudaMalloc(&d_H, H_row * H_col * sizeof(float)));
    CUDACHECK(cudaMemcpyAsync(d_H, h_H, H_row * H_col * sizeof(float), cudaMemcpyHostToDevice, s));
    
    // TODO: this memcpy is a major bottleneck, try memcpy2D, but requires adjusting data layout
    for (int i = 0; i < Y_n; i++)
    {
        CUDACHECK(cudaMemcpyAsync(d_Y_s + i * Y_row * Y_col + scan - 1, h_Y_s[i] + scan - 1, 1 * sizeof(float), cudaMemcpyHostToDevice, s));
    }

    // auto native_space_probe = std::chrono::high_resolution_clock::now();
    // auto native_space_probe_duration = std::chrono::duration_cast<std::chrono::milliseconds>(native_space_probe - native_space_start);
    // std::cout << "Native Space: Probing Point. Took " << native_space_probe_duration.count() << " ms" << std::endl;

    // intermediate matricies
    float *d_betaHat = nullptr,
          *d_XbetaHat = nullptr,
          *d_D = nullptr,
          *d_cXTXInverseXTD = nullptr,
          *d_cXTXInverseXTDXXTXInverse = nullptr,
          *d_cBetaHat = nullptr;

    CUDACHECK(cudaMalloc(&d_betaHat, Y_n * XTXInverseXT_row * Y_col * sizeof(float)));
    CUDACHECK(cudaMalloc(&d_XbetaHat, Y_n * X_row * Y_col * sizeof(float)));
    CUDACHECK(cudaMalloc(&d_D, Y_n * Y_row * sizeof(float)));
    CUDACHECK(cudaMalloc(&d_cXTXInverseXTD, Y_n * X_row * sizeof(float)));
    CUDACHECK(cudaMalloc(&d_cXTXInverseXTDXXTXInverse, Y_n * c_row * XXTXInverse_col * sizeof(float)));
    CUDACHECK(cudaMalloc(&d_cBetaHat, Y_n * c_row * Y_col * sizeof(float)));

    dim3 blockDims(256);                                  // Adjust block dimensions as needed
    dim3 gridDims((Y_n + blockDims.x - 1) / blockDims.x); // Calculate grid dimensions

    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));

    // Record the start event
    CUDACHECK(cudaEventRecord(start, s));

    compute_SPRT<<<gridDims, blockDims, 0, s>>>(d_sprt,
                                                d_C_s, C_n, c_row, c_col,
                                                d_X, X_row, X_col,
                                                d_Y_s, Y_n, Y_row, Y_col,
                                                d_XTXInverseXT, XTXInverseXT_row, XTXInverseXT_col,
                                                d_XXTXInverse, XXTXInverse_row, XXTXInverse_col,
                                                d_H, H_row, H_col,
                                                d_betaHat, d_XbetaHat, d_D, d_cXTXInverseXTD, d_cXTXInverseXTDXXTXInverse, d_cBetaHat);

    // Record the stop event
    CUDACHECK(cudaEventRecord(stop, s));

    // Synchronize the stream to wait for the stop event
    CUDACHECK(cudaStreamSynchronize(s));

    // Calculate the elapsed time
    float milliseconds = 0;
    CUDACHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

    CUDACHECK(cudaMemcpy(h_sprt, d_sprt, C_n * Y_n * sizeof(float), cudaMemcpyDeviceToHost));

    // The return object can be set according to your specific requirements
    // For example, returning the buffer address of HBuffer as a ByteBuffer
    jobject resultBuffer = env->NewDirectByteBuffer(h_sprt, C_n * Y_n * sizeof(float));
    initialized++;

    // cleanup

    CUDACHECK(cudaFree(d_X));
    CUDACHECK(cudaFree(d_XTXInverseXT));
    CUDACHECK(cudaFree(d_XXTXInverse));
    CUDACHECK(cudaFree(d_H));

    CUDACHECK(cudaFree(d_betaHat));
    CUDACHECK(cudaFree(d_XbetaHat));
    CUDACHECK(cudaFree(d_D));
    CUDACHECK(cudaFree(d_cXTXInverseXTD));
    CUDACHECK(cudaFree(d_cXTXInverseXTDXXTXInverse));
    CUDACHECK(cudaFree(d_cBetaHat));

    auto native_space_stop = std::chrono::high_resolution_clock::now();
    auto native_space_duration = std::chrono::duration_cast<std::chrono::milliseconds>(native_space_stop - native_space_start);
    std::cout << "Native Space: Existing. Took " << native_space_duration.count() << " ms" << std::endl;
    return resultBuffer;
}

/*
 * Class:     sprt_Algorithm
 * Method:    cleanup_CUDA
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_sprt_Algorithm_cleanup_1CUDA_1v2(JNIEnv *, jclass)
{

    CUDACHECK(cudaFree(d_C_s));
    CUDACHECK(cudaFree(d_sprt));
    CUDACHECK(cudaFree(d_Y_s));
    CUDACHECK(cudaStreamDestroy(s));
}