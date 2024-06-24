#include "sprt_Algorithm.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <jni.h>
#include <stdio.h>
#include <vector>

#ifndef NUM_CUDA_STREAMS
#define NUM_CUDA_STREAMS 8
#endif

#define CUDACHECK(err) (checkCudaError(err, __FILE__, __LINE__))
#define CUBLASCHECK(status) (checkCublasError(status, __FILE__, __LINE__))

static void checkCudaError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error at " << file << ":" << line << " - " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
}

static void checkCublasError(cublasStatus_t status, const char *file, int line)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "cuBLAS error at " << file << ":" << line << std::endl;
        exit(-1);
    }
}

static __global__ void generateD(float *d_R, float *d_Y, float *d_XBetaHat, float *d_H);
static __global__ void generatecXTXInverseXTD(float *d_cXTXInverseXT, float *D);
static __global__ void computeSPRT(float *SPRT, float *cBetaHat, float *c, float *cXTXInverseXTDXXTXInverse, float theta0, float theta1, int cCol);

// Initialize cuBLAS
static int initialized = 0;
static jclass arrayListClass = nullptr;
static jclass matrixClass = nullptr;
static jmethodID arrayListGet = nullptr;
static jmethodID arrayListSize = nullptr;
static jmethodID matrixGetRow = nullptr;
static jmethodID matrixGetCol = nullptr;
static jmethodID matrixGetDirectBuffer = nullptr;
static jint cListSize = 0, YListSize = 0;
static jint cRow = 0, cCol = 0;
static std::vector<float *> cBuffers(2);
static std::vector<float *> YBuffers(36 * 128 * 128);
static float **d_cBuffers = nullptr;
// Initialize cuBLAS handles and CUDA streams
static std::vector<cublasHandle_t> handles(NUM_CUDA_STREAMS);
static std::vector<cudaStream_t> streams(NUM_CUDA_STREAMS);
static float *d_SPRTBuffer = nullptr;
static float *SPRTBuffer = nullptr;

/*
 * Class:     sprt_Algorithm
 * Method:    computeSPRT_CUDA
 * Signature: (Ljava/util/ArrayList;Lsprt/Matrix;Ljava/util/ArrayList;Lsprt/Matrix;Lsprt/Matrix;Lsprt/Matrix;)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_sprt_Algorithm_computeSPRT_1CUDA(JNIEnv *env, jclass Algorithm, jobject cList, jobject X, jobject YList, jobject XTXInverseXT, jobject XXTXInverse, jobject H)
{
    if (!initialized)
    {
        for (int i = 0; i < NUM_CUDA_STREAMS; ++i)
        {
            CUBLASCHECK(cublasCreate(&handles[i]));
            CUDACHECK(cudaStreamCreate(&streams[i]));
            CUBLASCHECK(cublasSetStream(handles[i], streams[i]));
        }

        // Retrieve ArrayList class and methods
        arrayListClass = env->FindClass("java/util/ArrayList");
        arrayListGet = env->GetMethodID(arrayListClass, "get", "(I)Ljava/lang/Object;");
        arrayListSize = env->GetMethodID(arrayListClass, "size", "()I");

        // Retrieve the size of the ArrayList
        cListSize = env->CallIntMethod(cList, arrayListSize);
        YListSize = env->CallIntMethod(YList, arrayListSize);

        // Retrieve Matrix class and methods
        matrixClass = env->FindClass("sprt/Matrix");
        matrixGetRow = env->GetMethodID(matrixClass, "getRow", "()I");
        matrixGetCol = env->GetMethodID(matrixClass, "getCol", "()I");
        matrixGetDirectBuffer = env->GetMethodID(matrixClass, "getDirectBuffer", "()Ljava/nio/ByteBuffer;");

        cBuffers.resize(cListSize);
        YBuffers.resize(YListSize);

        // Iterate through ArrayList to get Matrix objects
        jobject cMatrix = env->CallObjectMethod(cList, arrayListGet, 0);
        cRow = env->CallIntMethod(cMatrix, matrixGetRow);
        cCol = env->CallIntMethod(cMatrix, matrixGetCol);
        cBuffers[0] = static_cast<float *>(env->GetDirectBufferAddress(env->CallObjectMethod(cMatrix, matrixGetDirectBuffer)));
        for (int i = 0; i < cListSize; ++i)
        {
            cMatrix = env->CallObjectMethod(cList, arrayListGet, i);

            // Retrieve row, col, and buffer for cMatrix
            cBuffers[i] = static_cast<float *>(env->GetDirectBufferAddress(env->CallObjectMethod(cMatrix, matrixGetDirectBuffer)));
            if (cBuffers[i] == nullptr)
            {
                std::cerr << "Failed to get direct buffer address for cMatrix at index " << i << std::endl;
                return NULL;
            }
        }
        d_cBuffers = (float **)malloc(cListSize * sizeof(float *));
        for (int i = 0; i < cListSize; i++)
        {
            CUDACHECK(cudaMalloc(&d_cBuffers[i], cRow * cCol * sizeof(float)));
            CUDACHECK(cudaMemcpy(d_cBuffers[i], cBuffers[i], cRow * cCol * sizeof(float), cudaMemcpyHostToDevice));
        }

        SPRTBuffer = (float *)malloc(cListSize * YListSize * sizeof(float));
        CUDACHECK(cudaMalloc(&d_SPRTBuffer, cListSize * YListSize * sizeof(float)));
    } // end initialization

    jobject YMatrix = env->CallObjectMethod(YList, arrayListGet, 0);
    jint YRow = env->CallIntMethod(YMatrix, matrixGetRow);
    jint YCol = env->CallIntMethod(YMatrix, matrixGetCol);

    // Retrieve row, col, and buffer for YMatrix
    if (!initialized)
    {
        YBuffers[0] = static_cast<float *>(env->GetDirectBufferAddress(env->CallObjectMethod(YMatrix, matrixGetDirectBuffer)));
        for (int i = 1; i < YListSize; i++)
        {
            YMatrix = env->CallObjectMethod(YList, arrayListGet, i);
            // Retrieve row, col, and buffer for XMatrix
            YBuffers[i] = static_cast<float *>(env->GetDirectBufferAddress(env->CallObjectMethod(YMatrix, matrixGetDirectBuffer)));
            if (YBuffers[i] == nullptr)
            {
                std::cerr << "Failed to get direct buffer address for XMatrix at index " << i << std::endl;
                return NULL;
            }
        }
    }

    // Retrieve dimensions and buffers for Y, XTXInverseXT, XXTXInverse, H
    int XRow = env->CallIntMethod(X, matrixGetRow);
    int XCol = env->CallIntMethod(X, matrixGetCol);
    float *XBuffer = static_cast<float *>(env->GetDirectBufferAddress(env->CallObjectMethod(X, matrixGetDirectBuffer)));
    float *d_XBuffer = nullptr;

    int XTXInverseXTRow = env->CallIntMethod(XTXInverseXT, matrixGetRow);
    int XTXInverseXTCol = env->CallIntMethod(XTXInverseXT, matrixGetCol);
    float *XTXInverseXTBuffer = static_cast<float *>(env->GetDirectBufferAddress(env->CallObjectMethod(XTXInverseXT, matrixGetDirectBuffer)));
    float *d_XTXInverseXTBuffer = nullptr;

    int XXTXInverseRow = env->CallIntMethod(XXTXInverse, matrixGetRow);
    int XXTXInverseCol = env->CallIntMethod(XXTXInverse, matrixGetCol);
    float *XXTXInverseBuffer = static_cast<float *>(env->GetDirectBufferAddress(env->CallObjectMethod(XXTXInverse, matrixGetDirectBuffer)));
    float *d_XXTXInverseBuffer = nullptr;

    int HRow = env->CallIntMethod(H, matrixGetRow);
    int HCol = env->CallIntMethod(H, matrixGetCol);
    float *HBuffer = static_cast<float *>(env->GetDirectBufferAddress(env->CallObjectMethod(H, matrixGetDirectBuffer)));
    float *d_HBuffer = nullptr;

    CUDACHECK(cudaMalloc(&d_XBuffer, XRow * XCol * sizeof(float)));
    CUDACHECK(cudaMemcpy(d_XBuffer, XBuffer, XRow * XCol * sizeof(float), cudaMemcpyHostToDevice));

    CUDACHECK(cudaMalloc(&d_XTXInverseXTBuffer, XTXInverseXTRow * XTXInverseXTCol * sizeof(float)));
    CUDACHECK(cudaMemcpy(d_XTXInverseXTBuffer, XTXInverseXTBuffer, XTXInverseXTRow * XTXInverseXTCol * sizeof(float), cudaMemcpyHostToDevice));

    CUDACHECK(cudaMalloc(&d_XXTXInverseBuffer, XXTXInverseRow * XXTXInverseCol * sizeof(float)));
    CUDACHECK(cudaMemcpy(d_XXTXInverseBuffer, XXTXInverseBuffer, XXTXInverseRow * XXTXInverseCol * sizeof(float), cudaMemcpyHostToDevice));

    CUDACHECK(cudaMalloc(&d_HBuffer, HRow * HCol * sizeof(float)));
    CUDACHECK(cudaMemcpy(d_HBuffer, HBuffer, HRow * HCol * sizeof(float), cudaMemcpyHostToDevice));

    // Perform necessary computations using cuBLAS with the retrieved buffers
    float alpha = 1.0f;
    float beta = 0.0f;

    float **d_YBuffers = (float **)malloc(NUM_CUDA_STREAMS * sizeof(float *));
    float **d_betaHatBuffers = (float **)malloc(NUM_CUDA_STREAMS * sizeof(float *));
    float **d_XBetaHatBuffers = (float **)malloc(NUM_CUDA_STREAMS * sizeof(float *));
    float **d_DBuffers = (float **)malloc(NUM_CUDA_STREAMS * sizeof(float *));
    float **d_cXTXInverseXTDBuffers = (float **)malloc(NUM_CUDA_STREAMS * sizeof(float *));
    float **d_cXTXInverseXTDXXTXInverseBuffers = (float **)malloc(NUM_CUDA_STREAMS * sizeof(float *));
    float **d_cBetaHatBuffers = (float **)malloc(NUM_CUDA_STREAMS * sizeof(float *));
    for (int i = 0; i < NUM_CUDA_STREAMS; i++)
    {
        CUDACHECK(cudaMalloc(&d_YBuffers[i], YRow * YCol * sizeof(float)));
        CUDACHECK(cudaMalloc(&d_betaHatBuffers[i], XTXInverseXTRow * YCol * sizeof(float)));
        CUDACHECK(cudaMalloc(&d_XBetaHatBuffers[i], XRow * YCol * sizeof(float)));
        CUDACHECK(cudaMalloc(&d_DBuffers[i], XRow * YCol * sizeof(float)));
        CUDACHECK(cudaMalloc(&d_cXTXInverseXTDBuffers[i], cRow * XTXInverseXTCol * sizeof(float)));
        CUDACHECK(cudaMalloc(&d_cXTXInverseXTDXXTXInverseBuffers[i], cRow * XXTXInverseCol * sizeof(float)));
        CUDACHECK(cudaMalloc(&d_cBetaHatBuffers[i], cRow * YCol * sizeof(float)));
    }
    for (int i = 0; i < YListSize; ++i)
    {
        int stream_index = i % NUM_CUDA_STREAMS;
        CUDACHECK(cudaMemcpyAsync(d_YBuffers[stream_index], YBuffers[i], YRow * YCol * sizeof(float), cudaMemcpyHostToDevice, streams[stream_index]));

        // compute beta_hat XTXInverseXTY (238 * 1)
        CUBLASCHECK(cublasSgemm(handles[stream_index],
                                       CUBLAS_OP_T, CUBLAS_OP_T,
                                       XTXInverseXTRow, YCol, YRow,
                                       &alpha,
                                       d_XTXInverseXTBuffer, XTXInverseXTCol,
                                       d_YBuffers[stream_index], YCol,
                                       &beta,
                                       d_betaHatBuffers[stream_index], XTXInverseXTRow));

        // compute D (238 * 1)
        CUBLASCHECK(cublasSgemm(handles[stream_index],
                                       CUBLAS_OP_T, CUBLAS_OP_N,
                                       XRow, YCol, XCol,
                                       &alpha,
                                       d_XBuffer, XCol,
                                       d_betaHatBuffers[stream_index], XTXInverseXTRow,
                                       &beta,
                                       d_XBetaHatBuffers[stream_index], XRow));
        generateD<<<1, XRow, 0, streams[stream_index]>>>(d_DBuffers[stream_index], d_YBuffers[stream_index], d_XBetaHatBuffers[stream_index], d_HBuffer);

        for (int j = 0; j < cListSize; ++j)
        {

            // compute c(XTXInverseXT)D (1 * 238)
            CUBLASCHECK(cublasSgemm(handles[stream_index],
                                           CUBLAS_OP_T, CUBLAS_OP_T,
                                           cCol, XTXInverseXTCol, XTXInverseXTRow,
                                           &alpha,
                                           d_cBuffers[j], cCol,
                                           d_XTXInverseXTBuffer, XTXInverseXTCol,
                                           &beta,
                                           d_cXTXInverseXTDBuffers[stream_index], cCol));

            generatecXTXInverseXTD<<<1, XRow, 0, streams[stream_index]>>>(d_cXTXInverseXTDBuffers[stream_index], d_DBuffers[stream_index]);

            // compute cXTXInverseXTDXXTXInverse
            CUBLASCHECK(cublasSgemm(handles[stream_index],
                                           CUBLAS_OP_N, CUBLAS_OP_T,
                                           cRow, XXTXInverseCol, XXTXInverseRow,
                                           &alpha,
                                           d_cXTXInverseXTDBuffers[stream_index], cCol,
                                           d_XXTXInverseBuffer, XXTXInverseCol,
                                           &beta,
                                           d_cXTXInverseXTDXXTXInverseBuffers[stream_index], cCol));

            // compute cBetaHat (1 * 1)
            CUBLASCHECK(cublasSgemm(handles[stream_index],
                                           CUBLAS_OP_T, CUBLAS_OP_N,
                                           cCol, YCol, XTXInverseXTRow,
                                           &alpha,
                                           d_cBuffers[j], cCol,
                                           d_betaHatBuffers[stream_index], XTXInverseXTRow,
                                           &beta,
                                           d_cBetaHatBuffers[stream_index], cCol));

            computeSPRT<<<1, 1, 0, streams[stream_index]>>>(&d_SPRTBuffer[i * cListSize + j], d_cBuffers[i], d_cBetaHatBuffers[stream_index], d_cXTXInverseXTDXXTXInverseBuffers[stream_index], 0.0f, 1.0f, 1);
        }
    }
    CUDACHECK(cudaMemcpy(SPRTBuffer, d_SPRTBuffer, cListSize * YListSize * sizeof(float), cudaMemcpyDeviceToHost));

    // The return object can be set according to your specific requirements
    // For example, returning the buffer address of HBuffer as a ByteBuffer
    jobject resultBuffer = env->NewDirectByteBuffer(SPRTBuffer, cListSize * YListSize * sizeof(float));
    initialized++;

    // cleanup

    CUDACHECK(cudaFree(d_XBuffer));
    CUDACHECK(cudaFree(d_XTXInverseXTBuffer));
    CUDACHECK(cudaFree(d_XXTXInverseBuffer));
    CUDACHECK(cudaFree(d_HBuffer));

    for (int i = 0; i < NUM_CUDA_STREAMS; i++)
    {
        CUDACHECK(cudaFree(d_YBuffers[i]));
        CUDACHECK(cudaFree(d_betaHatBuffers[i]));
        CUDACHECK(cudaFree(d_XBetaHatBuffers[i]));
        CUDACHECK(cudaFree(d_DBuffers[i]));
        CUDACHECK(cudaFree(d_cXTXInverseXTDBuffers[i]));
        CUDACHECK(cudaFree(d_cXTXInverseXTDXXTXInverseBuffers[i]));
        CUDACHECK(cudaFree(d_cBetaHatBuffers[i]));
    }
    free(d_YBuffers);
    free(d_betaHatBuffers);
    free(d_XBetaHatBuffers);
    free(d_DBuffers);
    free(d_cXTXInverseXTDBuffers);
    free(d_cXTXInverseXTDXXTXInverseBuffers);
    free(d_cBetaHatBuffers);

    return resultBuffer;
}

static __global__ void generateD(float *d_D, float *d_Y, float *d_XBetaHat, float *d_H)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    d_D[tid] = std::pow(d_Y[tid] - d_XBetaHat[tid], 2) / (1.0f - d_H[tid]);
    __syncthreads();
}

static __global__ void generatecXTXInverseXTD(float *d_cXTXInverseXT, float *D)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    d_cXTXInverseXT[tid] *= D[tid];
    __syncthreads();
}

static __global__ void computeSPRT(float *SPRT, float *cBetaHat, float *c, float *cXTXInverseXTDXXTXInverse, float theta0, float theta1, int cCol)
{
    float variance = 0.0f;
    for (int i = 0; i < cCol; i++)
    {
        variance += c[i] * cXTXInverseXTDXXTXInverse[i];
    }
    *SPRT = (std::pow(cBetaHat[0] - theta0, 2) - std::pow(cBetaHat[0] - theta1, 2)) / (2 * variance);
}

/*
 * Class:     sprt_Algorithm
 * Method:    cleanup_CUDA
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_sprt_Algorithm_cleanup_1CUDA(JNIEnv *, jclass)
{
    for (int i = 0; i < cListSize; i++)
    {
        CUDACHECK(cudaFree(d_cBuffers[i]));
    }
    free(d_cBuffers);

    CUDACHECK(cudaFree(d_SPRTBuffer));
}