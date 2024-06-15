#include <jni.h>
#include <iostream>
#include <chrono>
#include "sprt_benchmark_zerocopy_ThroughputTest.h"

JNIEXPORT jlong JNICALL Java_sprt_benchmark_zerocopy_ThroughputTest_testIntThroughput(JNIEnv *env, jclass clazz, jobject buffer) {
    // Get the direct buffer address
    void* directBuffer = env->GetDirectBufferAddress(buffer);
    if (directBuffer == nullptr) {
        std::cerr << "Failed to get direct buffer address" << std::endl;
        return -1;
    }

    // Get the buffer capacity
    jlong capacity = env->GetDirectBufferCapacity(buffer) / sizeof(int);
    if (capacity == -1) {
        std::cerr << "Failed to get buffer capacity" << std::endl;
        return -1;
    }

    // Measure the throughput
    int* nativeBuffer = static_cast<int*>(directBuffer);
    auto start = std::chrono::high_resolution_clock::now();

    // Read from buffer to measure read throughput
    volatile int sum = 0;
    for (jlong i = 0; i < capacity; ++i) {
        sum += nativeBuffer[i];
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::nanoseconds duration = end - start;

    return duration.count(); // Return duration in nanoseconds
}

JNIEXPORT jlong JNICALL Java_sprt_benchmark_zerocopy_ThroughputTest_testNonZeroCopyIntThroughput(JNIEnv *env, jclass clazz, jintArray buffer) {
    jboolean isCopy;
    jint* nativeBuffer = env->GetIntArrayElements(buffer, &isCopy);
    if (nativeBuffer == nullptr) {
        std::cerr << "Failed to get int array elements" << std::endl;
        return -1;
    }

    jsize capacity = env->GetArrayLength(buffer);

    auto start = std::chrono::high_resolution_clock::now();

    // Read from buffer to measure read throughput
    volatile int sum = 0;
    for (jsize i = 0; i < capacity; ++i) {
        sum += nativeBuffer[i];
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::nanoseconds duration = end - start;

    env->ReleaseIntArrayElements(buffer, nativeBuffer, JNI_ABORT);
    return duration.count(); // Return duration in nanoseconds
}

JNIEXPORT jlong JNICALL Java_sprt_benchmark_zerocopy_ThroughputTest_testByteThroughput(JNIEnv *env, jclass clazz, jobject buffer) {
    // Get the direct buffer address
    void* directBuffer = env->GetDirectBufferAddress(buffer);
    if (directBuffer == nullptr) {
        std::cerr << "Failed to get direct buffer address" << std::endl;
        return -1;
    }

    // Get the buffer capacity
    jlong capacity = env->GetDirectBufferCapacity(buffer) / sizeof(int);
    if (capacity == -1) {
        std::cerr << "Failed to get buffer capacity" << std::endl;
        return -1;
    }

    // Measure the throughput
    char* nativeBuffer = static_cast<char*>(directBuffer);
    auto start = std::chrono::high_resolution_clock::now();

    // Read from buffer to measure read throughput
    volatile char sum = 0;
    for (jlong i = 0; i < capacity; ++i) {
        sum += nativeBuffer[i];
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::nanoseconds duration = end - start;

    return duration.count(); // Return duration in nanoseconds
}

JNIEXPORT jlong JNICALL Java_sprt_benchmark_zerocopy_ThroughputTest_testNonZeroCopyByteThroughput(JNIEnv *env, jclass clazz, jbyteArray buffer) {
    jboolean isCopy;
    jbyte* nativeBuffer = env->GetByteArrayElements(buffer, &isCopy);
    if (nativeBuffer == nullptr) {
        std::cerr << "Failed to get int array elements" << std::endl;
        return -1;
    }

    jsize capacity = env->GetArrayLength(buffer);

    auto start = std::chrono::high_resolution_clock::now();

    // Read from buffer to measure read throughput
    volatile char sum = 0;
    for (jsize i = 0; i < capacity; ++i) {
        sum += nativeBuffer[i];
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::nanoseconds duration = end - start;

    env->ReleaseByteArrayElements(buffer, nativeBuffer, JNI_ABORT);
    return duration.count(); // Return duration in nanoseconds
}