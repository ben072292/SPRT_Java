/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class sprt_benchmark_zerocopy_ThroughputTest */

#ifndef _Included_sprt_benchmark_zerocopy_ThroughputTest
#define _Included_sprt_benchmark_zerocopy_ThroughputTest
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     sprt_benchmark_zerocopy_ThroughputTest
 * Method:    testIntThroughput
 * Signature: (Ljava/nio/IntBuffer;)J
 */
JNIEXPORT jlong JNICALL Java_sprt_benchmark_zerocopy_ThroughputTest_testIntThroughput
  (JNIEnv *, jclass, jobject);

/*
 * Class:     sprt_benchmark_zerocopy_ThroughputTest
 * Method:    testNonZeroCopyIntThroughput
 * Signature: ([I)J
 */
JNIEXPORT jlong JNICALL Java_sprt_benchmark_zerocopy_ThroughputTest_testNonZeroCopyIntThroughput
  (JNIEnv *, jclass, jintArray);

/*
 * Class:     sprt_benchmark_zerocopy_ThroughputTest
 * Method:    testByteThroughput
 * Signature: (Ljava/nio/ByteBuffer;)J
 */
JNIEXPORT jlong JNICALL Java_sprt_benchmark_zerocopy_ThroughputTest_testByteThroughput
  (JNIEnv *, jclass, jobject);

/*
 * Class:     sprt_benchmark_zerocopy_ThroughputTest
 * Method:    testNonZeroCopyByteThroughput
 * Signature: ([B)J
 */
JNIEXPORT jlong JNICALL Java_sprt_benchmark_zerocopy_ThroughputTest_testNonZeroCopyByteThroughput
  (JNIEnv *, jclass, jbyteArray);

#ifdef __cplusplus
}
#endif
#endif