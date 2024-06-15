package sprt.benchmark.zerocopy;

import java.nio.ByteBuffer;
import java.nio.IntBuffer;

public class ThroughputTest {
    static {
        try {
            System.loadLibrary("sprt_native"); // Load the native library
        } catch (UnsatisfiedLinkError e) {
            System.err.println("Native library failed to load: " + e);
            e.printStackTrace();
        } catch (Exception e) {
            System.err.println("Unexpected error occurred: " + e);
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        int intCapacity = 1024 * 1024 * 25; // 100 MB buffer, as integers
        int byteCapacity = intCapacity * Integer.BYTES; // 100 MB buffer, as bytes

        System.out.println("Buffer Size: " + byteCapacity / 1024 / 1024 + "MB\n");

        IntBuffer directIntBuffer = ByteBuffer.allocateDirect(intCapacity * Integer.BYTES).asIntBuffer();
        int[] heapIntArray = new int[intCapacity];

        ByteBuffer directByteBuffer = ByteBuffer.allocateDirect(byteCapacity);
        byte[] heapByteArray = new byte[byteCapacity];

        // Initialize buffers with some data
        for (int i = 0; i < intCapacity; i++) {
            directIntBuffer.put(i, i);
            heapIntArray[i] = i;
        }

        for (int i = 0; i < byteCapacity; i++) {
            directByteBuffer.put(i, (byte) (i % 256));
            heapByteArray[i] = (byte) (i % 256);
        }

        // Measure IntBuffer throughput using JNI
        long jniIntDuration = testIntThroughput(directIntBuffer);
        double jniIntThroughput = (intCapacity * Integer.BYTES / (1024.0 * 1024.0)) / (jniIntDuration / 1_000_000_000.0); // MB/s
        System.out.printf("JNI IntBuffer Throughput: %.2f MB/s%n", jniIntThroughput);

        // Measure int array throughput using non-zero copy JNI
        long nonZeroCopyJNIIntDuration = testNonZeroCopyIntThroughput(heapIntArray);
        double nonZeroCopyJNIIntThroughput = (intCapacity * Integer.BYTES / (1024.0 * 1024.0)) / (nonZeroCopyJNIIntDuration / 1_000_000_000.0); // MB/s
        System.out.printf("Non-Zero Copy JNI int[] Throughput: %.2f MB/s%n", nonZeroCopyJNIIntThroughput);

        // Measure Direct IntBuffer throughput purely in Java
        long javaDirectIntDuration = measureJavaDirectIntThroughput(directIntBuffer);
        double javaDirectIntThroughput = (intCapacity * Integer.BYTES / (1024.0 * 1024.0)) / (javaDirectIntDuration / 1_000_000_000.0); // MB/s
        System.out.printf("Java Direct IntBuffer Throughput: %.2f MB/s%n", javaDirectIntThroughput);

        // Measure heap int array throughput purely in Java
        long javaHeapIntDuration = measureJavaHeapIntThroughput(heapIntArray);
        double javaHeapIntThroughput = (intCapacity * Integer.BYTES / (1024.0 * 1024.0)) / (javaHeapIntDuration / 1_000_000_000.0); // MB/s
        System.out.printf("Java Heap int[] Throughput: %.2f MB/s%n", javaHeapIntThroughput);


        System.out.println();

        // Measure ByteBuffer throughput using JNI
        long jniByteDuration = testByteThroughput(directByteBuffer);
        double jniByteThroughput = (byteCapacity / (1024.0 * 1024.0)) / (jniByteDuration / 1_000_000_000.0); // MB/s
        System.out.printf("JNI ByteBuffer Throughput: %.2f MB/s%n", jniByteThroughput);

        // Measure byte array throughput using non-zero copy JNI
        long nonZeroCopyJNIByteDuration = testNonZeroCopyByteThroughput(heapByteArray);
        double nonZeroCopyJNIByteThroughput = (byteCapacity / (1024.0 * 1024.0)) / (nonZeroCopyJNIByteDuration / 1_000_000_000.0); // MB/s
        System.out.printf("Non-Zero Copy JNI byte[] Throughput: %.2f MB/s%n", nonZeroCopyJNIByteThroughput);

        // Measure Direct ByteBuffer throughput purely in Java
        long javaDirectByteDuration = measureJavaDirectByteThroughput(directByteBuffer);
        double javaDirectByteThroughput = (byteCapacity / (1024.0 * 1024.0)) / (javaDirectByteDuration / 1_000_000_000.0); // MB/s
        System.out.printf("Java Direct ByteBuffer Throughput: %.2f MB/s%n", javaDirectByteThroughput);

        // Measure heap byte array throughput purely in Java
        long javaHeapByteDuration = measureJavaHeapByteThroughput(heapByteArray);
        double javaHeapByteThroughput = (byteCapacity / (1024.0 * 1024.0)) / (javaHeapByteDuration / 1_000_000_000.0); // MB/s
        System.out.printf("Java Heap byte[] Throughput: %.2f MB/s%n", javaHeapByteThroughput);
    }

    // Native method declarations for int buffer
    private static native long testIntThroughput(IntBuffer buffer);
    private static native long testNonZeroCopyIntThroughput(int[] buffer);

    // Native method declarations for byte buffer
    private static native long testByteThroughput(ByteBuffer buffer);
    private static native long testNonZeroCopyByteThroughput(byte[] buffer);

    // Measure throughput purely in Java for Direct IntBuffer
    private static long measureJavaDirectIntThroughput(IntBuffer buffer) {
        buffer.clear();
        int capacity = buffer.capacity();

        long start = System.nanoTime();

        // Read from buffer to measure read throughput
        int sum = 0;
        for (int i = 0; i < capacity; i++) {
            sum += buffer.get(i);
        }

        long end = System.nanoTime();
        return end - start;
    }

    // Measure throughput purely in Java for Heap Int Array
    private static long measureJavaHeapIntThroughput(int[] buffer) {
        int capacity = buffer.length;

        long start = System.nanoTime();

        // Read from buffer to measure read throughput
        int sum = 0;
        for (int i = 0; i < capacity; i++) {
            sum += buffer[i];
        }

        long end = System.nanoTime();
        return end - start;
    }

    // Measure throughput purely in Java for Direct ByteBuffer
    private static long measureJavaDirectByteThroughput(ByteBuffer buffer) {
        buffer.clear();
        int capacity = buffer.capacity();

        long start = System.nanoTime();

        // Read from buffer to measure read throughput
        byte sum = 0;
        for (int i = 0; i < capacity; i++) {
            sum += buffer.get(i);
        }

        long end = System.nanoTime();
        return end - start;
    }

    // Measure throughput purely in Java for Heap Byte Array
    private static long measureJavaHeapByteThroughput(byte[] buffer) {
        int capacity = buffer.length;

        long start = System.nanoTime();

        // Read from buffer to measure read throughput
        byte sum = 0;
        for (int i = 0; i < capacity; i++) {
            sum += buffer[i];
        }

        long end = System.nanoTime();
        return end - start;
    }
}