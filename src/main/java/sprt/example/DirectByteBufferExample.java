package sprt.example;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.SparkConf;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

public class DirectByteBufferExample {
    public static void main(String[] args) {
        SparkConf conf = new SparkConf().setAppName("DirectByteBufferExample").setMaster("local[4]");
        try (JavaSparkContext sc = new JavaSparkContext(conf)) {
            // Create an ArrayList of DirectByteBuffer
            List<ByteBuffer> bufferList = new ArrayList<>();
            bufferList.add(ByteBuffer.allocateDirect(1024));
            bufferList.add(ByteBuffer.allocateDirect(2048));

            // Serialize DirectByteBuffer to byte[]
            List<byte[]> serializedBuffers = new ArrayList<>();
            for (ByteBuffer buffer : bufferList) {
                serializedBuffers.add(serializeDirectByteBuffer(buffer));
            }

            // Parallelize the serialized byte arrays
            JavaRDD<byte[]> rdd = sc.parallelize(serializedBuffers);

            // Perform a simple transformation and deserialize back to DirectByteBuffer
            JavaRDD<Integer> transformedRDD = rdd.map(bytes -> {
                ByteBuffer buffer = deserializeToDirectByteBuffer(bytes);
                // Example processing: just return the same buffer
                return 0;
            });

            // Collect the results
            List<Integer> resultList = transformedRDD.collect();

            // Print the results
            for (Integer buffer : resultList) {
                System.out.println("Buffer capacity: " + buffer.intValue());
            }

            sc.stop();
        }
    }

    public static byte[] serializeDirectByteBuffer(ByteBuffer buffer) {
        byte[] bytes = new byte[buffer.capacity()];
        buffer.position(0); // Ensure the buffer is at the start
        buffer.get(bytes);
        return bytes;
    }

    public static ByteBuffer deserializeToDirectByteBuffer(byte[] bytes) {
        ByteBuffer buffer = ByteBuffer.allocateDirect(bytes.length);
        buffer.put(bytes);
        buffer.flip(); // Prepare the buffer for reading
        return buffer;
    }
}
