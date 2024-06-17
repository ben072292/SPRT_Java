package sprt.example;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.SparkConf;
import org.bytedeco.javacpp.DoublePointer;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.nio.ByteBuffer;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;
import java.util.ArrayList;
import java.util.List;

public class PointerSerializationExample {

    static class CustomData implements Serializable {
        private transient double[] dataArray;
        private transient DoublePointer doublePointer;

        public CustomData(double[] dataArray) {
            this.dataArray = dataArray;
        }

        public double[] getDataArray() {
            return dataArray;
        }

        public void setDataArray(double[] dataArray) {
            this.dataArray = dataArray;
        }

        public DoublePointer getDoublePointer() {
            return doublePointer;
        }

        // private void writeObject(ObjectOutputStream out) throws IOException {
        //     out.defaultWriteObject();
        //     if (dataArray != null) {
        //         out.writeInt(dataArray.length);
        //         for (double value : dataArray) {
        //             out.writeDouble(value);
        //         }
        //     }
        // }

        private void writeObject(ObjectOutputStream out) throws IOException {
            out.defaultWriteObject();
            if (dataArray != null) {
                out.writeInt(dataArray.length);
                for (double value : dataArray) {
                    long bits = Double.doubleToRawLongBits(value);
                    long swappedBits = Long.reverseBytes(bits); // has to change the endianess (from big to little) to produce the correct result
                    out.writeLong(swappedBits);
                }
            }
        }

        private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
            in.defaultReadObject();
            int length = in.readInt();

            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(length * Double.BYTES);//.order(ByteOrder.nativeOrder());
            ReadableByteChannel channel = Channels.newChannel(in);
            channel.read(byteBuffer);
            byteBuffer.flip(); // Prepare the buffer for reading
            doublePointer = new DoublePointer(byteBuffer.asDoubleBuffer());
            // System.out.println("==================================== " + byteBuffer.asDoubleBuffer().get(0) + " " + doublePointer.get(0));
            dataArray = null; // Set dataArray to null after loading into DoublePointer
        }
    }

    public static void main(String[] args) {
        SparkConf conf = new SparkConf().setAppName("SparkDoublePointerExample").setMaster("local[*]");
        try (JavaSparkContext sc = new JavaSparkContext(conf)) {
            // Sample data
            List<CustomData> dataList = new ArrayList<>();
            dataList.add(new CustomData(new double[] { 1.0, 2.0, 3.0 }));
            dataList.add(new CustomData(new double[] { 4.0, 5.0, 6.0 }));

            // Create RDD
            JavaRDD<CustomData> rdd = sc.parallelize(dataList);

            // Map step: Convert DoublePointer to double[]
            JavaRDD<double[]> mappedRDD = rdd.map(customData -> {
                DoublePointer dp = customData.getDoublePointer();
                double[] array = new double[(int) dp.capacity()];
                dp.get(array);
                return array;
            });

            // Reduce step: Element-wise sum
            double[] elementWiseSum = mappedRDD.reduce((array1, array2) -> {
                double[] result = new double[array1.length];
                for (int i = 0; i < array1.length; i++) {
                    result[i] = array1[i] + array2[i];
                }
                return result;
            });

            // Print the result
            System.out.println("Element-wise sum:");
            for (double value : elementWiseSum) {
                System.out.print(value + " ");
            }
            System.out.println();

            sc.stop();
        }
    }

    // public static void main(String[] args) {
    // // Allocate a direct ByteBuffer with capacity for 10 doubles (8 bytes each)
    // int capacity = 10 * Double.BYTES; // Double.BYTES is 8
    // ByteBuffer buffer = ByteBuffer.allocateDirect(capacity);
    // buffer.order(ByteOrder.nativeOrder()); // Set byte order to native order

    // // Fill the ByteBuffer with double data
    // for (int i = 0; i < 10; i++) {
    // buffer.putDouble(i * 1.0); // Example: setting values
    // }

    // // Flip the buffer to prepare it for reading
    // buffer.flip();

    // // Create a DoublePointer using the direct ByteBuffer
    // DoublePointer doublePointer = new DoublePointer(buffer.asDoubleBuffer());

    // // Verify the values using the DoublePointer
    // for (int i = 0; i < 10; i++) {
    // System.out.println(doublePointer.get(i));
    // }
    // }
}
