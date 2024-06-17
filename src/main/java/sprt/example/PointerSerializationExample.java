package sprt.example;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.function.Function;
import org.bytedeco.javacpp.DoublePointer;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.Arrays;
import java.util.List;

public class PointerSerializationExample {
    static class MySerializableClass implements Serializable {
        private static final long serialVersionUID = 1L;
    
        private transient DoublePointer doublePointer;
        private int id;
        private String name;
    
        public MySerializableClass(int id, String name) {
            this.id = id;
            this.name = name;
            this.doublePointer = new DoublePointer(10);
            for(int i = 0; i < 10; i++){
                doublePointer.put(i, i+0.5);
            }
        }
    
        private void writeObject(ObjectOutputStream oos) throws IOException {
            oos.defaultWriteObject();
            // No need to serialize the DoublePointer
        }
    
        private void readObject(ObjectInputStream ois) throws ClassNotFoundException, IOException {
            ois.defaultReadObject();
            this.doublePointer = new DoublePointer(1.0); // Reinitialize after deserialization
        }
    
        public DoublePointer getDoublePointer() {
            return doublePointer;
        }
    
        public int getId() {
            return id;
        }
    
        public String getName() {
            return name;
        }
    
        public void setDoublePointer(DoublePointer doublePointer) {
            this.doublePointer = doublePointer;
        }
    }


    public static void main(String[] args) {
        SparkConf conf = new SparkConf().setAppName("SparkExample").setMaster("local[*]");
        JavaSparkContext sc = new JavaSparkContext(conf);

        // Create a list of MySerializableClass objects
        List<MySerializableClass> list = Arrays.asList(
                new MySerializableClass(1, "Object1"),
                new MySerializableClass(2, "Object2"),
                new MySerializableClass(3, "Object3"));

        // Parallelize the list
        JavaRDD<MySerializableClass> rdd = sc.parallelize(list);

        // Perform a map transformation
        JavaRDD<String> resultRDD = rdd.map(new Function<MySerializableClass, String>() {
            @Override
            public String call(MySerializableClass obj) throws Exception {
                // Access the DoublePointer (reinitialized after deserialization)
                DoublePointer p = new DoublePointer(10);
                for(int i = 0; i < 10; i++){
                    p.put(i, i+0.5+obj.getId());
                }
                obj.setDoublePointer(p);
                double value = obj.getDoublePointer().get(5);
                return "ID: " + obj.getId() + ", Name: " + obj.getName() + ", DoublePointer Value: " + value;
            }
        });

        // Collect and print the results
        List<String> results = resultRDD.collect();
        for (String result : results) {
            System.out.println(result);
        }

        sc.stop();
    }
}
