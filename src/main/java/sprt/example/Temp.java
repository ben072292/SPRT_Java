package sprt.example;

import java.util.ArrayList;

import sprt.Algorithm;
import sprt.Matrix;
import sprt.Matrix.MatrixStorageScope;

public class Temp {
    static {
        try {
            // Print the current library path for debugging
            System.out.println("java.library.path=" + System.getProperty("java.library.path"));

            // Set the java.library.path to include the target directory
            System.setProperty("java.library.path",
                    System.getProperty("java.library.path") + ":/home/weicong/Desktop/SPRT_Java/target");

            // Load the native libraries
            System.loadLibrary("sprt_native");
            System.loadLibrary("sprt_native_cuda");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        ArrayList<Matrix> cList = new ArrayList<>();
        cList.add(new Matrix(1, 8, MatrixStorageScope.NATIVE));
        Matrix X = new Matrix(238, 8, MatrixStorageScope.NATIVE);
        ArrayList<Matrix> YList = new ArrayList<>();
        for(int i = 0; i < (36 * 128 * 128); i++){
            YList.add(new Matrix(238, 1, MatrixStorageScope.NATIVE));
        }
        Matrix XTXInverseXT = new Matrix(8, 238, MatrixStorageScope.NATIVE);
        Matrix XXTXInverse = new Matrix(238, 8, MatrixStorageScope.NATIVE);
        Matrix H = new Matrix(238, 1, MatrixStorageScope.NATIVE);
        Algorithm.computeSPRT_CUDA(cList, X, YList, XTXInverseXT, XXTXInverse, H);
    }
}
