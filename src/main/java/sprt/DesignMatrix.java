package sprt;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Arrays;

import sprt.exception.FileFormatNotCorrectException;

/**
 * Processes design matrix files used for SPRT computation
 * 
 * @author Ben
 *
 */
public class DesignMatrix {
	private int row;
	private int col;
	private String path;
	private double[] arr;

	public DesignMatrix(String path, int row, int col) {
		this.path = path;
		this.row = row;
		this.col = col;
		this.arr = new double[this.row * this.col];
		try {
			readDesignMatrix(this.path, this.row, this.col);
		} catch (Exception e) {
			e.printStackTrace();
		}

	}

	public void readDesignMatrix(String path, int row, int col) throws FileFormatNotCorrectException {
		BufferedReader reader;
		int pos = 0;
		try {
			reader = new BufferedReader(new FileReader(path));
			String line = reader.readLine();
			while (line != null) {
				double[] array = Arrays.stream(line.split("\\s")).mapToDouble(Double::parseDouble).toArray();
				if (array.length != col) {
					reader.close();
					throw new FileFormatNotCorrectException("Design Matrix: Column Size Not Match!");
				}
				System.arraycopy(array, 0, this.arr, pos, array.length);
				pos+=array.length;
				line = reader.readLine();
			}
			reader.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public Matrix toMatrix(){
		return new Matrix(this.arr, this.row, this.col);
	}

}
