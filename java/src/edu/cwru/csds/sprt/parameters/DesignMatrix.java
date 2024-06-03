package edu.cwru.csds.sprt.parameters;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Arrays;

import edu.cwru.csds.sprt.exceptions.FileFormatNotCorrectException;
import edu.cwru.csds.sprt.numerical.Matrix;

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
	private double[][] designMatrix;

	public DesignMatrix(String path, int row, int col) {
		this.path = path;
		this.row = row;
		this.col = col;
		this.designMatrix = new double[this.row][this.col];
		try {
			readDesignMatrix(this.path, this.row, this.col);
		} catch (Exception e) {
			e.printStackTrace();
		}

	}

	public void readDesignMatrix(String path, int row, int col) throws FileFormatNotCorrectException {
		BufferedReader reader;
		int counter = 0;
		try {
			reader = new BufferedReader(new FileReader(path));
			String line = reader.readLine();
			while (line != null) {
				double[] array = Arrays.stream(line.split("\\s")).mapToDouble(Double::parseDouble).toArray();
				if (array.length != col) {
					reader.close();
					throw new FileFormatNotCorrectException("Design Matrix: Column Size Not Match!");
				}
				this.designMatrix[counter] = array;
				counter++;
				line = reader.readLine();
			}
			reader.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public Matrix toMatrix(int rowToSlice) {
		double[] arr = new double[rowToSlice * this.col];
		for (int i = 0; i < rowToSlice; i++) {
			for (int j = 0; j < this.col; j++) {
				arr[i * this.col + j] = this.designMatrix[i][j];
			}
		}
		return new Matrix(arr, rowToSlice, this.col);
	}

	// main
	public static void main(String[] args) {
		DesignMatrix designMatrix = new DesignMatrix("Latest_data/design_easy.txt", 238, 8);
		System.out.println(designMatrix.designMatrix[3][6]);

	}

}
