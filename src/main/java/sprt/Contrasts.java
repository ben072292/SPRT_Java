package sprt;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;

/**
 * Processes the contrast files used for SPRT
 * 
 * @author Ben
 *
 */
public class Contrasts implements Serializable {
	public int contrastLength = 0;
	public int numContrasts = 0;
	private float[][] contrasts;

	public Contrasts(int numOfContrasts, int contrastLength) {
		this.numContrasts = numOfContrasts;
		this.contrastLength = contrastLength;
		this.contrasts = new float[numOfContrasts][contrastLength];
		readStdout();
	}

	public Contrasts(String path) {
		readFile(path);
	}

	public Contrasts(String path, int numOfContrasts, int contrastLength) {
		this(path);
		if (this.numContrasts < numOfContrasts || this.contrastLength < contrastLength)
			System.out.println("Specified dimensions too large, will use dimensions of the whole file.");
		else {
			float[][] arr = new float[numOfContrasts][contrastLength];
			for (int i = 0; i < numOfContrasts; i++) {
				for (int j = 0; j < contrastLength; j++) {
					arr[i][j] = this.contrasts[i][j];
				}
			}
			this.numContrasts = numOfContrasts;
			this.contrastLength = contrastLength;
			this.contrasts = arr;
		}
	}

	public void readStdout() {
		Scanner scanner = new Scanner(System.in);
		for (int i = 1; i <= this.numContrasts; i++) {
			System.out.print(
					"Input contrast " + i + " Out Of " + this.numContrasts + " (Size " + this.contrastLength + ") :");
			double[] array = Arrays.stream(scanner.nextLine().split(" ")).mapToDouble(Integer::parseInt).toArray();
			if (array.length != this.contrastLength) {
				System.out.println("Size not match: Expect " + this.contrastLength + ", Actual: " + array.length + ".");
				scanner.close();
				i--;
				continue;
			}
			for(int j = 0; j < array.length; j++){
				this.contrasts[i-1][j] = (float)array[j];
			}
		}
		System.out.println("Read Contrast Complete");
		scanner.close();
	}

	public void readFile(String path) {
		BufferedReader reader;
		try {
			reader = new BufferedReader(new FileReader(path));
			String line = reader.readLine();
			this.numContrasts = Integer.parseInt(line);
			this.contrasts = new float[this.numContrasts][];
			for(int i = 0; i < this.numContrasts; i++) {
				line = reader.readLine();
				double[] array = Arrays.stream(line.split("\\s")).mapToDouble(Integer::parseInt).toArray();
				this.contrasts[i] = new float[array.length];
				for(int j = 0; j < array.length; j++){
					this.contrasts[i][j] = (float)array[j];
				}
			}
			this.contrastLength = this.contrasts[0].length;
			reader.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public ArrayList<Matrix> toMatrices() {
		ArrayList<Matrix> ret = new ArrayList<>();
		for(int i = 0; i < numContrasts; i++){
			ret.add(new Matrix(contrasts[i], 1, this.contrastLength));
		}
		return ret;
	}

}
