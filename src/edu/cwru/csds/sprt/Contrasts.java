package edu.cwru.csds.sprt;
import java.util.List;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;

/**
 * Processes the contrast files used for SPRT
 * @author Ben
 *
 */
public class Contrasts {
	private int contrastLength = 0;
	private int numOfContrasts = 0;
	private int[][] contrasts;
	
	public Contrasts(int numOfContrasts, int contrastLength) {
		this.numOfContrasts = numOfContrasts;
		this.contrastLength = contrastLength;
		this.contrasts = new int[numOfContrasts][contrastLength];
		readStdout();
	}
	
	public Contrasts(String path) {
		readFile(path);
	}
	
	public Contrasts(String path, int numOfContrasts, int contrastLength) {
		this(path);
		if(this.numOfContrasts < numOfContrasts || this.contrastLength < contrastLength) System.out.println("Specified dimensions too large, will use dimensions of the whole file.");
		else {
			int[][] arr = new int[numOfContrasts][contrastLength];
			for(int i = 0; i < numOfContrasts; i++) {
				for(int j = 0; j < contrastLength; j++) {
					arr[i][j] = this.contrasts[i][j];
				}
			}
			this.numOfContrasts = numOfContrasts;
			this.contrastLength = contrastLength;
			this.contrasts = arr;
		}
	}
	
	public void readStdout() {
		Scanner scanner = new Scanner(System.in);
		for(int i = 1; i <= this.numOfContrasts; i++) {
			System.out.print("Input contrast " + i + " Out Of " + this.numOfContrasts + " (Size " + this.contrastLength + ") :");
			int[] array = Arrays.stream(scanner.nextLine().split(" ")).mapToInt(Integer::parseInt).toArray();
			if(array.length != this.contrastLength) {
				System.out.println("Size not match: Expect " + this.contrastLength + ", Actual: " + array.length + ".");
				scanner.close();
				i--;
				continue;
			}
			this.contrasts[i-1] = array;
		}
		System.out.println("Read Contrast Complete");
		scanner.close();
	}
	
	public void readFile(String path)  {
		List<int[]> rowList = new ArrayList<>();
		BufferedReader reader;
		try {
			reader = new BufferedReader(new FileReader(path));
			String line = reader.readLine();
			while(line != null) {
				int[] array = Arrays.stream(line.split("\\s")).mapToInt(Integer::parseInt).toArray();
				if(this.contrastLength == 0) this.contrastLength = array.length;
				else assert(this.contrastLength == array.length) : "File contains contrasts with different sizes";
				rowList.add(array);
				line = reader.readLine();
			}
			reader.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		this.numOfContrasts = rowList.size();
		this.contrasts = new int[this.numOfContrasts][this.contrastLength];
		for(int i = 0; i < this.numOfContrasts; i++) {
			this.contrasts[i] = rowList.get(i);
		}
	}
	
	public Matrix toMatrix() {
		int[] arr = new int[this.numOfContrasts * this.contrastLength];
		for(int i = 0; i < this.numOfContrasts; i++) {
			for(int j = 0; j <this.contrastLength; j++) {
				arr[i*this.contrastLength+j] = this.contrasts[i][j];
			}
		}
		return new Matrix(arr, this.numOfContrasts, this.contrastLength);
	}
	
	public static void main(String[] args) {
//		Contrasts contrasts = new Contrasts(2, 3);
//		System.out.println(contrasts.contrasts[1][2]);
		
		Contrasts contrasts1 = new Contrasts("contrasts.txt", 2, 2);
		System.out.println(contrasts1.contrasts[1][1]);
	}
	
	
	
	
}
