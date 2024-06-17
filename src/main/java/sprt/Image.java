package sprt;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import sprt.exception.FileFormatNotCorrectException;
import sprt.exception.ImageException;

/**
 * Processes single bold response file
 * 
 * @author Ben
 *
 */
public class Image implements Serializable {
	private int x = 0;
	private int y = 0;
	private int z = 0;
	private ArrayList<Double> voxels;

	public Image(){}

	public Image(int x, int y, int z) {
		this.x = x;
		this.y = y;
		this.z = z;
		this.voxels = new ArrayList<>(x*y*z);
	}

	public Image(int x, int y, int z, boolean random) {
		this.x = x;
		this.y = y;
		this.z = z;
		this.voxels = new ArrayList<>(x*y*z);
		Random rand = new Random();
		for (int i = 0; i < x * y * z; i++) {
			this.voxels.set(i, rand.nextDouble());
		}
	}

	public Image(Image image) {
		this.x = image.getX();
		this.y = image.getY();
		this.z = image.getZ();
		this.voxels = image.voxels;
	}

	// getter
	public int getX() {
		return this.x;
	}

	public int getY() {
		return this.y;
	}

	public int getZ() {
		return this.z;
	}

	public ArrayList<Double> getVoxels() {
		return this.voxels;
	}

	public double getVoxel(int x, int y, int z) {
		return this.voxels.get(getLocation(x, y, z));
	}

	public double getVoxel(int pos) {
		return this.voxels.get(pos);
	}

	// helper function
	public int getLocation(int x, int y, int z) {
		return x * (this.y * this.z) + y * this.z + z;
	}

	// setter
	public void setX(int x) {
		this.x = x;
	}

	public void setY(int y) {
		this.y = y;
	}

	public void setZ(int z) {
		this.z = z;
	}

	// read in file after getting the path
	public void readImage(String path, int scanNumber) {
		BufferedReader reader;
		try {
			reader = new BufferedReader(new FileReader(path));
			String line = reader.readLine();
			if (line.contains("(")) {
				dimensionParser(line);
			} else {
				reader.close();
				throw new FileFormatNotCorrectException();
			}
			this.voxels = new ArrayList<>(x * y * z);
			line = reader.readLine();
			// int x = 0;
			int y = 0;
			while (line != null) {
				if (line.contains("Slice")) { // pass this line
					line = reader.readLine();
					continue;
				}
				double[] array = Arrays.stream(line.split(",")).mapToDouble(Double::parseDouble).toArray();
				for (int i = 0; i < array.length; i++) {
					voxels.add(array[i]);
					// System.out.println(x + " " + y + " " + i);
				}
				y++;
				if (y == this.y) {
					y = 0;
					// x++;
				}
				line = reader.readLine();
			}
			reader.close();
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(1);
		}

	}

	public void dimensionParser(String line) {
		try {
			int[] array = Arrays.stream(line.replaceAll("[()]", "").split(", ")).mapToInt(Integer::parseInt).toArray();
			if (array[0] == 0 || array[1] == 0 || array[2] == 0)
				throw new FileFormatNotCorrectException("Dimensions information in BOLD file cannot be 0");
			if (this.x == 0 && this.y == 0 && this.z == 0) {
				this.x = array[0];
				this.y = array[1];
				this.z = array[2];
			} else {
				if (this.x != array[0] || this.y != array[1] || this.z != array[2])
					throw new FileFormatNotCorrectException("Dimensions information not consistent through all scans");
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public void setVolume(double[] volume) throws ImageException {
		for (int i = 0; i < this.voxels.size(); i++) {
			this.voxels.set(i, volume[i]);
		}
	}

	public void setVoxel(double val, int x, int y, int z) {
		this.voxels.set(getLocation(x, y, z), val);
	}

	@Override
	public String toString() {
		int nonZeroCounter = 0;
		for (int x = 0; x < this.x; x++) {
			for (int y = 0; y < this.y; y++) {
				for (int z = 0; z < this.z; z++) {
					if (getVoxel(x, y, z) != 0.0)
						nonZeroCounter++;
				}
			}
		}
		return "Size: [" + x + " " + y + " " + z + "]. There are " + nonZeroCounter
				+ " Non-zero element in this volume";
	}

	public void outputToFile(String path) {
		try {
			FileWriter myWriter = new FileWriter(path);
			myWriter.write("\n\n");
			for (int x = 0; x < this.x; x++) {
				for (int y = 0; y < this.y; y++) {
					for (int z = 0; z < this.z; z++) {
						myWriter.write(Double.toString(getVoxel(x, y, z)) + ", ");
					}
					myWriter.write("\n");
				}
			}
			myWriter.close();
			System.out.println("Successfully wrote to the file.");
		} catch (IOException e) {
			System.out.println("An error occurred.");
			e.printStackTrace();
		}
	}

}