package edu.cwru.csds.sprt.data;

import java.io.FileWriter;
import java.io.IOException;
import java.io.Serializable;
import java.util.List;
import java.util.Random;

import com.google.common.primitives.Doubles;

import edu.cwru.csds.sprt.exceptions.VolumeNotMatchException;

/**
 * Processes single bold response file
 * 
 * @author Ben
 *
 */
public class Brain implements Serializable {
	private int scanNumber = 0;
	private int x = 0;
	private int y = 0;
	private int z = 0;
	private double[] voxels;

	public Brain(int scanNum, int x, int y, int z) {
		this.scanNumber = scanNum;
		this.x = x;
		this.y = y;
		this.z = z;
		this.voxels = new double[x * y * z];
	}

	public Brain(int scanNum, int x, int y, int z, boolean random) {
		this.scanNumber = scanNum;
		this.x = x;
		this.y = y;
		this.z = z;
		this.voxels = new double[x * y * z];
		Random rand = new Random();
		for (int i = 0; i < x * y * z; i++) {
			this.voxels[i] = rand.nextDouble();
		}
	}

	public Brain(Brain volume) {
		this.scanNumber = volume.getScanNumber();
		this.x = volume.getX();
		this.y = volume.getY();
		this.z = volume.getZ();
		this.voxels = volume.voxels;
	}

	// getter
	public int getScanNumber() {
		return this.scanNumber;
	}

	public int getX() {
		return this.x;
	}

	public int getY() {
		return this.y;
	}

	public int getZ() {
		return this.z;
	}

	public double[] getVoxels() {
		return this.voxels;
	}

	public List<Double> getVoxelsAsList() {
		return Doubles.asList(this.voxels);
	}

	public double getVoxel(int x, int y, int z) {
		return this.voxels[getLocation(x, y, z)];
	}

	public double getVoxel(int pos) {
		return this.voxels[pos];
	}

	// helper function
	public int getLocation(int x, int y, int z) {
		return x * (this.y * this.z) + y * this.z + z;
	}

	// setter
	public void setScanNumber(int scanNumber) {
		this.scanNumber = scanNumber;
	}

	public void setX(int x) {
		this.x = x;
	}

	public void setY(int y) {
		this.y = y;
	}

	public void setZ(int z) {
		this.z = z;
	}

	public void setVolume(double[] volume) throws VolumeNotMatchException {
		for (int i = 0; i < this.voxels.length; i++) {
			this.voxels[i] = volume[i];
		}
	}

	public void setVoxel(double signal, int x, int y, int z) {
		this.voxels[getLocation(x, y, z)] = signal;
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