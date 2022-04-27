package edu.cwru.csds.sprt.data;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import edu.cwru.csds.sprt.exceptions.VolumeNotMatchException;
import edu.cwru.csds.sprt.numerical.Matrix;

/**
 * Stores and handles data read from bold response files
 * 
 * @author Ben
 *
 */
public class Dataset implements Serializable {
	private int totalScans = 0;
	private int x = 0;
	private int y = 0;
	private int z = 0;
	private Brain[] brainVolumes;
	private int currentScans = 0;
	private boolean[] ROI;

	public Dataset(int scans, int x, int y, int z) {
		assert x > 0 : "X cannot be 0!";
		assert y > 0 : "Y cannot be 0!";
		assert z > 0 : "Z cannot be 0!";
		this.x = x;
		this.y = y;
		this.z = z;
		this.brainVolumes = new Brain[scans];
		for (int i = 0; i < scans; i++) {
			this.brainVolumes[i] = new Brain(i + 1, x, y, z);
		}
		currentScans = 0;
	}

	// Only allow sequential add
	public void addOneScan(Brain volume) {
		try {
			assert (currentScans == volume.getScanNumber() - 1) : "Need to add scan sequentially!";
			if (this.x != volume.getX() ||
					this.y != volume.getY() ||
					this.z != volume.getZ())
				throw new VolumeNotMatchException();
			this.brainVolumes[currentScans] = volume;
			currentScans++;
			isCompleted(this);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public void addMultipleScans(Brain[] volumes) {
		Arrays.sort(volumes, (Brain a, Brain b) -> a.getScanNumber() - b.getScanNumber());
		for (Brain volume : volumes)
			addOneScan(volume);
	}

	public void addWholeScans(Brain[] volumes) {
		for (Brain volume : volumes)
			addOneScan(volume);
		if (!isCompleted(this)) {
			System.err.println("Dataset Is Not Complete! Exiting...");
			System.exit(1);
		}
	}

	public ArrayList<DistributedDataset> toDistrbutedDataset() {
		ArrayList<DistributedDataset> ret = new ArrayList<>(this.x * this.y * this.z);
		Random rand = new Random();
		for (int x = 0; x < this.x; x++) {
			for (int y = 0; y < this.y; y++) {
				for (int z = 0; z < this.z; z++) {
					int pos = getLocation(x, y, z);
					// if(ROI[pos]) {
					// ret.add(new DistributedDataset(getBoldResponseAsArray(x, y, z), x, y, z, pos,
					// ROI[pos]));
					// }
					// else{
					double[] a = getBoldResponseAsArray(x, y, z);
					for (int i = 0; i < a.length; i++) {
						a[i] = rand.nextDouble();
					}
					ret.add(new DistributedDataset(a, x, y, z, pos, true));
					// }
				}
			}
		}
		return ret;
	}

	public ArrayList<DistributedDataset> toDistrbutedDataset(int dataExpand) {
		ArrayList<DistributedDataset> ret = new ArrayList<>(this.x * this.y * this.z);
		Random rand = new Random();
		for (int x = 0; x < this.x; x++) {
			for (int y = 0; y < this.y; y++) {
				for (int z = 0; z < this.z; z++) {
					int pos = getLocation(x, y, z);
					// if(ROI[pos]) {
					// ret.add(new DistributedDataset(getBoldResponseAsArray(x, y, z), x, y, z, pos,
					// ROI[pos]));
					// }
					// else{
					double[] a = getBoldResponseAsArray(x, y, z);
					for (int i = 0; i < a.length; i++) {
						a[i] = rand.nextDouble();
					}
					ret.add(new DistributedDataset(a, x, y, z, pos, true));
					// }
				}
			}
		}
		return ret;
	}

	// getter
	public int getTotalScanNumber() {
		return this.totalScans;
	}

	public int getCurrentScanNumber() {
		return this.currentScans;
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

	public int getLocation(int x, int y, int z) {
		return x * (this.y * this.z) + y * this.z + z;
	}

	public Brain getVolume(int scans) {
		return this.brainVolumes[scans - 1];
	}

	public double[] getVerticalVoxels(int x, int y, int z) {
		double[] res = new double[currentScans];
		for (int i = 0; i < currentScans; i++) {
			res[i] = brainVolumes[i].getVoxel(x, y, z);
		}
		return res;
	}

	public double getVoxel(int scanNumber, int x, int y, int z) {
		return this.brainVolumes[scanNumber - 1].getVoxel(x, y, z);
	}

	public double getSignal(int scanNumber, int x, int y, int z) {
		return this.brainVolumes[scanNumber - 1].getVoxel(x, y, z);
	}

	public boolean isCompleted(Dataset dataset) { // check whether the dataset is complete
		return currentScans == totalScans - 1;
	}

	public Matrix getBoldResponseAsMatrix(int x, int y, int z) {
		double[] arr = new double[this.currentScans];
		for (int i = 0; i < this.currentScans; i++) {
			arr[i] = getSignal(i + 1, x, y, z);
		}
		return new Matrix(arr, this.currentScans, 1);
	}

	public double[] getBoldResponseAsArray(int x, int y, int z) {
		double[] ret = new double[this.currentScans];
		for (int i = 0; i < this.currentScans; i++) {
			ret[i] = getSignal(i + 1, x, y, z);
		}
		return ret;
	}

	// setter
	public void setX(int x) {
		assert x > 0 : "X cannot be 0!";
		this.x = x;
	}

	public void setY(int y) {
		assert y > 0 : "Y cannot be 0!";
		this.y = y;
	}

	public void setZ(int z) {
		assert z > 0 : "Z cannot be 0!";
		this.z = z;
	}

	public void setSignal(int scanNumber, int x, int y, int z, double signal) {
		this.brainVolumes[scanNumber - 1].setVoxel(signal, x, y, z);
	}

	public void setROI(boolean[] ROI) {
		this.ROI = ROI;
	}

	public static void main(String[] args) {
		Dataset d = new Dataset(238, 36, 128, 128);
		System.out.println(d.getVolume(1));

	}
}
