package sprt;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.BitSet;

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
	private ArrayList<Image> brainVolumes;

	public Dataset(int x, int y, int z) {
		this.x = x;
		this.y = y;
		this.z = z;
		this.brainVolumes = new ArrayList<>();
	}

	public Dataset(ArrayList<Image> images) {
		this.x = images.get(0).getX();
		this.y = images.get(0).getY();
		this.z = images.get(0).getZ();
		for (Image volume : images) {
			add(volume);
		}
	}

	// Only allow sequential add
	public void add(Image image) {
		brainVolumes.add(image);
	}

	public ArrayList<DistributedDataset> toDistrbutedDataset(boolean enableROI, BitSet ROI) {
		ArrayList<DistributedDataset> ret = new ArrayList<>(ROI.cardinality());
		for (int x = 0; x < this.x; x++) {
			for (int y = 0; y < this.y; y++) {
				for (int z = 0; z < this.z; z++) {
					if (enableROI && ROI.get(getLocation(x, y, z))) { // comment this condition to disable ROI
						ret.add(new DistributedDataset(getBoldResponseAsArray(x, y, z), x, y, z));
					}
				}
			}
		}
		return ret;
	}

	// getter
	public int getTotalScanNumber() {
		return this.totalScans;
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

	public Image getVolume(int scans) {
		return this.brainVolumes.get(scans - 1);
	}

	public double getVoxel(int scanNumber, int x, int y, int z) {
		return this.brainVolumes.get(scanNumber - 1).getVoxel(x, y, z);
	}

	public Matrix getBoldResponseAsMatrix(int x, int y, int z) {
		int size = brainVolumes.size();
		double[] arr = new double[size];
		for (int i = 0; i < size; i++) {
			arr[i] = getVoxel(i + 1, x, y, z);
		}
		return new Matrix(arr, brainVolumes.size(), 1);
	}

	public double[] getBoldResponseAsArray(int x, int y, int z) {
		int size = brainVolumes.size();
		double[] arr = new double[size];
		for (int i = 0; i < size; i++) {
			arr[i] = getVoxel(i + 1, x, y, z);
		}
		return arr;
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

	public static void main(String[] args) {

	}
}
