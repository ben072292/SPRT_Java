package sprt;

import java.io.Serializable;
import java.util.ArrayList;

/**
 * Stores and handles data read from bold response files
 * 
 * @author Ben
 *
 */
public class Dataset implements Serializable {
	private ArrayList<Image> images;
	private Config config;

	public Dataset(Config config) {
		this.config = config;
		this.images = new ArrayList<>();
	}

	// Only allow sequential add
	public void add(Image image) {
		images.add(image);
	}

	public ArrayList<double[]> toDD() {
		ArrayList<double[]> ret = new ArrayList<>(config.getROI().cardinality());
		int size = config.getX() * config.getY() * config.getZ();
		for (int pos = 0; pos < size; pos++) {
			if (config.enableROI && config.getROI().get(pos)) { // comment this condition to disable ROI
				ret.add(getBoldResponseAsList(pos));
			} else if (!config.enableROI) {
				ret.add(getBoldResponseAsList(pos));
			}
		}
		return ret;

	}

	public void addImages(ArrayList<Image> images) {
		this.images.addAll(images);
	}

	public void setImages(ArrayList<Image> images) {
		this.images.clear();
		this.images = images;
	}

	public void addImage(Image image) {
		this.images.add(image);
	}

	public Image getImage(int scan) {
		return this.images.get(scan - 1);
	}

	public double getVoxel(int scan, int pos) {
		return this.images.get(scan - 1).getVoxel(pos);
	}

	public double[] getBoldResponseAsList(int pos) {
		int size = images.size();
		double[] arr = new double[size];
		for (int i = 0; i < size; i++) {
			arr[i] = getVoxel(i + 1, pos);
		}
		return arr;
	}
}
