package sprt;

import java.io.Serializable;
import java.util.BitSet;

import sprt.exception.ImageException;

/**
 * Handles all configurable parameters.
 * 
 * @author Ben
 *
 */
public class Config implements Serializable {
	public int MAX_SCAN = 238; // total scans in the session
	public int COL = 8; // # of columns in design matrix / contrasts
	public int K = 78; // first K scans used to collect data to estimate theta 1
	public double ZScore = 3.12; // Z values used for estimating theta 1
	public double theta0 = 0;
	public double theta1 = 1;
	public double alpha = 0.001; // used to decide SPRT boundary
	public double beta = 0.1; // used to decide SPRT boundary
	public double percentLower = 0.85; // percentage of SPRT to count as active
	public double percentUpper = 0.85; // percentage of SPRT to count as non-active
	public double SPRTUpperBound = Algorithm.SPRTUpperBound(alpha, beta);
	public double SPRTLowerBound = Algorithm.SPRTLowerBound(alpha, beta);
	public String BOLDPath = "dat/";
	public String BOLDPrefix = "bold";
	public Contrasts contrasts;
	public boolean enableROI = true;

	private int x = 0;
	private int y = 0;
	private int z = 0;
	private BitSet ROI; // default to false

	public Config() {
	}

	public String assemblyBOLDPath(int scanNumber) {
		return BOLDPath + BOLDPrefix + (scanNumber - 1) + ".txt";
	}

	public void setImageSpec(Image image) {
		try {
			if (this.x == 0 && this.y == 0 && this.z == 0) {
				this.x = image.getX();
				this.y = image.getY();
				this.z = image.getZ();

				// set ROI here since only first scan will enter this if statement
				this.ROI = setROI(image);
			} else {
				throw new ImageException("Image size has been set and cannot be modified!");
			}

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public void expandeImageSize(int x) {
		this.x *= x;
	}

	public void setContrasts(Contrasts contrasts) {
		this.contrasts = contrasts;
	}

	public BitSet setROI(Image image) {
		BitSet ROI = new BitSet(image.getX() * image.getY() * image.getZ());
		for (int i = 0; i < image.getX() * image.getY() * image.getZ(); i++) {
			if (image.getVoxel(i) != 0) {
				ROI.set(i);
			}
		}
		return ROI;
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

	public BitSet getROI() {
		return this.ROI;
	}

	public int getLocation(int x, int y, int z) {
		return x * this.y * this.z + y * this.z + z;
	}
}
