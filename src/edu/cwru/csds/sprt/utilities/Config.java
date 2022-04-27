package edu.cwru.csds.sprt.utilities;

import java.io.Serializable;

import edu.cwru.csds.sprt.data.Brain;
import edu.cwru.csds.sprt.exceptions.VolumeNotMatchException;
import edu.cwru.csds.sprt.numerical.Numerical;
import edu.cwru.csds.sprt.parameters.Contrasts;

/**
 * Handles all configurable parameters.
 * 
 * @author Ben
 *
 */
public class Config implements Serializable {
	public int ROW = 88; // total scans in the session
	public int COL = 8; // # of columns in design matrix / contrasts
	public int K = 78; // first K blocks used to collect data to estimate theta 1
	public double ZScore = 3.12; // Z values used for estimating theta 1
	public double theta0 = 0;
	public double theta1 = 1;
	public double alpha = 0.001; // used to decide SPRT boundary
	public double beta = 0.1; // used to decide SPRT boundary
	public double percentLower = 0.85; // percentage of SPRT to count as active
	public double percentUpper = 0.85; // percentage of SPRT to count as non-active
	public double SPRTUpperBound = Numerical.SPRTUpperBound(alpha, beta);
	public double SPRTLowerBound = Numerical.SPRTLowerBound(alpha, beta);
	public String BOLDPath = "Latest_data/";
	public String BOLDPrefix = "bold";
	public Contrasts contrasts;

	private int x = 0;
	private int y = 0;
	private int z = 0;
	private boolean[] ROI; // default to false

	public Config() {
	}

	public String assemblyBOLDPath(int scanNumber) {
		return BOLDPath + BOLDPrefix + (scanNumber - 1) + ".txt";
	}

	public void setVolumeSize(Brain volume) {
		try {
			if (this.x == 0 && this.y == 0 && this.z == 0) {
				this.x = volume.getX();
				this.y = volume.getY();
				this.z = volume.getZ();

				// set ROI here since only first scan will enter this if statement
				this.ROI = setROI(volume);
			} else {
				throw new VolumeNotMatchException("Brain volume size has been set and cannot be modified!");
			}

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public void expandeVolumeSize(int x) {
		this.x *= x;
	}

	public void setContrasts(Contrasts contrasts) {
		this.contrasts = contrasts;
	}

	public boolean[] setROI(Brain volume) {
		boolean[] ROI = new boolean[volume.getX() * volume.getY() * volume.getZ()];
		for (int i = 0; i < volume.getX() * volume.getY() * volume.getZ(); i++) {
			if (volume.getVoxel(i) != 0) {
				ROI[i] = true;
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

	public boolean[] getROI() {
		return this.ROI;
	}

	public int getLocation(int x, int y, int z) {
		return x * this.y * this.z + y * this.z + z;
	}
}
