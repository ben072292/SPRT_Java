package edu.cwru.csds.sprt.data;

import java.io.Serializable;

import edu.cwru.csds.sprt.numerical.Matrix;

/**
 * Dataset object that stores the parameters to be computed
 * by Spark executors
 * 
 * @author Ben
 *
 */
public class DistributedDataset implements Serializable {
	private double[] boldResponse;
	private int x;
	private int y;
	private int z;
	private int id;
	private boolean withinROI;

	public DistributedDataset() {
	}

	public DistributedDataset(double[] boldResponse, int x, int y, int z, int id, boolean withinROI) {
		this.boldResponse = boldResponse;
		this.x = x;
		this.y = y;
		this.z = z;
		this.id = id;
		this.withinROI = withinROI;
	}

	public DistributedDataset(double[] newBoldResponse, DistributedDataset oldDistributedDataset) {
		this.boldResponse = newBoldResponse;
		this.x = oldDistributedDataset.x;
		this.y = oldDistributedDataset.y;
		this.z = oldDistributedDataset.z;
		this.id = oldDistributedDataset.id;
		this.withinROI = oldDistributedDataset.withinROI;
	}

	public void setBoldResponse(double[] boldResponse) {
		this.boldResponse = boldResponse;
	}

	public Matrix getBoldResponseMatrix() {
		return new Matrix(boldResponse, boldResponse.length, 1);
	}

	public double[] getBoldResponse() {
		return this.boldResponse;
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

	public int getID() {
		return this.id;
	}

	public boolean getWithinROI() {
		return this.withinROI;
	}
}