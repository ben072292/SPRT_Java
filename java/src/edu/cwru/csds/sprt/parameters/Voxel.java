package edu.cwru.csds.sprt.parameters;

public class Voxel {

	private double signal;
	private int scanNumber = 0;
	private int x = Integer.MAX_VALUE;
	private int y = Integer.MAX_VALUE;
	private int z = Integer.MAX_VALUE;

	public Voxel() {
		this.signal = 0.0;
	}

	public Voxel(double signal) {
		this.signal = signal;
	}

	public Voxel(int scanNum, int x, int y, int z, double signal) {
		this.scanNumber = scanNum;
		this.x = x;
		this.y = y;
		this.z = z;
		this.signal = signal;
	}

	// Getters
	public int[] getPosition() {
		int[] position = new int[3];
		position[0] = this.x;
		position[1] = this.y;
		position[2] = this.z;
		return position;
	}

	public double getSignal() {
		return this.signal;
	}

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

	// Setters
	public void setSignal(double new_signal) {
		this.signal = new_signal;
	}

	public void setPosition(int[] pos) {
		this.x = pos[0];
		this.y = pos[1];
		this.z = pos[2];
	}

	public void setPositionX(int posX) {
		this.x = posX;
	}

	public void setPositionY(int posY) {
		this.y = posY;
	}

	public void setPositionZ(int posZ) {
		this.z = posZ;
	}

}
