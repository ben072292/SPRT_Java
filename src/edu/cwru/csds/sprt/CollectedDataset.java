package edu.cwru.csds.sprt;
import java.io.Serializable;

@SuppressWarnings("serial")
public class CollectedDataset implements Serializable{
	private int numOfC;
	private double[] variance;
	private double[] cBeta;
	private double[] ZScore;
	private double[] theta1;
	private double[] SPRT;
	private int[] SPRTActivationStatus;
	
	public CollectedDataset(int numOfC) {
		this.numOfC = numOfC;
		this.variance = new double[numOfC];
		this.cBeta = new double[numOfC];
		this.ZScore = new double[numOfC];
		this.theta1 = new double[numOfC];
		this.SPRT = new double[numOfC];
		this.SPRTActivationStatus = new int[numOfC];
	}
	
	public int getNumOfC() {
		return this.numOfC;
	}
	
	public double getVariance(int index) {
		return this.variance[index];
	}
	
	public void setVariance(int index, double value) {
		this.variance[index] = value;
	}
	
	public double getCBeta(int index) {
		return this.cBeta[index];
	}
	
	public void setCBeta(int index, double value) {
		this.cBeta[index] = value;
	}
	
	public double getZScore(int index) {
		return this.ZScore[index];
	}
	
	public void setZScore(int index, double value) {
		this.ZScore[index] = value;
	}
	
	public double getTheta1(int index) {
		return this.theta1[index];
	}
	
	public void setTheta1(int index, double value) {
		this.theta1[index] = value;
	}
	
	public double getSPRTe(int index) {
		return this.SPRT[index];
	}
	
	public void setSPRT(int index, double value) {
		this.SPRT[index] = value;
	}
	
	public int getSPRTActivationStatus(int index) {
		return this.SPRTActivationStatus[index];
	}
	
	public void setSPRTActivationStatus(int index, int value) {
		this.SPRTActivationStatus[index] = value;
	}
	
}
