package sprt;

import java.io.Serializable;

public class ReduceData implements Serializable {
	private double[] variance;
	private double[] cBeta;
	private double[] ZScore;
	private double[] theta1;
	private double[] SPRT;
	private int[] SPRTActivationStatus;
	private int[][] forecastedActivationStatus;

	public ReduceData(Config config) {
		this.variance = new double[config.contrasts.numContrasts];
		this.cBeta = new double[config.contrasts.numContrasts];
		this.ZScore = new double[config.contrasts.numContrasts];
		this.theta1 = new double[config.contrasts.numContrasts];
		this.SPRT = new double[config.contrasts.numContrasts];
		this.SPRTActivationStatus = new int[config.contrasts.numContrasts];
		// int[config.ROW][config.contrasts.numContrasts];
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

	public double getSPRT(int index) {
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

	public int getForecastedActivationStatus(int pos1, int pos2) {
		return this.forecastedActivationStatus[pos1][pos2];
	}

	public void setForecastedActivationStatus(int pos1, int pos2, int val) {
		this.forecastedActivationStatus[pos1][pos2] = val;
	}
}
