package sprt;

import java.io.Serializable;

public class ReduceData implements Serializable {
	private float[] variance;
	private float[] cBeta;
	private float[] ZScore;
	private float[] theta1;
	private float[] SPRT;
	private int[] SPRTActivationStatus;
	private int[][] forecastedActivationStatus;

	public ReduceData(Config config) {
		this.variance = new float[config.contrasts.numContrasts];
		this.cBeta = new float[config.contrasts.numContrasts];
		this.ZScore = new float[config.contrasts.numContrasts];
		this.theta1 = new float[config.contrasts.numContrasts];
		this.SPRT = new float[config.contrasts.numContrasts];
		this.SPRTActivationStatus = new int[config.contrasts.numContrasts];
		// int[config.ROW][config.contrasts.numContrasts];
	}

	public float getVariance(int index) {
		return this.variance[index];
	}

	public void setVariance(int index, float value) {
		this.variance[index] = value;
	}

	public float getCBeta(int index) {
		return this.cBeta[index];
	}

	public void setCBeta(int index, float value) {
		this.cBeta[index] = value;
	}

	public float getZScore(int index) {
		return this.ZScore[index];
	}

	public void setZScore(int index, float value) {
		this.ZScore[index] = value;
	}

	public float getTheta1(int index) {
		return this.theta1[index];
	}

	public void setTheta1(int index, float value) {
		this.theta1[index] = value;
	}

	public float getSPRT(int index) {
		return this.SPRT[index];
	}

	public void setSPRT(int index, float value) {
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
