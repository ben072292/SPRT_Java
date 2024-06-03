package edu.cwru.csds.sprt.data;

import java.io.Serializable;

public class ActivationResult implements Serializable {
    private int[][][] SPRTActivationResult;
    private int[][][][] forecastActivationResult;

    public ActivationResult(int[][][] SPRTActivationResult) {
        this.SPRTActivationResult = SPRTActivationResult;
    }

    public ActivationResult(int[][][] SPRTActivationResult, int[][][][] forecastActivationResult) {
        this.SPRTActivationResult = SPRTActivationResult;
        this.forecastActivationResult = forecastActivationResult;
    }

    public int[][][] getSPRTActivationResult() {
        return this.SPRTActivationResult;
    }

    public int[][][][] getForecastActivationResult() {
        return this.forecastActivationResult;
    }

    public ActivationResult merge(ActivationResult newResult) {
        if (SPRTActivationResult != null) {
            for (int i = 0; i < SPRTActivationResult.length; i++) {
                for (int j = 0; j < SPRTActivationResult[0].length; j++) {
                    for (int k = 0; k < SPRTActivationResult[0][0].length; k++) {
                        SPRTActivationResult[i][j][k] += newResult.getSPRTActivationResult()[i][j][k];
                    }
                }
            }
        }

        if (forecastActivationResult != null) {
            for (int i = 0; i < forecastActivationResult.length; i++) {
                for (int j = 0; j < forecastActivationResult[0].length; j++) {
                    for (int k = 0; k < forecastActivationResult[0][0].length; k++) {
                        for (int l = 0; l < forecastActivationResult[0][0][0].length; l++) {
                            forecastActivationResult[i][j][k][l] += newResult.getForecastActivationResult()[i][j][k][l];
                        }
                    }
                }
            }
        }

        return this;
    }
}
