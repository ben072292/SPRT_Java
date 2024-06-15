package sprt;

import java.io.Serializable;

public class SprtStat implements Serializable {
    private int[][][] sprtStat;
    private int[][][][] forecastStat;

    public SprtStat(int[][][] SPRTActivationResult) {
        this.sprtStat = SPRTActivationResult;
    }

    public SprtStat(int[][][] SPRTActivationResult, int[][][][] forecastActivationResult) {
        this.sprtStat = SPRTActivationResult;
        this.forecastStat = forecastActivationResult;
    }

    public int[][][] getSprtStat() {
        return this.sprtStat;
    }

    public int[][][][] getForecastStat() {
        return this.forecastStat;
    }

    public SprtStat merge(SprtStat newResult) {
        if (sprtStat != null) {
            for (int i = 0; i < sprtStat.length; i++) {
                for (int j = 0; j < sprtStat[0].length; j++) {
                    for (int k = 0; k < sprtStat[0][0].length; k++) {
                        sprtStat[i][j][k] += newResult.getSprtStat()[i][j][k];
                    }
                }
            }
        }

        if (forecastStat != null) {
            for (int i = 0; i < forecastStat.length; i++) {
                for (int j = 0; j < forecastStat[0].length; j++) {
                    for (int k = 0; k < forecastStat[0][0].length; k++) {
                        for (int l = 0; l < forecastStat[0][0][0].length; l++) {
                            forecastStat[i][j][k][l] += newResult.getForecastStat()[i][j][k][l];
                        }
                    }
                }
            }
        }
        return this;
    }
}
