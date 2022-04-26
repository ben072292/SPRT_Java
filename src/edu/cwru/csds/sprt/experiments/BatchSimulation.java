package edu.cwru.csds.sprt.experiments;

import static edu.cwru.csds.sprt.numerical.Numerical.*;
import static org.bytedeco.mkl.global.mkl_rt.*;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Date;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.broadcast.Broadcast;

import edu.cwru.csds.sprt.data.ActivationResult;
import edu.cwru.csds.sprt.data.Brain;
import edu.cwru.csds.sprt.data.CollectedDataset;
import edu.cwru.csds.sprt.data.Dataset;
import edu.cwru.csds.sprt.data.DistributedDataset;
import edu.cwru.csds.sprt.numerical.Matrix;
import edu.cwru.csds.sprt.numerical.Numerical;
import edu.cwru.csds.sprt.parameters.Contrasts;
import edu.cwru.csds.sprt.parameters.DesignMatrix;
import edu.cwru.csds.sprt.utilities.Config;
import edu.cwru.csds.sprt.utilities.VolumeReader;

/**
 * The Driver class using Apache Spark
 * 
 * @author Ben
 *
 */
public class BatchSimulation implements Serializable {

    public static void main(String[] args) {
        int batchSize = Integer.parseInt(args[0]);
        int dataExpand = Integer.parseInt(args[1]);
        long start, end;
        PrintStream out;
        try {
            out = new PrintStream(new FileOutputStream("output.txt"));
            System.setOut(out);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        // configure spark
        // SparkConf sparkConf = new
        // SparkConf().setAppName("DistributedSPRT").setMaster("local[8]")
        // .set("spark.driver.memory", "16g");
        // JavaSparkContext sc = new JavaSparkContext(sparkConf);

        JavaSparkContext sc = JavaSparkContext.fromSparkContext(SparkContext.getOrCreate(new SparkConf()));

        // load configuration and predefined data
        System.out.println("load configuration and predefined data");
        Config config = new Config();
        DesignMatrix designMatrix = new DesignMatrix("Latest_data/design_easy.txt", config.ROW, config.COL);
        Contrasts contrasts = new Contrasts("test/contrasts.txt");
        config.setContrasts(contrasts);
        VolumeReader volumeReader = new VolumeReader();
        Matrix C = contrasts.toMatrix();
        Broadcast<Matrix> broadcastC = sc.broadcast(C);
        Matrix X;
        System.out.println("Complete");

        // Generate simu
        System.out.println("Read in first scan to get some brain volume metadata");
        int scanNumber;
        String BOLDPath = config.assemblyBOLDPath(1);
        Brain volume = new Brain(1, 36 * dataExpand, 128, 128, true);
        config.setVolumeSize(volume);
        Dataset dataset = new Dataset(config.ROW, config.getX(), config.getY(), config.getZ());
        dataset.addOneSimulationData();
        System.out.println("Complete");
        boolean[] ROI = config.getROI();
        dataset.setROI(ROI);

        // setup broadcast variables
        Broadcast<Config> broadcastConfig = sc.broadcast(config);

        ArrayList<Matrix> XList = new ArrayList<>();
        ArrayList<Matrix> XTXInverseList = new ArrayList<>();
        ArrayList<Matrix> XTXInverseXTList = new ArrayList<>();
        ArrayList<Matrix> XXTXInverseList = new ArrayList<>();
        ArrayList<double[]> HList = new ArrayList<>();
        for (int i = 1; i <= config.ROW; i++) {
            if (i <= config.K) {
                XList.add(null);
                XTXInverseList.add(null);
                XTXInverseXTList.add(null);
                XXTXInverseList.add(null);
                HList.add(null);
            } else {
                X = designMatrix.toMatrix(i);
                Matrix XTXInverse = computeXTXInverse(X);
                Matrix XTXInverseXT = XTXInverse.multiplyTranspose(X);
                Matrix XXTXInverse = X.multiply(XTXInverse);
                // double[] CTXTXInverseC = new double[C.getRow()];
                // for (int j = 0; j < C.getRow(); j++) {
                // Matrix c = C.getRowSlice(j);
                // CTXTXInverseC[j] = c.transposeMultiply(XTXInverse).multiply(c).get();
                // }
                double[] H = computeH(XXTXInverse, X);
                XList.add(X);
                XTXInverseList.add(XTXInverse);

                XTXInverseXTList.add(XTXInverseXT);
                XXTXInverseList.add(XXTXInverse);
                HList.add(H);
            }
        }
        Broadcast<ArrayList<Matrix>> broadcastXList = sc.broadcast(XList);
        // Broadcast<ArrayList<Matrix>> broadcastXTXInverseList =
        // sc.broadcast(XTXInverseList);
        Broadcast<ArrayList<Matrix>> broadcastXTXInverseXTList = sc.broadcast(XTXInverseXTList);
        Broadcast<ArrayList<Matrix>> broadcastXXTXInverseList = sc.broadcast(XXTXInverseList);
        Broadcast<ArrayList<double[]>> broadcastHList = sc.broadcast(HList);

        double[] theta1 = new double[config.getX() * config.getY() * config.getZ()];
        // Continue reading till reaching the K-th scan
        for (scanNumber = 2; scanNumber <= config.K; scanNumber++) {
            dataset.addOneSimulationData();

            // formula update 09/01/2021: theta1 is only estimated once for all at scan K
            if (scanNumber == config.K) {
                Brain[] theta1Volume = Numerical.estimateTheta1(dataset, XList.get(config.K), C, config.ZScore,
                        ROI);
                for (int i = 0; i < theta1Volume.length; i++) {
                    theta1[i] = theta1Volume[i].getVoxel(i);
                }
            }
        }
        Broadcast<double[]> broadcastTheta1 = sc.broadcast(theta1);
        // System.out.println(dataset.getVolume(config.K));

        // Prepare
        System.out.println(
                new Date() + ": Successfully reading in first " + config.K + " scans, Now start SPRT estimation.");

        JavaRDD<DistributedDataset> distributedDataset = sc.parallelize(dataset.toDistrbutedDataset());

        start = System.nanoTime();

        while (scanNumber <= config.ROW) {
            int currentScanNumber = scanNumber;

            Broadcast<Integer> broadcastStartScanNumber = sc.broadcast(scanNumber);

            ArrayList<Brain> volumes = new ArrayList<>();
            for (; currentScanNumber < scanNumber + batchSize && currentScanNumber <= config.ROW; currentScanNumber++) {
                BOLDPath = config.assemblyBOLDPath(currentScanNumber);
                volumes.add(new Brain(currentScanNumber, config.getX(), config.getY(), config.getZ(), true));
            }

            Broadcast<ArrayList<Brain>> broadcastVolumes = sc.broadcast(volumes);

            distributedDataset = distributedDataset.map(new Function<DistributedDataset, DistributedDataset>() {
                public DistributedDataset call(DistributedDataset distributedDataset) {
                    for (int i = 0; i < broadcastVolumes.value().size(); i++) {
                        distributedDataset = new DistributedDataset(
                                ArrayUtils.add(distributedDataset.getBoldResponse(),
                                        broadcastVolumes.value().get(i).getVoxel(distributedDataset.getX(),
                                                distributedDataset.getY(), distributedDataset.getZ())),
                                distributedDataset);
                    }
                    return distributedDataset;
                }
            });

            Broadcast<Integer> broadcastRealBatchSize = sc.broadcast(volumes.size());

            JavaRDD<ArrayList<CollectedDataset>> collectedDatasets = distributedDataset
                    .map(new Function<DistributedDataset, ArrayList<CollectedDataset>>() {
                        public ArrayList<CollectedDataset> call(DistributedDataset distributedDataset) {
                            MKL_Set_Num_Threads(1);
                            ArrayList<CollectedDataset> ret = new ArrayList<>();
                            for (int i = broadcastStartScanNumber.value(); i < broadcastStartScanNumber.value()
                                    + broadcastRealBatchSize.value(); i++) {
                                double[] boldResponseRaw = new double[i];
                                System.arraycopy(distributedDataset.getBoldResponse(), 0, boldResponseRaw, 0, i);
                                Matrix boldResponse = new Matrix(boldResponseRaw, boldResponseRaw.length, 1);
                                CollectedDataset temp = new CollectedDataset(broadcastConfig.value());
                                Matrix beta = computeBeta2(broadcastXTXInverseXTList.value().get(i - 1),
                                        boldResponse);
                                double[] R = computeR(boldResponse, broadcastXList.value().get(i - 1), beta);
                                Matrix D = generateD(R, broadcastHList.value().get(i - 1));
                                for (int j = 0; j < broadcastC.value().getRow(); j++) {

                                    Matrix c = broadcastC.value().getRowSlice(j);
                                    double variance = Numerical.computeVarianceUsingMKLSparseRoutine3(c,
                                            broadcastXTXInverseXTList.value().get(i - 1),
                                            broadcastXXTXInverseList.value().get(i - 1), D);
                                    // double SandwichVariance = Numerical.computeVarianceUsingMKLSparseRoutine1(c,
                                    // X, D);
                                    // double variance = computeVariance(c, broadcastXList.value().get(i-1), D);
                                    // double variance = computeVarianceSandwich(c,
                                    // broadcastXTXInverseList.value().get(i-1), broadcastXList.value().get(i-1),
                                    // D);
                                    // double sigmaHatSquare = estimateSigmaHatSquare(boldResponseRaw,
                                    // broadcastXList.value().get(i-1), beta, i, broadcastConfig.value().COL);
                                    // double variance = computeVarianceUsingSigmaHatSquare(sigmaHatSquare, c,
                                    // broadcastXTXInverseList.value().get(i-1));
                                    double cBeta = computeCBeta(c, beta);
                                    // double ZScore = computeZ(cBeta, variance);
                                    // double theta1 = broadcastConfig.value().ZScore * Math.sqrt(variance);
                                    double theta1 = broadcastTheta1.value()[distributedDataset.getID()];
                                    double SPRT = compute_SPRT(cBeta, broadcastConfig.value().theta0, theta1,
                                            variance);
                                    int SPRTActivationStatus = computeActivationStatus(SPRT,
                                            broadcastConfig.value().SPRTUpperBound,
                                            broadcastConfig.value().SPRTLowerBound);
                                    temp.setVariance(j, variance);
                                    temp.setCBeta(j, cBeta);
                                    // temp.setZScore(j, ZScore);
                                    temp.setTheta1(j, theta1);
                                    temp.setSPRT(j, SPRT);
                                    temp.setSPRTActivationStatus(j, SPRTActivationStatus);

                                    // forecasting using confidence interval

                                    // for(int k = i; k < broadcastConfig.value().ROW; k++){
                                    // double forecastedVariance =
                                    // computeVarianceUsingSigmaHatSquare(sigmaHatSquare, c,
                                    // broadcastXTXInverseList.value().get(k));
                                    // // double forecastedZScore = computeZ(cBeta, forecastedVariance);
                                    // //double forecastedTheta1 = broadcastConfig.value().ZScore *
                                    // Math.sqrt(forecastedVariance);
                                    // int forecastedSPRTActivationStatus = evaluateConfidenceInterval(cBeta,
                                    // forecastedVariance, 1.96, theta1);
                                    // temp.setForecastedActivationStatus(k, j, forecastedSPRTActivationStatus);
                                    // }

                                    // forecasting using SPRT

                                    // for (int k = i; k < broadcastConfig.value().ROW; k++) {
                                    // double forecastedVariance =
                                    // Numerical.computeVarianceUsingMKLSparseRoutine3(c,
                                    // broadcastXTXInverseXTList.value().get(k),
                                    // broadcastXXTXInverseList.value().get(k), D);
                                    // // double forecastedVariance =
                                    // // computeVarianceUsingSigmaHatSquare(sigmaHatSquare, c,
                                    // // broadcastXTXInverseList.value().get(k));
                                    // // double forecastedZScore = computeZ(cBeta, forecastedVariance);
                                    // // double forecastedTheta1 = broadcastConfig.value().ZScore *
                                    // // Math.sqrt(forecastedVariance);
                                    // double forecastedSPRT = compute_SPRT(cBeta, broadcastConfig.value().theta0,
                                    // theta1, forecastedVariance);
                                    // int forecastedSPRTActivationStatus = computeActivationStatus(forecastedSPRT,
                                    // broadcastConfig.value().SPRTUpperBound,
                                    // broadcastConfig.value().SPRTLowerBound);
                                    // temp.setForecastedActivationStatus(k, j, forecastedSPRTActivationStatus);
                                    // }

                                }
                                ret.add(temp);
                            }
                            return ret;
                        }
                    });

            /**
             * Mean and Variance of variance estimation
             */

            // List<ArrayList<CollectedDataset>> all = collectedDatasets.collect();
            // double[][] variances = new double[all.get(0).size()][all.size()];
            // for(int i = 0; i < all.size(); i++){
            // for(int j = 0; j < all.get(0).size(); j++){
            // variances[j][i] = all.get(i).get(j).getVariance(1);
            // }
            // }

            // ArrayList<double[]> res = new ArrayList<>();
            // for(double[] arr : variances){
            // res.add(computeMeanAndVariance(arr));
            // }

            // System.out.println("Mean:");
            // for(double[] re : res){
            // System.out.println(re[0]);
            // }

            // System.out.println("\n\nVariance");
            // for(double[] re : res){
            // System.out.println(re[1]);
            // }

            // ArrayList<Double> ret = new ArrayList<>();
            // for(double[] arr : variances){
            // double total = 0.0;
            // for(double d : arr){
            // total += d;
            // }
            // ret.add(total);
            // }

            // for(double d : ret){
            // System.out.println(d);
            // }

            // 3. Get statistics from collected dataset

            ActivationResult result = collectedDatasets
                    .map(new Function<ArrayList<CollectedDataset>, ActivationResult>() {
                        public ActivationResult call(ArrayList<CollectedDataset> collectedDatasets) {
                            int[][][] SPRTActivationCounter = new int[collectedDatasets.size()][broadcastC.value()
                                    .getRow()][3];
                            for (int i = 0; i < collectedDatasets.size(); i++) {
                                for (int j = 0; j < broadcastC.value().getRow(); j++) {
                                    if (collectedDatasets.get(i).getSPRTActivationStatus(j) == -1) {
                                        SPRTActivationCounter[i][j][0]++;
                                    } else if (collectedDatasets.get(i).getSPRTActivationStatus(j) == 0) {
                                        SPRTActivationCounter[i][j][1]++;
                                    } else {
                                        SPRTActivationCounter[i][j][2]++;
                                    }

                                }
                            }
                            // int[][][][] forecastedActivationCounter = new int[collectedDatasets
                            // .size()][broadcastConfig.value().ROW][broadcastC.value().getRow()][3];

                            // for (int i = 0; i < collectedDatasets.size(); i++) {
                            // for (int j = broadcastStartScanNumber.value() + i; j < broadcastConfig
                            // .value().ROW; j++) {
                            // for (int k = 0; k < broadcastC.value().getRow(); k++) {
                            // if (collectedDatasets.get(i).getForecastedActivationStatus(j, k) == -1) {
                            // forecastedActivationCounter[i][j][k][0]++;
                            // } else if (collectedDatasets.get(i).getForecastedActivationStatus(j, k) == 0)
                            // {
                            // forecastedActivationCounter[i][j][k][1]++;
                            // } else {
                            // forecastedActivationCounter[i][j][k][2]++;
                            // }

                            // }
                            // }
                            // }
                            return new ActivationResult(SPRTActivationCounter);
                        }
                    }).reduce((a, b) -> {
                        return a.merge(b);
                    });

            for (int i = 0; i < result.getSPRTActivationResult().length; i++) {
                System.out.println("Scan " + (i + scanNumber));
                for (int j = 0; j < result.getSPRTActivationResult()[0].length; j++) {
                    System.out.println("Contrast "
                            + (j + 1)
                            + ": Cross Upper: "
                            + result.getSPRTActivationResult()[i][j][2]
                            + ", Cross Lower: "
                            + result.getSPRTActivationResult()[i][j][0]
                            + ", Within Bound: "
                            + result.getSPRTActivationResult()[i][j][1]);
                }
            }

            // for (int i = 0; i < result.getForecastActivationResult().length; i++) {
            // System.out.println("Forecasting at Scan " + (i + scanNumber));
            // for (int j = config.ROW - 1; j < config.ROW; j++) {
            // System.out.println("Forecasted Scan " + (j + 1));
            // for (int k = 0; k < result.getForecastActivationResult()[0][0].length; k++) {
            // System.out.println("Contrast "
            // + (k + 1)
            // + ": Cross Upper: "
            // + result.getForecastActivationResult()[i][j][k][2]
            // + ", Cross Lower: "
            // + result.getForecastActivationResult()[i][j][k][0]
            // + ", Within Bound: "
            // + result.getForecastActivationResult()[i][j][k][1]);
            // }
            // }
            // }

            scanNumber = currentScanNumber;

        }

        sc.close();
        end = System.nanoTime();
        System.out.println("Total Time Consumption: " + (end - start) / 1e9 + " seconds.");

    }
}