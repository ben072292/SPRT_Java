package sprt.experiment;

import static org.bytedeco.mkl.global.mkl_rt.*;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.io.Serializable;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.Iterator;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.broadcast.Broadcast;

import sprt.ReduceData;
import sprt.Contrasts;
import sprt.DesignMatrix;
import sprt.BOLD;
import sprt.Matrix;
import sprt.SprtStat;
import sprt.Config;
import static sprt.Algorithm.*;

/**
 * The Driver class using Apache Spark
 * 
 * @author Ben
 *
 */
public class Simulation implements Serializable {

    public static void main(String[] args) {
        MKL_Set_Num_Threads(1);
        int batchSize = Integer.parseInt(args[0]);
        int dataExpand = Integer.parseInt(args[1]);
        int numThreads = Integer.parseInt(args[2]);
        long start, end, scanStart, scanEnd;
        PrintStream out;
        try {
            Date now = new Date();
            SimpleDateFormat formatter = new SimpleDateFormat("yyyy_MM_dd_HH_mm_ss");
            String formattedDate = formatter.format(now);
            out = new PrintStream(new FileOutputStream("simulation_" + dataExpand + "x_" + formattedDate + ".txt"));
            System.setOut(out);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        JavaSparkContext sc = JavaSparkContext.fromSparkContext(SparkContext.getOrCreate(new SparkConf()));

        // load configuration and predefined data
        System.out.println("Generate All BOLD responses and complete setup configurations");
        Config config = new Config();
        config.setX(dataExpand * 36);
        config.setY(128);
        config.setZ(128);
        config.MAX_SCAN = 238;
        config.COL = 8;
        config.enableROI = false;
        DesignMatrix designMatrix = new DesignMatrix("dat/design_easy.txt", config.MAX_SCAN, config.COL);
        Contrasts contrasts = new Contrasts("dat/contrasts.txt");
        config.setContrasts(contrasts);
        ArrayList<Matrix> C = contrasts.toMatrices();
        Broadcast<ArrayList<Matrix>> bcastC = sc.broadcast(C);
        System.out.println("Complete");
        int scanNumber;
        int boldSize = dataExpand * 36 * 128 * 128; // real-subject image size is 36 * 128 * 128
        ArrayList<BOLD> bolds = new ArrayList<>(boldSize);
        for (int i = 0; i < boldSize; i++) {
            bolds.add(new BOLD(i, 238, true));
        }
        // setup broadcast variables
        Broadcast<Config> bcastConfig = sc.broadcast(config);

        Matrix X = designMatrix.toMatrix().toHeap();
        ArrayList<Matrix> XTXInverseList = new ArrayList<>();
        ArrayList<Matrix> XTXInverseXTList = new ArrayList<>();
        ArrayList<Matrix> XXTXInverseList = new ArrayList<>();
        ArrayList<Matrix> HList = new ArrayList<>();
        for (int i = 1; i <= config.MAX_SCAN; i++) {
            if (i <= config.K) {
                XTXInverseList.add(null);
                XTXInverseXTList.add(null);
                XXTXInverseList.add(null);
                HList.add(null);
            } else {
                X = X.setRow(i);
				Matrix XTXInverse = computeXTXInverse(X).toHeap();
				Matrix XTXInverseXT = XTXInverse.mmult(X).toHeap();
				Matrix XXTXInverse = X.mmul(XTXInverse).toHeap();
				Matrix H = computeH(XXTXInverse, X).toHeap();

                XTXInverseList.add(XTXInverse);
                XTXInverseXTList.add(XTXInverseXT);
                XXTXInverseList.add(XXTXInverse);
                HList.add(H);
            }
        }
        Broadcast<Matrix> bcastX = sc.broadcast(X);
        Broadcast<ArrayList<Matrix>> bcastXTXInverseXTList = sc.broadcast(XTXInverseXTList);
        Broadcast<ArrayList<Matrix>> bcastXXTXInverseList = sc.broadcast(XXTXInverseList);
        Broadcast<ArrayList<Matrix>> bcastHList = sc.broadcast(HList);
        Broadcast<Integer> bcastNumThreads = sc.broadcast(numThreads);
        Broadcast<Integer> bcastBatchSize = sc.broadcast(batchSize);

        // Prepare
        System.out.println(
                new Date() + ": Successfully reading in first " + config.K + " scans, Now start SPRT estimation.");

        JavaRDD<BOLD> BOLD_RDD = sc.parallelize(bolds);

        start = System.nanoTime();
        scanNumber = 79;
        while (scanNumber < config.MAX_SCAN) {
            scanStart = System.nanoTime();
            Broadcast<Integer> bcastStartScanNumber = sc.broadcast(scanNumber);
            SprtStat sprtStat = BOLD_RDD
                    .mapPartitions(new FlatMapFunction<Iterator<BOLD>, SprtStat>() {
                        public Iterator<SprtStat> call(Iterator<BOLD> boldIterator)
                                throws Exception {
                            ArrayList<BOLD> bolds = new ArrayList<>();
                            boldIterator.forEachRemaining(bolds::add);

                            ArrayList<SprtStat> SPRTActivation = new ArrayList<>();
                            ForkJoinPool myPool = new ForkJoinPool(bcastNumThreads.value());
                            try {
                                myPool.submit(() -> bolds.parallelStream().forEach(bold -> {
                                    ArrayList<ReduceData> ret = new ArrayList<>();
                                    for (int i = bcastStartScanNumber.value(); i < bcastStartScanNumber
                                            .value() + bcastBatchSize.value()
                                            && i < bcastConfig.value().MAX_SCAN; i++) {
                                        Matrix X = new Matrix(bcastX.value().getArray(), i, bcastX.value().getCol());
                                        // Matrix X = new Matrix(bcastX.value().getPointer(), i, bcastX.value().getCol());
                                        Matrix Y = new Matrix(bold.getPointer(), i, 1);
                                        ReduceData reduceData = new ReduceData(bcastConfig.value());
                                        Matrix beta = computeBetaHat(bcastXTXInverseXTList.value().get(i - 1), Y);
                                        Matrix R = computeR(Y, X, beta);
                                        Matrix D = generateD(R, bcastHList.value().get(i - 1));
                                        for (int j = 0; j < bcastC.value().size(); j++) {
                                            Matrix c = bcastC.value().get(j);
                                            float variance = compute_variance(c,
                                                    bcastXTXInverseXTList.value().get(i - 1),
                                                    bcastXXTXInverseList.value().get(i - 1), D);
                                            float cBeta = compute_cBetaHat(c, beta);
                                            float SPRT = compute_SPRT(cBeta, bcastConfig.value().theta0, 0.0f,
                                                    variance);
                                            int SPRTActivationStatus = compute_activation_stat(SPRT,
                                                    bcastConfig.value().SPRTUpperBound,
                                                    bcastConfig.value().SPRTLowerBound);
                                            reduceData.setVariance(j, variance);
                                            reduceData.setCBeta(j, cBeta);
                                            reduceData.setTheta1(j, 0.0f);
                                            reduceData.setSPRT(j, SPRT);
                                            reduceData.setSPRTActivationStatus(j, SPRTActivationStatus);
                                        }
                                        ret.add(reduceData);
                                    }

                                    int[][][] SPRTActivationCounter = new int[ret.size()][bcastC.value()
                                            .size()][3];

                                    for (int i = 0; i < ret.size(); i++) {
                                        for (int j = 0; j < bcastC.value().size(); j++) {
                                            if (ret.get(i).getSPRTActivationStatus(j) == -1) {
                                                SPRTActivationCounter[i][j][0]++;
                                            } else if (ret.get(i).getSPRTActivationStatus(j) == 0) {
                                                SPRTActivationCounter[i][j][1]++;
                                            } else {
                                                SPRTActivationCounter[i][j][2]++;
                                            }

                                        }
                                    }
                                    SPRTActivation.add(new SprtStat(SPRTActivationCounter));

                                })).get();
                            } catch (InterruptedException | ExecutionException e) {
                                throw new RuntimeException(e);
                            } finally {
                                myPool.shutdown();
                            }

                            return SPRTActivation.iterator();
                        }
                    }).reduce((a, b) -> {
                        return a.merge(b);
                    });

            for (int i = 0; i < sprtStat.getSprtStat().length; i++) {
                System.out.println("Scan " + (i + scanNumber));
                for (int j = 0; j < sprtStat.getSprtStat()[0].length; j++) {
                    System.out.println("Contrast "
                            + (j + 1)
                            + ": Cross Upper: "
                            + sprtStat.getSprtStat()[i][j][2]
                            + ", Cross Lower: "
                            + sprtStat.getSprtStat()[i][j][0]
                            + ", Within Bound: "
                            + sprtStat.getSprtStat()[i][j][1]);
                }
            }

            scanNumber = (scanNumber + batchSize > config.MAX_SCAN ? config.MAX_SCAN : scanNumber + batchSize);
            scanEnd = System.nanoTime();
            System.out.println("Execution Time:" + (scanEnd - scanStart) / 1e9 + " seconds.\n");
        }

        sc.close();
        end = System.nanoTime();
        System.out.println("Total Execution Time: " + (end - start) / 1e9 + " seconds.");

    }
}