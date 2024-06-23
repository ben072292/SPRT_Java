package sprt.experiment;

import static org.bytedeco.mkl.global.mkl_rt.*;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.io.Serializable;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.broadcast.Broadcast;

import sprt.ReduceData;
import sprt.Config;
import sprt.Contrasts;
import sprt.DesignMatrix;
import sprt.BOLD;
import sprt.Matrix;

import static sprt.Algorithm.*;
import sprt.SprtStat;

/**
 * The Driver class using Apache Spark
 * 
 * @author Ben
 *
 */
public class RealExperiment implements Serializable {

	public static void main(String[] args) {
		MKL_Set_Num_Threads(1);
		int batchSize = Integer.parseInt(args[0]);
		long start, end, scanStart, scanEnd;
		PrintStream out;
		try {
			Date now = new Date();
			SimpleDateFormat formatter = new SimpleDateFormat("yyyy_MM_dd_HH_mm_ss");
			String formattedDate = formatter.format(now);
			out = new PrintStream(new FileOutputStream("real_subject_experiment_" + formattedDate + ".txt"));
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
		DesignMatrix designMatrix = new DesignMatrix("dat/design_easy.txt", config.MAX_SCAN, config.COL);
		Contrasts contrasts = new Contrasts("dat/contrasts.txt");
		config.setContrasts(contrasts);
		ArrayList<Matrix> C = contrasts.toMatrices();
		Broadcast<ArrayList<Matrix>> bcastC = sc.broadcast(C);
		System.out.println("Complete");

		System.out.println("Read in first scan to complete configuration");
		int scanNumber = 1;
		String BOLD_Path = config.assemblyBOLDPath(scanNumber);
		ArrayList<BOLD> bolds = null;
		bolds = BOLD.read(BOLD_Path, config, bolds);
		System.out.println("Complete");

		// setup broadcast variables
		Broadcast<Config> bcastConfig = sc.broadcast(config);

		Matrix X = designMatrix.toMatrix().toHeap();
		Broadcast<Matrix> bcastX = sc.broadcast(X);

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
		Broadcast<ArrayList<Matrix>> bcastXTXInverseXTList = sc.broadcast(XTXInverseXTList);
		Broadcast<ArrayList<Matrix>> bcastXXTXInverseList = sc.broadcast(XXTXInverseList);
		Broadcast<ArrayList<Matrix>> bcastHList = sc.broadcast(HList);

		// Continue reading till reaching the K-th scan
		for (scanNumber = 2; scanNumber <= config.K; scanNumber++) {
			System.out.println("Reading Scan " + scanNumber);
			BOLD_Path = config.assemblyBOLDPath(scanNumber);
			bolds = BOLD.read(BOLD_Path, config, bolds);
		}

		// Prepare
		System.out.println(
				new Date() + ": Successfully reading in first " + config.K + " scans, Now start SPRT estimation.");

		JavaRDD<BOLD> BOLD_RDD = sc.parallelize(bolds).cache();

		List<Long> nativeAddr_RDD = BOLD_RDD.map(bold -> {
			return bold.getAddress();
		}).collect();

		for (int i = 0; i < bolds.size(); i++) {
			bolds.get(i).setPointerAddr(nativeAddr_RDD.get(i));
		}

		start = System.nanoTime();

		while (scanNumber < config.MAX_SCAN) {
			scanStart = System.nanoTime();
			int currentScanNumber = scanNumber;

			Broadcast<Integer> bcastStartScanNumber = sc.broadcast(scanNumber);

			for (; currentScanNumber < scanNumber + batchSize
					&& currentScanNumber <= config.MAX_SCAN; currentScanNumber++) {
				BOLD_Path = config.assemblyBOLDPath(currentScanNumber);
				bolds = BOLD.read(BOLD_Path, config, bolds);
			}

			JavaRDD<BOLD> inc_BOLD_RDD = sc.parallelize(bolds);

			JavaRDD<ArrayList<ReduceData>> reduce_RDD = inc_BOLD_RDD
					.map(new Function<BOLD, ArrayList<ReduceData>>() {
						public ArrayList<ReduceData> call(BOLD bold) {
							ArrayList<ReduceData> ret = new ArrayList<>();
							for (int i = bcastStartScanNumber.value(); i < bcastStartScanNumber.value()
									+ bold.getBatchSize(); i++) {
								Matrix X = new Matrix(bcastX.value().getPointer(), i, bcastX.value().getCol());
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
									// float variance = computeVariance(c,
									// bcastXList.getValue().get(i-1), D);
									// float variance = 1.0f;
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
							return ret;
						}
					});

			// 3. Get statistics from collected dataset

			SprtStat result = reduce_RDD
					.map(new Function<ArrayList<ReduceData>, SprtStat>() {
						public SprtStat call(ArrayList<ReduceData> collectedDatasets) {
							int[][][] SPRTActivationCounter = new int[collectedDatasets.size()][bcastC.value()
									.size()][3];
							for (int i = 0; i < collectedDatasets.size(); i++) {
								for (int j = 0; j < bcastC.value().size(); j++) {
									if (collectedDatasets.get(i).getSPRTActivationStatus(j) == -1) {
										SPRTActivationCounter[i][j][0]++;
									} else if (collectedDatasets.get(i).getSPRTActivationStatus(j) == 0) {
										SPRTActivationCounter[i][j][1]++;
									} else {
										SPRTActivationCounter[i][j][2]++;
									}
								}
							}
							return new SprtStat(SPRTActivationCounter);
						}
					}).reduce((a, b) -> {
						return a.merge(b);
					});

			for (int i = 0; i < result.getSprtStat().length; i++) {
				System.out.println("\nScan " + (i + scanNumber));
				for (int j = 0; j < result.getSprtStat()[0].length; j++) {
					System.out.println("Contrast "
							+ (j + 1)
							+ ": Cross Upper: "
							+ result.getSprtStat()[i][j][2]
							+ ", Cross Lower: "
							+ result.getSprtStat()[i][j][0]
							+ ", Within Bound: "
							+ result.getSprtStat()[i][j][1]);
				}
			}

			scanNumber = currentScanNumber;
			scanEnd = System.nanoTime();
			System.out.println("Execution Time:" + (scanEnd - scanStart) / 1e9 + " seconds.\n");
		}
		System.out.println();

		sc.close();
		end = System.nanoTime();
		System.out.println("Total Execution Time: " + (end - start) / 1e9 + " seconds.");

	}
}