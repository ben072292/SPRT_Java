package sprt.experiment;

import static org.bytedeco.mkl.global.mkl_rt.*;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Date;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.broadcast.Broadcast;

import sprt.CollectedDataset;
import sprt.Config;
import sprt.Contrasts;
import sprt.Dataset;
import sprt.DesignMatrix;
import sprt.Image;
import sprt.Matrix;
import sprt.Matrix.MatrixStorageScope;

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
			out = new PrintStream(new FileOutputStream("real_subject_experiment.txt"));
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
		Matrix C = contrasts.toMatrix(MatrixStorageScope.HEAP);
		Broadcast<Matrix> broadcastC = sc.broadcast(C);
		Matrix X;
		System.out.println("Complete");

		// Read in first scan to get some brain volume metadata
		System.out.println("Read in first scan to get some brain volume metadata");
		int scanNumber = 1;
		String BOLD_Path = config.assemblyBOLDPath(scanNumber);
		Image image = new Image();
		image.readImage(BOLD_Path, scanNumber);
		config.setImageSpec(image);
		Dataset BOLD_Dataset = new Dataset(config.getX(), config.getY(), config.getZ());
		BOLD_Dataset.add(image);
		System.out.println("Complete");

		// setup broadcast variables
		Broadcast<Config> broadcastConfig = sc.broadcast(config);

		ArrayList<Matrix> XList = new ArrayList<>();
		ArrayList<Matrix> XTXInverseList = new ArrayList<>();
		ArrayList<Matrix> XTXInverseXTList = new ArrayList<>();
		ArrayList<Matrix> XXTXInverseList = new ArrayList<>();
		ArrayList<double[]> HList = new ArrayList<>();
		for (int i = 1; i <= config.MAX_SCAN; i++) {
			if (i <= config.K) {
				XList.add(null);
				XTXInverseList.add(null);
				XTXInverseXTList.add(null);
				XXTXInverseList.add(null);
				HList.add(null);
			} else {
				X = designMatrix.toMatrix(i, MatrixStorageScope.HEAP);
				Matrix XTXInverse = computeXTXInverse(X);
				Matrix XTXInverseXT = XTXInverse.multiplyTranspose(X);
				Matrix XXTXInverse = X.multiply(XTXInverse);
				double[] H = computeH(XXTXInverse, X);
				XList.add(X);
				XTXInverseList.add(XTXInverse);

				XTXInverseXTList.add(XTXInverseXT);
				XXTXInverseList.add(XXTXInverse);
				HList.add(H);
			}
		}
		Broadcast<ArrayList<Matrix>> bcastXList = sc.broadcast(XList);
		Broadcast<ArrayList<Matrix>> bcastXTXInverseXTList = sc.broadcast(XTXInverseXTList);
		Broadcast<ArrayList<Matrix>> bcastXXTXInverseList = sc.broadcast(XXTXInverseList);
		Broadcast<ArrayList<double[]>> bcastHList = sc.broadcast(HList);

		// Continue reading till reaching the K-th scan
		for (scanNumber = 2; scanNumber <= config.K; scanNumber++) {
			System.out.println("Reading Scan " + scanNumber);
			BOLD_Path = config.assemblyBOLDPath(scanNumber);
			image.readImage(BOLD_Path, scanNumber);
			BOLD_Dataset.add(image);
		}
		// System.out.println(dataset.getVolume(config.K));

		// Prepare
		System.out.println(
				new Date() + ": Successfully reading in first " + config.K + " scans, Now start SPRT estimation.");

		JavaRDD<double[]> BOLD_RDD = sc.parallelize(BOLD_Dataset.toDD(config.enableROI, config.getROI()));

		start = System.nanoTime();

		while (scanNumber < config.MAX_SCAN) {
			scanStart = System.nanoTime();
			int currentScanNumber = scanNumber;

			Broadcast<Integer> broadcastStartScanNumber = sc.broadcast(scanNumber);

			ArrayList<Image> incImages = new ArrayList<>();
			for (; currentScanNumber < scanNumber + batchSize
					&& currentScanNumber <= config.MAX_SCAN; currentScanNumber++) {
				BOLD_Path = config.assemblyBOLDPath(currentScanNumber);
				image.readImage(BOLD_Path, currentScanNumber);
				incImages.add(image);
			}

			Dataset incDataset = new Dataset(incImages);

			JavaRDD<double[]> inc_RDD = sc.parallelize(incDataset.toDD(config.enableROI, config.getROI()));

			// Zip the RDDs
			JavaPairRDD<double[], double[]> zippedRDD = BOLD_RDD.zip(inc_RDD);

			// Merge each pair of ArrayLists
			BOLD_RDD = zippedRDD.map(tuple -> {
				double[] array1 = tuple._1;
				double[] array2 = tuple._2;
				double[] mergedArray = new double[array1.length + array2.length];
				System.arraycopy(array1, 0, mergedArray, 0, array1.length);
				System.arraycopy(array2, 0, mergedArray, array1.length, array2.length);
				return mergedArray;
			});


			Broadcast<Integer> bcastBatchSize = sc.broadcast(incImages.size());

			JavaRDD<ArrayList<CollectedDataset>> collectedDatasets = BOLD_RDD
					.map(new Function<double[], ArrayList<CollectedDataset>>() {
						public ArrayList<CollectedDataset> call(double[] dd) {
							ArrayList<CollectedDataset> ret = new ArrayList<>();
							ret.clear();
							for (int i = broadcastStartScanNumber.value(); i < broadcastStartScanNumber.value()
									+ bcastBatchSize.value(); i++) {
								Matrix boldResponse = new Matrix(dd, i, 1, MatrixStorageScope.NATIVE);
								CollectedDataset temp = new CollectedDataset(broadcastConfig.value());
								Matrix beta = computeBetaHat(bcastXTXInverseXTList.value().get(i - 1),
										boldResponse);
								double[] R = computeR(boldResponse, bcastXList.value().get(i - 1), beta);
								Matrix D = generateD(R, bcastHList.value().get(i - 1), MatrixStorageScope.NATIVE);
								// double[] D = generateD_array(R, broadcastHList.value().get(i - 1));
								for (int j = 0; j < broadcastC.value().getRow(); j++) {

									Matrix c = broadcastC.value().getRowSlice(j);
									double variance = compute_variance_sparse_fast(c,
											bcastXTXInverseXTList.value().get(i - 1),
											bcastXXTXInverseList.value().get(i - 1), D);
									// double variance = computeVariance(c,
									// broadcastXList.getValue().get(i-1), D);
									// double variance = 1.0;
									double cBeta = compute_cBetaHat(c, beta);
									double SPRT = compute_SPRT(cBeta, broadcastConfig.value().theta0, 0.0,
											variance);
									int SPRTActivationStatus = compute_activation_stat(SPRT,
											broadcastConfig.value().SPRTUpperBound,
											broadcastConfig.value().SPRTLowerBound);
									temp.setVariance(j, variance);
									temp.setCBeta(j, cBeta);
									temp.setTheta1(j, 0.0);
									temp.setSPRT(j, SPRT);
									temp.setSPRTActivationStatus(j, SPRTActivationStatus);
								}
								ret.add(temp);
							}
							return ret;
						}
					});

			// 3. Get statistics from collected dataset

			SprtStat result = collectedDatasets
					.map(new Function<ArrayList<CollectedDataset>, SprtStat>() {
						public SprtStat call(ArrayList<CollectedDataset> collectedDatasets) {
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
							return new SprtStat(SPRTActivationCounter);
						}
					}).reduce((a, b) -> {
						return a.merge(b);
					});

			for (int i = 0; i < result.getSprtStat().length; i++) {
				System.out.println("Scan " + (i + scanNumber));
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
			System.out.print((scanEnd - scanStart) / 1e9 + ", ");
		}
		System.out.println();

		sc.close();
		end = System.nanoTime();
		System.out.println("Total Time Consumption: " + (end - start) / 1e9 + " seconds.");

	}
}