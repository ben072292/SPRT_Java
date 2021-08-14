package edu.cwru.csds.sprt;

import static edu.cwru.csds.sprt.Numerical.*;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.io.Serializable;
import java.util.Date;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.broadcast.Broadcast;

/**
 * The Driver class using Apache Spark
 * 
 * @author Ben
 *
 */
@SuppressWarnings("serial")
public class SparkDriver implements Serializable {

	public static void main(String[] args) {
		long start, end;
		PrintStream out;
		try {
			out = new PrintStream(new FileOutputStream("output.txt"));
			System.setOut(out);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		// configure spark
		// SparkConf sparkConf = new SparkConf().setAppName("Distributed SPRT").setMaster("local[2]")
		// 		.set("spark.executor.memory", "32g");
		// sc = new JavaSparkContext(sparkConf);
		
		JavaSparkContext sc = JavaSparkContext.fromSparkContext(SparkContext.getOrCreate(new SparkConf()));

		// load configuration and predefined data
		System.out.println("load configuration and predefined data");
		Config config = new Config();
		DesignMatrix designMatrix = new DesignMatrix("Latest_data/design_easy.txt", config.ROW, config.COL);
		Contrasts contrasts = new Contrasts("test/contrasts.txt");
		VolumeReader volumeReader = new VolumeReader();
		Matrix C = contrasts.toMatrix();
		Broadcast<Matrix> broadcastC = sc.broadcast(C);
		Matrix X;
		System.out.println("Complete");

		// Read in first scan to get some brain volume metadata
		System.out.println("Read in first scan to get some brain volume metadata");
		int scanNumber;
		String BOLDPath = config.assemblyBOLDPath(1);
		Brain volume = volumeReader.readFile(BOLDPath, 1);
		config.setVolumeSize(volume);
		Dataset dataset = new Dataset(config.ROW, config.getX(), config.getY(), config.getZ());
		dataset.addOneScan(volume);
		System.out.println("Complete");
		boolean[] ROI = config.getROI();
		dataset.setROI(ROI);

		Broadcast<Config> broadcastConfig = sc.broadcast(config);

		// // Broadcasting global data for bootstrapping
		// X = designMatrix.toMatrix(config.ROW);
		// Matrix XTXInverse = computeXTXInverse(X);
		// Matrix XTXInverseXT = XTXInverse.multiplyTranspose(X);
		// Matrix XXTXInverse = X.multiply(XTXInverse);
		// double[] CTXTXInverseC = new double[C.getRow()];
		// for (int i = 0; i < C.getRow(); i++) {
		// 	Matrix c = C.getRowSlice(i);
		// 	CTXTXInverseC[i] = c.transposeMultiply(XTXInverse).multiply(c).get();
		// }
		// double[] H = computeH(XXTXInverse, X);

		// // Create Spark shared variables
		// Broadcast<Matrix> broadcastXComplete = sc.broadcast(X);
		// // Broadcast<Matrix> broadcastXTXInverseComplete = sc.broadcast(XTXInverse);
		// Broadcast<Matrix> broadcastXTXInverseXTComplete = sc.broadcast(XTXInverseXT);
		// Broadcast<Matrix> broadcastXXTXInverseComplete = sc.broadcast(XXTXInverse);
		// // Broadcast<double[]> broadcastCTXTXInverseCComplete =
		// // sc.broadcast(CTXTXInverseC);
		// Broadcast<double[]> broadcastHComplete = sc.broadcast(H);

		// Continue reading till reaching the K-th scan
		for (scanNumber = 2; scanNumber <= config.K; scanNumber++) {
			System.out.println("Reading Scan " + scanNumber);
			BOLDPath = config.assemblyBOLDPath(scanNumber);
			volume = volumeReader.readFile(BOLDPath, scanNumber);
			dataset.addOneScan(volume);
		}
		// System.out.println(dataset.getVolume(config.K));

		// Prepare
		System.out.println(new Date() + ": Successfully reading in first " + config.K + " scans, Now start SPRT estimation.");

		JavaRDD<DistributedDataset> distributedDataset = sc.parallelize(dataset.toDistrbutedDataset()).cache();

		start = System.nanoTime();
		for (scanNumber = config.K + 1; scanNumber <= config.ROW; scanNumber++) {
			System.out.println("Reading Scan " + scanNumber);
			BOLDPath = config.assemblyBOLDPath(scanNumber);
			volume = volumeReader.readFile(BOLDPath, scanNumber);

			//dataset.addOneScan(volume);
			System.out.println(new Date() + ": Round " + scanNumber + ": Initializing broadcast variables");

			X = designMatrix.toMatrix(scanNumber);

			Matrix XTXInverse = computeXTXInverse(X);
			Matrix XTXInverseXT = XTXInverse.multiplyTranspose(X);
			Matrix XXTXInverse = X.multiply(XTXInverse);
			double[] CTXTXInverseC = new double[C.getRow()];
			for (int i = 0; i < C.getRow(); i++) {
				Matrix c = C.getRowSlice(i);
				CTXTXInverseC[i] = c.transposeMultiply(XTXInverse).multiply(c).get();
			}
			double[] H = computeH(XXTXInverse, X);

			// Create Spark shared variables
			Broadcast<Matrix> broadcastX = sc.broadcast(X);
			// Broadcast<Matrix> broadcastXTXInverse = sc.broadcast(XTXInverse);
			Broadcast<Matrix> broadcastXTXInverseXT = sc.broadcast(XTXInverseXT);
			Broadcast<Matrix> broadcastXXTXInverse = sc.broadcast(XXTXInverse);
			// Broadcast<double[]> broadcastCTXTXInverseC = sc.broadcast(CTXTXInverseC);
			Broadcast<double[]> broadcastH = sc.broadcast(H);
			Broadcast<Brain> broadcastVolume = sc.broadcast(volume);

			// Spark logic
			// 1. add new scan to distributedDataset

			// Not sure broadcast new brain volume would be a good way.
			// Might cause performance issue.
			distributedDataset = distributedDataset.map(new Function<DistributedDataset, DistributedDataset>() {

				public DistributedDataset call(DistributedDataset distributedDataset) {
					return new DistributedDataset(
							ArrayUtils
									.add(distributedDataset.getBoldResponse(),
											broadcastVolume.value().getVoxel(distributedDataset.getX(),
													distributedDataset.getY(), distributedDataset.getZ())),
							distributedDataset);
				}

			});

			// 2. Perform computation
			System.out.println(new Date() + ": Round " + scanNumber + ": Starting computation in workers");
			JavaRDD<CollectedDataset> collectedDataset = distributedDataset
					.map(new Function<DistributedDataset, CollectedDataset>() {
						public CollectedDataset call(DistributedDataset distributedDataset) {
							CollectedDataset ret = new CollectedDataset(broadcastConfig.value());

							Matrix beta = computeBeta2(broadcastXTXInverseXT.value(),
									distributedDataset.getBoldResponseMatrix());
							double[] R = computeR(distributedDataset.getBoldResponseMatrix(), broadcastX.value(), beta);
							Matrix D = generateD(R, broadcastH.value());

							for (int i = 0; i < broadcastC.value().getRow(); i++) {
								Matrix c = broadcastC.value().getRowSlice(i);
								double variance = Numerical.computeVarianceUsingMKLSparseRoutine3(c,
								broadcastXTXInverseXT.value(), broadcastXXTXInverse.value(), D);
								// double variance = Numerical.computeVarianceUsingMKLSparseRoutine1(c,
								// broadcastX.value(), D);
								// double variance = computeVariance(c, broadcastX.value(), D);
								double cBeta = computeCBeta(c, beta);
								double ZScore = computeZ(cBeta, variance);
								double theta1 = broadcastConfig.value().ZScore * Math.sqrt(variance);
								double SPRT = compute_SPRT(cBeta, config.theta0, theta1, variance);
								int SPRTActivationStatus = computeActivationStatus(SPRT, config.SPRTUpperBound,
										config.SPRTLowerBound);

								ret.setVariance(i, variance);
								ret.setCBeta(i, cBeta);
								ret.setZScore(i, ZScore);
								ret.setTheta1(i, theta1);
								ret.setSPRT(i, SPRT);
								ret.setSPRTActivationStatus(i, SPRTActivationStatus);
							}
							return ret;
						}
					});

			// System.out.println(new Date() + ": Round " + scanNumber + ": Count elemnets
			// for collectedDataset RDD: " + collectedDataset.count());
			// 3. Get statistics from collected dataset
			System.out.println(new Date() + ": Round " + scanNumber + ": Transform to activation map Rdd");
			int activationCounter = collectedDataset.map(new Function<CollectedDataset, Integer>() {
				public Integer call(CollectedDataset collectedDataset) {
					int sum = 0;
					for (int i = 0; i < broadcastC.value().getRow(); i++) {
						if (collectedDataset.getSPRTActivationStatus(i) == 1)
							sum++;
					}
					return Integer.valueOf(sum);
				}
			}).reduce((a, b) -> a + b);
			// System.out.println(new Date() + ": Round " + scanNumber + ": Count elemnets
			// for activationMap RDD: " + activationMap.count());

			System.out.println(new Date() + ": " + activationCounter);

			// 3.1 Testing rdd.take()
//			System.out.println(new Date() + ": Round " + scanNumber + ": Retriving data from activationMap RDD using take()");
//			collectedDataset.take(1);
//			System.out.println(new Date() + ": Round " + scanNumber + ": Retrived");
		}
		sc.close();
		end = System.nanoTime();
		System.out.println("Total Time Consumption: " + (end-start)/1e9 + " seconds.");

	}
}