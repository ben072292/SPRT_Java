import java.io.Serializable;
import java.util.Date;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.broadcast.Broadcast;

import scala.Tuple2;

/**
 * The Driver class using Apache Spark
 * 
 * @author Ben
 *
 */
@SuppressWarnings("serial")
public class SparkDriver implements Serializable {
	private static JavaSparkContext sc;

	public static void main(String[] args) {
		// configure spark
		SparkConf sparkConf = new SparkConf().setAppName("Distributed SPRT").setMaster("local[4]")
				.set("spark.executor.memory", "2g");
		sc = new JavaSparkContext(sparkConf);

		// load configuration and predefined data
		System.out.println("---------------------------------------------------------------------------");
		System.out.println("load configuration and predefined data");
		System.out.println("---------------------------------------------------------------------------");
		Config config = new Config();
		DesignMatrix designMatrix = new DesignMatrix("Latest_data/design_easy.txt", config.ROW, config.COL);
		Contrasts contrasts = new Contrasts("test/contrasts.txt", config.numContrasts, config.COL);
		VolumeReader volumeReader = new VolumeReader();
		Matrix C = contrasts.toMatrix();
		Broadcast<Matrix> broadcastC = sc.broadcast(C);
		Matrix X;
		System.out.println("Complete");

		// Read in first scan to get some brain volume metadata
		System.out.println("---------------------------------------------------------------------------");
		System.out.println("Read in first scan to get some brain volume metadata");
		System.out.println("---------------------------------------------------------------------------");
		int scanNumber = 1;
		String BOLDPath = config.assemblyBOLDPath(1);
		Brain volume = volumeReader.readFile(BOLDPath, 1);
		config.setVolumeSize(volume);
		Dataset dataset = new Dataset(config.ROW, config.getX(), config.getY(), config.getZ());
		dataset.addOneScan(volume);
		System.out.println("Complete");
		boolean[] ROI = config.getROI();
		dataset.setROI(ROI);
		System.out.println("---------------------------------------------------------------------------");
		int effectiveVoxelNumber = 0;
		for (boolean b : ROI) {
			if (b)
				effectiveVoxelNumber++;
		}
		System.out.println("Effective Voxel Number: " + effectiveVoxelNumber);
		System.out.println("---------------------------------------------------------------------------");

		Broadcast<Config> broadcastConfig = sc.broadcast(config);

		// Continue reading till reaching the K-th scan
		for (scanNumber = 2; scanNumber <= config.K; scanNumber++) {
			System.out.println("Reading Scan " + scanNumber);
			BOLDPath = config.assemblyBOLDPath(scanNumber);
			volume = volumeReader.readFile(BOLDPath, scanNumber);
			dataset.addOneScan(volume);
		}
		// System.out.println(dataset.getVolume(config.K));

		// Prepare
		System.out.println("---------------------------------------------------------------------------");
		System.out
				.println(new Date() + ": Successfully reading in first " + 238 + " scans, Now start SPRT estimation.");
		System.out.println("---------------------------------------------------------------------------");

		JavaRDD<DistributedDataset> distributedDataset = sc.parallelize(dataset.toDistrbutedDataset()).cache();

		for (scanNumber = config.K + 1; scanNumber <= config.ROW; scanNumber++) {
			System.out.println("Reading Scan " + scanNumber);
			BOLDPath = config.assemblyBOLDPath(scanNumber);
			volume = volumeReader.readFile(BOLDPath, scanNumber);

			dataset.addOneScan(volume);
			System.out.println("---------------------------------------------------------------------------");
			System.out.println(new Date() + ": Round " + scanNumber + ": Initializing broadcast variables");
			System.out.println("---------------------------------------------------------------------------");

			X = designMatrix.toMatrix(scanNumber);

			Matrix XTXInverse = Numerical.computeXTXInverse(X);
			Matrix XTXInverseXT = XTXInverse.multiplyTranspose(X);
			Matrix XXTXInverse = X.multiply(XTXInverse);
			double[] CTXTXInverseC = new double[C.getRow()];
			for (int i = 0; i < C.getRow(); i++) {
				Matrix c = C.getRowSlice(i);
				CTXTXInverseC[i] = c.transposeMultiply(XTXInverse).multiply(c).get();
			}
			double[] H = Numerical.computeH(XXTXInverse, X);

			// Create Spark shared variables
			Broadcast<Matrix> broadcastX = sc.broadcast(X);
			// Broadcast<Matrix> broadcastXTXInverse = sc.broadcast(XTXInverse);
			Broadcast<Matrix> broadcastXTXInverseXT = sc.broadcast(XTXInverseXT);
			Broadcast<Matrix> broadcastXXTXInverse = sc.broadcast(XXTXInverse);
			// Broadcast<double[]> broadcastCTXTXInverseC = sc.broadcast(CTXTXInverseC);
			Broadcast<double[]> broadcastH = sc.broadcast(H);

			// Spark logic
			// 1.1 Parallelize newly added scan
			JavaRDD<DistributedDataset> newDistributedDataset = sc.parallelize(dataset.toDistrbutedDataset(scanNumber));

			// 1.2 Zip up newly added scan to existing scans
			JavaPairRDD<DistributedDataset, DistributedDataset> pairedDataset = distributedDataset
					.zip(newDistributedDataset);

			// 1.3 Using map() to flatten.
			distributedDataset = pairedDataset
					.map(new Function<Tuple2<DistributedDataset, DistributedDataset>, DistributedDataset>() {

						@Override
						public DistributedDataset call(Tuple2<DistributedDataset, DistributedDataset> v1)
								throws Exception {
							return new DistributedDataset(
									ArrayUtils.add(v1._1.getBoldResponse(), v1._2.getBoldResponse()[0]), v1._1);
						}

					});

			// 2. Perform computation
			System.out.println("---------------------------------------------------------------------------");
			System.out.println(new Date() + ": Round " + scanNumber + ": Starting computation in workers");
			System.out.println("---------------------------------------------------------------------------");
			JavaRDD<CollectedDataset> collectedDataset = distributedDataset
					.map(new Function<DistributedDataset, CollectedDataset>() {

						@Override
						public CollectedDataset call(DistributedDataset distributedDataset) {
							CollectedDataset ret = new CollectedDataset(broadcastC.value().getRow());

							Matrix beta = Numerical.computeBeta2(broadcastXTXInverseXT.value(),
									distributedDataset.getBoldResponseMatrix());
							double[] R = Numerical.computeR(distributedDataset.getBoldResponseMatrix(),
									broadcastX.value(), beta);
							Matrix D = Numerical.generateD(R, broadcastH.value());

							for (int i = 0; i < broadcastC.value().getRow(); i++) {
								Matrix c = broadcastC.value().getRowSlice(i);
								double variance = Numerical.computeVarianceUsingMKLSparseRoutine2(c,
										broadcastXTXInverseXT.value(), broadcastXXTXInverse.value(), D);
								// double variance = Numerical.computeVariance(c, broadcastX.value(), D);
								double cBeta = Numerical.computeCBeta(c, beta);
								double ZScore = Numerical.computeZ(cBeta, variance);
								double theta1 = broadcastConfig.value().ZScore * Math.sqrt(variance);
								double SPRT = Numerical.compute_SPRT(cBeta, config.theta0, theta1, variance);
								int SPRTActivationStatus = Numerical.computeActivationStatus(SPRT,
										config.SPRTUpperBound, config.SPRTLowerBound);

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
			System.out.println("---------------------------------------------------------------------------");
			System.out.println(new Date() + ": Round " + scanNumber + ": Transform to activation map Rdd");
			System.out.println("---------------------------------------------------------------------------");
			int activationCounter = collectedDataset.map(new Function<CollectedDataset, Integer>() {

				@Override
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
			System.out.println("---------------------------------------------------------------------------");
			System.out.println(new Date() + ": Number of Activated Voxels: " + activationCounter);
			System.out.println("---------------------------------------------------------------------------");
		}
		sc.close();
	}
}