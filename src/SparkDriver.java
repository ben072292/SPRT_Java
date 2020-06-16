import java.io.Serializable;
import java.util.Arrays;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.*;
import org.apache.spark.broadcast.Broadcast;
import org.spark_project.guava.primitives.Doubles;

/**
 * The Driver class using Apache Spark
 * @author Ben
 *
 */
@SuppressWarnings("serial")
public class SparkDriver implements Serializable{
	public static void main(String[] args) {
		// configure spark
        SparkConf sparkConf = new SparkConf().setAppName("Distributed SPRT")
                                        .setMaster("local[8]").set("spark.executor.memory","2g");
        // start a spark context
        JavaSparkContext sc = new JavaSparkContext(sparkConf);
        
		// load configuration and predefined data
		System.out.println("load configuration and predefined data");
		Config config = new Config();
		DesignMatrix designMatrix = new DesignMatrix("Latest_data/design_easy.txt", config.ROW, config.COL);
		Contrasts contrasts = new Contrasts("test/contrasts.txt", config.numContrasts, config.COL);
		VolumeReader volumeReader = new VolumeReader();
		Matrix C = contrasts.toMatrix();
		Broadcast<Matrix> broadcastC = sc.broadcast(C);
		Matrix X;
		System.out.println("Complete");
		
		// Read in first scan to get some brain volume metadata
		System.out.println("Read in first scan to get some brain volume metadata");
		int scanNumber = 1;
		String BOLDPath = config.assemblyBOLDPath(1);
		Brain volume = volumeReader.readFile(BOLDPath,  1);
		config.setVolumeSize(volume);
		Dataset dataset = new Dataset(config.ROW, config.getX(), config.getY(), config.getZ());
		dataset.addOneScan(volume);
		System.out.println("Complete");
		boolean[] ROI = config.getROI();
		dataset.setROI(ROI);
		
		Broadcast<boolean[]> broadcastROI = sc.broadcast(ROI);
		Broadcast<Config> broadcastConfig = sc.broadcast(config);
		
		// Continue reading till reaching the K-th scan
		for(scanNumber = 2; scanNumber <= config.K; scanNumber++) {
			System.out.println("Reading Scan " + scanNumber);
			BOLDPath = config.assemblyBOLDPath(scanNumber);
			volume = volumeReader.readFile(BOLDPath, scanNumber);
			dataset.addOneScan(volume);
		}
		//System.out.println(dataset.getVolume(config.K));
		
		// Prepare
		System.out.println("Successfully reading in first " + config.K + " scans, Now start SPRT estimation.");
		
		JavaRDD<DistributedDataset> distributedDataset = sc.parallelize(dataset.toDistrbutedDataset()).cache();
		
		for(scanNumber = config.K+1; scanNumber <= config.ROW; scanNumber++) {
			System.out.println("Reading Scan " + scanNumber);
			BOLDPath = config.assemblyBOLDPath(scanNumber);
			volume = volumeReader.readFile(BOLDPath, scanNumber);
			
			dataset.addOneScan(volume);
			
			X = designMatrix.toMatrix(scanNumber);
			
			Matrix XTXInverse = Numerical.computeXTXInverse(X);
			Matrix XTXInverseXT = XTXInverse.multiplyTranspose(X);
			Matrix XXTXInverse = X.multiply(XTXInverse);
			double[] CTXTXInverseC = new double[C.getRow()];
			for(int i = 0; i < C.getRow(); i++) {
				Matrix c = C.getRowSlice(i);
				CTXTXInverseC[i] = c.transposeMultiply(XTXInverse).multiply(c).get(); 
			}
			double[] H = Numerical.computeH(XXTXInverse, X);
			
			
			// Create Spark shared variables
			Broadcast<Matrix> broadcastX = sc.broadcast(X);
			Broadcast<Matrix> broadcastXTXInverse = sc.broadcast(XTXInverse);
			Broadcast<Matrix> broadcastXTXInverseXT = sc.broadcast(XTXInverseXT);
			Broadcast<Matrix> broadcastXXTXInverse = sc.broadcast(XXTXInverse);
			Broadcast<double[]> broadcastCTXTXInverseC = sc.broadcast(CTXTXInverseC);
			Broadcast<double[]> broadcastH = sc.broadcast(H);
			Broadcast<Brain> broadcastVolume = sc.broadcast(volume);
			
			// Spark logic
			// 1. add new scan to distributedDataset

			// Not sure broadcast new brain volume would be a good way.
			// Might cause performance issue.
			distributedDataset = distributedDataset.map(new Function<DistributedDataset, DistributedDataset>(){
				public DistributedDataset call(DistributedDataset distributedDataset) {
					int len = distributedDataset.getBoldResponse().length;
					double[] array = new double[len+1];
					for(int i = 0; i < len; i++) {
						array[i] = distributedDataset.getBoldResponse()[i];
					}
					array[len] = broadcastVolume.value().getVoxel(distributedDataset.getX(), distributedDataset.getY(), distributedDataset.getZ());
					
					return new DistributedDataset(array, distributedDataset.getX(), distributedDataset.getY(), distributedDataset.getZ());
				}
			});
			
			// 2. Perform computation
			JavaRDD<CollectedDataset> res = distributedDataset.map(new Function<DistributedDataset, CollectedDataset>() {
				public CollectedDataset call(DistributedDataset distributedDataset) {
					CollectedDataset ret = new CollectedDataset(broadcastC.value().getRow());
					
					Matrix beta = Numerical.computeBeta2(broadcastXTXInverseXT.value(), distributedDataset.getBoldResponseMatrix());
					double[] R = Numerical.computeR(distributedDataset.getBoldResponseMatrix(), broadcastX.value(), beta);
					Matrix D = Numerical.generateD(R, broadcastH.value());
					
					if(broadcastROI.value()[broadcastVolume.value().getLocation(distributedDataset.getX(), distributedDataset.getY(), distributedDataset.getZ())]) {
						for(int i = 0; i < broadcastC.value().getRow(); i++) {
							Matrix c = broadcastC.value().getRowSlice(i);
							//double variance = Numerical.computeVarianceUsingMKLSparseRoutine2(c, broadcastXTXInverseXT.value(), broadcastXXTXInverse.value(), D);
							double variance = Numerical.computeVariance(c, broadcastX.value(), D);
							double cBeta = Numerical.computeCBeta(c, beta);
							double ZScore = Numerical.computeZ(cBeta, variance);
							double theta1 = broadcastConfig.value().ZScore * Math.sqrt(variance);
							double SPRT = Numerical.compute_SPRT(cBeta, config.theta0, theta1, variance);
							double SPRTActivationStatus = Numerical.computeActivationStatus(SPRT, config.SPRTUpperBound, config.SPRTLowerBound);
							
							ret.setVariance(i, variance);
							ret.setCBeta(i, cBeta);
							ret.setZScore(i, ZScore);
							ret.setTheta1(i, theta1);
							ret.setSPRT(i, SPRT);
							ret.setSPRTActivationStatus(i, SPRTActivationStatus);
							
						}
					}
					
					return ret;
				}
			});
			
			int counter = 0;
			List<CollectedDataset> list = res.collect();
			for(CollectedDataset cd : list) {
				for(int i = 0; i < cd.getNumOfC(); i++) {
					if(cd.getSPRTActivationStatus(i) == 1) counter++;
				}
			}
			System.out.println(counter);
		}
		sc.close();
	}
}
