package edu.cwru.csds.sprt;

import static edu.cwru.csds.sprt.Numerical.computeActivationStatus;
import static edu.cwru.csds.sprt.Numerical.computeBeta2;
import static edu.cwru.csds.sprt.Numerical.computeCBeta;
import static edu.cwru.csds.sprt.Numerical.computeH;
import static edu.cwru.csds.sprt.Numerical.computeR;
import static edu.cwru.csds.sprt.Numerical.computeVariance;
import static edu.cwru.csds.sprt.Numerical.computeXTXInverse;
import static edu.cwru.csds.sprt.Numerical.computeZ;
import static edu.cwru.csds.sprt.Numerical.compute_SPRT;
import static edu.cwru.csds.sprt.Numerical.generateD;
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

/**
 * The Driver class using Apache Spark
 * 
 * @author Ben
 *
 */
@SuppressWarnings("serial")
public class SparkDriverBatchAnalysis implements Serializable {

	public static void main(String[] args) {
		int batchSize = Integer.parseInt(args[0]);
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
		Contrasts contrasts = new Contrasts("test/contrasts.txt", config.numContrasts, config.COL);
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

        ArrayList<Matrix> designMatrixList = new ArrayList<>();
        //ArrayList<Matrix> XTXInverseList = new ArrayList<>();
        ArrayList<Matrix> XTXInverseXTList = new ArrayList<>();
        ArrayList<Matrix> XXTXInverseList = new ArrayList<>();
        ArrayList<double[]> HList = new ArrayList<>();

		// Continue reading till reaching the K-th scan
		for (scanNumber = 2; scanNumber <= config.K; scanNumber++) {
			System.out.println("Reading Scan " + scanNumber);
			BOLDPath = config.assemblyBOLDPath(scanNumber);
			volume = volumeReader.readFile(BOLDPath, scanNumber);
			dataset.addOneScan(volume);
		}
		// System.out.println(dataset.getVolume(config.K));

		// Prepare
		System.out.println(new Date() + ": Successfully reading in first " + 238 + " scans, Now start SPRT estimation.");

		JavaRDD<DistributedDataset> distributedDataset = sc.parallelize(dataset.toDistrbutedDataset());

		start = System.nanoTime();

		while(scanNumber <= config.ROW){
			int currentScanNumber = scanNumber;
			designMatrixList.clear();
			//XTXInverseList.clear();
			XTXInverseXTList.clear();
			XXTXInverseList.clear();
			HList.clear();
			Broadcast<Integer> broadcastStartScanNumber = sc.broadcast(scanNumber);
			Broadcast<Integer> broadcastEndScanNumber = sc.broadcast(Math.min(scanNumber+batchSize, config.ROW));

			ArrayList<Brain> volumes = new ArrayList<>();
			volumes.clear();
			for(; currentScanNumber < scanNumber+batchSize && currentScanNumber <= config.ROW; currentScanNumber++){
				BOLDPath = config.assemblyBOLDPath(currentScanNumber);
				volumes.add(volumeReader.readFile(BOLDPath, currentScanNumber));

				X = designMatrix.toMatrix(currentScanNumber);
		    	Matrix XTXInverse = computeXTXInverse(X);
		    	Matrix XTXInverseXT = XTXInverse.multiplyTranspose(X);
		    	Matrix XXTXInverse = X.multiply(XTXInverse);
		    	double[] CTXTXInverseC = new double[C.getRow()];
		    	for (int j = 0; j < C.getRow(); j++) {
				    Matrix c = C.getRowSlice(j);
				    CTXTXInverseC[j] = c.transposeMultiply(XTXInverse).multiply(c).get();
		    	}
		    	double[] H = computeH(XXTXInverse, X);
            	designMatrixList.add(X);
            	//XTXInverseList.add(XTXInverse);
            	XTXInverseXTList.add(XTXInverseXT);
            	XXTXInverseList.add(XXTXInverse);
            	HList.add(H);
			}
			
			Broadcast<ArrayList<Brain>> broadcastVolumes = sc.broadcast(volumes);
			
			distributedDataset = distributedDataset.map(new Function<DistributedDataset, DistributedDataset>() {

				public DistributedDataset call(DistributedDataset distributedDataset) {
						for(int i = 0; i < broadcastVolumes.value().size(); i++){
							distributedDataset = new DistributedDataset(ArrayUtils.add(distributedDataset.getBoldResponse(),
											broadcastVolumes.value().get(i).getVoxel(distributedDataset.getX(),
													distributedDataset.getY(), distributedDataset.getZ())),
							distributedDataset);
						}
						return distributedDataset;
				}

			});

			Broadcast<ArrayList<Matrix>> broadcastDesignMatrixList = sc.broadcast(designMatrixList);
        	//Broadcast<ArrayList<Matrix>> broadcastXTXInverseList = sc.broadcast(XTXInverseList);
        	Broadcast<ArrayList<Matrix>> broadcastXTXInverseXTList = sc.broadcast(XTXInverseXTList);
        	Broadcast<ArrayList<Matrix>> broadcastXXTXInverseList = sc.broadcast(XXTXInverseList);
        	Broadcast<ArrayList<double[]>> broadcastHList = sc.broadcast(HList);
			Broadcast<Integer> broadcastRealBatchSize = sc.broadcast(HList.size());


			JavaRDD<ArrayList<CollectedDataset>> collectedDatasets = distributedDataset
					.map(new Function<DistributedDataset, ArrayList<CollectedDataset>>() {
						public ArrayList<CollectedDataset> call(DistributedDataset distributedDataset) {
							MKL_Set_Num_Threads(1);
							ArrayList<CollectedDataset> ret = new ArrayList<>();
                            for(int i = 0; i < broadcastRealBatchSize.value(); i++){
                                double[] boldResponseRaw = new double[broadcastStartScanNumber.value() + i];
                                System.arraycopy(distributedDataset.getBoldResponse(), 0, boldResponseRaw, 0, broadcastStartScanNumber.value() + i);
                                Matrix boldResponse = new Matrix(boldResponseRaw, boldResponseRaw.length, 1);
                                // Matrix X = broadcastXComplete.value().getFirstNumberOfRows(i+1);

			                    // Matrix XTXInverse = computeXTXInverse(X);
			                    // Matrix XTXInverseXT = XTXInverse.multiplyTranspose(X);
			                    // Matrix XXTXInverse = X.multiply(XTXInverse);
			                    // double[] CTXTXInverseC = new double[broadcastC.value().getRow()];
			                    // for (int j = 0; j < C.getRow(); j++) {
				                //     Matrix c = C.getRowSlice(j);
				                //     CTXTXInverseC[j] = c.transposeMultiply(XTXInverse).multiply(c).get();
			                    // }
			                    // double[] H = computeH(XXTXInverse, X);

                                CollectedDataset temp = new CollectedDataset(broadcastC.value().getRow());
                                Matrix beta = computeBeta2(broadcastXTXInverseXTList.value().get(i),boldResponse);
                                double[] R = computeR(boldResponse, broadcastDesignMatrixList.value().get(i), beta);
                                
                                Matrix D = generateD(R, broadcastHList.value().get(i));

                                for (int j = 0; j < broadcastC.value().getRow(); j++) {
                                    Matrix c = broadcastC.value().getRowSlice(j);
                                    // double variance = Numerical.computeVarianceUsingMKLSparseRoutine3(c,
									// broadcastXTXInverseXTList.value().get(i), broadcastXXTXInverseList.value().get(i), D);
                                    // double variance = Numerical.computeVarianceUsingMKLSparseRoutine1(c,
                                    // X, D);
                                    double variance = computeVariance(c, broadcastDesignMatrixList.value().get(i), D);
                                    double cBeta = computeCBeta(c, beta);
                                    double ZScore = computeZ(cBeta, variance);
                                    double theta1 = broadcastConfig.value().ZScore * Math.sqrt(variance);
                                    double SPRT = compute_SPRT(cBeta, broadcastConfig.value().theta0, theta1, variance);
                                    int SPRTActivationStatus = computeActivationStatus(SPRT, broadcastConfig.value().SPRTUpperBound,
                                    broadcastConfig.value().SPRTLowerBound);
    
                                    temp.setVariance(j, variance);
                                    temp.setCBeta(j, cBeta);
                                    temp.setZScore(j, ZScore);
                                    temp.setTheta1(j, theta1);
                                    temp.setSPRT(j, SPRT);
                                    temp.setSPRTActivationStatus(j, SPRTActivationStatus);
                                }

                                ret.add(temp);
                            }
                            return ret;
                        }});

			

			// System.out.println(new Date() + ": Round " + scanNumber + ": Count elemnets
			// for collectedDataset RDD: " + collectedDataset.count());
			// 3. Get statistics from collected dataset
			ArrayList<Integer> activationCounter = collectedDatasets.map(new Function<ArrayList<CollectedDataset>, ArrayList<Integer>>() {
				public ArrayList<Integer> call(ArrayList<CollectedDataset> collectedDatasets) {
                    ArrayList<Integer> ret = new ArrayList<>();
                    for(CollectedDataset cd : collectedDatasets){
                        int sum = 0;
					    for (int i = 0; i < broadcastC.value().getRow(); i++) {
						    if (cd.getSPRTActivationStatus(i) == 1)
							sum++;
					    }
					    ret.add(sum);
                    }
					return ret;
				}
			}).reduce((a, b) -> {
                for(int i = 0; i < a.size(); i++){
                    a.set(i, a.get(i) + b.get(i));
                }
                return a;
            });
			// System.out.println(new Date() + ": Round " + scanNumber + ": Count elemnets
			// for activationMap RDD: " + activationMap.count());

            for(int i : activationCounter){
                System.out.println(new Date() + ": " + i);
            }

			scanNumber = currentScanNumber;

		}

		sc.close();
		end = System.nanoTime();
		System.out.println("Total Time Consumption: " + (end-start)/1e9 + " seconds.");

	}
}