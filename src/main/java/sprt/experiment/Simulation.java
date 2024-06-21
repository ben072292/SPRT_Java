// package sprt.experiment;

// import static org.bytedeco.mkl.global.mkl_rt.*;

// import java.io.FileNotFoundException;
// import java.io.FileOutputStream;
// import java.io.PrintStream;
// import java.io.Serializable;
// import java.util.ArrayList;
// import java.util.Date;
// import java.util.Iterator;
// import java.util.concurrent.ExecutionException;
// import java.util.concurrent.ForkJoinPool;

// import org.apache.spark.SparkConf;
// import org.apache.spark.api.java.JavaPairRDD;
// import org.apache.spark.api.java.JavaRDD;
// import org.apache.spark.SparkContext;
// import org.apache.spark.api.java.JavaSparkContext;
// import org.apache.spark.api.java.function.FlatMapFunction;
// import org.apache.spark.broadcast.Broadcast;

// import sprt.ReduceData;
// import sprt.Contrasts;
// import sprt.Dataset;
// import sprt.DesignMatrix;
// import sprt.BOLD;
// import sprt.Matrix;
// import sprt.Matrix.MatrixStorageScope;
// import sprt.SprtStat;
// import sprt.Config;
// import static sprt.Algorithm.*;

// /**
//  * The Driver class using Apache Spark
//  * 
//  * @author Ben
//  *
//  */
// public class Simulation implements Serializable {

//     public static void main(String[] args) {
//         MKL_Set_Num_Threads(1);
//         int batchSize = Integer.parseInt(args[0]);
//         int dataExpand = Integer.parseInt(args[1]);
//         int numThreads = Integer.parseInt(args[2]);
//         long start, end;
//         PrintStream out;
//         try {
//             out = new PrintStream(new FileOutputStream("output-" + dataExpand + ".txt"));
//             System.setOut(out);
//         } catch (FileNotFoundException e) {
//             e.printStackTrace();
//         }

//         JavaSparkContext sc = JavaSparkContext.fromSparkContext(SparkContext.getOrCreate(new SparkConf()));

//         // load configuration and predefined data
//         System.out.println("load configuration and predefined data");
//         Config config = new Config();
//         DesignMatrix designMatrix = new DesignMatrix("Latest_data/design_easy.txt", config.MAX_SCAN, config.COL);
//         Contrasts contrasts = new Contrasts("test/contrasts.txt");
//         config.setContrasts(contrasts);
//         ArrayList<Matrix> C = contrasts.toMatrices();
//         Broadcast<ArrayList<Matrix>> broadcastC = sc.broadcast(C);
//         System.out.println("Complete");

//         System.out.println("Read in first scan to get some brain image metadata");
//         int scanNumber;
//         BOLD image = new BOLD(dataExpand * 36, 100, 100, true);
//         config.setImageSpec(image);
//         Dataset BOLD_dataset = new Dataset(config);
//         BOLD_dataset.add(image);
//         System.out.println("Complete");

//         // setup broadcast variables
//         Broadcast<Config> bcastConfig = sc.broadcast(config);

//         Matrix X = designMatrix.toMatrix();
// 		ArrayList<Matrix> XTXInverseList = new ArrayList<>();
// 		ArrayList<Matrix> XTXInverseXTList = new ArrayList<>();
// 		ArrayList<Matrix> XXTXInverseList = new ArrayList<>();
// 		ArrayList<double[]> HList = new ArrayList<>();
// 		for (int i = 1; i <= config.MAX_SCAN; i++) {
// 			if (i <= config.K) {
// 				XTXInverseList.add(null);
// 				XTXInverseXTList.add(null);
// 				XXTXInverseList.add(null);
// 				HList.add(null);
// 			} else {
// 				Matrix XTXInverse = computeXTXInverse(X.setRow(i));
// 				Matrix XTXInverseXT = XTXInverse.mmult(X.setRow(i));
// 				Matrix XXTXInverse = X.setRow(i).mmul(XTXInverse);
// 				double[] H = computeH(XXTXInverse, X.setRow(i));
				
// 				XTXInverseList.add(XTXInverse);
// 				XTXInverseXTList.add(XTXInverseXT);
// 				XXTXInverseList.add(XXTXInverse);
// 				HList.add(H);
// 			}
// 		}
// 		Broadcast<Matrix> bcastX = sc.broadcast(X);
//         Broadcast<ArrayList<Matrix>> bcastXTXInverseXTList = sc.broadcast(XTXInverseXTList);
//         Broadcast<ArrayList<Matrix>> bcastXXTXInverseList = sc.broadcast(XXTXInverseList);
//         Broadcast<ArrayList<double[]>> bcastHList = sc.broadcast(HList);
//         Broadcast<Integer> bcastNumThreads = sc.broadcast(numThreads);

//         // Continue reading till reaching the K-th scan
//         for (scanNumber = 2; scanNumber <= config.K; scanNumber++) {
//             BOLD_dataset.add(image);
//         }

//         // Prepare
//         System.out.println(
//                 new Date() + ": Successfully reading in first " + config.K + " scans, Now start SPRT estimation.");

//         JavaRDD<double[]> BOLD_RDD = sc
//                 .parallelize(BOLD_dataset.toDD());

//         start = System.nanoTime();

//         while (scanNumber <= 99) {
//             int currentScanNumber = scanNumber;

//             Broadcast<Integer> bcastStartScanNumber = sc.broadcast(scanNumber);

//             ArrayList<BOLD> incImages = new ArrayList<>();
//             for (; currentScanNumber < scanNumber + batchSize
//                     && currentScanNumber <= config.MAX_SCAN; currentScanNumber++) {
//                 incImages.add(image);
//             }

//             Broadcast<Integer> bcastRealBatchSize = sc.broadcast(incImages.size());

//             BOLD_dataset.setImages(incImages);

//             JavaRDD<double[]> inc_RDD = sc.parallelize(BOLD_dataset.toDD());

//             // Zip the RDDs
//             JavaPairRDD<double[], double[]> zippedRDD = BOLD_RDD.zip(inc_RDD);

//             // Merge each pair of ArrayLists
//             BOLD_RDD = zippedRDD.map(tuple -> {
//                 double[] array1 = tuple._1;
//                 double[] array2 = tuple._2;
//                 double[] mergedArray = new double[array1.length + array2.length];
//                 System.arraycopy(array1, 0, mergedArray, 0, array1.length);
//                 System.arraycopy(array2, 0, mergedArray, array1.length, array2.length);
//                 return mergedArray;
//             });

//             SprtStat sprtStat = BOLD_RDD
//                     .mapPartitions(new FlatMapFunction<Iterator<double[]>, SprtStat>() {
//                         public Iterator<SprtStat> call(Iterator<double[]> dd)
//                                 throws Exception {
//                             ArrayList<double[]> distributedData = new ArrayList<>();
//                             dd.forEachRemaining(distributedData::add);

//                             ArrayList<SprtStat> SPRTActivation = new ArrayList<>();
//                             ForkJoinPool myPool = new ForkJoinPool(bcastNumThreads.value());
//                             try {
//                                 myPool.submit(() -> distributedData.parallelStream().forEach(d -> {
//                                     ArrayList<ReduceData> ret = new ArrayList<>();
//                                     for (int i = bcastStartScanNumber.value(); i < bcastStartScanNumber
//                                             .value()
//                                             + bcastRealBatchSize.value(); i++) {
//                                         Matrix boldResponse = new Matrix(d, d.length,
//                                                 1, MatrixStorageScope.NATIVE);
//                                         ReduceData temp = new ReduceData(bcastConfig.value());
//                                         Matrix beta = computeBetaHat(bcastXTXInverseXTList.value().get(i - 1),
//                                                 boldResponse);
//                                         double[] R = computeR(boldResponse, bcastX.value().setRow(i),
//                                                 beta);
//                                         Matrix D = generateD(R, bcastHList.value().get(i - 1),
//                                                 MatrixStorageScope.NATIVE);
//                                         for (int j = 0; j < broadcastC.value().size(); j++) {

//                                             Matrix c = broadcastC.value().get(j);
//                                             double variance = compute_variance_sparse_fastest(c,
//                                                     bcastXTXInverseXTList.value().get(i - 1),
//                                                     bcastXXTXInverseList.value().get(i - 1), D);
//                                             double cBeta = compute_cBetaHat(c, beta);
//                                             double SPRT = compute_SPRT(cBeta, bcastConfig.value().theta0, 0.0,
//                                                     variance);
//                                             int SPRTActivationStatus = compute_activation_stat(SPRT,
//                                                     bcastConfig.value().SPRTUpperBound,
//                                                     bcastConfig.value().SPRTLowerBound);
//                                             temp.setVariance(j, variance);
//                                             temp.setCBeta(j, cBeta);
//                                             temp.setTheta1(j, 0.0);
//                                             temp.setSPRT(j, SPRT);
//                                             temp.setSPRTActivationStatus(j, SPRTActivationStatus);
//                                         }
//                                         ret.add(temp);
//                                     }

//                                     int[][][] SPRTActivationCounter = new int[ret.size()][broadcastC.value()
//                                             .size()][3];

//                                     for (int i = 0; i < ret.size(); i++) {
//                                         for (int j = 0; j < broadcastC.value().size(); j++) {
//                                             if (ret.get(i).getSPRTActivationStatus(j) == -1) {
//                                                 SPRTActivationCounter[i][j][0]++;
//                                             } else if (ret.get(i).getSPRTActivationStatus(j) == 0) {
//                                                 SPRTActivationCounter[i][j][1]++;
//                                             } else {
//                                                 SPRTActivationCounter[i][j][2]++;
//                                             }

//                                         }
//                                     }
//                                     SPRTActivation.add(new SprtStat(SPRTActivationCounter));

//                                 })).get();
//                             } catch (InterruptedException | ExecutionException e) {
//                                 throw new RuntimeException(e);
//                             } finally {
//                                 myPool.shutdown();
//                             }

//                             return SPRTActivation.iterator();
//                         }
//                     }).reduce((a, b) -> {
//                         return a.merge(b);
//                     });

//             for (int i = 0; i < sprtStat.getSprtStat().length; i++) {
//                 System.out.println("Scan " + (i + scanNumber));
//                 for (int j = 0; j < sprtStat.getSprtStat()[0].length; j++) {
//                     System.out.println("Contrast "
//                             + (j + 1)
//                             + ": Cross Upper: "
//                             + sprtStat.getSprtStat()[i][j][2]
//                             + ", Cross Lower: "
//                             + sprtStat.getSprtStat()[i][j][0]
//                             + ", Within Bound: "
//                             + sprtStat.getSprtStat()[i][j][1]);
//                 }
//             }

//             scanNumber = currentScanNumber;

//         }

//         sc.close();
//         end = System.nanoTime();
//         System.out.println("Total Time Consumption: " + (end - start) / 1e9 + " seconds.");

//     }
// }