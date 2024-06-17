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
// import org.apache.spark.api.java.JavaRDD;
// import org.apache.spark.SparkContext;
// import org.apache.spark.api.java.JavaSparkContext;
// import org.apache.spark.api.java.function.FlatMapFunction;
// import org.apache.spark.api.java.function.Function;
// import org.apache.spark.broadcast.Broadcast;

// import sprt.CollectedDataset;
// import sprt.Contrasts;
// import sprt.Dataset;
// import sprt.DesignMatrix;
// import sprt.DistributedDataset;
// import sprt.Image;
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
//         // configure spark
//         // SparkConf sparkConf = new
//         // SparkConf().setAppName("DistributedSPRT").setMaster("local[8]")
//         // .set("spark.driver.memory", "16g");
//         // JavaSparkContext sc = new JavaSparkContext(sparkConf);

//         JavaSparkContext sc = JavaSparkContext.fromSparkContext(SparkContext.getOrCreate(new SparkConf()));

//         // load configuration and predefined data
//         System.out.println("load configuration and predefined data");
//         Config config = new Config();
//         DesignMatrix designMatrix = new DesignMatrix("Latest_data/design_easy.txt", config.MAX_SCAN, config.COL);
//         Contrasts contrasts = new Contrasts("test/contrasts.txt");
//         config.setContrasts(contrasts);
//         Matrix C = contrasts.toMatrix(MatrixStorageScope.HEAP);
//         Broadcast<Matrix> broadcastC = sc.broadcast(C);
//         Matrix X;
//         System.out.println("Complete");

//         // Read in first scan to get some brain volume metadata
//         System.out.println("Read in first scan to get some brain volume metadata");
//         int scanNumber;
//         Image volume = new Image(dataExpand * 36, 100, 100, true);
//         config.setVolumeSize(volume);
//         Dataset dataset = new Dataset(config.getX(), config.getY(), config.getZ());
//         dataset.add(volume);
//         System.out.println("Complete");

//         // setup broadcast variables
//         Broadcast<Config> broadcastConfig = sc.broadcast(config);

//         ArrayList<Matrix> XList = new ArrayList<>();
//         ArrayList<Matrix> XTXInverseList = new ArrayList<>();
//         ArrayList<Matrix> XTXInverseXTList = new ArrayList<>();
//         ArrayList<Matrix> XXTXInverseList = new ArrayList<>();
//         ArrayList<double[]> HList = new ArrayList<>();
//         for (int i = 1; i <= config.MAX_SCAN; i++) {
//             if (i <= config.K) {
//                 XList.add(null);
//                 XTXInverseList.add(null);
//                 XTXInverseXTList.add(null);
//                 XXTXInverseList.add(null);
//                 HList.add(null);
//             } else {
//                 X = designMatrix.toMatrix(i, MatrixStorageScope.HEAP);
//                 Matrix XTXInverse = computeXTXInverse(X);
//                 Matrix XTXInverseXT = XTXInverse.multiplyTranspose(X);
//                 Matrix XXTXInverse = X.multiply(XTXInverse);
//                 double[] H = computeH(XXTXInverse, X);
//                 XList.add(X);
//                 XTXInverseList.add(XTXInverse);

//                 XTXInverseXTList.add(XTXInverseXT);
//                 XXTXInverseList.add(XXTXInverse);
//                 HList.add(H);
//             }
//         }
//         Broadcast<ArrayList<Matrix>> broadcastXList = sc.broadcast(XList);
//         Broadcast<ArrayList<Matrix>> broadcastXTXInverseXTList = sc.broadcast(XTXInverseXTList);
//         Broadcast<ArrayList<Matrix>> broadcastXXTXInverseList = sc.broadcast(XXTXInverseList);
//         Broadcast<ArrayList<double[]>> broadcastHList = sc.broadcast(HList);
//         Broadcast<Integer> broadcastNumThreads = sc.broadcast(numThreads);

//         // Continue reading till reaching the K-th scan
//         for (scanNumber = 2; scanNumber <= config.K; scanNumber++) {
//             dataset.add(volume);
//         }
//         // System.out.println(dataset.getVolume(config.K));

//         // Prepare
//         System.out.println(
//                 new Date() + ": Successfully reading in first " + config.K + " scans, Now start SPRT estimation.");

//         JavaRDD<DistributedDataset> distributedDataset = sc
//                 .parallelize(dataset.toDD(config.enableROI, config.getROI()));

//         start = System.nanoTime();

//         while (scanNumber <= 99) {
//             int currentScanNumber = scanNumber;

//             Broadcast<Integer> broadcastStartScanNumber = sc.broadcast(scanNumber);

//             ArrayList<Image> volumes = new ArrayList<>();
//             for (; currentScanNumber < scanNumber + batchSize && currentScanNumber <= config.MAX_SCAN; currentScanNumber++) {
//                 volumes.add(volume);
//             }

//             Broadcast<ArrayList<Image>> broadcastVolumes = sc.broadcast(volumes);
//             Broadcast<Integer> broadcastRealBatchSize = sc.broadcast(volumes.size());

//             distributedDataset = distributedDataset.map(new Function<DistributedDataset, DistributedDataset>() {
//                 public DistributedDataset call(DistributedDataset distributedDataset) {
//                     int a = distributedDataset.getBoldResponse().length;
//                     int b = broadcastVolumes.value().size();
//                     int x = distributedDataset.getX();
//                     int y = distributedDataset.getY();
//                     int z = distributedDataset.getZ();

//                     double[] newBoldResponse = new double[a + b];
//                     for (int i = 0; i < a; i++) {
//                         newBoldResponse[i] = distributedDataset.getBoldResponse()[i];
//                     }
//                     for (int i = 0; i < b; i++) {
//                         newBoldResponse[a + i] = broadcastVolumes.value().get(i).getVoxel(x, y, z);
//                     }
//                     return new DistributedDataset(newBoldResponse, x, y, z);
//                 }
//             });

//             SprtStat result = distributedDataset
//                     .mapPartitions(new FlatMapFunction<Iterator<DistributedDataset>, SprtStat>() {
//                         public Iterator<SprtStat> call(Iterator<DistributedDataset> distributedDataset)
//                                 throws Exception {
//                             ArrayList<DistributedDataset> distributedData = new ArrayList<>();
//                             distributedDataset.forEachRemaining(distributedData::add);

//                             ArrayList<SprtStat> SPRTActivation = new ArrayList<>();
//                             try (ForkJoinPool myPool = new ForkJoinPool(broadcastNumThreads.value())) {
//                                 try {
//                                     myPool.submit(() -> distributedData.parallelStream().forEach(d -> {
//                                         ArrayList<CollectedDataset> ret = new ArrayList<>();
//                                         for (int i = broadcastStartScanNumber.value(); i < broadcastStartScanNumber
//                                                 .value()
//                                                 + broadcastRealBatchSize.value(); i++) {
//                                             double[] boldResponseRaw = new double[i];
//                                             System.arraycopy(d.getBoldResponse(), 0, boldResponseRaw, 0,
//                                                     i);
//                                             Matrix boldResponse = new Matrix(boldResponseRaw, boldResponseRaw.length,
//                                                     1, MatrixStorageScope.BUF);
//                                             CollectedDataset temp = new CollectedDataset(broadcastConfig.value());
//                                             Matrix beta = computeBeta2(broadcastXTXInverseXTList.value().get(i - 1),
//                                                     boldResponse);
//                                             double[] R = computeR(boldResponse, broadcastXList.value().get(i - 1),
//                                                     beta);
//                                             Matrix D = generateD(R, broadcastHList.value().get(i - 1), MatrixStorageScope.BUF);
//                                             for (int j = 0; j < broadcastC.value().getRow(); j++) {

//                                                 Matrix c = broadcastC.value().getRowSlice(j);
//                                                 double variance = computeVarianceSparseFast(c,
//                                                         broadcastXTXInverseXTList.value().get(i - 1),
//                                                         broadcastXXTXInverseList.value().get(i - 1), D);
//                                                 double cBeta = computeCBeta(c, beta);
//                                                 double SPRT = compute_SPRT(cBeta, broadcastConfig.value().theta0, 0.0,
//                                                         variance);
//                                                 int SPRTActivationStatus = computeActivationStatus(SPRT,
//                                                         broadcastConfig.value().SPRTUpperBound,
//                                                         broadcastConfig.value().SPRTLowerBound);
//                                                 temp.setVariance(j, variance);
//                                                 temp.setCBeta(j, cBeta);
//                                                 temp.setTheta1(j, 0.0);
//                                                 temp.setSPRT(j, SPRT);
//                                                 temp.setSPRTActivationStatus(j, SPRTActivationStatus);
//                                             }
//                                             ret.add(temp);
//                                         }

//                                         int[][][] SPRTActivationCounter = new int[ret.size()][broadcastC.value()
//                                                 .getRow()][3];

//                                         for (int i = 0; i < ret.size(); i++) {
//                                             for (int j = 0; j < broadcastC.value().getRow(); j++) {
//                                                 if (ret.get(i).getSPRTActivationStatus(j) == -1) {
//                                                     SPRTActivationCounter[i][j][0]++;
//                                                 } else if (ret.get(i).getSPRTActivationStatus(j) == 0) {
//                                                     SPRTActivationCounter[i][j][1]++;
//                                                 } else {
//                                                     SPRTActivationCounter[i][j][2]++;
//                                                 }

//                                             }
//                                         }
//                                         SPRTActivation.add(new SprtStat(SPRTActivationCounter));

//                                     })).get();
//                                 } catch (InterruptedException | ExecutionException e) {
//                                     throw new RuntimeException(e);
//                                 } finally {
//                                     myPool.shutdown();
//                                 }
//                             }

//                             return SPRTActivation.iterator();
//                         }
//                     }).reduce((a, b) -> {
//                         return a.merge(b);
//                     });

//             for (int i = 0; i < result.getSprtStat().length; i++) {
//                 System.out.println("Scan " + (i + scanNumber));
//                 for (int j = 0; j < result.getSprtStat()[0].length; j++) {
//                     System.out.println("Contrast "
//                             + (j + 1)
//                             + ": Cross Upper: "
//                             + result.getSprtStat()[i][j][2]
//                             + ", Cross Lower: "
//                             + result.getSprtStat()[i][j][0]
//                             + ", Within Bound: "
//                             + result.getSprtStat()[i][j][1]);
//                 }
//             }

//             scanNumber = currentScanNumber;

//         }

//         sc.close();
//         end = System.nanoTime();
//         System.out.println("Total Time Consumption: " + (end - start) / 1e9 + " seconds.");

//     }
// }