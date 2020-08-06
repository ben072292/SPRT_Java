package edu.cwru.csds.sprt;
public class Driver {
	public static void main(String[] args) {
		// load configuration and predefined data
		System.out.println("load configuration and predefined data");
		Config config = new Config();
		DesignMatrix designMatrix = new DesignMatrix("Latest_data/design_easy.txt", config.ROW, config.COL);
		Contrasts contrasts = new Contrasts("test/contrasts.txt", config.numContrasts, config.COL);
		VolumeReader volumeReader = new VolumeReader();
		Matrix C = contrasts.toMatrix();
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
		
		// Continue reading till reaching the K-th scan
		for(scanNumber = 2; scanNumber <= config.K; scanNumber++) {
			System.out.println("Reading Scan " + scanNumber);
			BOLDPath = config.assemblyBOLDPath(scanNumber);
			volume = volumeReader.readFile(BOLDPath, scanNumber);
			dataset.addOneScan(volume);
		}
		//System.out.println(dataset.getVolume(config.K));
		
		// Prepare
		System.out.println("Successfully reading in first " + config.K + " scans, Now start preparing SPRT estimation.");
		Dataset[] activationDataset = new Dataset[config.numContrasts];
		Dataset[] theta1Dataset = new Dataset[config.numContrasts];
		Dataset[] ZScoreDataset = new Dataset[config.numContrasts];
		Dataset[] varianceDataset = new Dataset[config.numContrasts];
		Dataset[] SPRTDataset = new Dataset[config.numContrasts];
		Dataset[] betaDataset = new Dataset[config.numContrasts];
		for(int i = 0; i < config.numContrasts; i++) {
			activationDataset[i] = new Dataset(config.ROW, config.getX(), config.getY(), config.getZ());
//			theta1Dataset[i] = new Dataset(config.ROW, config.getX(), config.getY(), config.getZ());
//			ZScoreDataset[i] = new Dataset(config.ROW, config.getX(), config.getY(), config.getZ());
//			varianceDataset[i] = new Dataset(config.ROW, config.getX(), config.getY(), config.getZ());
//			SPRTDataset[i] = new Dataset(config.ROW, config.getX(), config.getY(), config.getZ());
//			betaDataset[i] = new Dataset(config.ROW, config.getX(), config.getY(), config.getZ());
		}
		System.out.println("Preparation completed, now starting SPRT estimation!");
		
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
			int counter = 0;
			for(int x = 0; x < dataset.getX(); x++) {
				for(int y = 0; y < dataset.getY(); y++) {
					for(int z = 0; z < dataset.getZ(); z++) {
						if(ROI[x * dataset.getY() * dataset.getZ() + y * dataset.getZ() + z]) {
							Matrix Y = dataset.getBoldResponseAsMatrix(x, y, z);
							Matrix beta = Numerical.computeBeta2(XTXInverseXT, Y);
							double[] R = Numerical.computeR(Y, X, beta);
							Matrix D = Numerical.generateD(R, H);
							
							for(int i = 0; i < C.getRow(); i++) {
								Matrix c = C.getRowSlice(i);
								//double variance = Numerical.computeVarianceUsingMKLSparseRoutine2(c, XTXInverseXT, XXTXInverse, D);
								double variance = Numerical.computeVariance(c, X, D);
								double cBeta = Numerical.computeCBeta(c, beta);
								double ZScore = Numerical.computeZ(cBeta, variance);
								double theta1 = config.ZScore * Math.sqrt(variance);
								double SPRT = Numerical.compute_SPRT(cBeta, config.theta0, theta1, variance);
								double SPRTActivationStatus = Numerical.computeActivationStatus(SPRT, config.SPRTUpperBound, config.SPRTLowerBound);
								if(SPRTActivationStatus == 1) counter++;
								
//								varianceDataset[i].setSignal(scanNumber, x, y, z, variance);
//								betaDataset[i].setSignal(scanNumber, x, y, z, cBeta);
//								ZScoreDataset[i].setSignal(scanNumber, x, y, z, ZScore);
//								theta1Dataset[i].setSignal(scanNumber, x, y, z, theta1);
//								SPRTDataset[i].setSignal(scanNumber, x, y, z, SPRT);
								activationDataset[i].setSignal(scanNumber, x, y, z, SPRTActivationStatus);
							}
						}
					}
				}
			}
			System.out.println("haha " + counter);
			for(int i = 0; i < C.getRow(); i++) {
				int activatedCounter = 0;
				for(int x = 0; x < dataset.getX(); x++) {
					for(int y = 0; y < dataset.getY(); y++) {
						for(int z = 0; z < dataset.getZ(); z++) {
							if(activationDataset[i].getSignal(scanNumber, x, y, z) == 1) activatedCounter++;
						}
					}
				}
				System.out.println("Activated for Contrast " + (i+1) +": " + activatedCounter);
			}
		}
	}
}
