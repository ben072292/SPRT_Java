/**
 * Numerical methods used for SPRT
 * 
 * @author Ben
 *
 */
public class Numerical {
	public static Matrix computeBeta1(Matrix X, Matrix Y) {
		try {
			return X.transposeMultiply(X).inverse().multiplyTranspose(X).multiply(Y);
		} catch (MatrixComputationErrorException e) {
			e.printStackTrace();
		}
		System.err.println("Error: You should never see this line, go debug");
		return X;
	}

	public static Matrix computeBeta2(Matrix XTXInverseXT, Matrix Y) {
		return XTXInverseXT.multiply(Y);
	}

	public static double computeCBeta(Matrix c, Matrix beta) {
		return c.multiply(beta).get();
	}

	public static Matrix computeXTXInverse(Matrix X) {
		try {
			return X.transposeMultiply(X).inverse();
		} catch (MatrixComputationErrorException e) {
			e.printStackTrace();
		}
		System.err.println("Error: You should never see this line, go debug");
		return X;
	}

	public static double computeZ(Matrix c, Matrix beta, double variance) {
		return computeCBeta(c, beta) / Math.sqrt(variance);
	}

	public static double computeZ(double cBeta, double variance) {
		return cBeta / variance;
	}

	/*
	 * update 2019-04-12: Deprecated due to slowness, use the sparse version instead
	 * var(c_beta_hat) = c(X'X)^(-1)X'D_r^(*)X(X'X)^(-1)c'
	 */
	public static double computeVariance(Matrix c, Matrix X, Matrix D) {
		try {
			return c.multiply(X.transposeMultiply(X).inverse()).multiplyTranspose(X).multiply(D).multiply(X)
					.multiply(X.transposeMultiply(X).inverse()).multiplyTranspose(c).get();
		} catch (MatrixComputationErrorException e) {
			e.printStackTrace();
		}
		System.err.println("Error: You should never see this line, go debug");
		return 0;
	}

	/*
	 * use Inspector-Executor Sparse BLAS routines for fast sparse matrix
	 * computation ~200 times faster than without using sparse BLAS routines.
	 */
	public static double computeVarianceUsingMKLSparseRoutine1(Matrix c, Matrix X, Matrix D) {
		try {
			return c.multiply(X.transposeMultiply(X).inverse()).multiplyTranspose(X)
					.multiply(D.sparseDiagonalMultiplyDense(X.multiply(X.transposeMultiply(X).inverse())))
					.multiplyTranspose(c).get();
		} catch (MatrixComputationErrorException e) {
			e.printStackTrace();
		}
		System.err.println("Error: You should never see this line, go debug");
		return 0;
	}

	/*
	 * use Inspector-Executor Sparse BLAS routines for fast sparse matrix
	 * computation ~200 times faster than without using sparse BLAS routines.
	 */
	public static double computeVarianceUsingMKLSparseRoutine2(Matrix c, Matrix XTXInverseXT, Matrix XXTXInverse,
			Matrix D) {
		return c.multiply(XTXInverseXT).multiply(D.sparseDiagonalMultiplyDense(XXTXInverse)).multiplyTranspose(c).get();
	}

	/*
	 * Output: SPRT result Functionality: this function is to handle the weight
	 * lifting part and computes the formula 3 in paper "Dynamic adjustment of
	 * stimuli in real time functional magnetic resonance imaging" Formula:
	 * {(c*beta_hat-theta_0)'* Var(c*beta_hat)^-1 * (c*beta_hat-theta_0) -
	 * (c*beta_hat-theta_1)'* Var(c*beta_hat)^-1 * (c*beta_hat-theta_1)} / 2
	 */
	public static double compute_SPRT(double cBetaHat, double thetaZero, double thetaOne, double variance) {
		return (Math.pow((cBetaHat - thetaZero), 2) - Math.pow((cBetaHat - thetaOne), 2)) / (2 * variance);
	}

	/**
	 * Params: double -- alpha, double -- beta Output: the stopping rule's
	 * boundaries [A, B] Functionality: this method takes in two parameters, alpha
	 * and bate, and computes the stopping rule's boundary values A and B
	 */
	public static double SPRTUpperBound(double alpha, double beta) {
		double A = Math.log((1 - beta) / alpha);
		double B = Math.log(beta / (1 - alpha));
		return Math.max(A, B);
	}

	public static double SPRTLowerBound(double alpha, double beta) {
		double A = Math.log((1 - beta) / alpha);
		double B = Math.log(beta / (1 - alpha));
		return Math.min(A, B);
	}

	public static int computeActivationStatus(double SPRT, double upper, double lower) {
		if (SPRT > upper)
			return 1;
		else if (SPRT < lower)
			return -1;
		else
			return 0;
	}

	public static Brain[] estimateTheta1(Dataset dataset, Matrix X, Matrix C, double Z, boolean[] ROI) {
		Brain ret[] = new Brain[C.getRow()];
		for (int i = 0; i < C.getRow(); i++) {
			ret[i] = new Brain(dataset.getCurrentScanNumber(), dataset.getX(), dataset.getY(), dataset.getZ());
		}
		Matrix XTXInverse = computeXTXInverse(X);
		Matrix XTXInverseXT = XTXInverse.multiplyTranspose(X);
		Matrix XXTXInverse = X.multiply(XTXInverse);
		double[] CTXTXInverseC = new double[C.getRow()];
		for (int i = 0; i < C.getRow(); i++) {
			Matrix c = C.getRowSlice(i);
			CTXTXInverseC[i] = c.transposeMultiply(XTXInverse).multiply(c).get();
		}
		double[] H = computeH(XXTXInverse, X);
		for (int x = 0; x < dataset.getX(); x++) {
			for (int y = 0; y < dataset.getY(); y++) {
				for (int z = 0; z < dataset.getZ(); z++) {
					if (ROI[x * dataset.getY() * dataset.getZ() + y * dataset.getZ() + z]) {
						Matrix Y = dataset.getBoldResponseAsMatrix(x, y, z);
						Matrix beta = computeBeta2(XTXInverseXT, Y);
						double[] R = computeR(Y, X, beta);
						Matrix D = generateD(R, H);
						double[] variance = new double[C.getRow()];
						for (int i = 0; i < C.getRow(); i++) {
							Matrix c = C.getRowSlice(i);
							variance[i] = computeVarianceUsingMKLSparseRoutine2(c, XTXInverseXT, XXTXInverse, D);
							ret[i].setVoxel(Z * Math.sqrt(variance[i]), x, y, z);
						}
					}
				}
			}
		}
		return ret;
	}

	public static double computeTheta1(double Z, double variance) {
		return Z * Math.sqrt(variance);
	}

	/*
	 * compute H matrix and store its diagonal value into a vector H = X * (X'X)^-1
	 * * X'
	 */
	public static double[] computeH(Matrix XXTXInverse, Matrix X) {
		Matrix H = XXTXInverse.multiplyTranspose(X);
		assert (H.getRow() == H.getCol()) : "Not a square matrix";
		double[] ret = new double[H.getRow()];
		for (int i = 0; i < H.getRow(); i++) {
			ret[i] = H.get(i, i);
		}
		return ret;
	}

	/*
	 * r_i = Y_i - X * beta_hat r_i is used to compute (D_r)^*
	 */
	public static double[] computeR(Matrix Y, Matrix X, Matrix beta) {
		double[] ret = new double[Y.getRow()];
		Matrix XBeta = X.multiply(beta);
		for (int i = 0; i < Y.getRow(); i++) {
			ret[i] = Y.get(i, 0) - XBeta.get(i, 0);
		}
		return ret;
	}

	/*
	 * D_values is a single vector is used as a parameter of sparse matrix
	 * computation
	 */
	public static Matrix generateD(double[] R, double[] H) {
		assert (R.length == H.length) : "R and H not the same size";
		double[] D = new double[R.length * R.length];
		for (int i = 0; i < R.length; i++) {

			D[i * R.length + i] = Math.pow(R[i], 2) / (1 - H[i]);
		}
		return new Matrix(D, R.length, R.length);
	}

}
