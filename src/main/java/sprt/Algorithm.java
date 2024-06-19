package sprt;

import sprt.Matrix.MatrixStorageScope;
import sprt.exception.MatrixComputationErrorException;

/**
 * Numerical methods used for SPRT
 * 
 * @author Ben
 *
 */
public class Algorithm {
	public static Matrix computeBetaHat(Matrix XTXInverseXT, Matrix Y) {
		return XTXInverseXT.mmul(Y);
	}

	public static double compute_cBetaHat(Matrix c, Matrix betaHat) {
		return c.mmul(betaHat).get(0);
	}

	public static Matrix computeXTXInverse(Matrix X) {
		try {
			return X.tmmul(X).minv();
		} catch (MatrixComputationErrorException e) {
			e.printStackTrace();
			System.exit(1);
		}
		return null;
	}

	public static double computeZ(Matrix c, Matrix beta, double variance) {
		return compute_cBetaHat(c, beta) / Math.sqrt(variance);
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
			return c.mmul(X.tmmul(X).minv())
					.mmult(X).mmul(D)
					.mmul(X).mmul(X.tmmul(X).minv())
					.mmult(c).get(0);
		} catch (MatrixComputationErrorException e) {
			e.printStackTrace();
			System.exit(1);
		}
		return -1;
	}

	/*
	 * use Inspector-Executor Sparse BLAS routines for fast sparse matrix
	 * computation
	 * ~200 times faster than without using sparse BLAS routines.
	 */
	public static double computeVarianceSparse(Matrix c, Matrix X, Matrix D) {
		try {
			return c.mmul(X.tmmul(X).minv()).mmult(X)
					.mmul(D.smmul(X.mmul(X.tmmul(X).minv()))).mmult(c)
					.get(0);
		} catch (MatrixComputationErrorException e) {
			e.printStackTrace();
			System.exit(1);
		}
		return -1;
	}

	/*
	 * use Inspector-Executor Sparse BLAS routines for fast sparse matrix
	 * computation
	 * ~200 times faster than without using sparse BLAS routines.
	 */
	public static double compute_variance_sparse_fast(Matrix c, Matrix XTXInverseXT, Matrix XXTXInverse,
			Matrix D) {
		return c.mmul(XTXInverseXT).mmul(D.smmul(XXTXInverse)).mmult(c).get(0);
	}

	public static double compute_variance_sparse_fastest(Matrix c, Matrix XTXInverseXT, Matrix XXTXInverse,
			Matrix D) {
		return c.mmul(XTXInverseXT).mmul(D.dmmul(XXTXInverse)).mmult(c).get(0);
	}

	/*
	 * Output: SPRT result
	 * Functionality: this function is to handle the weight lifting part and
	 * computes
	 * the formula 3 in paper "Dynamic adjustment of stimuli in real time functional
	 * magnetic resonance imaging"
	 * Formula: {(c*beta_hat-theta_0)'* Var(c*beta_hat)^-1 * (c*beta_hat-theta_0)
	 * - (c*beta_hat-theta_1)'* Var(c*beta_hat)^-1 * (c*beta_hat-theta_1)} / 2
	 */
	public static double compute_SPRT(double cBetaHat, double thetaZero, double thetaOne, double variance) {
		return (Math.pow((cBetaHat - thetaZero), 2) - Math.pow((cBetaHat - thetaOne), 2)) / (2 * variance);
	}

	/**
	 * Params: double -- alpha, double -- beta
	 * Output: the stopping rule's boundaries [A, B]
	 * Functionality: this method takes in two parameters, alpha and bate, and
	 * computes the stopping rule's boundary values A and B
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

	public static int compute_activation_stat(double SPRT, double upper, double lower) {
		if (SPRT > upper)
			return 1;
		else if (SPRT < lower)
			return -1;
		else
			return 0;
	}

	public static double computeTheta1(double Z, double variance) {
		return Z * Math.sqrt(variance);
	}

	/*
	 * compute H matrix and store its diagonal value into a vector
	 * H = X * (X'X)^-1 * X'
	 */
	public static double[] computeH(Matrix XXTXInverse, Matrix X) {
		Matrix H = XXTXInverse.mmult(X);
		double[] ret = new double[H.getRow()];
		for (int i = 0; i < H.getRow(); i++) {
			ret[i] = H.get(i, i);
		}
		return ret;
	}

	/*
	 * r_i = Y_i - X * beta_hat
	 * r_i is used to compute (D_r)^*
	 */
	public static double[] computeR(Matrix Y, Matrix X, Matrix beta) {
		double[] ret = new double[Y.getRow()];
		Matrix XBeta = X.mmul(beta);
		for (int i = 0; i < Y.getRow(); i++) {
			ret[i] = Y.get(i, 0) - XBeta.get(i, 0);
		}
		return ret;
	}

	/*
	 * D_values is a single vector is used as a parameter of sparse matrix
	 * computation
	 */
	public static Matrix generateD(double[] R, double[] H, MatrixStorageScope datatype) {
		Matrix D = new Matrix(R.length, R.length, datatype);
		for (int i = 0; i < R.length; i++) {
			for (int j = 0; j < R.length; j++) {
				D.put(i * R.length + j,  Math.pow(R[i], 2) / (1 - H[i]));
			}
		}
		return D;
	}

	/*
	 * estimate sigma_hat^2
	 * sigma_hat^2 = sum{r_i^2} / (# of scans - # of parameters)
	 * r_i = Y_i - X * beta_hat
	 */
	@Deprecated
	public static double estimateSigmaHatSquare(double[] response, Matrix X, Matrix beta, int scanNumber, int col) {
		Matrix XBeta = X.mmul(beta);
		double[] xBetaArray = XBeta.getArr();
		double ret = 0.0;
		for (int i = 0; i < response.length; i++) {
			ret += Math.pow((response[i] - xBetaArray[i]), 2);
		}
		ret /= (scanNumber - col);
		return ret;
	}

	@Deprecated
	public static double computeVarianceUsingSigmaHatSquare(double sigmaHatSquare, Matrix c, Matrix XTXInverse) {
		return sigmaHatSquare * c.mmul(XTXInverse).mmult(c).get(0);
	}

	public static int evaluateConfidenceInterval(double cBeta, double variance, double CI, double theta) {
		if (cBeta + CI * Math.sqrt(variance) < theta)
			return -1;
		else if (cBeta - CI * Math.sqrt(variance) > 0)
			return 1;
		else
			return 0;
	}

	public static double[] computeMeanAndVariance(double[] array) {
		double[] ret = new double[2];
		double total = 0.0;
		for (double d : array) {
			total += d;
		}
		ret[0] = total / array.length;
		double variance = 0.0;
		for (double d : array) {
			variance += Math.pow(d - ret[0], 2);
		}
		ret[1] = variance / array.length;
		return ret;
	}
}
