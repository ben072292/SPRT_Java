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
	// static {
    //     try {
    //         System.loadLibrary("sprt_native"); // Load the native library
    //     } catch (UnsatisfiedLinkError e) {
    //         System.err.println("Native library failed to load: " + e);
    //         e.printStackTrace();
    //     } catch (Exception e) {
    //         System.err.println("Unexpected error occurred: " + e);
    //         e.printStackTrace();
    //     }
    // }

	public static Matrix computeBetaHat(Matrix XTXInverseXT, Matrix Y) {
		return XTXInverseXT.mmul(Y);
	}

	public static float compute_cBetaHat(Matrix c, Matrix betaHat) {
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

	public static float computeZ(Matrix c, Matrix beta, float variance) {
		return compute_cBetaHat(c, beta) / (float)Math.sqrt(variance);
	}

	public static float computeZ(float cBeta, float variance) {
		return cBeta / variance;
	}

	/*
	 * use Inspector-Executor Sparse BLAS routines for fast sparse matrix
	 * computation
	 * ~200 times faster than without using sparse BLAS routines.
	 */
	public static float compute_variance(Matrix c, Matrix XTXInverseXT, Matrix XXTXInverse,
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
	public static float compute_SPRT(float cBetaHat, float thetaZero, float thetaOne, float variance) {
		return (float)(Math.pow((cBetaHat - thetaZero), 2) - Math.pow((cBetaHat - thetaOne), 2)) / (2 * variance);
	}

	/**
	 * Params: float -- alpha, float -- beta
	 * Output: the stopping rule's boundaries [A, B]
	 * Functionality: this method takes in two parameters, alpha and bate, and
	 * computes the stopping rule's boundary values A and B
	 */
	public static float SPRTUpperBound(float alpha, float beta) {
		float A = (float)Math.log((1 - beta) / alpha);
		float B = (float)Math.log(beta / (1 - alpha));
		return Math.max(A, B);
	}

	public static float SPRTLowerBound(float alpha, float beta) {
		float A = (float)Math.log((1 - beta) / alpha);
		float B = (float)Math.log(beta / (1 - alpha));
		return Math.min(A, B);
	}

	public static int compute_activation_stat(float SPRT, float upper, float lower) {
		if (SPRT > upper)
			return 1;
		else if (SPRT < lower)
			return -1;
		else
			return 0;
	}

	public static float computeTheta1(float Z, float variance) {
		return Z * (float)Math.sqrt(variance);
	}

	/*
	 * compute H matrix and store its diagonal value into a vector
	 * H = X * (X'X)^-1 * X'
	 */
	public static Matrix computeH(Matrix XXTXInverse, Matrix X) {
		Matrix H = XXTXInverse.mmult(X);
		Matrix ret = new Matrix(H.getRow(), 1, MatrixStorageScope.HEAP);
		for(int i = 0; i < H.getRow(); i++){
			ret.put(i, H.get(i * H.getRow() + i));
		}
		H.getPointer().deallocate();
		return ret;
	}

	/*
	 * r_i = Y_i - X * beta_hat
	 * r_i is used to compute (D_r)^*
	 */
	public static Matrix computeR(Matrix Y, Matrix X, Matrix beta) {
		return Y.vsub(X.mmul(beta));
	}

	/*
	 * D_values is a single vector is used as a parameter of sparse matrix
	 * computation
	 */
	public static Matrix generateD(Matrix R, Matrix H) {
		for(int i = 0; i < R.getRow(); i++){
			R.put(i, (float)(Math.pow(R.get(i), 2) / (1.0f - H.get(i))));
		}
		return R;
	}

	/*
	 * estimate sigma_hat^2
	 * sigma_hat^2 = sum{r_i^2} / (# of scans - # of parameters)
	 * r_i = Y_i - X * beta_hat
	 */
	@Deprecated
	public static float estimateSigmaHatSquare(float[] response, Matrix X, Matrix beta, int scanNumber, int col) {
		Matrix XBeta = X.mmul(beta);
		float[] xBetaArray = XBeta.getArray();
		float ret = 0.0f;
		for (int i = 0; i < response.length; i++) {
			ret += Math.pow((response[i] - xBetaArray[i]), 2);
		}
		ret /= (scanNumber - col);
		return ret;
	}

	@Deprecated
	public static float computeVarianceUsingSigmaHatSquare(float sigmaHatSquare, Matrix c, Matrix XTXInverse) {
		return sigmaHatSquare * c.mmul(XTXInverse).mmult(c).get(0);
	}

	public static int evaluateConfidenceInterval(float cBeta, float variance, float CI, float theta) {
		if (cBeta + CI * Math.sqrt(variance) < theta)
			return -1;
		else if (cBeta - CI * Math.sqrt(variance) > 0)
			return 1;
		else
			return 0;
	}

	public static float[] computeMeanAndVariance(float[] array) {
		float[] ret = new float[2];
		float total = 0.0f;
		for (float d : array) {
			total += d;
		}
		ret[0] = total / array.length;
		float variance = 0.0f;
		for (float d : array) {
			variance += Math.pow(d - ret[0], 2);
		}
		ret[1] = variance / array.length;
		return ret;
	}

	// public static ReduceData computeSPRT_CUDA(Matrix c, Matrix X, Matrix Y, Matrix XTXInverseXT, Matrix XXTXInverse, Matrix H, Config config){
	// 	FloatBuffer buf = computeSPRT_CUDA(c.getPointer().asBuffer(), X.getPointer().asBuffer(), Y.getPointer().asBuffer(), XTXInverseXT.getPointer().asBuffer(), XXTXInverse.getPointer().asBuffer(), H.getPointer().asBuffer());
	// 	ReduceData ret = new ReduceData(config);
	// 	for(int i = 0; i < config.)
	// }

	// public static native FloatBuffer computeSPRT_CUDA(FloatBuffer c, FloatBuffer X, FloatBuffer Y, FloatBuffer XTXInverseXT, FloatBuffer XXTXInverse, FloatBuffer H);
}
