package sprt;

import org.bytedeco.javacpp.*;

import sprt.exception.MatrixComputationErrorException;

import static org.bytedeco.mkl.global.mkl_rt.*;

import java.io.FileWriter;
import java.io.Serializable;
import java.util.Random;

/**
 * Matrix is the elemental computing unit in matrix computation. All
 * datasets/volumes
 * need to tranform to Matrix objects in order to perform any computation
 * 
 * @author Ben
 *
 */
public class Matrix implements Serializable {
	private static final double alpha = 1.0;
	private static final double beta = 0.0;

	private double[] data;
	private int row;
	private int col;

	public Matrix(int row, int col) {
		this.row = row;
		this.col = col;
		this.data = new double[row * col];
	}

	public Matrix(double[] array, int row, int col) {
		this.row = row;
		this.col = col;
		this.data = array;
	}

	public Matrix(int[] array, int row, int col) {
		this.row = row;
		this.col = col;
		double[] arr = new double[row * col];
		for (int i = 0; i < row * col; i++) {
			arr[i] = array[i];
		}
		this.data = arr;
	}

	public Matrix(Matrix matrix) {
		this.row = matrix.row;
		this.col = matrix.col;
		this.data = matrix.data.clone();
	}

	public Matrix transpose() {
		Matrix res = new Matrix(this.col, this.row);
		for (int i = 0; i < this.row; i++) {
			for (int j = 0; j < this.col; j++) {
				res.data[j * this.row + i] = this.data[i * this.col + j];
			}
		}
		return res;
	}

	public Matrix multiply(Matrix matrix) {
		Matrix res = new Matrix(this.row, matrix.col);
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				this.row, matrix.col, this.col, alpha, this.data, this.col, matrix.data, matrix.col, beta, res.data,
				matrix.col);
		return res;
	}

	public Matrix transposeMultiply(Matrix matrix) {
		double[] res = new double[this.col * matrix.col];
		cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
				this.col, matrix.col, this.row, alpha, this.data, this.col, matrix.data, matrix.col, beta, res,
				matrix.col);
		return new Matrix(res, this.col, matrix.col);
	}

	public Matrix multiplyTranspose(Matrix matrix) {
		double[] res = new double[this.row * matrix.row];
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
				this.row, matrix.row, this.col, alpha, this.data, this.col, matrix.data, matrix.col, beta, res,
				matrix.row);
		return new Matrix(res, this.row, matrix.row);
	}

	public Matrix sparseMultiplyDense(Matrix matrix) {
		try {
			return sparseMultiplyDense(matrix, SPARSE_MATRIX_TYPE_DIAGONAL, SPARSE_FILL_MODE_LOWER,
					SPARSE_DIAG_NON_UNIT);
		} catch (MatrixComputationErrorException e) {
			e.printStackTrace();
		}
		return matrix;
	}

	public Matrix diagnalMultiplyDense(Matrix matrix) {
		double[] res = new double[this.row * matrix.col];
		for (int i = 0; i < this.row; i++) {
			double val = this.get(i, i);
			for (int j = 0; j < matrix.col; j++) {
				res[i * matrix.col + j] = matrix.get(i, j) * val;
			}
		}
		return new Matrix(res, this.row, matrix.col);
	}

	public Matrix sparseMultiplyDense(Matrix matrix, int type, int mode, int diag)
			throws MatrixComputationErrorException { // this: sparse, matrix: dense
		double[] res = new double[this.row * matrix.col];
		CSR csr = new CSR(this);
		sparse_matrix D = new sparse_matrix();
		int resCSR = mkl_sparse_d_create_csr(D, SPARSE_INDEX_BASE_ZERO, this.row, this.col, csr.rowsStart, csr.rowsEnd,
				csr.colIndex, csr.values);
		if (resCSR != SPARSE_STATUS_SUCCESS)
			throw new MatrixComputationErrorException("mkl_sparse_d_create_csr: Error " + resCSR);

		// Analyze sparse matrix; choose proper kernels and workload balancing strategy
		// mkl_sparse_optimize (D);

		matrix_descr descr = new matrix_descr();
		descr.type(type).mode(mode).diag(diag);
		int resMultiply = mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, D, descr, SPARSE_LAYOUT_ROW_MAJOR,
				matrix.data, matrix.col, matrix.col, 0.0, res, matrix.col);
		if (resMultiply != SPARSE_STATUS_SUCCESS)
			throw new MatrixComputationErrorException("mkl_sparse_d_mm: Error " + resMultiply);

		mkl_sparse_destroy(D);
		csr.free();
		descr.deallocate();
		return new Matrix(res, this.row, matrix.col);
	}

	public class CSR {
		DoublePointer values;
		IntPointer rowsStart;
		IntPointer rowsEnd;
		IntPointer colIndex;

		public CSR(Matrix matrix) {
			double[] values = new double[matrix.row * matrix.col];
			int[] rowsIndex = new int[matrix.row + 1];
			int[] colIndex = new int[matrix.row * matrix.col];
			int valsIndex = 0;
			int rsIndex = 1;
			int ciIndex = 0;
			rowsIndex[0] = 0;
			int rowCounter = 0;
			for (int i = 0; i < matrix.row; i++) {
				for (int j = 0; j < matrix.col; j++) {
					if (matrix.get(i, j) != 0.0) {
						values[valsIndex++] = matrix.get(i, j);
						colIndex[ciIndex++] = j;
						rowCounter++;
					}
				}
				rowsIndex[rsIndex++] = rowCounter;
			}

			this.values = new DoublePointer(MKL_malloc((valsIndex) * Double.BYTES, 64));
			for (int i = 0; i < valsIndex; i++) {
				this.values.put(i, values[i]);
			}

			this.rowsStart = new IntPointer(MKL_malloc(matrix.row * Integer.BYTES, 32));
			for (int i = 0; i < matrix.row; i++) {
				this.rowsStart.put(i, rowsIndex[i]);
			}

			this.rowsEnd = new IntPointer(MKL_malloc(matrix.row * Integer.BYTES, 32));
			for (int i = 0; i < matrix.row; i++) {
				this.rowsEnd.put(i, rowsIndex[i + 1]);
			}

			this.colIndex = new IntPointer(MKL_malloc((ciIndex) * Integer.BYTES, 32));
			for (int i = 0; i < ciIndex; i++) {
				this.colIndex.put(i, colIndex[i]);
			}
		}

		public void free() {
			MKL_free(this.values);
			MKL_free(this.rowsStart);
			MKL_free(this.rowsEnd);
			MKL_free(this.colIndex);
		}
	}

	// public class CSR {
	// 	double[] values;
	// 	int[] rowsStart;
	// 	int[] rowsEnd;
	// 	int[] colIndex;

	// 	public CSR(Matrix matrix) {
	// 		values = new double[matrix.row * matrix.col];
	// 		int[] rowsIndex = new int[matrix.row + 1];
	// 		colIndex = new int[matrix.row * matrix.col];
	// 		int valsIndex = 0;
	// 		int rsIndex = 1;
	// 		int ciIndex = 0;
	// 		rowsIndex[0] = 0;
	// 		int rowCounter = 0;
	// 		for (int i = 0; i < matrix.row; i++) {
	// 			for (int j = 0; j < matrix.col; j++) {
	// 				if (matrix.get(i, j) != 0.0) {
	// 					values[valsIndex++] = matrix.get(i, j);
	// 					colIndex[ciIndex++] = j;
	// 					rowCounter++;
	// 				}
	// 			}
	// 			rowsIndex[rsIndex++] = rowCounter;
	// 		}

	// 		this.rowsStart = new int[matrix.row];
	// 		for (int i = 0; i < matrix.row; i++) {
	// 			this.rowsStart[i] = rowsIndex[i];
	// 		}

	// 		this.rowsEnd = new int[matrix.row];
	// 		for (int i = 0; i < matrix.row; i++) {
	// 			this.rowsEnd[i] = rowsIndex[i+1];
	// 		}

	// 	}
	// }
	// TO DO: Sparse-Sparse Matrix Multiplication

	public Matrix inverse() throws MatrixComputationErrorException {
		int result;
		Matrix matrix = new Matrix(this);
		int[] I = new int[Math.max(1, Math.min(matrix.row, matrix.col))];
		/*
		 * https://software.intel.com/en-us/mkl-developer-reference-c-getrf#E4779E02-
		 * 346C-4670-92AB-C67BD8559051
		 */
		result = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, matrix.row, matrix.col, matrix.data, matrix.col, I); // factorization
		if (result < 0)
			throw new MatrixComputationErrorException(
					"Matrix Inverse Factorization: parameter " + result + " had an illegal value.");
		/* https://software.intel.com/en-us/mkl-developer-reference-c-getri */
		result = LAPACKE_dgetri(LAPACK_ROW_MAJOR, matrix.row, matrix.data, matrix.row, I); // inverse
		if (result < 0)
			throw new MatrixComputationErrorException("Matrix Inverse: parameter " + result + " had an illegal value.");
		else if (result > 0)
			throw new MatrixComputationErrorException("the " + result
					+ "-th diagonal element of the factor U is zero, U is singular, and the inversion could not be completed.");
		return matrix;
	}

	public Matrix multiply(double d) {
		for (int i = 0; i < row * col; i++) {
			this.data[i] *= d;
		}
		return this;
	}

	public Matrix divide(double d) {
		for (int i = 0; i < row * col; i++) {
			this.data[i] /= d;
		}
		return this;
	}

	public void showMatrix(int howMany, String s) {
		System.out.println("Show Matrix: Size: " + this.row + "*" + this.col + s);
		for (int i = 0; i < Math.min(this.row, howMany); i++) {
			for (int j = 0; j < Math.min(this.col, howMany); j++) {
				System.out.printf("%12.5G", this.data[i * this.col + j]);
			}
			System.out.print("\n");
		}
		System.out.println();

	}

	public void showMatrix(int howMany) {
		showMatrix(howMany, "");
	}

	public void outputWholeMatrixToTestFile(String filename, String descr) {
		try {
			FileWriter writer = new FileWriter(filename + ".csv");
			writer.write(filename + ".csv\n");
			writer.write(descr + "\n");
			writer.write(",");
			for (int i = 0; i < this.col; i++) {
				writer.write("Column " + (i + 1) + ",");
			}
			writer.write("\n");

			for (int i = 0; i < this.row; i++) {
				writer.write("Row" + (i + 1) + ",");
				for (int j = 0; j < this.col; j++) {
					writer.write(String.valueOf(this.data[i * this.row + j]) + ",");
				}
				writer.write("\n");
			}
			writer.close();
		} catch (Exception e) {
			e.printStackTrace();
		}

	}

	public void showMatrix(String s) {
		showMatrix(5, s);
	}

	public void showMatrix() {
		showMatrix(5, "");
	}

	public int getRow() {
		return this.row;
	}

	public int getCol() {
		return this.col;
	}

	public double get() { // get first value
		return this.data[0];
	}

	public double[] getData() {
		return this.data;
	}

	public double get(int x, int y) {
		return this.data[x * col + y];
	}

	public Matrix getRowSlice(int row) {
		double[] arr = new double[this.getCol()];
		for (int i = 0; i < this.getCol(); i++) {
			arr[i] = this.get(row, i);
		}
		return new Matrix(arr, 1, this.getCol());
	}

	public Matrix getColSlice(int col) {
		double[] arr = new double[this.getRow()];
		for (int i = 0; i < this.getRow(); i++) {
			arr[i] = this.get(i, col);
		}
		return new Matrix(arr, this.getRow(), 1);
	}

	public Matrix getFirstNumberOfRows(int number) {
		double[] arr = new double[number * this.getCol()];
		for (int i = 0; i < arr.length; i++) {
			arr[i] = this.data[i];
		}
		return new Matrix(arr, number, this.getCol());
	}

	public static void main(String[] args) throws Exception {

		Random rand = new Random();
		int a = 3000;
		int b = 1;

		double[] cc = new double[a * a];
		for (int i = 0; i < a; i++) {
			cc[i * a + i] = rand.nextDouble();
		}

		double[] ccc = new double[a * a];
		for (int i = 0; i < a; i++) {
			for (int j = 0; j < a; j++) {
				ccc[i * a + j] = rand.nextDouble();
			}
		}

		Matrix c = new Matrix(cc, a, a);
		// c.showMatrix();
		Matrix bb = new Matrix(ccc, a, a);

		long startTime = System.nanoTime();
		for (int i = 0; i < b; i++) {
			// c.inverse().multiply(c);
			c.multiply(bb).showMatrix();
		}
		long endTime = System.nanoTime();
		long timeElapsed = endTime - startTime;
		System.out.println("1: Execution time in us: " + timeElapsed / 1e6);

		startTime = System.nanoTime();
		for (int i = 0; i < b; i++) {
			c.sparseMultiplyDense(bb, SPARSE_MATRIX_TYPE_DIAGONAL, SPARSE_FILL_MODE_LOWER, SPARSE_DIAG_NON_UNIT)
					.showMatrix();
			;
		}
		endTime = System.nanoTime();
		timeElapsed = endTime - startTime;
		System.out.println("2: Execution time in us: " + timeElapsed / 1e6);

		startTime = System.nanoTime();
		for (int i = 0; i < b; i++) {
			c.diagnalMultiplyDense(bb)
					.showMatrix();
			;
		}
		endTime = System.nanoTime();
		timeElapsed = endTime - startTime;
		System.out.println("3: Execution time in us: " + timeElapsed / 1e6);

		// System.out.println(Numerical.computeVarianceUsingMKLSparseRoutine1(c, c, c));

		// c.sparseMultiplyDense(c).outputWholeMatrixToTestFile("ddd", "niubi");

	}

}