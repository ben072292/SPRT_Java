import static org.bytedeco.mkl.global.mkl_rt.CblasNoTrans;
import static org.bytedeco.mkl.global.mkl_rt.CblasRowMajor;
import static org.bytedeco.mkl.global.mkl_rt.CblasTrans;
import static org.bytedeco.mkl.global.mkl_rt.LAPACKE_dgetrf;
import static org.bytedeco.mkl.global.mkl_rt.LAPACKE_dgetri;
import static org.bytedeco.mkl.global.mkl_rt.LAPACK_ROW_MAJOR;
import static org.bytedeco.mkl.global.mkl_rt.SPARSE_DIAG_NON_UNIT;
import static org.bytedeco.mkl.global.mkl_rt.SPARSE_FILL_MODE_FULL;
import static org.bytedeco.mkl.global.mkl_rt.SPARSE_FILL_MODE_LOWER;
import static org.bytedeco.mkl.global.mkl_rt.SPARSE_INDEX_BASE_ZERO;
import static org.bytedeco.mkl.global.mkl_rt.SPARSE_LAYOUT_ROW_MAJOR;
import static org.bytedeco.mkl.global.mkl_rt.SPARSE_MATRIX_TYPE_DIAGONAL;
import static org.bytedeco.mkl.global.mkl_rt.SPARSE_OPERATION_NON_TRANSPOSE;
import static org.bytedeco.mkl.global.mkl_rt.SPARSE_STATUS_SUCCESS;
import static org.bytedeco.mkl.global.mkl_rt.cblas_dgemm;
import static org.bytedeco.mkl.global.mkl_rt.mkl_sparse_d_create_csr;
import static org.bytedeco.mkl.global.mkl_rt.mkl_sparse_d_mm;
import static org.bytedeco.mkl.global.mkl_rt.mkl_sparse_destroy;

import java.io.FileWriter;
import java.io.Serializable;
import java.util.Random;

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.mkl.global.mkl_rt.matrix_descr;
import org.bytedeco.mkl.global.mkl_rt.sparse_matrix;

/**
 * Matrix is the elemental computing unit in matrix computation. All
 * datasets/volumes need to tranform to Matrix objects in order to perform any
 * computation
 * 
 * @author Ben
 *
 */
@SuppressWarnings("serial")
public class Matrix implements Serializable {
	private static final double alpha = 1.0;
	private static final double beta = 0.0;

	private double[] array;
	private int row;
	private int col;

	public Matrix(int row, int col) {
		this.row = row;
		this.col = col;
		this.array = new double[row * col];
	}

	public Matrix(double[] array, int row, int col) {
		assert (array.length == row * col) : "Matrix defined size not match the array";
		this.row = row;
		this.col = col;
		this.array = array;
	}

	public Matrix(int[] array, int row, int col) {
		assert (array.length == row * col) : "Matrix defined size not match the array";
		this.row = row;
		this.col = col;
		double[] arr = new double[row * col];
		for (int i = 0; i < row * col; i++) {
			arr[i] = array[i];
		}
		this.array = arr;
	}

	public Matrix(Matrix matrix) {
		this.row = matrix.row;
		this.col = matrix.col;
		this.array = matrix.array.clone();
	}

	/**
	 * Perform dense matrix computation C = A * B
	 * 
	 * @param matrix B
	 * @return Matrix C
	 */
	public Matrix multiply(Matrix matrix) {
		assert matrix.row == this.col : "Matrix Multiplication: size incorrect!";
		Matrix res = new Matrix(this.row, matrix.col);
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, this.row, matrix.col, this.col, alpha, this.array,
				this.col, matrix.array, matrix.col, beta, res.array, matrix.col);
		return res;
	}

	/**
	 * Perform dense matrix computation C = A' * B
	 * 
	 * @param matrix B
	 * @return matrix C
	 */
	public Matrix transposeMultiply(Matrix matrix) {
		assert this.row == matrix.row : "Matrix Multiplication: size incorrect!";
		double[] res = new double[this.col * matrix.col];
		cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, this.col, matrix.col, this.row, alpha, this.array,
				this.col, matrix.array, matrix.col, beta, res, matrix.col);
		return new Matrix(res, this.col, matrix.col);
	}

	/**
	 * Perform dense matrix computation C = A * B'
	 * 
	 * @param matrix B
	 * @return matrix C
	 */
	public Matrix multiplyTranspose(Matrix matrix) {
		assert this.col == matrix.col : "Matrix Multiplication: size incorrect!";
		double[] res = new double[this.row * matrix.row];
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, this.row, matrix.row, this.col, alpha, this.array,
				this.col, matrix.array, matrix.col, beta, res, matrix.row);
		return new Matrix(res, this.row, matrix.row);
	}

	/**
	 * Perform sparse diagonal matrix computation C = A * B, where A is a sparse
	 * matrix that only has diagonal unit.
	 * 
	 * @param matrix B
	 * @return matrix C
	 */
	public Matrix sparseDiagonalMultiplyDense(Matrix matrix) {
		try {
			return sparseMultiplyDense(matrix, SPARSE_MATRIX_TYPE_DIAGONAL, SPARSE_FILL_MODE_FULL,
					SPARSE_DIAG_NON_UNIT);
		} catch (MatrixComputationErrorException e) {
			e.printStackTrace();
		}
		return matrix;
	}

	/**
	 * Perform sparse matrix computation C = A * B, where A is a sparse matrix of
	 * specified type, mode and diag.
	 * 
	 * @param matrix B
	 * @param type
	 * @param mode
	 * @param diag
	 * @return matrix C
	 * @throws MatrixComputationErrorException
	 */
	public Matrix sparseMultiplyDense(Matrix matrix, int type, int mode, int diag)
			throws MatrixComputationErrorException { // this: sparse, matrix: dense
		assert this.col == matrix.row : "Matrix Multiplication: size incorrect!";
		double[] res = new double[this.row * matrix.col];
		CSR csr = new CSR(this);
		sparse_matrix D = new sparse_matrix();
		int resCSR = mkl_sparse_d_create_csr(D, SPARSE_INDEX_BASE_ZERO, this.row, this.col, csr.rowsStart, csr.rowsEnd,
				csr.colIndex, csr.values);
		if (resCSR != SPARSE_STATUS_SUCCESS)
			throw new MatrixComputationErrorException("mkl_sparse_d_create_csr: Error " + resCSR);
		// Analyze sparse matrix; choose proper kernels and workload balancing strategy
		// mkl_sparse_optimize(D);
		matrix_descr descr = new matrix_descr();
		descr.type(type).mode(mode).diag(diag);
		int resMultiply = mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, D, descr, SPARSE_LAYOUT_ROW_MAJOR,
				matrix.array, matrix.col, matrix.col, 0.0, res, matrix.col);
		if (resMultiply != SPARSE_STATUS_SUCCESS)
			throw new MatrixComputationErrorException("mkl_sparse_d_mm: Error " + resMultiply);

		mkl_sparse_destroy(D);
		return new Matrix(res, this.row, matrix.col);
	}

	/**
	 * CSR format reference:
	 * https://software.intel.com/content/www/us/en/develop/documentation/mkl-
	 * developer-reference-c/top/appendix-a-linear-solvers-basics/sparse-matrix-
	 * storage-formats/sparse-blas-csr-matrix-storage-format.html
	 */
	public static class CSR {
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
					if (matrix.get(i, j) != 0) {
						values[valsIndex++] = matrix.get(i, j);
						colIndex[ciIndex++] = j;
						rowCounter++;
					}
				}
				rowsIndex[rsIndex++] = rowCounter;
			}
			double[] newValues = new double[--valsIndex];
			System.arraycopy(values, 0, newValues, 0, valsIndex);
			this.values = new DoublePointer(newValues);

			int[] rowsStart = new int[matrix.row];
			System.arraycopy(rowsIndex, 0, rowsStart, 0, matrix.row);
			this.rowsStart = new IntPointer(rowsStart);

			int[] rowsEnd = new int[matrix.row];
			System.arraycopy(rowsIndex, 1, rowsEnd, 0, matrix.row);
			this.rowsEnd = new IntPointer(rowsEnd);

			int[] newColIndex = new int[--ciIndex];
			System.arraycopy(colIndex, 0, newColIndex, 0, ciIndex);
			this.colIndex = new IntPointer(newColIndex);
		}
	}
	// TO DO: Sparse-Sparse Matrix Multiplication

	/**
	 * Inverse a square matrix
	 * 
	 * @return the matrix that is inversed
	 * @throws MatrixComputationErrorException
	 */
	public Matrix inverse() throws MatrixComputationErrorException {
		int result;
		assert this.row == this.col : "Matrix Inverse: not a square matrix";
		Matrix matrix = new Matrix(this);
		int[] I = new int[Math.max(1, Math.min(matrix.row, matrix.col))];
		/*
		 * https://software.intel.com/en-us/mkl-developer-reference-c-getrf#E4779E02-
		 * 346C-4670-92AB-C67BD8559051
		 */
		result = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, matrix.row, matrix.col, matrix.array, matrix.col, I); // factorization
		if (result < 0)
			throw new MatrixComputationErrorException(
					"Matrix Inverse Factorization: parameter " + result + " had an illegal value.");
		/* https://software.intel.com/en-us/mkl-developer-reference-c-getri */
		result = LAPACKE_dgetri(LAPACK_ROW_MAJOR, matrix.row, matrix.array, matrix.row, I); // inverse
		if (result < 0)
			throw new MatrixComputationErrorException("Matrix Inverse: parameter " + result + " had an illegal value.");
		else if (result > 0)
			throw new MatrixComputationErrorException("the " + result
					+ "-th diagonal element of the factor U is zero, U is singular, and the inversion could not be completed.");
		return matrix;
	}

	/**
	 * Matrix element-wise multiply a constant.
	 * 
	 * @param d
	 * @return the new Matrix that is multiplied.
	 */
	public Matrix multiply(double d) {
		for (int i = 0; i < row * col; i++) {
			this.array[i] *= d;
		}
		return this;
	}

	/**
	 * Matrix element-wise divide a constant.
	 * 
	 * @param d
	 * @return the new Matrix that is divided.
	 */
	public Matrix divide(double d) {
		assert d != 0 : "Cannot Divide By 0!";
		for (int i = 0; i < row * col; i++) {
			this.array[i] /= d;
		}
		return this;
	}

	/**
	 * Output the matrix
	 * 
	 * @param howMany
	 * @param s
	 */
	public void showMatrix(int howMany, String s) {
		System.out.println("Show Matrix: Size: " + this.row + "*" + this.col + s);
		for (int i = 0; i < Math.min(this.row, howMany); i++) {
			for (int j = 0; j < Math.min(this.col, howMany); j++) {
				System.out.printf("%12.7G", this.array[i * this.col + j]);
			}
			System.out.print("\n");
		}
		System.out.println();

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
					writer.write(String.valueOf(this.array[i * this.row + j]) + ",");
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
		return this.array[0];
	}

	public double get(int x, int y) {
		assert x < row : "Row out of bounds!";
		assert y < col : "Column out of bounds!";
		return this.array[x * col + y];
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

	public static void main(String[] args) throws Exception {

		Random rand = new Random();
		int a = 6000;
		int b = 1;

		double[] cc = new double[a * a];
		for (int i = 0; i < a; i++) {
			cc[i * a + i] = rand.nextDouble();
		}

		Matrix c = new Matrix(cc, a, a);
		// c.showMatrix();

		long startTime = System.nanoTime();
		for (int i = 0; i < b; i++) {
			// c.inverse().multiply(c);
			c.multiply(c).showMatrix(10, "");
			;
		}
		long endTime = System.nanoTime();
		long timeElapsed = endTime - startTime;
		System.out.println("1: Execution time in milliseconds: " + timeElapsed / 1e6);

		startTime = System.nanoTime();
		for (int i = 0; i < b; i++) {
			c.sparseMultiplyDense(c, SPARSE_MATRIX_TYPE_DIAGONAL, SPARSE_FILL_MODE_LOWER, SPARSE_DIAG_NON_UNIT)
					.showMatrix(10, "");
			;
		}
		endTime = System.nanoTime();
		timeElapsed = endTime - startTime;
		System.out.println("2: Execution time in milliseconds: " + timeElapsed / 1e6);

		System.out.println(Numerical.computeVarianceUsingMKLSparseRoutine1(c, c, c));

		// c.sparseDiagonalMultiplyDense(c).outputWholeMatrixToTestFile("ddd", "niubi");

	}

}
