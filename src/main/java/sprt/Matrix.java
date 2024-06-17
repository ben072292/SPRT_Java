package sprt;

import org.bytedeco.javacpp.*;
import org.bytedeco.mkl.global.mkl_rt.matrix_descr;
import org.bytedeco.mkl.global.mkl_rt.sparse_matrix;

import sprt.exception.MatrixComputationErrorException;

import static org.bytedeco.mkl.global.mkl_rt.*;

import java.io.FileWriter;
import java.io.Serializable;

/**
 * Matrix is the elemental computing unit in matrix computation. All
 * datasets/volumes
 * need to tranform to Matrix objects in order to perform any computation
 * 
 * @author Ben
 *
 */
public class Matrix implements Serializable {
	public enum MatrixStorageScope {
		HEAP, NATIVE
	}

	private static final double alpha = 1.0;
	private static final double beta = 0.0;

	private double[] arr;
	private transient DoublePointer nativeBuf;
	private int row;
	private int col;
	private MatrixStorageScope datatype = MatrixStorageScope.HEAP;

	public Matrix(int row, int col, MatrixStorageScope datatype) {
		this.row = row;
		this.col = col;
		this.datatype = datatype;
		switch (datatype) {
			case HEAP:
				this.arr = new double[row * col];
			case NATIVE:
				this.nativeBuf = new DoublePointer(row * col);
		}
	}

	public Matrix(double[] array, int row, int col, MatrixStorageScope datatype) {
		this.row = row;
		this.col = col;
		this.datatype = datatype;
		switch (datatype) {
			case HEAP:
				this.arr = array;
			case NATIVE:
				this.nativeBuf = new DoublePointer(array);
		}
	}

	public Matrix(Matrix matrix) {
		this.row = matrix.row;
		this.col = matrix.col;
		this.datatype = matrix.datatype;
		switch (datatype) {
			case HEAP:
				this.arr = matrix.arr.clone();
			case NATIVE:
				this.nativeBuf = new DoublePointer(matrix.nativeBuf);
		}
	}

	public Matrix deserialize(){
		if(this.arr != null && this.nativeBuf == null){
			this.nativeBuf = new DoublePointer(this.arr);
			this.arr = null;
			this.datatype = MatrixStorageScope.NATIVE;
		}
		return this;
	}

	public Matrix multiply(Matrix matrix) {
		this.deserialize();
		matrix.deserialize();
		Matrix res = new Matrix(this.row, matrix.col, MatrixStorageScope.NATIVE);
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				this.row, matrix.col, this.col, alpha, this.nativeBuf, this.col, matrix.nativeBuf, matrix.col, beta,
				res.nativeBuf,
				matrix.col);
		return res;
	}

	public Matrix transposeMultiply(Matrix matrix) {
		this.deserialize();
		matrix.deserialize();
		Matrix res = new Matrix(this.col, matrix.col, MatrixStorageScope.NATIVE);
		cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
				this.col, matrix.col, this.row, alpha, this.nativeBuf, this.col, matrix.nativeBuf, matrix.col, beta,
				res.nativeBuf,
				matrix.col);
		return res;
	}

	public Matrix multiplyTranspose(Matrix matrix) {
		this.deserialize();
		matrix.deserialize();
		Matrix res = new Matrix(this.row, matrix.row, MatrixStorageScope.NATIVE);
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
				this.row, matrix.row, this.col, alpha, this.nativeBuf, this.col, matrix.nativeBuf, matrix.col, beta,
				res.nativeBuf,
				matrix.row);
		return res;
	}

	public Matrix sparseMultiplyDense(Matrix matrix) {
		this.deserialize();
		matrix.deserialize();
		try {
			return sparseMultiplyDense(matrix, SPARSE_MATRIX_TYPE_DIAGONAL, SPARSE_FILL_MODE_LOWER,
					SPARSE_DIAG_NON_UNIT);
		} catch (MatrixComputationErrorException e) {
			e.printStackTrace();
		}
		return matrix;
	}

	public Matrix diagnalMultiplyDense(Matrix matrix) {
		this.deserialize();
		matrix.deserialize();
		Matrix res = new Matrix(this.row, matrix.col, MatrixStorageScope.NATIVE);
		for (int i = 0; i < this.row; i++) {
			double val = this.get(i, i);
			for (int j = 0; j < matrix.col; j++) {
				res.nativeBuf.put(i * matrix.col + j, matrix.get(i, j) * val);
			}
		}
		return res;
	}

	public Matrix sparseMultiplyDense(Matrix matrix, int type, int mode, int diag)
			throws MatrixComputationErrorException { // this: sparse, matrix: dense
		Matrix res = new Matrix(this.row, matrix.col, MatrixStorageScope.NATIVE);
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
				matrix.nativeBuf, matrix.col, matrix.col, 0.0, res.nativeBuf, matrix.col);
		if (resMultiply != SPARSE_STATUS_SUCCESS)
			throw new MatrixComputationErrorException("mkl_sparse_d_mm: Error " + resMultiply);

		mkl_sparse_destroy(D);
		csr.free();
		descr.deallocate();
		return res;
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

	public Matrix inverse() throws MatrixComputationErrorException {
		int result;
		Matrix matrix = new Matrix(this);
		IntPointer I = new IntPointer(Math.max(1, Math.min(matrix.row, matrix.col)));
		/*
		 * https://software.intel.com/en-us/mkl-developer-reference-c-getrf#E4779E02-
		 * 346C-4670-92AB-C67BD8559051
		 */
		result = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, matrix.row, matrix.col, matrix.nativeBuf, matrix.col, I); // factorization
		if (result < 0)
			throw new MatrixComputationErrorException(
					"Matrix Inverse Factorization: parameter " + result + " had an illegal value.");
		/* https://software.intel.com/en-us/mkl-developer-reference-c-getri */
		result = LAPACKE_dgetri(LAPACK_ROW_MAJOR, matrix.row, matrix.nativeBuf, matrix.row, I); // inverse
		I.deallocate();
		if (result < 0)
			throw new MatrixComputationErrorException("Matrix Inverse: parameter " + result + " had an illegal value.");
		else if (result > 0)
			throw new MatrixComputationErrorException("the " + result
					+ "-th diagonal element of the factor U is zero, U is singular, and the inversion could not be completed.");
		return matrix;
	}

	public Matrix multiply(double d) {
		for (int i = 0; i < row * col; i++) {
			this.put(i, this.get(i) * d);
		}
		return this;
	}

	public void showMatrix(int howMany, String s) {
		System.out.println("Show Matrix: Size: " + this.row + "*" + this.col + s);
		for (int i = 0; i < Math.min(this.row, howMany); i++) {
			for (int j = 0; j < Math.min(this.col, howMany); j++) {
				System.out.printf("%12.5G", this.get(i * this.col + j));
			}
			System.out.print("\n");
		}
		System.out.println();

	}

	public void showMatrix(int howMany) {
		showMatrix(howMany, "");
	}

	public void write(String filename, String descr) {
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
					writer.write(String.valueOf(this.get(i * this.row + j)) + ",");
				}
				writer.write("\n");
			}
			writer.close();
		} catch (Exception e) {
			e.printStackTrace();
		}

	}

	public int getRow() {
		return this.row;
	}

	public int getCol() {
		return this.col;
	}

	public double[] getArr() {
		return this.arr;
	}

	public DoublePointer getNativeBuf() {
		return this.nativeBuf;
	}

	public double get(int pos) {
		switch (datatype) {
			case HEAP:
				return this.arr[pos];
			case NATIVE:
				return this.nativeBuf.get(pos);
			default:
				return 0.0;
		}
	}

	public double get(int x, int y) {
		switch (datatype) {
			case HEAP:
				return this.arr[x * col + y];
			case NATIVE:
				return this.nativeBuf.get(x * col + y);
			default:
				return 0.0;
		}
	}

	public void put(int pos, double val) {
		switch (datatype) {
			case HEAP:
				this.arr[pos] = val;
			case NATIVE:
				this.nativeBuf.put(pos, val);
		}
	}

	public void put(int x, int y, double val) {
		int pos = x * col + y;
		put(pos, val);
	}

	public Matrix getRowSlice(int row) {
		Matrix res = new Matrix(1, this.col, this.datatype);
		for (int i = 0; i < this.getCol(); i++) {
			res.put(i, this.get(row, i));
		}
		return res;
	}

	public Matrix getColSlice(int col) {
		Matrix res = new Matrix(this.row, 1, this.datatype);
		for (int i = 0; i < this.getRow(); i++) {
			res.put(i, this.get(i, col));
		}
		return res;
	}

}
