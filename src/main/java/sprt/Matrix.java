package sprt;

import org.bytedeco.javacpp.*;
import org.bytedeco.mkl.global.mkl_rt.matrix_descr;
import org.bytedeco.mkl.global.mkl_rt.sparse_matrix;

import sprt.exception.MatrixComputationErrorException;

import static org.bytedeco.mkl.global.mkl_rt.*;

import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;

/**
 * Matrix is the elemental computing unit in matrix computation. All
 * datasets need to tranform to Matrix objects in order to perform any
 * computation
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
	private transient double[] array;
	private transient DoublePointer pointer;
	private int row;
	private int col;
	private MatrixStorageScope storageScope = MatrixStorageScope.HEAP;

	public Matrix(int row, int col, MatrixStorageScope datatype) {
		this.row = row;
		this.col = col;
		this.storageScope = datatype;
		switch (datatype) {
			case HEAP:
				this.array = new double[row * col];
			case NATIVE:
				this.pointer = new DoublePointer(row * col);
		}
	}

	public Matrix(double[] array, int row, int col, MatrixStorageScope datatype) {
		this.row = row;
		this.col = col;
		this.storageScope = datatype;
		switch (datatype) {
			case HEAP:
				this.array = array;
			case NATIVE:
				this.pointer = new DoublePointer(array);
		}
	}

	public Matrix(DoublePointer buf, int row, int col, MatrixStorageScope datatype) {
		this.row = row;
		this.col = col;
		this.storageScope = datatype;
		switch (datatype) {
			case HEAP:
				buf.put(this.array);
			case NATIVE:
				this.pointer = buf;
		}
	}

	public Matrix(DoubleBuffer buf, int row, int col, MatrixStorageScope datatype) {
		this.row = row;
		this.col = col;
		this.storageScope = datatype;
		switch (datatype) {
			case HEAP:
				buf.put(this.array);
			case NATIVE:
				this.pointer = new DoublePointer(buf);
		}
	}

	public Matrix(double[] array, int row, int col) {
		this.row = row;
		this.col = col;
		this.storageScope = MatrixStorageScope.HEAP;
		this.array = array;

	}

	public Matrix(DoublePointer buf, int row, int col) {
		this.row = row;
		this.col = col;
		this.storageScope = MatrixStorageScope.NATIVE;
		this.pointer = buf;

	}

	public Matrix(DoubleBuffer buf, int row, int col) {
		this.row = row;
		this.col = col;
		this.storageScope = MatrixStorageScope.NATIVE;
		this.pointer = new DoublePointer(buf);

	}

	public Matrix(Matrix matrix) {
		this.row = matrix.row;
		this.col = matrix.col;
		this.storageScope = matrix.storageScope;
		switch (storageScope) {
			case HEAP:
				this.array = matrix.array.clone();
			case NATIVE:
				this.pointer = new DoublePointer(matrix.pointer);
		}
	}

	public Matrix toHeap() {
		if (this.array == null) {
			this.array = new double[this.row * this.col];
			this.pointer.get(this.array);
			this.pointer.deallocate();
			this.storageScope = MatrixStorageScope.HEAP;
			this.pointer = null;
		}
		return this;
	}

	public Matrix toNative() {
		if (this.pointer == null) {
			this.pointer = new DoublePointer(this.array);
			this.array = null;
			this.storageScope = MatrixStorageScope.NATIVE;
			this.array = null;
		}
		return this;
	}

	private void writeObject(ObjectOutputStream out) throws IOException {
		this.toHeap();
		out.defaultWriteObject();
		if (this.array != null) {
			out.writeInt(this.array.length);
			for (double value : this.array) {
				long bits = Double.doubleToRawLongBits(value);
				long swappedBits = Long.reverseBytes(bits); // has to change the endianess (from big to little) to
															// produce the correct result
				out.writeLong(swappedBits);
			}
		}
	}

	private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
		in.defaultReadObject();
		int length = in.readInt();

		ByteBuffer byteBuffer = ByteBuffer.allocateDirect(length * Double.BYTES);// .order(ByteOrder.nativeOrder());
		ReadableByteChannel channel = Channels.newChannel(in);
		channel.read(byteBuffer);
		byteBuffer.flip(); // Prepare the buffer for reading
		this.pointer = new DoublePointer(byteBuffer.asDoubleBuffer());
		this.storageScope = MatrixStorageScope.NATIVE;
		// System.out.println("Deserialize");
	}

	public Matrix mmul(Matrix matrix) {
		this.toNative();
		matrix.toNative();
		Matrix res = new Matrix(this.row, matrix.col, MatrixStorageScope.NATIVE);
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				this.row, matrix.col, this.col, alpha, this.pointer, this.col, matrix.pointer, matrix.col,
				beta,
				res.pointer,
				matrix.col);
		return res;
	}

	public Matrix tmmul(Matrix matrix) {
		this.toNative();
		matrix.toNative();
		Matrix res = new Matrix(this.col, matrix.col, MatrixStorageScope.NATIVE);
		cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
				this.col, matrix.col, this.row, alpha, this.pointer, this.col, matrix.pointer, matrix.col,
				beta,
				res.pointer,
				matrix.col);
		return res;
	}

	public Matrix mmult(Matrix matrix) {
		this.toNative();
		matrix.toNative();
		Matrix res = new Matrix(this.row, matrix.row, MatrixStorageScope.NATIVE);
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
				this.row, matrix.row, this.col, alpha, this.pointer, this.col, matrix.pointer, matrix.col,
				beta,
				res.pointer,
				matrix.row);
		return res;
	}

	public Matrix smmul(Matrix matrix) {
		this.toNative();
		matrix.toNative();
		try {
			return smmul(matrix, SPARSE_MATRIX_TYPE_DIAGONAL, SPARSE_FILL_MODE_LOWER,
					SPARSE_DIAG_NON_UNIT);
		} catch (MatrixComputationErrorException e) {
			e.printStackTrace();
		}
		return matrix;
	}

	public Matrix dmmul(Matrix matrix) {
		this.toNative();
		matrix.toNative();
		Matrix res = new Matrix(this.row, matrix.col, this.storageScope());
		for (int i = 0; i < this.row; i++) {
			double val = this.get(i, i);
			for (int j = 0; j < matrix.col; j++) {
				res.put(i * matrix.col + j, matrix.get(i, j) * val);
			}
		}
		return res;
	}

	public Matrix smmul(Matrix matrix, int type, int mode, int diag)
			throws MatrixComputationErrorException { // this: sparse, matrix: dense
		this.toNative();
		matrix.toNative();
		Matrix res = new Matrix(this.row, matrix.col, this.storageScope());
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
				matrix.pointer, matrix.col, matrix.col, 0.0, res.pointer, matrix.col);
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
			this.values.put(values);

			this.rowsStart = new IntPointer(MKL_malloc(matrix.row * Integer.BYTES, 32));
			this.rowsStart.put(rowsIndex, 0, matrix.row);

			this.rowsEnd = new IntPointer(MKL_malloc(matrix.row * Integer.BYTES, 32));
			this.rowsEnd.put(rowsIndex, 1, matrix.row);

			this.colIndex = new IntPointer(MKL_malloc((ciIndex) * Integer.BYTES, 32));
			this.colIndex.put(colIndex);
		}

		public void free() {
			MKL_free(this.values);
			MKL_free(this.rowsStart);
			MKL_free(this.rowsEnd);
			MKL_free(this.colIndex);
		}
	}

	public Matrix minv() throws MatrixComputationErrorException {
		this.toNative();
		int result;
		Matrix matrix = new Matrix(this);
		IntPointer I = new IntPointer(Math.max(1, Math.min(matrix.row, matrix.col)));
		/*
		 * https://software.intel.com/en-us/mkl-developer-reference-c-getrf#E4779E02-
		 * 346C-4670-92AB-C67BD8559051
		 */
		result = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, matrix.row, matrix.col, matrix.pointer, matrix.col, I); // factorization
		if (result < 0)
			throw new MatrixComputationErrorException(
					"Matrix Inverse Factorization: parameter " + result + " had an illegal value.");
		/* https://software.intel.com/en-us/mkl-developer-reference-c-getri */
		result = LAPACKE_dgetri(LAPACK_ROW_MAJOR, matrix.row, matrix.pointer, matrix.row, I); // inverse
		I.deallocate();
		if (result < 0)
			throw new MatrixComputationErrorException("Matrix Inverse: parameter " + result + " had an illegal value.");
		else if (result > 0)
			throw new MatrixComputationErrorException("the " + result
					+ "-th diagonal element of the factor U is zero, U is singular, and the inversion could not be completed.");
		return matrix;
	}

	public Matrix mmul(double d) {
		this.toNative();
		for (int i = 0; i < row * col; i++) {
			this.put(i, this.get(i) * d);
		}
		return this;
	}

	public void print(int howMany, String s) {
		System.out.println("On " + (pointer == null ? "native memory." : "heap memory."));
		System.out.println("Show Matrix: Size: " + this.row + "*" + this.col + s);
		for (int i = 0; i < Math.min(this.row, howMany); i++) {
			for (int j = 0; j < Math.min(this.col, howMany); j++) {
				System.out.printf("%12.5G", this.get(i * this.col + j));
			}
			System.out.print("\n");
		}
		System.out.println();

	}

	public void print(int howMany) {
		print(howMany, "");
	}

	public void print(){
		print(10);
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

	public double[] getArray() {
		return this.array;
	}

	public DoublePointer getPointer() {
		return this.pointer;
	}

	public double get(int pos) {
		switch (storageScope) {
			case HEAP:
				return this.array[pos];
			case NATIVE:
				return this.pointer.get(pos);
			default:
				return 0.0;
		}
	}

	public double get(int x, int y) {
		switch (storageScope) {
			case HEAP:
				return this.array[x * col + y];
			case NATIVE:
				return this.pointer.get(x * col + y);
			default:
				return 0.0;
		}
	}

	public void put(int pos, double val) {
		switch (storageScope) {
			case HEAP:
				this.array[pos] = val;
			case NATIVE:
				this.pointer.put(pos, val);
		}
	}

	public void put(int x, int y, double val) {
		int pos = x * col + y;
		put(pos, val);
	}

	public Matrix getRowSlice(int row) {
		Matrix res = new Matrix(1, this.col, this.storageScope);
		for (int i = 0; i < this.getCol(); i++) {
			res.put(i, this.get(row, i));
		}
		return res;
	}

	public Matrix getColSlice(int col) {
		Matrix res = new Matrix(this.row, 1, this.storageScope);
		for (int i = 0; i < this.getRow(); i++) {
			res.put(i, this.get(i, col));
		}
		return res;
	}

	public Matrix setRow(int row) {
		this.row = row;
		return this;
	}

	public Matrix setCol(int col) {
		this.col = col;
		return this;
	}

	public MatrixStorageScope storageScope() {
		return this.storageScope;
	}

}
