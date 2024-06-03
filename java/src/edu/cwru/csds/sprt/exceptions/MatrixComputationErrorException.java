package edu.cwru.csds.sprt.exceptions;

/**
 * Handles all exceptions during matrix computation
 * 
 * @author Ben
 *
 */
public class MatrixComputationErrorException extends Exception {

	/**
	 * 
	 */
	private static final long serialVersionUID = 3L;

	public MatrixComputationErrorException() {
		super("File Format Not As Expected!");
	}

	public MatrixComputationErrorException(String errorMessage) {
		super(errorMessage);
	}

	public MatrixComputationErrorException(String errorMessage, Throwable err) {
		super(errorMessage, err);
	}

}
