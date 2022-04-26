package edu.cwru.csds.sprt.exceptions;

/**
 * Handles all exceptions during parsing brain volumes
 * 
 * @author Ben
 *
 */
public class VolumeNotMatchException extends Exception {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public VolumeNotMatchException() {
		super("The Actual Volume Does Not Match The Specified Size!");
	}

	public VolumeNotMatchException(String errorMessage) {
		super(errorMessage);
	}
}
