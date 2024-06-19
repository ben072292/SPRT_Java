package sprt.exception;

/**
 * Handles all exceptions during parsing brain volumes
 * 
 * @author Ben
 *
 */
public class ImageException extends Exception {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public ImageException() {
		super("The Actual Image Does Not Match The Specified Size!");
	}

	public ImageException(String errorMessage) {
		super(errorMessage);
	}
}
