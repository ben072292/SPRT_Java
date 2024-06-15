package sprt.exception;

/**
 * Handles all exceptions in reading/writing files
 * 
 * @author Ben
 *
 */
public class FileFormatNotCorrectException extends Exception {

	public FileFormatNotCorrectException() {
		super("File Format Not As Expected!");
	}

	public FileFormatNotCorrectException(String errorMessage) {
		super(errorMessage);
	}

	public FileFormatNotCorrectException(String errorMessage, Throwable err) {
		super(errorMessage, err);
	}

}
