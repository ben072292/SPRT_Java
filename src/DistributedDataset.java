import java.io.Serializable;

/**
 * Dataset object that stores the parameters to be computed
 * by Spark executors
 * @author Ben
 *
 */
@SuppressWarnings("serial")
public class DistributedDataset implements Serializable {
	private double[] boldResponse;
	private int x;
	private int y;
	private int z;
	private int id;
	
	public DistributedDataset(double[] boldResponse, int x, int y, int z, int id) {
		this.boldResponse = boldResponse;
		this.x = x;
		this.y = y;
		this.z = z;
		this.id = id;
	}
	
	public void setBoldResponse(double[] boldResponse) {
		this.boldResponse = boldResponse;
	}
	
	public Matrix getBoldResponseMatrix() {
		return new Matrix(boldResponse, boldResponse.length, 1);
	}
	
	public double[] getBoldResponse() {
		return this.boldResponse;
	}
	
	public int getX() {
		return this.x;
	}
	
	public int getY() {
		return this.y;
	}
	
	public int getZ() {
		return this.z;
	}
	
	public int getID() {
		return this.id;
	}
}
