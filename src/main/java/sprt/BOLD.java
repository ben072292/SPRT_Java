package sprt;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Random;

import org.bytedeco.javacpp.FloatPointer;

import sprt.exception.FileFormatNotCorrectException;

/**
 * Processes single bold response file
 * 
 * @author Ben
 *
 */
public class BOLD implements Serializable {
	private int id;
	private int nativeStartOffset = 0;
	private int batchSize = 0;
	private int scan;
	private transient FloatPointer pointer;
	private long pointerAddr = -1L;
	private transient ArrayList<Float> bold = null;

	public BOLD(){}

	public BOLD(int id, int scan) {
		this.id = id;
		this.scan = scan;
		this.bold = new ArrayList<>(scan);
	}

	public BOLD(int id, int scan, float val) {
		this(id, scan);
		this.bold.add(val);
	}

	public BOLD(int id, int scan, boolean random) {
		this(id, scan);
		Random rand = new Random();
		for (int i = 0; i < scan; i++) {
			this.bold.add(i, rand.nextFloat());
		}
	}

	public int getStartScan(){
		return this.nativeStartOffset;
	}

	public int getBatchSize(){
		return this.batchSize;
	}

	public FloatPointer getPointer(){
		return this.pointer;
	}

	public void setPointerAddr(long addr) {
		this.pointerAddr = addr;
	}

	public long getAddress() {
		return this.pointerAddr;
	}

	public ArrayList<Float> getBOLD() {
		return this.bold;
	}

	public int getID(){
		return this.id;
	}

	// read in file after getting the path
	public static ArrayList<BOLD> read(String path, Config config, ArrayList<BOLD> bolds) {
		boolean firstRead = false;
		BufferedReader reader;
		try {
			reader = new BufferedReader(new FileReader(path));
			String line = reader.readLine();
			if (line.contains("(")) {
				dimensionParser(line, config);
			} else {
				reader.close();
				throw new FileFormatNotCorrectException();
			}
			int size = config.getX() * config.getY() * config.getZ();
			if (bolds == null) {
				bolds = config.enableROI ? new ArrayList<>(size / 10) : new ArrayList<>(size);
				firstRead = true;
				config.setROI(new BitSet(size));
			}
			line = reader.readLine();
			int count = 0;
			while (line != null) {
				if (line.contains("Slice")) { // pass this line
					line = reader.readLine();
					continue;
				}
				double[] array = config.enableROI
						? Arrays.stream(line.split(",")).mapToDouble(Double::parseDouble).filter(val -> val != 0.0)
								.toArray()
						: Arrays.stream(line.split(",")).mapToDouble(Double::parseDouble).toArray();
				for (int i = 0; i < array.length; i++) {
					if (firstRead) {
						if (array[i] != 0.0) {
							config.getROI().set(count + i);
							bolds.add(new BOLD(count + i, config.MAX_SCAN, (float)array[i]));
						} else {
							if (!config.enableROI) {
								bolds.add(new BOLD(count + i, config.MAX_SCAN, (float)array[i]));
							}
						}
					} else {
						if (config.getROI().get(count + i)) {
							bolds.get(count + i).getBOLD().add((float)array[i]);
						} else {
							if (!config.enableROI) {
								bolds.get(count + i).getBOLD().add((float)array[i]);
							}
						}
					}
					bolds.get(count + i).batchSize++;
				}
				count += array.length;
				line = reader.readLine();
			}
			reader.close();
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(1);
		}
		return bolds;
	}

	public static void dimensionParser(String line, Config config) {
		try {
			int[] array = Arrays.stream(line.replaceAll("[()]", "").split(", ")).mapToInt(Integer::parseInt).toArray();
			if (array[0] == 0 || array[1] == 0 || array[2] == 0)
				throw new FileFormatNotCorrectException("Dimensions information in BOLD file cannot be 0");
			if (config.getX() == 0 || config.getY() == 0 || config.getZ() == 0) {
				config.setX(array[0]);
				config.setY(array[1]);
				config.setZ(array[2]);
			} else {
				if (config.getX() != array[0] || config.getY() != array[1] || config.getZ() != array[2])
					throw new FileFormatNotCorrectException("Dimensions information not consistent through all scans");
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public static void setROI(Config config, ArrayList<BOLD> bolds) {

	}

	// private void writeObject(ObjectOutputStream out) throws IOException {
	// 	out.defaultWriteObject();
	// 	for (int i = this.nativeStartOffset; i < this.nativeStartOffset + this.batchSize; i++) {
	// 		long bits = Double.floatToRawLongBits(this.bold.get(i));
	// 		long swappedBits = Long.reverseBytes(bits); // has to change the endianess (from big to little) to produce
	// 													// the correct result
	// 		out.writeLong(swappedBits);
	// 	}
	// 	this.nativeStartOffset += this.batchSize;
	// 	this.batchSize = 0;
	// }

	// private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
	// 	in.defaultReadObject();
	// 	this.bold = null;
	// 	if (this.pointerAddr == -1L) {
	// 		this.bold_pointer = new DoublePointer(this.scan);
	// 		this.pointerAddr = this.bold_pointer.address();
	// 	} else {
	// 		long addr = this.pointerAddr;
	// 		this.bold_pointer = new DoublePointer() {
	// 			{
	// 				this.address = addr;
	// 			}
	// 		};
	// 	}
	// 	ReadableByteChannel channel = Channels.newChannel(in);
	// 	channel.read(this.bold_pointer.position(this.nativeStartOffset).asByteBuffer());
	// 	this.bold_pointer.position(0); // rewind internal bytebuffer
	// }

	private void writeObject(ObjectOutputStream out) throws IOException {
		out.defaultWriteObject();
		for (int i = this.nativeStartOffset; i < this.nativeStartOffset + this.batchSize; i++) {
			out.writeFloat(this.bold.get(i));
		}
		this.nativeStartOffset += this.batchSize;
		this.batchSize = 0;
	}

	private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
		in.defaultReadObject();
		this.bold = null;
		if (this.pointerAddr == -1L) {
			this.pointer = new FloatPointer(this.scan);
			this.pointerAddr = this.pointer.address();
		} else {
			long addr = this.pointerAddr;
			this.pointer = new FloatPointer() {
				{
					this.address = addr;
				}
			};
		}
		for (int i = this.nativeStartOffset; i < this.nativeStartOffset + this.batchSize; i++) {
			this.pointer.put(i,  in.readFloat());
		}
	}

	public static void main(String[] args){
		ArrayList<BOLD> bolds = null;
		Config config = new Config();
		int scanNumber = 1;
		String BOLD_Path = config.assemblyBOLDPath(scanNumber);
		bolds = BOLD.read(BOLD_Path, config, bolds);

		scanNumber = 2;
		BOLD_Path = config.assemblyBOLDPath(scanNumber);
		bolds = BOLD.read(BOLD_Path, config, bolds);

		System.out.println("================ " + bolds.size());
		System.out.println("================ " + bolds.get(0).bold.size());
	}

}