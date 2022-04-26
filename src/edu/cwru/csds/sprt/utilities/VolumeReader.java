package edu.cwru.csds.sprt.utilities;
// This is a reader to read in a whole brain volume

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Arrays;

import edu.cwru.csds.sprt.data.Brain;
import edu.cwru.csds.sprt.exceptions.FileFormatNotCorrectException;

public class VolumeReader {
	private int x = 0;
	private int y = 0;
	private int z = 0;

	public VolumeReader() {
	}

	// read in file after getting the path
	public Brain readFile(String path, int scanNumber) {
		BufferedReader reader;
		try {
			reader = new BufferedReader(new FileReader(path));
			String line = reader.readLine();
			if (line.contains("(")) {
				dimensionParser(line);
			} else {
				reader.close();
				throw new FileFormatNotCorrectException();
			}
			Brain ret = new Brain(scanNumber, this.x, this.y, this.z);
			line = reader.readLine();
			int x = 0;
			int y = 0;
			while (line != null) {
				if (line.contains("Slice")) { // pass this line
					line = reader.readLine();
					continue;
				}
				double[] array = Arrays.stream(line.split(",")).mapToDouble(Double::parseDouble).toArray();
				for (int i = 0; i < array.length; i++) {
					ret.setVoxel(array[i], x, y, i);
					// System.out.println(x + " " + y + " " + i);
				}
				y++;
				if (y == this.y) {
					y = 0;
					x++;
				}
				line = reader.readLine();
			}
			reader.close();
			return ret;
		} catch (Exception e) {
			e.printStackTrace();
		}
		System.err.println("Error reaching this step");
		return null;

	}

	public void dimensionParser(String line) {
		try {
			int[] array = Arrays.stream(line.replaceAll("[()]", "").split(", ")).mapToInt(Integer::parseInt).toArray();
			if (array[0] == 0 || array[1] == 0 || array[2] == 0)
				throw new FileFormatNotCorrectException("Dimensions information in BOLD file cannot be 0");
			if (this.x == 0 && this.y == 0 && this.z == 0) {
				this.x = array[0];
				this.y = array[1];
				this.z = array[2];
			} else {
				if (this.x != array[0] || this.y != array[1] || this.z != array[2])
					throw new FileFormatNotCorrectException("Dimensions information not consistent through all scans");
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	// main
	public static void main(String[] args) {
		VolumeReader volumeReader = new VolumeReader();
		Brain volume = volumeReader.readFile("Latest_data/bold0.txt", 1);
		System.out.println(volumeReader.x + " " + volumeReader.y + " " + volumeReader.z);
		System.out.println(volume);
		volume.outputToFile("bold00.txt");

	}
}
