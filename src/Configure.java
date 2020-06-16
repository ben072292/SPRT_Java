import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * Handles all configurable parameters.
 * @author Ben
 *
 */
public class Configure {
	private String path;
	private Map<String, Object> config;
	
	public Configure() {
		this.config = new HashMap<>();
	}
	
	public void loadFromConfigFile(String path) {
		this.path = path;
		config.put("path", this.path);
		BufferedReader reader;
		String key;
		try {
			reader = new BufferedReader(new FileReader(path));
			String line = reader.readLine();
			while(line != null) {
				if(line.startsWith("##")) { // instruction section
					line = reader.readLine();
					continue;
				}
				else if(line.startsWith("#")) {
					key = line.substring(4);
					line = reader.readLine();
				}
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	}
	
	public Object get(String key) {
		return this.config.get(key);
	}
}
