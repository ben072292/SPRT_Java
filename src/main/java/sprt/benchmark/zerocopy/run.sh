# Compile the Java class & Generate the header file (superseds javah since Java-10)
javac -h . ThroughputTest.java

g++ -O3 --std=c++11 -I/opt/homebrew/Cellar/openjdk/22.0.1/include -shared -fPIC -o libnativeLib.dylib nativeLib.cpp

java -cp ../../../../java sprt.example.zerocopy.ThroughputTest