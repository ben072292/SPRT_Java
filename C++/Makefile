P_NAME = SPRT_P
S_NAME = SPRT_S

all: $(P_NAME) $(S_NAME)

$(P_NAME): main.cpp numerical.cpp
	icpc -mkl -std=c++11 -fopenmp main.cpp numerical.cpp -o $(P_NAME)

$(S_NAME): main.cpp numerical.cpp
	icpc -mkl -cilk-serialize -std=c++11 -fopenmp main.cpp numerical.cpp -o $(S_NAME)

clean:
	rm $(P_NAME) $(S_NAME)

run:
	@$(MAKE) && ./$(P_NAME) 238 8