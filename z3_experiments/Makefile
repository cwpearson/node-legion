# https://github.com/Z3Prover/z3/releases
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:z3-4.8.8-x64-ubuntu-16.04/bin

all: main.cpp qap
	g++ -I z3-4.8.8-x64-ubuntu-16.04/include \
	    -L z3-4.8.8-x64-ubuntu-16.04/bin  \
	    -Wall -Wextra -Wpedantic main.cpp -o main -lz3

qap: qap.cpp
	g++ -I z3-4.8.8-x64-ubuntu-16.04/include \
	    -L z3-4.8.8-x64-ubuntu-16.04/bin  \
	    -Wall -Wextra -Wpedantic $< -o $@ -lz3