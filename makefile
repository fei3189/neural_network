all: nn clean
lib: libnn.a clean
nn: main.c libnn.a
	gcc main.c -L . -lnn -lm -O2 -Wall -o nn
libnn.a: alloc.o neuron.o neural_network.o
	ar csrv libnn.a alloc.o neuron.o neural_network.o
alloc.o: alloc.c alloc.h
	gcc -c alloc.c
neuron.o: neuron.c neuron.h
	gcc -c neuron.c
neural_network.o: neural_network.c neural_network.h
	gcc -c neural_network.c
clean:
	rm -f *.o
