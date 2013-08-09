nn: *.c *.h
	gcc alloc.c neuron.c neural_network.c main.c -lm -O2 -Wall -o nn
