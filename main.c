/*************************************************************************
	> File Name: main.cpp
	> Author: Felix Jiang
	> Mail: f91.jiang@gmail.com 
	> Created Time: Aug 9th, 2013
 ************************************************************************/

/*
 * Encode sparse vectors, e.g. 
 *
 * 1 0 0 0 0 0 0 0 => 0 1 0
 * 0 1 0 0 0 0 0 0 => 0 0 1
 * 0 0 1 0 0 0 0 0 => 0 1 1
 * 0 0 0 1 0 0 0 0 => 1 1 1
 * 0 0 0 0 1 0 0 0 => 1 0 0
 * 0 0 0 0 0 1 0 0 => 1 0 1
 * 0 0 0 0 0 0 1 0 => 1 1 0
 * 0 0 0 0 0 0 0 1 => 0 0 0
 *
 * The output will not be strictly 0 or 1,
 * but some proximate values, say 0.1 or 0.9
 * */

#include <stdio.h>
#include <string.h>
#include <math.h>
#include "neural_network.h"
#include "alloc.h"

int main() {
	size_t i, j;
	size_t nencode = 4, ninput = pow(2, nencode);  //Set the number of vectors
	size_t iter_times = 15000; // Complex network requires more iteration times
	double rate = 0.5, momentum = 0.5; // Learning parameters;
	size_t hidden[1] = { nencode };
	char *ch = "sigmoid";
	char *chs[1] = {ch};

	double **input = alloc_2d(ninput, ninput), **output = alloc_2d(ninput, ninput);
	for (i = 0; i < ninput; ++i) {
		for (j = 0; j < ninput; ++j) {
			if (i == j)
				input[i][j] = output[i][j] = 1;
			else
				input[i][j] = output[i][j] = 0;
		}
	}

	/* Set the configuration of neural network */
	struct nn_config config = {
		.dim_h = hidden,
		.ahfunc = chs,
		.n_hidden = 1,
		.dim_i = ninput,
		.dim_o = ninput,
		.aofunc = ch
	};

	struct neural_network* nn = create_nn(&config);
	train(nn, input, output, ninput, rate, momentum, iter_times);
	
	double *res = alloc_1d(ninput);

	for (i = 0; i < ninput; ++i) {
		for (j = 0; j < ninput; ++j)
			printf("%d ", (int)input[i][j]);
		printf("=> ");
		predict_hidden(nn, input[i], res, 0);
		for (j = 0; j < nencode; ++j) {
			printf("%.3f ", res[j]);    // Actually value
		//	printf("%d ", (int)(res[j]+0.5)); // 0-1 value
		}
		predict(nn, input[i], res);
		printf("=> ");
		for (j = 0; j < ninput; ++j)
			printf("%.3f ", res[j]);
		printf("\n");
	}
	free_2d(input, ninput);
	free_2d(output, ninput);
	free_1d(res);
	destroy_nn(nn);
	return 0;
}
