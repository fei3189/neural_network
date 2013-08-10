/*************************************************************************
	> File Name: main.cpp
	> Author: Felix Jiang
	> Mail: f91.jiang@gmail.com 
	> Created Time: 2013年08月08日 星期四 16时05分51秒
 ************************************************************************/

#include <stdio.h>
#include <string.h>
#include "neural_network.h"
#include "alloc.h"

double _input[8][8] = {
	{ 1, 0, 0, 0, 0, 0, 0, 0 },
	{ 0, 1, 0, 0, 0, 0, 0, 0 },
	{ 0, 0, 1, 0, 0, 0, 0, 0 },
	{ 0, 0, 0, 1, 0, 0, 0, 0 },
	{ 0, 0, 0, 0, 1, 0, 0, 0 },
	{ 0, 0, 0, 0, 0, 1, 0, 0 },
	{ 0, 0, 0, 0, 0, 0, 1, 0 },
	{ 0, 0, 0, 0, 0, 0, 0, 1 }
};

int main() {
	int i, j;
	size_t hidden[1] = { 3 };
	char *ch = "sigmoid";
	char *chs[1] = {ch};

	double **input = alloc_2d(8, 8), **output = alloc_2d(8, 8);
	for (i = 0; i < 8; ++i) {
		memcpy(input[i], _input[i], sizeof(double) * 8);
		memcpy(output[i], _input[i], sizeof(double) * 8);
	}
	struct nn_config config = {
		.dim_h = hidden,
		.ahfunc = chs,
		.n_hidden = 1,
		.dim_i = 8,
		.dim_o = 8,
		.aofunc = ch
	};
	struct neural_network* nn = create_nn(&config);
	train(nn, input, output, 8, 0.3, 0, 5000);
	
	double res[8];

	for (i = 0; i < 8; ++i) {
		for (j = 0; j < 8; ++j)
			printf("%d ", (int)input[i][j]);
		printf("=> ");
		predict_hidden(nn, input[i], res, 0);
		for (j = 0; j < 3; ++j)
			printf("%.3f ", res[j]);
		predict(nn, input[i], res);
		printf("=> ");
		for (j = 0; j < 8; ++j)
			printf("%.3f ", res[j]);
		printf("\n");
	}
	free_2d(input, 8);
	free_2d(output, 8);
	destroy_nn(nn);
	return 0;
}
