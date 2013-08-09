/*************************************************************************
	> File Name: neural_network.h
	> Author: Felix Jiang
	> Mail: f91.jiang@gmail.com 
	> Created Time: Aug 9th, 2013
 ************************************************************************/

#if !defined(_NEURAL_NETWORK_H_)
#define _NEURAL_NETWORK_H_

#include "neuron.h"

struct neural_network {
	struct neuron **layers;
	size_t dim_i;
	size_t *dims;
	size_t nlayer;
	size_t max_dim;
};

struct nn_config {
	size_t *dim_h;
	char **ahfunc;
	size_t n_hidden;
	size_t dim_i;
	size_t dim_o;
	char *aofunc;
};

struct neural_network* create_nn(struct nn_config *config);

void train(struct neural_network *nn, double **input, double **output, size_t ndata, double rate, double momentum, size_t n_iter);

void predict(struct neural_network *nn, double *input, double *output);

#endif

