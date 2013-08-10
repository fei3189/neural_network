/*************************************************************************
	> File Name: neural_network.c
	> Author: Felix Jiang
	> Mail: f91.jiang@gmail.com 
	> Created Time: Aug 9th, 2013
 ************************************************************************/

#include <string.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "alloc.h"
#include "neuron.h"
#include "neural_network.h"

struct neural_network* create_nn(struct nn_config *config)
{
	struct neural_network *nn;
	size_t i, j, k;

	if (config == NULL)
		return NULL;
	if (config->dim_i == 0 || config->dim_o == 0)
		return NULL;

	srand((unsigned)time(NULL));
	nn = (struct neural_network*)malloc(sizeof(struct neural_network));
	nn->nlayer = config->n_hidden + 2;
	nn->dim_i = config->dim_i;
	nn->dims = (size_t*)malloc(sizeof(size_t) * nn->nlayer);
	nn->max_dim = 0;

	nn->dims[0] = config->dim_i;
	nn->layers = (struct neuron**)malloc(sizeof(struct neuron*) * nn->nlayer);
	nn->layers[0] = (struct neuron*)malloc(sizeof(struct neuron) * nn->dims[0]);
	/* the input layer will not be used, so it is ok to be empty */
	for (i = 0; i < nn->dims[0]; ++i) {
		nn->layers[0][i].nw = 0;
		nn->layers[0][i].weights = NULL;
		nn->layers[0][i].odw = NULL;
	}
	for (i = 0; i < config->n_hidden + 1; ++i) {
		char *func;
		if (i == config->n_hidden) {
			nn->dims[i+1] = config->dim_o;
			func = config->aofunc;
		} else {
			nn->dims[i+1] = config->dim_h[i];
			func = config->ahfunc[i];
		}
		if (nn->dims[i+1] > nn->max_dim)
			nn->max_dim = nn->dims[i+1];
		nn->layers[i+1] = (struct neuron*)malloc(sizeof(struct neuron) * nn->dims[i+1]);
		for (j = 0; j < nn->dims[i+1];++j) {
			struct neuron *n = &(nn->layers[i+1][j]);
			n->nw = nn->dims[i] + 1;
			n->weights = alloc_1d(n->nw);
			n->odw = alloc_1d(n->nw);
			for (k = 0; k < n->nw; ++k) {
				n->weights[k] = 0.2 * (rand() - RAND_MAX / 2) / RAND_MAX;
				n->odw[k] = 0.0;
				if (strcmp(func, "sigmoid") == 0) {
					n->compute= sigmoid;
					n->derivate = sigmoid_d;
				} else if (strcmp(func, "tanh") == 0) {
					n->compute= tgh;
					n->derivate = tgh_d;
				} else {
					printf("unknown activation function!");
					return NULL;
				}
				n->update = update;
			}
		}
	}

	nn->values = alloc_2dv(nn->dims, nn->nlayer);

	return nn;
}

void forward(struct neural_network *nn, double *input)
{
	size_t i, j;
	double **values = nn->values;
	memcpy(values[0], input, sizeof(double) * (nn->dim_i));
	for (i = 1; i < nn->nlayer; ++i) {
		for (j = 0; j < nn->dims[i]; ++j) {
			struct neuron *nr = &(nn->layers[i][j]);
			values[i][j] = nr->compute(nr, values[i - 1]);
		}
	}
}

double compute_hidden_delta(struct neural_network *nn, double *delta, size_t m, size_t n)
{
	size_t i, dim = nn->dims[m + 1];
	double del = 0.0;
	struct neuron *layer = nn->layers[m + 1];
	for (i = 0; i < dim; ++i) {
		del += delta[i] * layer[i].weights[n];
	}
	return del;
}

void train(struct neural_network *nn, double **input, double **output, size_t ndata, double rate, double momentum, size_t n_iter)
{
	size_t i, j, k;
	double **values = nn->values,
			*delta = alloc_1d(nn->max_dim),
			*odelta = alloc_1d(nn->max_dim);
	while (n_iter--) {
		for (i = 0; i < ndata; ++i) {
			double *x = input[i], *y = output[i];
			forward(nn, x);

			for (k = nn->nlayer-1; k > 0; --k) {
				for (j = 0; j < nn->dims[k]; ++j) {
					struct neuron *nr = &nn->layers[k][j];
					double op = values[k][j],
						   d = (k == nn->nlayer - 1 ? y[j] - op : compute_hidden_delta(nn, odelta, k, j)), 
						   del_w;
					delta[j] = d * nr->derivate(nr, values[k-1], op);
					del_w = rate * delta[j];
					nr->update(nr, values[k-1], del_w, momentum);
				}
				memcpy(odelta, delta, sizeof(double) * nn->dims[k]);
			}
		}
		/* the error check code, just for debugging */
/*
		double sum = 0;
		double res[8];
		for (i = 0; i < 8; ++i) {
			predict(nn, input[i], res);
			for (j = 0; j < 8; ++j) {
				double e = res[j] - output[i][j];
				sum += e * e;
			}
				
		}
		printf("%f\n", sum); */
	}
	free_1d(delta);
	free_1d(odelta);
}

void predict(struct neural_network *nn, double *input, double *output)
{
	double **values = nn->values;
	forward(nn, input);
	memcpy(output, values[nn->nlayer-1], sizeof(double) * nn->dims[nn->nlayer-1]);
}

void predict_hidden(struct neural_network *nn, double *input, double *hidden, size_t n)
{
	double **values = nn->values;
	forward(nn, input);
	memcpy(hidden, values[n+1], sizeof(double) * nn->dims[n+1]);
}

void destroy_nn(struct neural_network *nn)
{
	size_t i, j, nlayer = nn->nlayer, *dims = nn->dims;
	struct neuron **layers = nn->layers;
	for (i = 0; i < nlayer; ++i) {
		for (j = 0; j < dims[i]; ++j) {
			double *w = layers[i][j].weights, *odw = layers[i][j].odw;
			if (w != NULL)
				free_1d(w);
			if (odw != NULL)
				free_1d(odw);
		}
		free(layers[i]);
	}
	free_2d(nn->values, nn->nlayer);
	free(layers);
	free(dims);
	free(nn);
}
