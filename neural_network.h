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
	/* all layers, layers[0] is the input layer, which is meanless
	 * in compution, layers[nlayer-1] is the output layer
	 * */
	struct neuron **layers;

	/* dim of input layer */
	size_t dim_i;

	/* dims of each layer, dims[0] is the dim of input layer */
	size_t *dims;

	/* nlayer = one input layer + hidden layers + output layer*/
	size_t nlayer;

	/* max of all dims */
	size_t max_dim;

	double **values;
};

struct nn_config {
	/* dims of hidden layers */
	size_t *dim_h;
	/* activation function of each hidden layer,
	 * currently sigmoid function and tanh function
	 * are support 
	 * */
	char **ahfunc;

	/* the number of hidden layers */
	size_t n_hidden;

	/* dim of input layer */
	size_t dim_i;

	/* dim of output layer */
	size_t dim_o;

	/* activation function of output layer,
	 * sigmoid and tanh function supported
	 * */
	char *aofunc;
};

/* create neural network using the config struct */
struct neural_network* create_nn(struct nn_config *config);

/* training neural network by stochastic gradient descent.
 *
 * nn: neural network created by create_nn function
 * input, output: pointers to a (double*) array, each array
 *				  is one training data
 * ndata: number of training data, that is the length of input
 *		  and output
 * rate: learning rate, the coefficient of bp algorithm
 * momentum: momentum of bp algorithm
 * n_iter: iter times, it is not a good idea for bp algorithm
 * to be ended after convergency
 * */
void train(struct neural_network *nn, double **input, double **output, size_t ndata, double rate, double momentum, size_t n_iter);

/* the size of input data should be equal with the dim of input layer,
 * the output array should be large enough to hold the output values
 * */
void predict(struct neural_network *nn, double *input, double *output);

/* output values of the h-th hidden units
 * the size of input data should be equal with the dim of input layer,
 * the hidden array should be large enough to hold the hidden values,
 * */
void predict_hidden(struct neural_network *nn, double *input, double *hidden, size_t n);

void destroy_nn(struct neural_network *nn);

#endif

