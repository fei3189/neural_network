/*************************************************************************
	> File Name: neuron.c
	> Author: Felix Jiang
	> Mail: f91.jiang@gmail.com 
	> Created Time: Aug 9th, 2013
 ************************************************************************/

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include "neuron.h"

/* size(x) == n, size(pn->weights) == n+1
 * The last element of pn->weights is the threshold
 */
double linear(struct neuron *pn, double *x)
{
	double y = 0.0, *w = pn->weights;
	size_t i;
	for (i = 0; i < pn->nw - 1; ++i) {
		y += w[i] * x[i];
	}
	y += w[pn->nw-1];
	return y;
}

double sigmoid(struct neuron *pn, double *x)
{
	return 1.0 / (1.0 + exp(-linear(pn, x)));
}

double sigmoid_d(struct neuron *n, double *x, double y)
{
	return (1.0 - y) * y;
}

double tgh(struct neuron *n, double *x)
{
	double y = linear(n, x);
	double a = exp(y), b = exp(-y);
	return (a - b) / (a + b);
}

double tanh_d(struct neuron *n, double *x, double y)
{
	return 1.0 - y * y;
}

void update(struct neuron *n, double *x, double del_w, double momentum)
{
	size_t i;
	double w;
	for (i = 0; i < n->nw - 1; ++i) {
		w = x[i] * del_w; 
		n->weights[i] += w + momentum * n->odw[i];
		n->odw[i] = w;
	}
	n->weights[n->nw - 1] += del_w + momentum * n->odw[n->nw - 1];
	n->odw[n->nw - 1] = del_w;
}
