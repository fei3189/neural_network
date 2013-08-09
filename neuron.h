/*************************************************************************
	> File Name: neuron.h
	> Author: Felix Jiang
	> Mail: f91.jiang@gmail.com 
	> Created Time: Aug 9th, 2013
 ************************************************************************/
#if !defined(_NEURON_H_)
#define _NEURON_H_

struct neuron {
	double* weights;
	double* odw;
	size_t nw;
	double (*compute)(struct neuron *n, double *x);
	double (*derivate)(struct neuron *n, double *x, double o);
	void (*update)(struct neuron *n, double *x, double del_w, double momentum);
};

double sigmoid(struct neuron *n, double *x);

double sigmoid_d(struct neuron *n, double *x, double y);

double tgh(struct neuron *n, double *x);

double tanh_d(struct neuron *n, double *x, double y);

void update(struct neuron *n, double *x, double del_w, double momentum);

#endif
