/*************************************************************************
	> File Name: neuron.h
	> Author: Felix Jiang
	> Mail: f91.jiang@gmail.com 
	> Created Time: Aug 9th, 2013
 ************************************************************************/
#if !defined(_NEURON_H_)
#define _NEURON_H_

#if defined(__cplusplus)
extern "C" {
#endif

struct neuron {
	/* weights of the linear part, the last element is the threshold*/
	double* weights;

	/* old delta_weight, for momentum in learning */
	double* odw;

	/* size of weights */
	size_t nw;

	/* activate function */
	double (*compute)(const struct neuron *n, const double *x);
	/* derivation of activate function */

	double (*derivate)(const struct neuron *n, const double *x, double o);

	void (*update)(struct neuron *n, const double *x, double del_w, double momentum);
};

/* sigmoid function */
double sigmoid(const struct neuron *n, const double *x);

/* derivation of sigmoid function */
double sigmoid_d(const struct neuron *n, const double *x, double y);

/* tanh function */
double tgh(const struct neuron *n, const double *x);

/* derivation of tanh function */
double tgh_d(const struct neuron *n, const double *x, double y);

/* update the weight of neurons,
 * x: the input vector,
 * weight += x * del_w 
 * */
void update(struct neuron *n, const double *x, double del_w, double momentum);

#if defined(__cplusplus)
}
#endif

#endif
