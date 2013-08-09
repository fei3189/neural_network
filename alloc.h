/*************************************************************************
	> File Name: alloc.h
	> Author: Felix Jiang
	> Mail: f91.jiang@gmail.com 
	> Created Time: Aug 9th, 2013
 ************************************************************************/

#if !defined(_ALLOC_H_)
#define _ALLOC_H_

#include <stdlib.h>

double* alloc_1d(size_t size);

double** alloc_2d(size_t d1, size_t d2);

double** alloc_2dv(size_t *dim, size_t len);

double*** alloc_3d(size_t d1, size_t d2, size_t d3);

void free_1d(double *buf);

void free_2d(double **buf, size_t d1);

void free_3d(double ***buf, size_t d1, size_t d2);
#endif
