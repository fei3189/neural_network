/*************************************************************************
    > File Name: alloc.c
    > Author: Felix Jiang
    > Mail: f91.jiang@gmail.com 
    > Created Time: Aug 9th, 2013
 ************************************************************************/

#include <stdlib.h>

double* alloc_1d(size_t d)
{
    double* buf = (double*)malloc(sizeof(double) * d);
    return buf;
}

double** alloc_2d(size_t d1, size_t d2)
{
    double** buf = (double**)malloc(sizeof(double*) * d1);
    int i;
    for (i = 0; i < d1; ++i)
        buf[i] = (double*)malloc(sizeof(double) * d2);
    return buf;
}

double** alloc_2dv(const size_t *dim, int len)
{
    double** buf = (double**)malloc(sizeof(double*) * len);
    int i;
    for (i = 0; i < len; ++i)
        buf[i] = (double*)malloc(sizeof(double) * dim[i]);
    return buf;
}

double*** alloc_3d(size_t d1, size_t d2, size_t d3)
{
    double*** buf = (double***)malloc(sizeof(double**) * d1);
    int i, j;
    for (i = 0; i < d1; ++i) {
        buf[i] = (double**)malloc(sizeof(double*) * d2);
        for (j = 0; j < d2; ++j)
            buf[i][j] = (double*)malloc(sizeof(double) * d3);
    }
    return buf;
}

void free_1d(double *buf)
{
    free(buf);
}

void free_2d(double **buf, size_t d1)
{
    int i;
    for (i = 0; i < d1; ++i)
        free(buf[i]);
    free(buf);
}

void free_3d(double ***buf, size_t d1, size_t d2)
{
    int i, j;
    for (i = 0; i < d1; ++i) {
        for (j = 0; j < d2; ++j)
            free(buf[i][j]);
        free(buf[i]);
    }
    free(buf);
}

