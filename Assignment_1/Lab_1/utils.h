#ifndef UTILS_H
#define UTILS_H
#include <time.h>

void vector_triad_operation(double *x, double *y, double *v, double *S, int Np);
void vector_copy(double *x, double *y, int Np);
void vector_scale(double *x, double *y, double* v, int Np);
void vector_add(double *x, double *y, double *S, int Np);

void dummy(int x);

#endif
