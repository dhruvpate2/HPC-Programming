#ifndef INIT_H
#define INIT_H

void init_matrices(int Np, double ***m1, double ***m2, double ***result);
void free_matrices(int Np, double** m1, double** m2, double** result);
double** alloc_matrix(int N);
void free_matrix(double** m, int N);

#endif
