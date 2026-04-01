#include <math.h>
#include "utils.h"

// Problem 01
void matrix_multiplication(double** m1, double** m2, double** result, int N)
{
    int i, j, k;

    // Standard i-j-k loop order
    for (j = 0; j < N; j++) {
        for (k = 0; k < N; k++) {
            for (i = 0; i < N; i++) {
                result[i][j] += m1[i][k] * m2[k][j];
            }
        }
    }
}

// Problem 02
void transpose(double** m, double** mt, int N)
{
    int i, j;

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            mt[j][i] = m[i][j];
        }
    }
}

void transposed_matrix_multiplication(double** m1, double** m2, double** result, int N)
{
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += m1[i][k] * m2[j][k];
            }
            result[i][j] = sum;
        }
    }
}

// Problem 03
void block_matrix_multiplication(double** m1, double** m2, double** result, int B, int N)
{

    // Blocked multiplication
    for (int ii = 0; ii < N; ii += B) {
        for (int jj = 0; jj < N; jj += B) {
            for (int kk = 0; kk < N; kk += B) {

                for (int i = ii; i < ii + B && i < N; i++) {
                    for (int j = jj; j < jj + B && j < N; j++) {
                    	double sum = 0.0;
                        for (int k = kk; k < kk + B && k < N; k++) {
                            sum += m1[i][k] * m2[k][j];
                        }
                        result[i][j] = sum;
                    }
                }

            }
        }
    }
}
