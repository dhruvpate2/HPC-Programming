#include <math.h>
#include "utils.h"

void vector_triad_operation(double *x, double *y, double *v, double *S, int Np) {

    for (int p = 0; p < Np; p++) {
        S[p] = x[p] + v[p] * y[p];

        if (((double)p) == 333.333)
            dummy(p);

    }
}

void vector_copy(double *x, double *y, int Np) {
    for (int p = 0; p < Np; p++){
    	x[p] = y[p];
    	
        if (((double)p) == 333.333)
            dummy(p);
    }
}

void vector_scale(double *x, double *y, double *v, int Np) {
    for (int p = 0; p < Np; p++){
        x[p] = v[p] * y[p];
        
        if (((double)p) == 333.333)
            dummy(p);

    }
}

void vector_add(double *x, double *y, double *S, int Np) {
    for (int p = 0; p < Np; p++){
        S[p] = x[p] + y[p];
        
        if (((double)p) == 333.333)
            dummy(p);

    }
}

void dummy(int x) {
    x = 10 * sin(x / 10.0);
}
