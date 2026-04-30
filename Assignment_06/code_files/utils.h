#ifndef UTILS_H
#define UTILS_H

#include "init.h"

void interpolation_serial(double *mesh_value, Points *points);
void interpolation_parallel(double *mesh_value, Points *points);
void interpolation_domain_decomp(double *mesh_value, Points *points);
void interpolation_atomic(double *mesh_value, Points *points);
void save_mesh(double *mesh_value);

#endif
