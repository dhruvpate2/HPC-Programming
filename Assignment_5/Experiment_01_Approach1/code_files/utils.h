#ifndef UTILS_H
#define UTILS_H
#include <time.h>
#include <stdbool.h>
#include "init.h"

void interpolation(double *mesh_value, Points *points);
int mover_immediate_serial(Points *points, double deltaX, double deltaY);
int mover_immediate_parallel(Points *points, double deltaX, double deltaY);
int mover_deferred_serial(Points *points, double deltaX, double deltaY);
int mover_deferred_parallel(Points *points, double deltaX, double deltaY, bool *is_void);
void save_mesh(double *mesh_value);

#endif
