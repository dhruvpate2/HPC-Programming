#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "utils.h"

extern int GRID_X, GRID_Y, NX, NY;
extern int NUM_Points;
extern double dx, dy;

void interpolation(double *mesh_value, Points *points)
{
    const double inv_dx = 1.0 / dx;
    const double inv_dy = 1.0 / dy;

    for (int p = 0; p < NUM_Points; p++)
    {
        double x = points[p].x;
        double y = points[p].y;
        double f = 1.0;   // As specified in assignment (fi = 1)

        // Compute cell indices
        int i = (int)(x * inv_dx);
        int j = (int)(y * inv_dy);

        // Boundary handling (important!)
        if (i >= NX) i = NX - 1;
        if (j >= NY) j = NY - 1;
        if (i < 0) i = 0;
        if (j < 0) j = 0;

        // Grid node coordinates
        double Xi = i * dx;
        double Yj = j * dy;

        // Local distances
        double lx = x - Xi;
        double ly = y - Yj;

        // Bilinear weights
        double w00 = (dx - lx) * (dy - ly);
        double w10 = lx * (dy - ly);
        double w01 = (dx - lx) * ly;
        double w11 = lx * ly;

        // Convert 2D to 1D indexing
        int idx00 = j * GRID_X + i;
        int idx10 = j * GRID_X + (i + 1);
        int idx01 = (j + 1) * GRID_X + i;
        int idx11 = (j + 1) * GRID_X + (i + 1);

        // Accumulate values
        mesh_value[idx00] += w00 * f;
        mesh_value[idx10] += w10 * f;
        mesh_value[idx01] += w01 * f;
        mesh_value[idx11] += w11 * f;
    }
}


// Write mesh to file
void save_mesh(double *mesh_value) {

    FILE *fd = fopen("Mesh.out", "w");
    if (!fd) {
        printf("Error creating Mesh.out\n");
        exit(1);
    }

    for (int i = 0; i < GRID_Y; i++) {
        for (int j = 0; j < GRID_X; j++) {
            fprintf(fd, "%lf ", mesh_value[i * GRID_X + j]);
        }
        fprintf(fd, "\n");
    }

    fclose(fd);
}
