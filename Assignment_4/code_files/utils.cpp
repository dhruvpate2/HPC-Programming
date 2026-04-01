#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <math.h>
#include "utils.h"

// Interpolation (Serial Code)
void interpolation(double *mesh_value, Points *points) {

    // Reset mesh
    memset(mesh_value, 0, GRID_X * GRID_Y * sizeof(double));

    for (long p = 0; p < NUM_Points; p++) {

        // Convert particle position to grid coordinates
        double gx = points[p].x * NX;
        double gy = points[p].y * NY;

        int i = (int)gx;
        int j = (int)gy;

        // Ensure valid cell index
        if (i >= NX) i = NX - 1;
        if (j >= NY) j = NY - 1;

        double wx = gx - i;
        double wy = gy - j;

        // Bilinear weights
        double w1 = (1.0 - wx) * (1.0 - wy);
        double w2 = wx * (1.0 - wy);
        double w3 = (1.0 - wx) * wy;
        double w4 = wx * wy;

        // Update four surrounding grid points
        mesh_value[j * GRID_X + i]         += w1;
        mesh_value[j * GRID_X + (i + 1)]   += w2;
        mesh_value[(j + 1) * GRID_X + i]   += w3;
        mesh_value[(j + 1) * GRID_X + i+1] += w4;
    }
}


// Stochastic Mover (Serial Code)
void mover_serial(Points *points, double deltaX, double deltaY) {

    for (long p = 0; p < NUM_Points; p++) {

        double newx, newy;

        do {
            double rx = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
            double ry = ((double)rand() / RAND_MAX) * 2.0 - 1.0;

            newx = points[p].x + rx * deltaX;
            newy = points[p].y + ry * deltaY;

        } while (newx <= 0.0 || newx >= 1.0 ||
                 newy <= 0.0 || newy >= 1.0);

        points[p].x = newx;
        points[p].y = newy;
    }
}


// Stochastic Mover (Parallel Code)
void mover_parallel(Points *points, double deltaX, double deltaY) {

    #pragma omp parallel for
    for (long p = 0; p < NUM_Points; p++) {

        unsigned int seed = omp_get_thread_num() + p;

        double newx, newy;

        do {
            double rx = ((double)rand_r(&seed) / RAND_MAX) * 2.0 - 1.0;
            double ry = ((double)rand_r(&seed) / RAND_MAX) * 2.0 - 1.0;

            newx = points[p].x + rx * deltaX;
            newy = points[p].y + ry * deltaY;

        } while (newx <= 0.0 || newx >= 1.0 ||
                 newy <= 0.0 || newy >= 1.0);

        points[p].x = newx;
        points[p].y = newy;
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
