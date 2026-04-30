#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <mpi.h> // Include MPI
#include <float.h>
#include "utils.h"

extern int GRID_X, GRID_Y, NX, NY;
extern double dx, dy;
double min_val, max_val;

// Interpolation using OpenMP Atomics + MPI_Allreduce
void interpolation(double *mesh_value, Points *points, int my_NUM_Points) {
    // 1. Clear the grid
    memset(mesh_value, 0, GRID_X * GRID_Y * sizeof(double));

    // 2. Local OpenMP Parallelization (Switched to atomics to fix memory bottleneck)
    #pragma omp parallel for
    for (int p = 0; p < my_NUM_Points; p++) {
        if (points[p].is_void) continue;

        double x = points[p].x;
        double y = points[p].y;

        int i = (int)(x / dx);
        int j = (int)(y / dy);

        if (i < 0 || i >= NX || j < 0 || j >= NY) continue;

        double lx = x - (i * dx);
        double ly = y - (j * dy);

        double w00 = (dx - lx) * (dy - ly);
        double w10 = lx * (dy - ly);
        double w01 = (dx - lx) * ly;
        double w11 = lx * ly;

        #pragma omp atomic
        mesh_value[j * GRID_X + i] += w00;
        #pragma omp atomic
        mesh_value[j * GRID_X + (i + 1)] += w10;
        #pragma omp atomic
        mesh_value[(j + 1) * GRID_X + i] += w01;
        #pragma omp atomic
        mesh_value[(j + 1) * GRID_X + (i + 1)] += w11;
    }

    // 3. MPI Data Communication across nodes
    // Sum everyone's partial grid into a fully accurate global grid 
    MPI_Allreduce(MPI_IN_PLACE, mesh_value, GRID_X * GRID_Y, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

// Normalize mesh values to [-1, 1] using OpenMP Reductions
void normalization(double *mesh_value) {
    min_val = DBL_MAX;
    max_val = -DBL_MAX;

    // Find min and max using parallel reduction
    #pragma omp parallel for reduction(min:min_val) reduction(max:max_val)
    for (int k = 0; k < GRID_X * GRID_Y; k++) {
        if (mesh_value[k] < min_val) min_val = mesh_value[k];
        if (mesh_value[k] > max_val) max_val = mesh_value[k];
    }

    if (max_val == min_val) return; // Prevent division by zero

    // Normalize to [-1, 1]
    #pragma omp parallel for
    for (int k = 0; k < GRID_X * GRID_Y; k++) {
        mesh_value[k] = 2.0 * ((mesh_value[k] - min_val) / (max_val - min_val)) - 1.0;
    }
}

// Mover (Gather): Mesh -> Point using Particle Decomposition
void mover(double *mesh_value, Points *points, int my_NUM_Points) {
    // Embarrassingly parallel: Each particle reads grid and updates itself independently
    #pragma omp parallel for
    for (int p = 0; p < my_NUM_Points; p++) {
        if (points[p].is_void) continue;

        double x = points[p].x;
        double y = points[p].y;

        int i = (int)(x / dx);
        int j = (int)(y / dy);

        if (i < 0 || i >= NX || j < 0 || j >= NY) continue;

        double lx = x - (i * dx);
        double ly = y - (j * dy);

        double w00 = (dx - lx) * (dy - ly);
        double w10 = lx * (dy - ly);
        double w01 = (dx - lx) * ly;
        double w11 = lx * ly;

        // Gather normalized grid values
        double F = w00 * mesh_value[j * GRID_X + i] +
                   w10 * mesh_value[j * GRID_X + (i + 1)] +
                   w01 * mesh_value[(j + 1) * GRID_X + i] +
                   w11 * mesh_value[(j + 1) * GRID_X + (i + 1)];

        // Update particle positions
        points[p].x += F * dx;
        points[p].y += F * dy;

        // Mark inactive if outside domain
        if (points[p].x < 0.0 || points[p].x > 1.0 || points[p].y < 0.0 || points[p].y > 1.0) {
            points[p].is_void = true;
        }
    }
}

// Denormalize the mesh back to original bounds
void denormalization(double *mesh_value) {
    if (max_val == min_val) return;

    #pragma omp parallel for
    for (int k = 0; k < GRID_X * GRID_Y; k++) {
        mesh_value[k] = ((mesh_value[k] + 1.0) / 2.0) * (max_val - min_val) + min_val;
    }
}

long long int void_count(Points *points, int my_NUM_Points) {
    long long int voids = 0;
    #pragma omp parallel for reduction(+:voids)
    for (int i = 0; i < my_NUM_Points; i++) {
        if(points[i].is_void) voids++;
    }
    return voids;
}

// Write mesh to file (Serial operation, I/O bound)
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
