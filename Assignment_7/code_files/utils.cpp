#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <float.h>
#include "utils.h"

// Bring in global variables from main.cpp
extern int GRID_X, GRID_Y, NX, NY;
extern int NUM_Points, Maxiter;
extern double dx, dy;

double min_val, max_val;

// Interpolation (Scatter): Point -> Mesh using Privatization
void interpolation(double *mesh_value, Points *points) {
    // 1. Clear the global mesh first
    memset(mesh_value, 0, GRID_X * GRID_Y * sizeof(double));

    // 2. Parallelize using Privatization
    #pragma omp parallel
    {
        // Allocate a local mesh for each thread initialized to 0
        double *local_mesh = (double *)calloc(GRID_X * GRID_Y, sizeof(double));

        #pragma omp for
        for (int p = 0; p < NUM_Points; p++) {
            if (points[p].is_void) continue;

            double x = points[p].x;
            double y = points[p].y;

            // Find bottom-left grid indices
            int i = (int)(x / dx);
            int j = (int)(y / dy);

            // Boundary safeguard
            if (i < 0 || i >= NX || j < 0 || j >= NY) continue;

            // Local distances within the cell
            double lx = x - (i * dx);
            double ly = y - (j * dy);

            // Calculate weights
            double w00 = (dx - lx) * (dy - ly);
            double w10 = lx * (dy - ly);
            double w01 = (dx - lx) * ly;
            double w11 = lx * ly;

            // f_i = 1, so we just add the weights
            local_mesh[j * GRID_X + i] += w00;
            local_mesh[j * GRID_X + (i + 1)] += w10;
            local_mesh[(j + 1) * GRID_X + i] += w01;
            local_mesh[(j + 1) * GRID_X + (i + 1)] += w11;
        }

        // 3. Reduce local meshes back to the global mesh
        #pragma omp critical
        {
            for (int k = 0; k < GRID_X * GRID_Y; k++) {
                mesh_value[k] += local_mesh[k];
            }
        }
        free(local_mesh);
    }
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

// Serial Mover (Gather): Mesh -> Point
void mover_serial(double *mesh_value, Points *points) {
    for (int p = 0; p < NUM_Points; p++) {
        // Skip particles that have already left the domain
        if (points[p].is_void) continue;

        double x = points[p].x;
        double y = points[p].y;

        // Find bottom-left grid indices
        int i = (int)(x / dx);
        int j = (int)(y / dy);

        // Boundary safeguard to prevent segmentation faults
        if (i < 0 || i >= NX || j < 0 || j >= NY) continue;

        // Local distances within the cell
        double lx = x - (i * dx);
        double ly = y - (j * dy);

        // Calculate weights
        double w00 = (dx - lx) * (dy - ly);
        double w10 = lx * (dy - ly);
        double w01 = (dx - lx) * ly;
        double w11 = lx * ly;

        // Gather normalized grid values [cite: 119]
        double F = w00 * mesh_value[j * GRID_X + i] +
                   w10 * mesh_value[j * GRID_X + (i + 1)] +
                   w01 * mesh_value[(j + 1) * GRID_X + i] +
                   w11 * mesh_value[(j + 1) * GRID_X + (i + 1)];

        // Update particle positions [cite: 120]
        points[p].x += F * dx;
        points[p].y += F * dy;

        // Mark inactive if outside domain (domain is 1x1) 
        if (points[p].x < 0.0 || points[p].x > 1.0 || points[p].y < 0.0 || points[p].y > 1.0) {
            points[p].is_void = true;
        }
    }
}

// Mover (Gather): Mesh -> Point using Particle Decomposition
void mover(double *mesh_value, Points *points) {
    // Embarrassingly parallel: Each particle reads grid and updates itself independently
    #pragma omp parallel for
    for (int p = 0; p < NUM_Points; p++) {
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

// Count particles that went beyond the domain using Reduction
long long int void_count(Points *points) {
    long long int voids = 0;
    
    #pragma omp parallel for reduction(+:voids)
    for (int i = 0; i < NUM_Points; i++) {
        if(points[i].is_void) {
            voids++;
        }
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
