#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "utils.h"

extern int GRID_X, GRID_Y, NX, NY;
extern int NUM_Points;
extern double dx, dy;

void interpolation_serial(double *mesh_value, Points *points)
{
    const double inv_dx = 1.0 / dx;
    const double inv_dy = 1.0 / dy;

    for (int p = 0; p < NUM_Points; p++)
    {
        double x = points[p].x;
        double y = points[p].y;
        double f = 1.0;   

        int i = (int)(x * inv_dx);
        int j = (int)(y * inv_dy);

        if (i >= NX) i = NX - 1;
        if (j >= NY) j = NY - 1;
        if (i < 0) i = 0;
        if (j < 0) j = 0;

        double Xi = i * dx;
        double Yj = j * dy;
        double lx = x - Xi;
        double ly = y - Yj;

        double w00 = (dx - lx) * (dy - ly);
        double w10 = lx * (dy - ly);
        double w01 = (dx - lx) * ly;
        double w11 = lx * ly;

        int idx00 = j * GRID_X + i;
        int idx10 = j * GRID_X + (i + 1);
        int idx01 = (j + 1) * GRID_X + i;
        int idx11 = (j + 1) * GRID_X + (i + 1);

        mesh_value[idx00] += w00 * f;
        mesh_value[idx10] += w10 * f;
        mesh_value[idx01] += w01 * f;
        mesh_value[idx11] += w11 * f;
    }
}

void interpolation_parallel(double *mesh_value, Points *points)
{
    const double inv_dx = 1.0 / dx;
    const double inv_dy = 1.0 / dy;

    #pragma omp parallel
    {
        double *local_mesh = (double *)calloc(GRID_X * GRID_Y, sizeof(double));

        #pragma omp for
        for (int p = 0; p < NUM_Points; p++)
        {
            double x = points[p].x;
            double y = points[p].y;
            double f = 1.0;   

            int i = (int)(x * inv_dx);
            int j = (int)(y * inv_dy);

            if (i >= NX) i = NX - 1;
            if (j >= NY) j = NY - 1;
            if (i < 0) i = 0;
            if (j < 0) j = 0;

            double Xi = i * dx;
            double Yj = j * dy;
            double lx = x - Xi;
            double ly = y - Yj;

            double w00 = (dx - lx) * (dy - ly);
            double w10 = lx * (dy - ly);
            double w01 = (dx - lx) * ly;
            double w11 = lx * ly;

            int idx00 = j * GRID_X + i;
            int idx10 = j * GRID_X + (i + 1);
            int idx01 = (j + 1) * GRID_X + i;
            int idx11 = (j + 1) * GRID_X + (i + 1);

            // Accumulate onto local mesh (no race conditions)
            local_mesh[idx00] += w00 * f;
            local_mesh[idx10] += w10 * f;
            local_mesh[idx01] += w01 * f;
            local_mesh[idx11] += w11 * f;
        }

        // Merge local grids into global grid
        #pragma omp critical
        {
            for (int i = 0; i < GRID_X * GRID_Y; i++) {
                mesh_value[i] += local_mesh[i];
            }
        }
        free(local_mesh);
    }
}

void interpolation_domain_decomp(double *mesh_value, Points *points)
{
    const double inv_dx = 1.0 / dx;
    const double inv_dy = 1.0 / dy;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        int rows_per_thread = GRID_Y / num_threads;
        int start_J = tid * rows_per_thread;
        int end_J = (tid == num_threads - 1) ? GRID_Y : start_J + rows_per_thread;

        for (int p = 0; p < NUM_Points; p++)
        {
            double x = points[p].x;
            double y = points[p].y;
            double f = 1.0;

            int i = (int)(x * inv_dx);
            int j = (int)(y * inv_dy);

            if (i >= NX) i = NX - 1;
            if (j >= NY) j = NY - 1;
            if (i < 0) i = 0;
            if (j < 0) j = 0;

            if ((j >= start_J && j < end_J) || (j + 1 >= start_J && j + 1 < end_J))
            {
                double Xi = i * dx;
                double Yj = j * dy;
                double lx = x - Xi;
                double ly = y - Yj;

                double w00 = (dx - lx) * (dy - ly);
                double w10 = lx * (dy - ly);
                double w01 = (dx - lx) * ly;
                double w11 = lx * ly;

                int idx00 = j * GRID_X + i;
                int idx10 = j * GRID_X + (i + 1);
                int idx01 = (j + 1) * GRID_X + i;
                int idx11 = (j + 1) * GRID_X + (i + 1);
                
                if (j >= start_J && j < end_J) {
                    mesh_value[idx00] += w00 * f;
                    mesh_value[idx10] += w10 * f;
                }
                
                if (j + 1 >= start_J && j + 1 < end_J) {
                    mesh_value[idx01] += w01 * f;
                    mesh_value[idx11] += w11 * f;
                }
            }
        }
    }
}

void interpolation_atomic(double *mesh_value, Points *points)
{
    const double inv_dx = 1.0 / dx;
    const double inv_dy = 1.0 / dy;

    #pragma omp parallel for
    for (int p = 0; p < NUM_Points; p++)
    {
        double x = points[p].x;
        double y = points[p].y;
        double f = 1.0;   

        // Compute cell indices
        int i = (int)(x * inv_dx);
        int j = (int)(y * inv_dy);

        // Boundary handling
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

        // ATOMIC UPDATES
        #pragma omp atomic
        mesh_value[idx00] += w00 * f;
        
        #pragma omp atomic
        mesh_value[idx10] += w10 * f;
        
        #pragma omp atomic
        mesh_value[idx01] += w01 * f;
        
        #pragma omp atomic
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
