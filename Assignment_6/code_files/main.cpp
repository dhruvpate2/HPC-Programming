#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "init.h"
#include "utils.h"

// Global variables
int GRID_X, GRID_Y, NX, NY;
int NUM_Points, Maxiter;
double dx, dy;

int main(int argc, char **argv) {

    if (argc != 3) {
        printf("Usage: %s <input_file> <num_threads>\n", argv[0]);
        printf("\nExample: %s input.bin 8 \n", argv[0]);
        return 1;
    }

    int num_threads = atoi(argv[2]);
    if (num_threads < 1) num_threads = 1;

    FILE *file = fopen(argv[1], "rb");
    if (!file) {
        printf("Error opening input file\n");
        exit(1);
    }

    fread(&NX, sizeof(int), 1, file);
    fread(&NY, sizeof(int), 1, file);

    fread(&NUM_Points, sizeof(int), 1, file);
    fread(&Maxiter, sizeof(int), 1, file);

    GRID_X = NX + 1;
    GRID_Y = NY + 1;
    dx = 1.0 / NX;
    dy = 1.0 / NY;

    double *mesh_value = (double *) calloc(GRID_X * GRID_Y, sizeof(double));
    Points *points = (Points *) calloc(NUM_Points, sizeof(Points));

    double total_time = 0.0;

    omp_set_num_threads(num_threads);

    // Determine the mode name for logging
    const char* mode_name = "";
    if (num_threads == 1) {
        mode_name = "Serial";
        num_threads = 1; 
    }
    else{
        mode_name = "Privatization";
    }

    printf("Running %s mode with %d threads...\n", mode_name, num_threads);

    for (int iter = 0; iter < Maxiter; iter++) {
        
        read_points(file, points);
        
        memset(mesh_value, 0, sizeof(double) * GRID_X * GRID_Y);

        double start = omp_get_wtime();

        // Execute the chosen approach
        if (num_threads == 1) {
            interpolation_serial(mesh_value, points);
        } else{
            interpolation_parallel(mesh_value, points);
        }

        // Stop timing
        double end = omp_get_wtime();
        total_time += (end - start);
    }

    save_mesh(mesh_value);
    
    printf("Mode: %-15s | Threads: %-2d | Time: %lf sec\n", mode_name, num_threads, total_time);

    FILE *csv_file = fopen("results.csv", "a");
    if (csv_file) {
        fseek(csv_file, 0, SEEK_END);
        long size = ftell(csv_file);
        if (size == 0) {
            fprintf(csv_file, "NX,NY,Particles,Iterations,Mode,Threads,Time_sec\n");
        }
        
        fprintf(csv_file, "%d,%d,%d,%d,%s,%d,%lf\n", 
                NX, NY, NUM_Points, Maxiter, mode_name, num_threads, total_time);
        
        fclose(csv_file);
        printf("Result appended to results.csv\n");
    } else {
        printf("Warning: Could not open results.csv for writing.\n");
    }

    free(mesh_value);
    free(points);
    fclose(file);

    return 0;
}
