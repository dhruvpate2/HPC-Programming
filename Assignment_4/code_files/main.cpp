#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "init.h"
#include "utils.h"

int GRID_X, GRID_Y, NX, NY;
int NUM_Points, Maxiter;
double dx, dy;

int main(int argc, char **argv) {

    int experiment_id = 0;

    /*
    // --- EXPERIMENT 01 CONFIGURATION ---
    experiment_id = 1;
    NX = 1000; // Modify to: 250, 500, or 1000
    NY = 400; // Modify to: 100, 200, or 400
    NUM_Points = 1e8; // Modify to: 10^2, 10^4, 10^6, 10^8, or 10^9
    Maxiter = 10;
    */

    /*
    // --- EXPERIMENT 02 CONFIGURATION ---
    experiment_id = 2;
    NX = 1000; // Modify to: 250, 500, or 1000
    NY = 400; // Modify to: 100, 200, or 400
    NUM_Points = 1e8; // Strictly fixed to 10^8 for Exp 2
    Maxiter = 10;
    */

    ///*
    // --- EXPERIMENT 03 CONFIGURATION ---
    experiment_id = 3;
    NX = 1000; // Fixed for Exp 3
    NY = 400; // Fixed for Exp 3
    NUM_Points = 14000000; // Fixed at 14 million for Exp 3
    Maxiter = 10; // Fixed
    omp_set_num_threads(4); // Fixed to 4 threads for the parallel phase
    //*/

    // Grid nodes are 1 more than the number of cells
    GRID_X = NX + 1;
    GRID_Y = NY + 1;
    dx = 1.0 / NX;
    dy = 1.0 / NY;

    double *mesh_value = (double *) calloc(GRID_X * GRID_Y, sizeof(double));
    Points *points = (Points *) calloc(NUM_Points, sizeof(Points));

    double total_interp_time = 0.0;
    double total_mover_time = 0.0;

    printf("Running Exp %d: NX=%d, NY=%d, Particles=%d, Maxiter=%d\n", 
            experiment_id, NX, NY, NUM_Points, Maxiter);

    if (experiment_id == 2 || experiment_id == 3) {
        initializepoints(points);
    }

    // --- CSV FILE SETUP FOR EXP 3 ---
    FILE *csv_file = NULL;
    const char* filename = "exp3_parallel_hpc.csv"; 

    if (experiment_id == 3) {
        printf("Iter\tInterp_Time\tMover_Time\tTotal_Time\n");
        csv_file = fopen(filename, "w");
        if (csv_file == NULL) {
            printf("Error opening %s for writing!\n", filename);
            free(mesh_value);
            free(points);
            return 1;
        }
        // Write the CSV header
        fprintf(csv_file, "Iteration,Interpolation_Time,Mover_Time,Total_Time\n");
    }

    // MAIN EXECUTION LOOP
    for (int iter = 0; iter < Maxiter; iter++) {

        if (experiment_id == 1) {
            initializepoints(points);
        }

        // --- 1. INTERPOLATION PHASE (Runs for all experiments) ---
        double start_interp = omp_get_wtime();
        interpolation(mesh_value, points);
        double end_interp = omp_get_wtime();
        
        double interp_time = end_interp - start_interp;
        total_interp_time += interp_time;

        // --- 2. MOVER PHASE (Runs ONLY for Experiment 3) ---
        if (experiment_id == 3) {
            double start_move = omp_get_wtime();
            
            // Toggle between Serial and Parallel Mover here:
            //mover_serial(points, dx, dy);
            mover_parallel(points, dx, dy);
            
            double end_move = omp_get_wtime();
            
            double move_time = end_move - start_move;
            double iter_total_time = interp_time + move_time;
            
            total_mover_time += move_time;

            printf("%d\t%lf\t%lf\t%lf\n", iter + 1, interp_time, move_time, iter_total_time);
            
            // Write to CSV file
            fprintf(csv_file, "%d,%lf,%lf,%lf\n", iter + 1, interp_time, move_time, iter_total_time);
        }
    }

    // FINAL RESULTS OUTPUT
    printf("\n--- Final Accumulations ---\n");
    if (experiment_id == 1 || experiment_id == 2) {
        printf("Total Accumulated Interpolation Time: %lf seconds\n", total_interp_time);
    } else if (experiment_id == 3) {
        printf("Total Interpolation Time: %lf seconds\n", total_interp_time);
        printf("Total Mover Time: %lf seconds\n", total_mover_time);
        printf("Overall Total Time: %lf seconds\n", total_interp_time + total_mover_time);
        
        // Close the CSV file safely
        fclose(csv_file);
        printf("\nData successfully written to %s\n", filename);
    }

    free(mesh_value);
    free(points);

    return 0;
}
