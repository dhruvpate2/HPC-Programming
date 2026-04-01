#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <stdbool.h> 

#include "init.h"
#include "utils.h"

int GRID_X, GRID_Y, NX, NY;
int NUM_Points, Maxiter;
double dx, dy;

int main(int argc, char **argv) {
    // HARDCODED FOR EXP 02: DEFERRED (APPROACH 1)
    int experiment_id = 2;                  
    const char* system_name = "Lab";        
    const char* approach_name = "Deferred"; 
    int config_id = 3;                      // Set to 1, 2, or 3
    
    NX = 1000;              
    NY = 400;               
    NUM_Points = 14000000;  // Fixed for Exp 2
    Maxiter = 10;           
    int num_threads = 4;    // Change to 2, 4, 8, 16 for testing

    omp_set_num_threads(num_threads); 

    GRID_X = NX + 1;
    GRID_Y = NY + 1;
    dx = 1.0 / NX;
    dy = 1.0 / NY;

    double *mesh_value = (double *) calloc(GRID_X * GRID_Y, sizeof(double));
    Points *points = (Points *) calloc(NUM_Points, sizeof(Points));
    bool *is_void = (bool *) calloc(NUM_Points, sizeof(bool));

    initializepoints(points);

    double total_interp_time = 0.0;
    double total_mover_time = 0.0;

    printf("Running Exp %d: %s System | %s Approach | Config %d | Particles %d | Threads %d\n", 
           experiment_id, system_name, approach_name, config_id, NUM_Points, num_threads);
    printf("Iter\tInterp\t\tMover\t\tTotal\t\tDeleted\n");
    
    for (int iter = 0; iter < Maxiter; iter++) {
        double start_interp = omp_get_wtime();
        interpolation(mesh_value, points);
        double interp_time = omp_get_wtime() - start_interp;

        double start_move = omp_get_wtime();
        int deleted_count = 0;
        
        if (num_threads == 1) deleted_count = mover_deferred_serial(points, dx, dy);
        else deleted_count = mover_deferred_parallel(points, dx, dy, is_void);
        
        double move_time = omp_get_wtime() - start_move;
        total_interp_time += interp_time;
        total_mover_time += move_time;

        printf("%d\t%lf\t%lf\t%lf\t%d\n", iter+1, interp_time, move_time, interp_time + move_time, deleted_count);
    }
    
    save_mesh(mesh_value);
    
    double overall_time = total_interp_time + total_mover_time;
    printf("\nOVERALL EXECUTION TIME: %lf seconds\n", overall_time);

    const char* filename = "exp2_speedup_results.csv";
    FILE *csv_file = fopen(filename, "a");
    if (csv_file != NULL) {
        fseek(csv_file, 0, SEEK_END);
        if (ftell(csv_file) == 0) {
            fprintf(csv_file, "System,Approach,Config,Particles,Threads,Total_Interp,Total_Mover,Overall_Time\n");
        }
        fprintf(csv_file, "%s,%s,%d,%d,%d,%lf,%lf,%lf\n", 
                system_name, approach_name, config_id, NUM_Points, num_threads, 
                total_interp_time, total_mover_time, overall_time);
        fclose(csv_file);
    }
    
    free(mesh_value);
    free(points);
    free(is_void);
    return 0;
}