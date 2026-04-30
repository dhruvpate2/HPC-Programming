#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <mpi.h> // Include MPI

#include "init.h"
#include "utils.h"

int GRID_X, GRID_Y, NX, NY;
int Maxiter;
double dx, dy;
// We change NUM_Points to global, and add a local variable
int NUM_Points; 
int my_NUM_Points; 

int main(int argc, char **argv) {
    // 1. Initialize MPI
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int num_threads = omp_get_max_threads();
    if (argc == 3) {
        num_threads = atoi(argv[2]);
    }
    omp_set_num_threads(num_threads);

    FILE *file = fopen(argv[1], "rb");
    if (!file) {
        if(rank == 0) printf("Error opening input file\n");
        MPI_Finalize();
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

    // 2. Particle Distribution Logic
    int base_points = NUM_Points / size;
    int remainder = NUM_Points % size;
    my_NUM_Points = base_points + (rank < remainder ? 1 : 0);
    int my_offset = rank * base_points + (rank < remainder ? rank : remainder);

    // Every rank allocates the full grid, but ONLY its share of particles
    double *mesh_value = (double *) calloc(GRID_X * GRID_Y, sizeof(double));
    Points *points = (Points *) calloc(my_NUM_Points, sizeof(Points));

    // 3. Offset the file pointer to read local chunk (16 bytes for the 4 header ints)
    fseek(file, 16 + (my_offset * 2 * sizeof(double)), SEEK_SET);
    for (int i = 0; i < my_NUM_Points; i++) {
        fread(&points[i].x, sizeof(double), 1, file);
        fread(&points[i].y, sizeof(double), 1, file);
        points[i].is_void = false;   
    }
    fclose(file); // Close file after reading

    double total_int_time = 0.0, total_norm_time = 0.0;
    double total_move_time = 0.0, total_denorm_time = 0.0;

    for (int iter = 0; iter < Maxiter; iter++) {
        double t0 = omp_get_wtime();
        
        // Pass my_NUM_Points instead of global NUM_Points
        interpolation(mesh_value, points, my_NUM_Points); 

        double t1 = omp_get_wtime();
        normalization(mesh_value);
        
        double t2 = omp_get_wtime();
        mover(mesh_value, points, my_NUM_Points);
        
        double t3 = omp_get_wtime();
        denormalization(mesh_value);
        
        double t4 = omp_get_wtime();

        total_int_time += (t1 - t0);
        total_norm_time += (t2 - t1);
        total_move_time += (t3 - t2);
        total_denorm_time += (t4 - t3);
    }

    // Only rank 0 saves the mesh
    if (rank == 0) {
        save_mesh(mesh_value);
    }
    
    // Gather voids globally from all ranks
    long long int local_voids = void_count(points, my_NUM_Points);
    long long int global_voids = 0;
    MPI_Reduce(&local_voids, &global_voids, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // RANK 0: PRINT CONSOLE AND SAVE TO CSV
    if (rank == 0) {
        double total_alg_time = total_int_time + total_norm_time + total_move_time + total_denorm_time;
        
        // Print to console
        printf("Total Interpolation Time = %lf seconds\n", total_int_time);
        printf("Total Normalization Time = %lf seconds\n", total_norm_time);
        printf("Total Mover Time = %lf seconds\n", total_move_time);
        printf("Total Denormalization Time = %lf seconds\n", total_denorm_time);
        printf("Total Algorithm Time = %lf seconds\n", total_alg_time);
        printf("Total Number of Voids = %lld\n", global_voids);

        // CSV Logging Logic
        bool write_header = false;
        FILE *check_file = fopen("performance_results.csv", "r");
        if (!check_file) {
            write_header = true;
        } else {
            fclose(check_file);
        }

        FILE *csv_file = fopen("performance_results.csv", "a");
        if (csv_file) {
            if (write_header) {
                fprintf(csv_file, "Mode,MPI_Ranks,OMP_Threads,InputFile,InterpolationTime,NormalizationTime,MoverTime,DenormalizationTime,TotalTime,Voids\n");
            }
            
            // Log the hybrid performance data
            fprintf(csv_file, "Hybrid(MPI+OMP),%d,%d,%s,%lf,%lf,%lf,%lf,%lf,%lld\n",
                    size, num_threads, argv[1], total_int_time, total_norm_time, total_move_time, total_denorm_time, total_alg_time, global_voids);
            
            fclose(csv_file);
            printf("-> Performance data saved to performance_results.csv\n");
        } else {
            printf("-> Warning: Could not open performance_results.csv for writing.\n");
        }
    }

    // Free memory
    free(mesh_value);
    free(points);
    
    // Finalize MPI environment
    MPI_Finalize(); 
    return 0;
}
