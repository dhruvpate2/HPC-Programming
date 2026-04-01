#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h> // Required for NAN and isnan()
#include <omp.h>
#include <stdbool.h>
#include "utils.h"

extern int GRID_X, GRID_Y;
extern int NUM_Points;
extern double dx, dy;


void interpolation(double *mesh_value, Points *points) {
    memset(mesh_value, 0, GRID_X * GRID_Y * sizeof(double));
    for (int p = 0; p < NUM_Points; p++) {
        double x = points[p].x;
        double y = points[p].y;
        int i = (int)(x / dx);
        int j = (int)(y / dy);
        double wx = (x - i * dx) / dx;
        double wy = (y - j * dy) / dy;
        
        if (i < GRID_X - 1 && j < GRID_Y - 1) {
            mesh_value[j * GRID_X + i] += (1.0 - wx) * (1.0 - wy);
            mesh_value[j * GRID_X + (i + 1)] += wx * (1.0 - wy);
            mesh_value[(j + 1) * GRID_X + i] += (1.0 - wx) * wy;
            mesh_value[(j + 1) * GRID_X + (i + 1)] += wx * wy;
        }
    }
}


int mover_immediate_serial(Points *points, double deltaX, double deltaY) {
    int deleted_count = 0;
    for (int p = 0; p < NUM_Points; p++) {
        double disp_x = ((double)rand() / RAND_MAX * 2.0 - 1.0) * deltaX;
        double disp_y = ((double)rand() / RAND_MAX * 2.0 - 1.0) * deltaY;
        
        double new_x = points[p].x + disp_x;
        double new_y = points[p].y + disp_y;
        
        if (new_x < 0.0 || new_x > 1.0 || new_y < 0.0 || new_y > 1.0) {
            // Immediate replacement inside the domain
            points[p].x = (double)rand() / RAND_MAX;
            points[p].y = (double)rand() / RAND_MAX;
            deleted_count++;
        } else {
            points[p].x = new_x;
            points[p].y = new_y;
        }
    }
    return deleted_count;
}

int mover_immediate_parallel(Points *points, double deltaX, double deltaY) {
    int global_deleted_count = 0;

    #pragma omp parallel 
    {
        unsigned int seed = 2525 ^ omp_get_thread_num(); 
        int local_deleted = 0;
        
        #pragma omp for
        for (int p = 0; p < NUM_Points; p++) {
            double disp_x = ((double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0) * deltaX;
            double disp_y = ((double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0) * deltaY;
            
            double new_x = points[p].x + disp_x;
            double new_y = points[p].y + disp_y;
            
            if (new_x < 0.0 || new_x > 1.0 || new_y < 0.0 || new_y > 1.0) {
                points[p].x = (double)rand_r(&seed) / RAND_MAX;
                points[p].y = (double)rand_r(&seed) / RAND_MAX;
                local_deleted++;
            } else {
                points[p].x = new_x;
                points[p].y = new_y;
            }
        }
        
        // Safely accumulate the deleted counts
        #pragma omp atomic
        global_deleted_count += local_deleted;
    }
    return global_deleted_count;
}

int mover_deferred_serial(Points *points, double deltaX, double deltaY) {
    unsigned int seed = 12345; 
    double scaleX = 2.0 * deltaX / RAND_MAX;
    double scaleY = 2.0 * deltaY / RAND_MAX;

    int left = 0;
    int right = NUM_Points - 1;

    // PHASE 1 & 2 MERGED: Dynamic On-The-Fly Compaction
    while (left <= right) {
        // Evaluate the particle at the left pointer
        double disp_x = (rand_r(&seed) * scaleX) - deltaX;
        double disp_y = (rand_r(&seed) * scaleY) - deltaY;
        double new_x = points[left].x + disp_x;
        double new_y = points[left].y + disp_y;

        if (new_x >= 0.0 && new_x <= 1.0 && new_y >= 0.0 && new_y <= 1.0) {
            // Particle is valid. Update in-place and advance left pointer.
            points[left].x = new_x;
            points[left].y = new_y;
            left++;
        } else {
            // Particle at 'left' went out of bounds (creating a void).
            // Evaluate particles from the 'right' to find a valid one to swap into 'left'.
            while (left < right) {
                double dr_x = (rand_r(&seed) * scaleX) - deltaX;
                double dr_y = (rand_r(&seed) * scaleY) - deltaY;
                double nr_x = points[right].x + dr_x;
                double nr_y = points[right].y + dr_y;

                if (nr_x >= 0.0 && nr_x <= 1.0 && nr_y >= 0.0 && nr_y <= 1.0) {
                    // Found a valid particle on the right! 
                    // Move it directly into the void slot at 'left'.
                    points[left].x = nr_x;
                    points[left].y = nr_y;
                    left++;
                    right--;
                    break; // Hole filled, return to outer loop
                } else {
                    // Particle at 'right' is also a void. 
                    // It is already sitting at the end of the array, so just decrement right.
                    right--;
                }
            }
            
            if (left == right) {
                break; 
            }
        }
    }

    // 'left' perfectly represents the index where the contiguous block of voids begins
    int deleted_count = NUM_Points - left;
    double scale_insert = 1.0 / RAND_MAX;

    // PHASE 3: Insert new particles strictly into the void block at the end
    for (int p = left; p < NUM_Points; p++) {
        points[p].x = rand_r(&seed) * scale_insert;
        points[p].y = rand_r(&seed) * scale_insert;
    }

    return deleted_count;
}


int mover_deferred_parallel(Points *points, double deltaX, double deltaY, bool *is_void) {
    int num_threads = omp_get_max_threads();
    int *valid_counts = (int*)malloc(num_threads * sizeof(int));

    #pragma omp parallel 
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        unsigned int seed = 2525 ^ tid;
        
        int chunk_size = NUM_Points / nthreads;
        int start_idx = tid * chunk_size;
        int end_idx = (tid == nthreads - 1) ? NUM_Points : start_idx + chunk_size;
        
        for (int p = start_idx; p < end_idx; p++) {
            double disp_x = ((double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0) * deltaX;
            double disp_y = ((double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0) * deltaY;

            double new_x = points[p].x + disp_x;
            double new_y = points[p].y + disp_y;

            if (new_x >= 0.0 && new_x <= 1.0 && new_y >= 0.0 && new_y <= 1.0) {
                points[p].x = new_x;
                points[p].y = new_y;
                is_void[p] = false; 
            } else {
                is_void[p] = true;  
            }
        }
        
        int left = start_idx;
        int right = end_idx - 1;
        
        while (left <= right) {
            while (left <= right && !is_void[left]) left++;
            while (left <= right && is_void[right]) right--;
            
            if (left < right) {
                points[left] = points[right];
                is_void[left] = false;
                is_void[right] = true;
                left++;
                right--;
            }
        }
        
        valid_counts[tid] = left - start_idx;
    }
    
    int global_valid = 0;
    for (int i = 0; i < num_threads; i++) {
        global_valid += valid_counts[i];
    }
    
    int deleted_count = NUM_Points - global_valid; 
    
    int fill_ptr = 0; 
    int extract_ptr = NUM_Points - 1; 
    
    while (fill_ptr < global_valid && extract_ptr >= global_valid) {
        while (fill_ptr < global_valid && !is_void[fill_ptr]) fill_ptr++;
        while (extract_ptr >= global_valid && is_void[extract_ptr]) extract_ptr--;
        
        if (fill_ptr < global_valid && extract_ptr >= global_valid) {
            points[fill_ptr] = points[extract_ptr];
            is_void[fill_ptr] = false;
            is_void[extract_ptr] = true;
            fill_ptr++;
            extract_ptr--;
        }
    }

    #pragma omp parallel
    {
        unsigned int seed = 2525 ^ omp_get_thread_num();
        
        #pragma omp for
        for(int p = global_valid; p < NUM_Points; p++) {
            points[p].x = (double)rand_r(&seed) / RAND_MAX;
            points[p].y = (double)rand_r(&seed) / RAND_MAX;
        }
    }
    
    free(valid_counts);
    
    return deleted_count;
}

// Write mesh to file (Unchanged)
void save_mesh(double *mesh_value) {
    FILE *fd = fopen("Mesh.out", "w");
    if (fd) {
        for (int i = 0; i < GRID_Y; i++) {
            for (int j = 0; j < GRID_X; j++) {
                fprintf(fd, "%lf ", mesh_value[i * GRID_X + j]);
            }
            fprintf(fd, "\n");
        }
        fclose(fd);
    }
}
