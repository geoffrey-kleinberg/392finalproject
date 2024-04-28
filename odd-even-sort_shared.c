/*
 * Sorts a list of numbers using the odd-even sort algorithm.
 * incorperates OpenMP to parallelize the sorting process.
 * 
 * Code from https://github.com/eduardnegru/Parallel-Odd-Even-Transposition-Sort/blob/master/openmp.c
 * with some modifications.
 * 
 * Compile (local) using: gcc-13 -Wall -O3 -fopenmp odd-even-sort_shared.c -o odd-even-sort_shared
 * Compile (mucluster) using: gcc -Wall -O3 -fopenmp odd-even-sort_shared.c -o odd-even-sort_shared
 * Run with: ./odd-even-sort_shared array-length (num-threads)
 * where:
 *       array-length: the length of the array to be sorted
 *       num-threads: the number of threads to use (optional, default is half the number of cores)
 *
*/

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <omp.h>

#include "utilities.h"


void odd_even_sort(double* arr, int n, int num_threads) {

#pragma omp parallel num_threads(num_threads) default(none) shared(arr, n)
    for (int phase = 0; phase < n; phase++) {
        // even phase
        if (phase % 2 == 0) {
#pragma omp for
            for (int i = 1; i < n; i += 2) {
                if (arr[i - 1] > arr[i]) {
                    swap(arr, i - 1, i);
                }
            }
        }

        // odd phase
        else {
#pragma omp for
            for (int i = 1; i < n - 1; i += 2) {
                if (arr[i] > arr[i + 1]) {
                    swap(arr, i, i + 1);
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
    int num_threads = omp_get_max_threads() / 2;
    if (argc != 2 && argc != 3) {
        printf("Usage: %s array-length (num-threads)\n", argv[0]);
        return 1;
    }

    struct timespec start, end;

    int n = atoi(argv[1]);
    double* arr = create_array(n);
    if (arr == NULL) {
        printf("Error: Unable to allocate memory\n");
        return 1;
    }
    if (argc == 3) {
        num_threads = atoi(argv[2]);
    }

    // warmup the CPU
    double* arr2 = create_array(n);
    odd_even_sort(arr2, n, num_threads);

    clock_gettime(CLOCK_MONOTONIC, &start);

    odd_even_sort(arr, n, num_threads);

    clock_gettime(CLOCK_MONOTONIC, &end);

    if (!is_sorted(arr, n)) {
        printf("Error: Array is not sorted\n");
        return 1;
    }

    double time_taken = end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) / 1000000000.0;

    printf("Time taken: %f\n", time_taken);

    free(arr);

    return 0;
}