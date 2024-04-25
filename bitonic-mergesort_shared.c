/**
 * Sorts a list of numbers using the bitonic merge sort algorithm.
 * 
 * Most of this code is based on https://github.com/richursa/cpuBitonicSort/blob/master/bitonicSort.cpp
 * and ideas from https://hwlang.de/algorithmen/sortieren/bitonic/oddn.htm,
 * with some modifications to make it work with OpenMP.
 * 
 * Compile (local) with: gcc-13 -Wall -O3 -fopenmp bitonic-mergesort_shared.c -o bitonic-mergesort_shared
 * Compile (cluster) with: gcc -Wall -O3 -fopenmp bitonic-mergesort_shared.c -o bitonic-mergesort_shared
 * Run with: ./bitonic-mergesort_shared length (num-threads)
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>

#include <omp.h>

#include "utilities.h"

void bitonic_merge(double *arr, int start, int length, bool ascending, int num_threads) {
    if (length <= 1) {
        return;
    }

    int m = greatest_power_of_two(length);
    for (int i = start; i < start + length - m; i++) {
        if (((arr[i] > arr[i + m]) == ascending)) {
            swap(arr, i, i + m);
        }
    }
    
    bitonic_merge(arr, start, m, ascending, num_threads);
    bitonic_merge(arr, start + m, length - m, ascending, num_threads);
}

bool bitonic_merge_sort_openMP(double *arr, int start, int n, int num_threads, bool ascending) {
    if (n <= 1) {
        return true;
    }
    int power = greatest_power_of_two(n);

    if (power == n / 2) {
        power = n;
    }

    for (int j = 2; j <= power; j = j*2) {
        #pragma omp parallel for num_threads(num_threads) default(none) shared(arr, start, n, j, ascending, power)
        for (int i = start; i < start + power; i += j) {
            if (((i-start)/j) % 2 == 0) {
                bitonic_merge(arr, i, j, !ascending, n);
            } else {
                bitonic_merge(arr, i, j, ascending, n);
            }
        }
    }

    bitonic_merge_sort_openMP(arr, start + power, n - power, num_threads, ascending);
    bitonic_merge(arr, start, n, ascending, num_threads);

    return true;
    
}

int main(int argc, char *argv[]) {
    int num_threads = omp_get_max_threads() / 2;
    if (argc != 2 && argc != 3) {
        printf("Usage: %s array-length (num-threads)\n", argv[0]);
        return 1;
    }

    struct timespec start, end;

    int n = atoi(argv[1]);
    double *arr = create_array(n);
    if (arr == NULL) {
        printf("Error: Unable to allocate memory\n");
        return 1;
    }
    if (argc == 3) {
        num_threads = atoi(argv[2]);
    }

    // warmup the CPU
    double* arr2 = create_array(n);
    bitonic_merge_sort_openMP(arr2, 0, n, num_threads, true);

    clock_gettime(CLOCK_MONOTONIC, &start);

    bitonic_merge_sort_openMP(arr, 0, n, num_threads, true);

    clock_gettime(CLOCK_MONOTONIC, &end);

    if (!is_sorted(arr, n)) {
        printf("Error: Array is not sorted\n");
        return 1;
    }

    double time_taken = end.tv_sec-start.tv_sec+(end.tv_nsec-start.tv_nsec)/1000000000.0;

    printf("Time taken: %f\n", time_taken);

    free(arr);

    return 0;


}