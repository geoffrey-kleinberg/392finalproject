/**
 * Sorts a list of numbers using the bitonic merge sort algorithm on the GPU.
 * 
 * Code mostly from https://gist.github.com/mre/1392067 with some modifications.
 * 
 * Compile with: nvcc -arch=sm_86 -O3 bitonic-mergesort_gpu.cu -o bitonic-mergesort_gpu
 * Run with: ./bitonic-mergesort_gpu array-length
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>

#include "utilities.h"

#include <cuda_runtime.h>

void extend_array(double *arr, int n, int m) {

    for (int i = n; i < m; i++) {
        arr[i] = INFINITY;
    }
}

__global__ void bitonic_sort_step(double *d_values, int j, int k) {
    unsigned int i, ixj; /* Sorting partners: i and ixj */
    i = threadIdx.x + blockDim.x * blockIdx.x;
    ixj = i^j;

    /* The threads with the lowest ids sort the array. */
    if ((ixj)>i) {
        if ((i&k)==0) {
            /* Sort ascending */
            if (d_values[i]>d_values[ixj]) {
                /* exchange(i,ixj); */
                swap(d_values, i, ixj);
            }
        }
        if ((i&k)!=0) {
            /* Sort descending */
            if (d_values[i]<d_values[ixj]) {
                /* exchange(i,ixj); */
                swap(d_values, i, ixj);
            }
        }
    }
}

void bitonic_sort(double *values, int n, int block_size) {
    double *d_values;
    size_t size = n * sizeof(double);

    CHECK(cudaMalloc((void**) &d_values, size));
    CHECK(cudaMemcpy(d_values, values, size, cudaMemcpyHostToDevice));

    int block = (block_size < n) ? block_size : n;

    dim3 blocks(block, 1);
    dim3 threads(n / block, 1);

    int j, k;
    /* Major step */
    for (k = 2; k <= n; k <<= 1) {
        /* Minor step */
        for (j=k>>1; j>0; j=j>>1) {
            bitonic_sort_step<<<threads, blocks>>>(d_values, j, k);
        }
    }

    CHECK(cudaMemcpy(values, d_values, size, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_values));
}

void bitonic_sort_mem_only(double *values, int n, int block_size) {
    double *d_values;
    size_t size = n * sizeof(double);

    CHECK(cudaMalloc((void**) &d_values, size));
    CHECK(cudaMemcpy(d_values, values, size, cudaMemcpyHostToDevice));

    CHECK(cudaMemcpy(values, d_values, size, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_values));
}

int main(int argc, char *argv[]) {
    int block_size = 512;
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
        block_size = atoi(argv[2]);
    }

    // warm up the GPU
    int new_n = lowest_power_of_two(n);
    double* arr2 = create_array(new_n);
    bitonic_sort(arr2, new_n, block_size);

    clock_gettime(CLOCK_MONOTONIC, &start);

    // make the array length power of two by filling with INFINITY
    int m = lowest_power_of_two(n);
    arr = (double *)realloc(arr, m * sizeof(double));
    extend_array(arr, n, m);

    bitonic_sort(arr, m, block_size);

    clock_gettime(CLOCK_MONOTONIC, &end);

    if (!is_sorted(arr, n)) {
        printf("Error: Array is not sorted\n");
        return 1;
    }

    double time_taken = end.tv_sec-start.tv_sec+(end.tv_nsec-start.tv_nsec)/1000000000.0;

    // get the time for memory only
    clock_gettime(CLOCK_MONOTONIC, &start);
    bitonic_sort_mem_only(arr, m, block_size);
    clock_gettime(CLOCK_MONOTONIC, &end);

    double time_taken_mem = end.tv_sec-start.tv_sec+(end.tv_nsec-start.tv_nsec)/1000000000.0;

    printf("Time taken: %f\n", time_taken);
    printf("Time taken for memory only: %f\n", time_taken_mem);
    printf("Time for sorting: %f\n", time_taken - time_taken_mem);


    CHECK(cudaDeviceReset());
    free(arr);

    return 0;


}