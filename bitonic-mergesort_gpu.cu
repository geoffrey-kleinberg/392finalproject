/**
 * Sorts a list of numbers using the bitonic merge sort algorithm on the GPU.
 * 
 * Code mostly from https://gist.github.com/mre/1392067 with some modifications.
 * 
 * Compile with: nvcc -arch=sm_86 -O3 utilities.c bitonic-mergesort_gpu.cu -o bitonic-mergesort_gpu
 * Run with: ./bitonic-mergesort_gpu array-length
 * 
 * Do we include memory copy in the time taken?
 * Can we use C headers in CUDA?
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>

// #include "utilities.h"

#include <cuda_runtime.h>

#define BLOCK_SIZE 512

double* create_array(int n) {
    double *arr = (double *)malloc(n * sizeof(double));
    if (arr == NULL) {
        printf("Error: Unable to allocate memory\n");
        return NULL;
    }

    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        arr[i] = rand() / (double)RAND_MAX;
    }

    return arr;
}

bool is_sorted(double *arr, int n) {
    for (int i = 0; i < n - 1; i++) {
        if (arr[i] > arr[i + 1]) {
            return false;
        }
    }

    return true;
}

int lowest_power_of_two(int n) {
    int m = 1;
    while (m < n) {
        m *= 2;
    }
    return m;
}

void extend_array(double *arr, int n, int m) {

    for (int i = n; i < m; i++) {
        arr[i] = INFINITY;
    }
}

__global__ void bitonic_sort_step(double *dev_values, int j, int k) {
    unsigned int i, ixj; /* Sorting partners: i and ixj */
    i = threadIdx.x + blockDim.x * blockIdx.x;
    ixj = i^j;

    /* The threads with the lowest ids sort the array. */
    if ((ixj)>i) {
        if ((i&k)==0) {
            /* Sort ascending */
            if (dev_values[i]>dev_values[ixj]) {
                /* exchange(i,ixj); */
                double temp = dev_values[i];
                dev_values[i] = dev_values[ixj];
                dev_values[ixj] = temp;
            }
        }
        if ((i&k)!=0) {
            /* Sort descending */
            if (dev_values[i]<dev_values[ixj]) {
                /* exchange(i,ixj); */
                double temp = dev_values[i];
                dev_values[i] = dev_values[ixj];
                dev_values[ixj] = temp;
            }
        }
    }
}

void bitonic_sort(double *values, int n) {
    double *dev_values;
    size_t size = n * sizeof(double);

    cudaMalloc((void**) &dev_values, size);
    cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);

    int block = (BLOCK_SIZE < n) ? BLOCK_SIZE : n;

    dim3 blocks(block, 1);
    dim3 threads(n / block, 1);

    int j, k;
    /* Major step */
    for (k = 2; k <= n; k <<= 1) {
        /* Minor step */
        for (j=k>>1; j>0; j=j>>1) {
            bitonic_sort_step<<<threads, blocks>>>(dev_values, j, k);
        }
    }

    cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
    cudaFree(dev_values);
}

void bitonic_sort_mem_only(double *values, int n) {
    double *dev_values;
    size_t size = n * sizeof(double);

    cudaMalloc((void**) &dev_values, size);
    cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);

    cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
    cudaFree(dev_values);
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s array-length\n", argv[0]);
        return 1;
    }

    struct timespec start, end;

    int n = atoi(argv[1]);
    double *arr = create_array(n);
    if (arr == NULL) {
        printf("Error: Unable to allocate memory\n");
        return 1;
    }

    // warm up the GPU
    int new_n = lowest_power_of_two(n);
    double* arr_copy = (double *)malloc(new_n * sizeof(double));
    memcpy(arr_copy, arr, n * sizeof(double));
    extend_array(arr_copy, n, new_n);
    bitonic_sort(arr_copy, new_n);

    clock_gettime(CLOCK_MONOTONIC, &start);

    // make the array length power of two by filling with INFINITY
    int m = lowest_power_of_two(n);
    arr = (double *)realloc(arr, m * sizeof(double));
    extend_array(arr, n, m);

    bitonic_sort(arr, m);

    clock_gettime(CLOCK_MONOTONIC, &end);

    if (!is_sorted(arr, n)) {
        printf("Error: Array is not sorted\n");
        return 1;
    }

    double time_taken = end.tv_sec-start.tv_sec+(end.tv_nsec-start.tv_nsec)/1000000000.0;

    // get the time for memory only
    clock_gettime(CLOCK_MONOTONIC, &start);
    bitonic_sort_mem_only(arr, m);
    clock_gettime(CLOCK_MONOTONIC, &end);

    double time_taken_mem = end.tv_sec-start.tv_sec+(end.tv_nsec-start.tv_nsec)/1000000000.0;

    printf("Time taken: %f\n", time_taken);
    printf("Time taken for memory only: %f\n", time_taken_mem);
    printf("Time for sorting: %f\n", time_taken - time_taken_mem);


    cudaDeviceReset();
    free(arr);

    return 0;


}