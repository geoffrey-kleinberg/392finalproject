/**
 * Implement the odd-even sort algorithm on the GPU.
 * 
 * Compile with: nvcc -arch=sm_86 -O3 odd-even-sort_gpu.cu -o odd-even-sort_gpu
 * Run with: ./odd-even-sort_gpu array-length
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>

#include "utilities.h"

#include <cuda_runtime.h>

#define BLOCK_SIZE 512

__global__ void odd_even_sort_kernel(double* arr, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int max = (blockIdx.x + 1) * blockDim.x;

    for (int j = 0; j < blockDim.x; j++) {
        //even phase
        if (i % 2 == 0 && i < max - 1 && i < n - 1) {
            if (arr[i] > arr[i + 1]) {
                swap(arr, i, i + 1);
            }
        }
        __syncthreads();
        //odd phase
        if (i % 2 == 1 && i < max - 1 && i < n - 1) {
            if (arr[i] > arr[i + 1]) {
                swap(arr, i, i + 1);
            }
        }
        __syncthreads();
    }
}

__global__ void merge_kernel(double* arr, int n, int merge_length, double* temp) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    size_t left = tid * merge_length;
    size_t middle = left + merge_length / 2;
    size_t right = (left + merge_length < n) ? left + merge_length : n;

    if (left >= n) {
        return;
    }

    if (middle >= n) {
        return;
    }

    size_t i = left;
    size_t j = middle;
    size_t k = left;

    while (i < middle && j < right) 
    {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
        }
    }

    while (i < middle) {
        temp[k++] = arr[i++];
    }
    while (j < right) {
        temp[k++] = arr[j++];
    }

    for (int x = left; x < right; x++) {
        arr[x] = temp[x];
    }

}

void odd_even_sort(double* arr, int n) {
    double *d_arr;
    cudaMalloc(&d_arr, n * sizeof(double));
    cudaMemcpy(d_arr, arr, n * sizeof(double), cudaMemcpyHostToDevice);

    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    odd_even_sort_kernel<<<num_blocks, BLOCK_SIZE>>>(d_arr, n);

    double* d_temp;
    cudaMalloc(&d_temp, n * sizeof(double));

    for (int merge_length = 2 * BLOCK_SIZE; merge_length < 2 * n; merge_length *= 2) {
        merge_kernel<<<num_blocks, BLOCK_SIZE>>>(d_arr, n, merge_length, d_temp);
    }

    cudaMemcpy(arr, d_arr, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
    cudaFree(d_temp);

}

void odd_even_sort_mem_only(double* arr, int n) {
    double *d_arr;
    cudaMalloc(&d_arr, n * sizeof(double));
    cudaMemcpy(d_arr, arr, n * sizeof(double), cudaMemcpyHostToDevice);

    double* d_temp;
    cudaMalloc(&d_temp, n * sizeof(double));

    cudaMemcpy(arr, d_arr, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
    cudaFree(d_temp);
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
    double* arr2 = create_array(n);
    odd_even_sort(arr2, n);

    // get the time for the full algorithm
    clock_gettime(CLOCK_MONOTONIC, &start);

    odd_even_sort(arr, n);

    clock_gettime(CLOCK_MONOTONIC, &end);

    if (!is_sorted(arr, n)) {
        printf("Error: Array is not sorted\n");
        return 1;
    }

    double time_taken = end.tv_sec-start.tv_sec+(end.tv_nsec-start.tv_nsec)/1000000000.0;

    // get the time for memory only
    clock_gettime(CLOCK_MONOTONIC, &start);
    odd_even_sort_mem_only(arr, n);
    clock_gettime(CLOCK_MONOTONIC, &end);

    double time_taken_mem = end.tv_sec-start.tv_sec+(end.tv_nsec-start.tv_nsec)/1000000000.0;

    printf("Time taken: %f\n", time_taken);
    printf("Time taken for memory only: %f\n", time_taken_mem);
    printf("Time for sorting: %f\n", time_taken - time_taken_mem);


    cudaDeviceReset();
    free(arr);

    return 0;
}