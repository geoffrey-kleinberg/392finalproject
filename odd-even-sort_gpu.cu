/**
 * Implement the odd-even sort algorithm on the GPU.
 * 
 * Compile with: nvcc -arch=sm_86 -O3 odd-even-sort_gpu.cu -o odd-even-sort_gpu
 * Run with: ./odd-even-sort_gpu array-length
 * 
 * Current constraints: array-length <= 2^19
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
    __shared__ int is_sorted;
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    is_sorted = 0;
    while (!is_sorted) {
    // for (int j = 0; j < n; j++) {
        is_sorted = 1;
        //even phase
        if (i % 2 == 0 && i < n - 1) {
            if (arr[i] > arr[i + 1]) {
                swap(arr, i, i + 1);
                is_sorted = 0;
            }
        }
        __syncthreads();
        //odd phase
        if (i % 2 == 1 && i < n - 1) {
            if (arr[i] > arr[i + 1]) {
                swap(arr, i, i + 1);
                is_sorted = 0;
            }
        }
        __syncthreads();
    }
}

__global__ void merge_kernel(double* arr, int n, int merge_length) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    int start = i * merge_length;
    int mid = start + merge_length / 2;
    int end = start + merge_length;

    if (mid >= n) {
        return;
    }

    if (end > n) {
        end = n;
    }

    double* temp = (double*)malloc(merge_length * sizeof(double));
    int left = start;
    int right = mid;
    int k = 0;

    while (left < mid && right < end) {
        if (arr[left] < arr[right]) {
            temp[k] = arr[left];
            left++;
        } else {
            temp[k] = arr[right];
            right++;
        }
        k++;
    }

    while (left < mid) {
        temp[k] = arr[left];
        left++;
        k++;
    }

    while (right < end) {
        temp[k] = arr[right];
        right++;
        k++;
    }

    for (int j = 0; j < k; j++) {
        arr[start + j] = temp[j];
    }

    free(temp);

}

void odd_even_sort(double* arr, int n) {
    double *d_arr;
    cudaMalloc(&d_arr, n * sizeof(double));
    cudaMemcpy(d_arr, arr, n * sizeof(double), cudaMemcpyHostToDevice);

    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    odd_even_sort_kernel<<<num_blocks, BLOCK_SIZE>>>(d_arr, n);

    // int merge_length = BLOCK_SIZE;

    // while (merge_length < n) {
    //     merge_length *= 2;
    //     merge_kernel<<<num_blocks, BLOCK_SIZE>>>(d_arr, n, merge_length);
    // }

    // merge blocks on the cpu
    // double* temp = (double*)malloc(n * sizeof(double));
    // int* offsets = (int*)malloc(num_blocks * sizeof(int));
    // for (int i = 0; i < num_blocks; i++) {
    //     offsets[i] = i * BLOCK_SIZE;
    // }

    // for (int i = 0; i < n; i++) {
    //     double min = INFINITY;
    //     int min_block = -1;
    //     for (int j = 0; j < num_blocks; j++) {
    //         if (offsets[j] < (j + 1) * BLOCK_SIZE && d_arr[offsets[j]] < min) {
    //             min = d_arr[offsets[j]];
    //             min_block = j;
    //         }
    //     }

    //     temp[i] = min;
    //     offsets[min_block]++;
    // }

    // memcpy(arr, temp, n * sizeof(double));

    // free(temp);

    cudaMemcpy(arr, d_arr, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
}

void odd_even_sort_mem_only(double* arr, int n) {
    double *d_arr;
    cudaMalloc(&d_arr, n * sizeof(double));
    cudaMemcpy(d_arr, arr, n * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(arr, d_arr, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
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

    // print before
    for (int i = 0; i < n; i++) {
        printf("%f ", arr[i]);
    }
    printf("\n");

    // get the time for the full algorithm
    clock_gettime(CLOCK_MONOTONIC, &start);

    odd_even_sort(arr, n);

    clock_gettime(CLOCK_MONOTONIC, &end);

    for (int i = 0; i < n; i++) {
        printf("%f ", arr[i]);
    }
    printf("\n");


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