#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>

static double* create_array(int n) {
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

static bool is_sorted(double *arr, int n) {
    for (int i = 0; i < n - 1; i++) {
        if (arr[i] > arr[i + 1]) {
            return false;
        }
    }

    return true;
}

static int lowest_power_of_two(int n) {
    int m = 1;
    while (m < n) {
        m *= 2;
    }
    return m;
}

static int greatest_power_of_two(int n) {
    return lowest_power_of_two(n) / 2;
}



#ifdef __CUDACC__
__device__ void swap(double* arr, int i, int j) {
    double temp = arr[i];
    arr[i] = arr[j];
    arr[j] = temp;
}
#else
static void swap(double* arr, int i, int j) {
    double temp = arr[i];
    arr[i] = arr[j];
    arr[j] = temp;
}
#endif

#ifdef __CUDACC__
#define CHECK(call)                                                       \
{                                                                         \
   const cudaError_t error = call;                                        \
   if (error != cudaSuccess)                                              \
   {                                                                      \
      printf("Error: %s:%d, ", __FILE__, __LINE__);                       \
      printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
      exit(1);                                                            \
   }                                                                      \
}
#endif