#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>


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