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