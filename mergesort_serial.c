/**
 * Sorts a list of numbers using the merge sort algorithm.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>

#include "utilities.h"

bool merge(double *arr, double *left, int left_len, double *right, int right_len) {
    int i = 0, j = 0, k = 0;

    while (i < left_len && j < right_len) {
        if (left[i] < right[j]) {
            arr[k] = left[i];
            i++;
        } else {
            arr[k] = right[j];
            j++;
        }
        k++;
    }

    while (i < left_len) {
        arr[k] = left[i];
        i++;
        k++;
    }

    while (j < right_len) {
        arr[k] = right[j];
        j++;
        k++;
    }

    return true;
}

bool merge_sort(double *arr, int n) {

    if (n <= 1) {
        return true;
    }

    int mid = n / 2;
    double *left = (double *)malloc(mid * sizeof(double));
    double *right = (double *)malloc((n - mid) * sizeof(double));

    if (left == NULL || right == NULL) {
        printf("Error: Unable to allocate memory\n");
        return false;
    }

    memcpy(left, arr, mid * sizeof(double));
    memcpy(right, arr + mid, (n - mid) * sizeof(double));

    merge_sort(left, mid);
    merge_sort(right, n - mid);

    merge(arr, left, mid, right, n - mid);
    
    free(left);
    free(right);

    return true;
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

    clock_gettime(CLOCK_MONOTONIC, &start);

    merge_sort(arr, n);

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