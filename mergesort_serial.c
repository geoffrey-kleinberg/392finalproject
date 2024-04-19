/**
 * Sorts a list of numbers using the merge sort algorithm.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>

#include "utilities.h"

bool merge(double *left, int left_len, double *right, int right_len) {
    int i = 0, j = 0, k = 0;

    while (i < left_len && j < right_len) {
        if (left[i] < right[j]) {
            left[k] = left[i];
            i++;
        } else {
            left[k] = right[j];
            j++;
        }
        k++;
    }

    while (i < left_len) {
        left[k] = left[i];
        i++;
        k++;
    }

    while (j < right_len) {
        left[k] = right[j];
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
    double *left = arr;
    double *right = arr + mid;

    merge_sort(left, mid);
    merge_sort(right, n - mid);

    merge(left, mid, right, n - mid);

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