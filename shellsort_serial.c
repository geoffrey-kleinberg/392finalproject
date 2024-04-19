/**
 * Sorts a list of numbers using the shell sort algorithm.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>

#include "utilities.h"

int get_gap_count(int n) {
    int start = 701 * 9 / 4;
    int count = 0;
    while (start < n) {
        start = start * 9 / 4;
        count++;
    }

    return count + 8;
}

bool shell_sort(double *arr, int n) {
    // create gap sequence
    int count_gaps = get_gap_count(n);
    int* gaps = (int *)malloc(count_gaps * sizeof(int));
    if (gaps == NULL) {
        printf("Error: Unable to allocate memory\n");
        return false;
    }

    // Ciura gap sequence
    gaps[0] = 1;
    gaps[1] = 4;
    gaps[2] = 10;
    gaps[3] = 23;
    gaps[4] = 57;
    gaps[5] = 132;
    gaps[6] = 301;
    gaps[7] = 701;
    int next = 701 * 9 / 4;
    for (int i = 8; i < count_gaps + 8; i++) {
        gaps[i] = next;
        next = next * 9 / 4;
    }


    for (int g = count_gaps - 1; g >= 0; g--) {
        int gap = gaps[g];
        for (int i = gap; i < n; i++) {
            double temp = arr[i];
            int j = i;

            while (j >= gap && arr[j - gap] > temp) {
                arr[j] = arr[j - gap];
                j -= gap;
            }

            arr[j] = temp;
        }
        
    }

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

    shell_sort(arr, n);

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