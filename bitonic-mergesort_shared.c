/**
 * Sorts a list of numbers using the bitonic merge sort algorithm.
 * 
 * Most of this code is based on https://github.com/richursa/cpuBitonicSort/blob/master/bitonicSort.cpp
 * with help from copilot.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>

#include <omp.h>

#include "utilities.h"

int greatest_power_of_two(int n) {
    int power = 1;
    while (power < n) {
        power *= 2;
    }
    return power / 2;
}

void bitonic_merge(double *arr, int start, int length, bool ascending, int num_threads) {
    if (length <= 1) {
        return;
    }

    // if (start > actual_length) {
    //     return;
    // }

    // int half_length = length / 2;
    // for (int i = start; i < start + half_length; i++) {
    //     if ((i + half_length < actual_length) && ((arr[i] > arr[i + half_length]) == ascending)) {
    //         double temp = arr[i];
    //         arr[i] = arr[i + half_length];
    //         arr[i + half_length] = temp;
    //     }
    //     // if (i + half_length > actual_length && i < actual_length) {
    //     //     return;
    //     // }
    // }

    // bitonic_merge(arr, start, half_length, ascending, actual_length);
    // bitonic_merge(arr, start + half_length, half_length, ascending, actual_length);

    int m = greatest_power_of_two(length);

    // #pragma omp parallel for num_threads(num_threads) default(none) shared(arr, start, length, m, ascending)
    // #pragma omp for
    for (int i = start; i < start + length - m; i++) {
        if (((arr[i] > arr[i + m]) == ascending)) {
            double temp = arr[i];
            arr[i] = arr[i + m];
            arr[i + m] = temp;
        }
    }
    
    bitonic_merge(arr, start, m, ascending, num_threads);
    bitonic_merge(arr, start + m, length - m, ascending, num_threads);
}

int get_count(int n) {
    int count = 0;
    int m = n;
    while (m > 1) {
        count += 1;
        m = m / 2;
    }
    return count + 1;

}

void make_list(int* list, int n, int count) {
    // list goes n / 2, (n / 2) / 2, (n / 2) / 2 / 2, ... 1

    for (int i = 0; i < count; i++) {
        list[i] = n;
        n = n / 2;
    }

}

bool bitonic_merge_sort_openMP(double *arr, int start, int n, int num_threads, bool ascending) {
    if (n <= 1) {
        return true;
    }
    // int power = lowest_power_of_two(n);

    // printf("Power: %d\n", power);

    // // j is length, i is startindex, ascending is direction
    // for (int j = 2; j <= power; j = j*2) {
    //     #pragma omp parallel for num_threads(num_threads) default(none) shared(arr, n, j, ascending, power)
    //     for (int i = 0; i < power; i += j) {
    //         if ((i/j) % 2 == 0) {
    //             bitonic_merge(arr, i, j, ascending, n);
    //         } else {
    //             bitonic_merge(arr, i, j, !ascending, n);
    //         }
    //     }
    // }

    // return true;

    int m = n / 2;

    // make a list of n / 2, then (n / 2) / 2, and so on and then reverse it

    int count = get_count(n);
    int* list = (int*) malloc(count * sizeof(int));
    if (list == NULL) {
        return false;
    }

    make_list(list, n, count);

    // print the list
    for (int i = 0; i < count; i++) {
        printf("%d ", list[i]);
    }
    printf("\n");
    
    
    // bitonic_merge_sort_openMP(arr, start, m, num_threads, !ascending);
    // bitonic_merge_sort_openMP(arr, start + m, n - m, num_threads, ascending);

    for (int j = count - 1; j >= 0; j--) {
        #pragma omp parallel for num_threads(num_threads) default(none) shared(arr, n, j, ascending, num_threads, list)
        for (int i = 0; i < n; i += list[j]) {
            if ((i/list[j]) % 2 == 0) {
                bitonic_merge(arr, i, list[j], ascending, num_threads);
            } else {
                bitonic_merge(arr, i, list[j], !ascending, num_threads);
            }
        }
    }

    // #pragma omp parallel 
    // bitonic_merge(arr, start, n, ascending, num_threads);

    return true;
    
}

int main(int argc, char *argv[]) {
    int num_threads = omp_get_max_threads() / 2;
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
        num_threads = atoi(argv[2]);
    }

    // print array
    for (int i = 0; i < n; i++) {
        printf("%f ", arr[i]);
    }
    printf("\n");

    clock_gettime(CLOCK_MONOTONIC, &start);

    bitonic_merge_sort_openMP(arr, 0, n, num_threads, true);

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

    printf("Time taken: %f\n", time_taken);

    free(arr);

    return 0;


}