#!/bin/bash

# Check if output argument is provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <array_size> <outfile>"
    exit 1
fi

outfile=$2

if [ ! -p /dev/stdin ]; then
    echo "Please pipe in the output from one of the programs"
fi

if ! test -f "$outfile"; then
    echo "Array Size, Time" >$outfile
fi



# Extracting the array size value from the argument
array_size=$1

# Extracting the time taken value from the argument
output=$(cat)
time_taken=$(echo $output | cut -d' ' -f3)

# Writing the data to a CSV file
echo "$array_size,$time_taken" >>$outfile
