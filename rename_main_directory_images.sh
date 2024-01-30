#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <main_directory>"
    exit 1
fi

main_directory="$1"
counter=1

cd "$main_directory" || { echo "Error: Unable to enter directory $main_directory"; exit 1; }

for subdir in */; do
    cd "$subdir" || continue  # Skip to the next directory if unable to enter

    for file in *; do
        if [ -f "$file" ]; then
            new_name=$(printf "%05d.jpg" "$counter")
            mv -- "$file" "$new_name" || { echo "Error renaming file $file"; exit 1; }
            counter=$((counter + 1))
        fi
    done

    cd ..
done

