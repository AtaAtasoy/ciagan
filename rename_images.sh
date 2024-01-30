#!/bin/bash
# Renames the images in the directory as  00001.jpg 000002.jpg etc. (CelebA dataset format)


if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <target_directory>"
    exit 1
fi

target_directory="$1"
counter=1

cd "$target_directory" || { echo "Error: Unable to enter directory $target_directory"; exit 1; }

for file in *; do
    if [ -f "$file" ]; then
        new_name=$(printf "%05d.jpg" "$counter")
        mv -- "$file" "$new_name" || { echo "Error renaming file $file"; exit 1; }
        counter=$((counter+1))
    fi
done

