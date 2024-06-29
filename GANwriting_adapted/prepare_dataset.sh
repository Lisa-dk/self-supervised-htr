#!/bin/sh

# find $1 -mindepth 2 -type f -exec mv -t $1 -i '{}' + $1
# find $1 -type d -empty -delete

find "$1" -mindepth 2 -type f -exec mv -i '{}' "$1" \;
find "$1" -type d -empty -delete
