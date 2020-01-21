#!/bin/bash
# Skip files in $1 which are symlinks
for f in `find $1` 
do
    # Suppress details of differences
    diff -rq $f $2/${f##$1} 
done
