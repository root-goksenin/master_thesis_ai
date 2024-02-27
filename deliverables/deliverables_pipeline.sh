#!/bin/bash



python3 make_lattex.py "$@"
for var in "$@"
do
    python3 convert_to_latex.py table.json "$var"
done
