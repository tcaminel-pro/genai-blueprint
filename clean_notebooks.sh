#!/bin/bash

# Clean Notebook Outputs Script

# Check if nbconvert is installed
if ! command -v jupyter-nbconvert &> /dev/null
then
    echo "jupyter-nbconvert could not be found. Please install it first:"
    echo "pip install nbconvert"
    exit 1
fi

# Directory containing notebooks (default to current directory)
NOTEBOOK_DIR=${1:-.}

# Find all .ipynb files and clean their outputs
find "$NOTEBOOK_DIR" -name "*.ipynb" | while read -r notebook; do
    echo "Cleaning outputs from: $notebook"
    jupyter-nbconvert --clear-output --inplace "$notebook"
done

echo "Notebook outputs cleaned successfully!"
