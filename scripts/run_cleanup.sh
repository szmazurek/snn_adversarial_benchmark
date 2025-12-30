#!/bin/bash

# This script compresses the results directory for a specific model and dataset into a tar.gz file
# and then removes the original directory.

DATASET="$1"
MODEL="$2"
RESULTS_DIR="$3"
ARCHIVE_DIR="$4"

if [ -z "$DATASET" ] || [ -z "$MODEL" ]; then
  echo "Usage: $0 <dataset> <model> <results_dir> <archive_dir>"
  exit 1
fi

RESULTS_DIR=${RESULTS_DIR:-"./results_${DATASET}_${MODEL}"}
ARCHIVE_DIR=${ARCHIVE_DIR:-"raw_results_permute_attack_success"}

mkdir -p "$ARCHIVE_DIR"

if [ -d "$RESULTS_DIR" ]; then
    echo "Compressing results for ${DATASET} with model ${MODEL}..."
    tar -czf "${ARCHIVE_DIR}/results_${DATASET}_${MODEL}.tar.gz" -C "$RESULTS_DIR" .
    if [ $? -eq 0 ]; then
        echo "Compression successful. Removing ${RESULTS_DIR}..."
        rm -rf "$RESULTS_DIR"
    else
        echo "Compression failed! Not removing ${RESULTS_DIR}."
    fi
else
    echo "Directory ${RESULTS_DIR} does not exist. Skipping compression."
fi
