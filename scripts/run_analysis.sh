#!/bin/bash

EXPERIMENT_NAME="$1"
DATASET="$2"
MODEL="$3"
RESULTS_DIR="$4"
OUTPUT_DIR="$5"
OUTPUT_BASE_DIR="$6"

if [ -z "$EXPERIMENT_NAME" ] || [ -z "$DATASET" ] || [ -z "$MODEL" ]; then
  echo "Usage: $0 <experiment_name> <dataset> <model> <results_dir> <output_dir> <output_base_dir>"
  exit 1
fi

RESULTS_DIR=${RESULTS_DIR:-"./results_${DATASET}_${MODEL}"}
OUTPUT_DIR=${OUTPUT_DIR:-"./avg_corr_results_${DATASET}_${MODEL}"}
OUTPUT_BASE_DIR=${OUTPUT_BASE_DIR:-"./output_files"}

mkdir -p "${OUTPUT_BASE_DIR}/out"
mkdir -p "${OUTPUT_BASE_DIR}/err"
mkdir -p "$OUTPUT_DIR"

echo "==> Launching average correlation job: ${EXPERIMENT_NAME} on dataset ${DATASET} with model ${MODEL}"

srun --ntasks=1 --cpus-per-task=16 --gpus-per-task=1 --mem-per-cpu=4GB \
  --output="${OUTPUT_BASE_DIR}/out/avg_corr_%j.out" \
  --error="${OUTPUT_BASE_DIR}/err/avg_corr_%j.err" \
  python src/average_correlation_scores.py \
    --root-data-path "$RESULTS_DIR" \
    --save-dir "$OUTPUT_DIR" &
