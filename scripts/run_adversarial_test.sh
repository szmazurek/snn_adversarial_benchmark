#!/bin/bash

EXPERIMENT_NAME="$1"
DATASET="$2"
MODEL="$3"
DATA_DIR="$4"
RESULTS_DIR="$5"
OUTPUT_BASE_DIR="$6"

if [ -z "$EXPERIMENT_NAME" ] || [ -z "$DATASET" ] || [ -z "$MODEL" ]; then
  echo "Usage: $0 <experiment_name> <dataset> <model> <data_dir> <results_dir> <output_base_dir>"
  exit 1
fi

DATA_DIR=${DATA_DIR:-"./data"}
RESULTS_DIR=${RESULTS_DIR:-"./results_${DATASET}_${MODEL}"}
OUTPUT_BASE_DIR=${OUTPUT_BASE_DIR:-"./output_files"}

mkdir -p "${OUTPUT_BASE_DIR}/out"
mkdir -p "${OUTPUT_BASE_DIR}/err"
mkdir -p "$RESULTS_DIR"

echo "==> Launching adversarial test job: ${EXPERIMENT_NAME} on dataset ${DATASET} with model ${MODEL}"

srun --ntasks=1 --cpus-per-task=16 --gpus-per-task=1 --mem-per-cpu=4GB \
  --output="${OUTPUT_BASE_DIR}/out/${EXPERIMENT_NAME}_adv_%j.out" \
  --error="${OUTPUT_BASE_DIR}/err/${EXPERIMENT_NAME}_adv_%j.err" \
  python src/adversarial_test.py \
    --experiment_name "$EXPERIMENT_NAME" \
    --dataset "$DATASET" \
    --results_dir "$RESULTS_DIR" \
    --data_dir "$DATA_DIR" \
    --model "$MODEL" &
