#!/bin/bash

EXPERIMENT_NAME="$1"
DATASET="$2"
MODEL="$3"
BATCH_SIZE="$4"
EPOCHS="$5"
DATA_DIR="$6"
OUTPUT_BASE_DIR="$7"

if [ -z "$EXPERIMENT_NAME" ] || [ -z "$DATASET" ] || [ -z "$MODEL" ]; then
  echo "Usage: $0 <experiment_name> <dataset> <model> <batch_size> <epochs> <data_dir> <output_base_dir>"
  exit 1
fi

BATCH_SIZE=${BATCH_SIZE:-32}
EPOCHS=${EPOCHS:-100}
DATA_DIR=${DATA_DIR:-"./data"}
OUTPUT_BASE_DIR=${OUTPUT_BASE_DIR:-"./output_files"}

mkdir -p "${OUTPUT_BASE_DIR}/out"
mkdir -p "${OUTPUT_BASE_DIR}/err"

echo "==> Launching training job: ${EXPERIMENT_NAME} on dataset ${DATASET} with model ${MODEL}"

srun --ntasks=1 --cpus-per-task=16 --gpus-per-task=1 --mem-per-cpu=4GB \
  --output="${OUTPUT_BASE_DIR}/out/${EXPERIMENT_NAME}_%j.out" \
  --error="${OUTPUT_BASE_DIR}/err/${EXPERIMENT_NAME}_%j.err" \
  python src/train_network.py \
    --experiment_name "$EXPERIMENT_NAME" \
    --dataset "$DATASET" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --data_dir "$DATA_DIR" \
    --model "$MODEL" &
