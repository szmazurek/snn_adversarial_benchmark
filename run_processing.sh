#!/bin/bash -l
## Name
#SBATCH -J test  
## Number of nodes
#SBATCH -N 1
## Tasks (porcesses to be launched) per node
#SBATCH --ntasks-per-node=6
## Number of CPU cores per task
#SBATCH --cpus-per-task=16
## Total RAM memory allocated
#SBATCH --mem-per-cpu=4GB
## Max allocated time (format HH:MM:SS)
#SBATCH --time=24:00:00
## Grant name, this is ours for Athena cluster
#SBATCH -A plgdyplomanci7-gpu-a100
## Partition name, this is the one in Athena cluster
#SBATCH --partition plgrid-gpu-a100
## Number of GPUs
#SBATCH --gpus-per-task=1
#SBATCH -C memfs
## File with standard output - change to your needs
#SBATCH --output="/net/tscratch/people/plgmazurekagh/snn_adversarial_benchmark/output_files/out/stdout_%j.out"
## File with stderr output - change to your needs
#SBATCH --error="/net/tscratch/people/plgmazurekagh/snn_adversarial_benchmark/output_files/err/stderr_%j.err"

# Default values
DATASET=""
MODEL=""
STEP="all"
DATA_DIR="./data"
OUTPUT_BASE_DIR="/net/tscratch/people/plgmazurekagh/snn_adversarial_benchmark/output_files"
SCRATCH_DIR="$SCRATCH/snn_adversarial_benchmark"

# Helper function to print usage
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -d, --dataset <name>    Dataset name (required)"
    echo "  -m, --model <name>      Model name (required, use 'all' for all models)"
    echo "  -s, --step <step>       Step to run: train, test, analyze, cleanup, or all (default: all)"
    echo "  --data_dir <path>       Path to data directory (default: ./data)"
    echo "  --output_dir <path>     Path for output files (default: $OUTPUT_BASE_DIR)"
    exit 1
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -d|--dataset) DATASET="$2"; shift ;;
        -m|--model) MODEL="$2"; shift ;;
        -s|--step) STEP="$2"; shift ;;
        --data_dir) DATA_DIR="$2"; shift ;;
        --output_dir) OUTPUT_BASE_DIR="$2"; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

if [ -z "$DATASET" ]; then
    echo "Error: Dataset is required."
    usage
fi

if [ -z "$MODEL" ]; then
    echo "Error: Model is required."
    usage
fi

cd "$SCRATCH_DIR" || exit 1
source set-up-env.sh

# Define model list if 'all' is selected
if [ "$MODEL" == "all" ]; then
    MODELS=(simple_conv_snn simple_mlp_snn simple_mlp_snn_recurrent simple_conv_snn_recurrent sew_resnet spiking_vgg)
else
    MODELS=("$MODEL")
fi

BATCH_SIZE=32
EPOCHS=100

SCRIPT_DIR="snn_stability/scripts"

for CURRENT_MODEL in "${MODELS[@]}"; do
    EXPERIMENT_NAME="${DATASET}_test_${CURRENT_MODEL}"
    RESULTS_DIR="./results_${DATASET}_${CURRENT_MODEL}"
    AVG_CORR_DIR="./avg_corr_results_${DATASET}_${CURRENT_MODEL}"

    # 1. Training
    if [[ "$STEP" == "train" || "$STEP" == "all" ]]; then
        echo "Starting Training for $CURRENT_MODEL on $DATASET..."
        "$SCRIPT_DIR/run_training.sh" "$EXPERIMENT_NAME" "$DATASET" "$CURRENT_MODEL" "$BATCH_SIZE" "$EPOCHS" "$DATA_DIR" "$OUTPUT_BASE_DIR"
    fi
    
    # Wait for training? 
    # Note: srun with & in previous script implies async execution within the allocation.
    # If we want to wait for training before testing, we should wait here.
    # However, standard practice inside sbatch is usually sequential or parallel with wait.
    # The original script launched multiple trainings in background and waited.
    # Here we iterate. If we want parallel, we should loop inside the step block.
    # BUT, srun inside an sbatch allocation consumes resources.
    # If we run sequentially, it's safer.
    # If STEP is 'all', we MUST wait for training to finish before testing.
    
    if [[ "$STEP" == "all" ]]; then
        wait
    fi

    # 2. Adversarial Test
    if [[ "$STEP" == "test" || "$STEP" == "all" ]]; then
        echo "Starting Adversarial Test for $CURRENT_MODEL on $DATASET..."
        "$SCRIPT_DIR/run_adversarial_test.sh" "$EXPERIMENT_NAME" "$DATASET" "$CURRENT_MODEL" "$DATA_DIR" "$RESULTS_DIR" "$OUTPUT_BASE_DIR"
    fi

    if [[ "$STEP" == "all" ]]; then
        wait
    fi

    # 3. Analysis (Average Correlation)
    if [[ "$STEP" == "analyze" || "$STEP" == "all" ]]; then
        echo "Starting Analysis for $CURRENT_MODEL on $DATASET..."
        "$SCRIPT_DIR/run_analysis.sh" "$EXPERIMENT_NAME" "$DATASET" "$CURRENT_MODEL" "$RESULTS_DIR" "$AVG_CORR_DIR" "$OUTPUT_BASE_DIR"
    fi
    
    if [[ "$STEP" == "all" ]]; then
        wait
    fi

    # 4. Cleanup
    if [[ "$STEP" == "cleanup" || "$STEP" == "all" ]]; then
         # Only run cleanup if explicitly requested or part of 'all' AFTER everything else
         echo "Running cleanup for $CURRENT_MODEL on $DATASET..."
         "$SCRIPT_DIR/run_cleanup.sh" "$DATASET" "$CURRENT_MODEL" "$RESULTS_DIR" "raw_results_permute_attack_success"
    fi
done

wait
echo "All tasks completed."