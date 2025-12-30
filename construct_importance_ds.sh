#!/bin/bash -l
## Name
#SBATCH -J rgb_dvs  
## Number of nodes
#SBATCH -N 1
## Tasks (porcesses to be launched) per node
#SBATCH --ntasks-per-node=6
## Number of CPU cores per task
#SBATCH --cpus-per-task=16
## Total RAM memory allocated
#SBATCH --mem-per-cpu=6GB
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

process_single_dataset() {
  local dataset_tar="$1"
  
    echo "==> Processing dataset tar file: ${dataset_tar}"
    dir_name=$(basename "$dataset_tar" .tar.gz)
    mkdir -p $MEMFS/$dir_name
    tar -xzf $dataset_tar -C $MEMFS/$dir_name
    srun --ntasks=1 --cpus-per-task=16 --gpus-per-task=1 --mem-per-cpu=6GB \
    python src/construct_importance_dataset.py --input_dir $MEMFS/$dir_name --save_dir $ROOT_STATS_DIR
    rm -rf $MEMFS/$dir_name
}


cd $SCRATCH/snn_adversarial_benchmark

source set-up-env.sh

ROOT_STATS_DIR="/net/tscratch/people/plgmazurekagh/snn_adversarial_benchmark/stats_permute_attack_success"
RAW_RESULTS_DIR="/net/tscratch/people/plgmazurekagh/snn_adversarial_benchmark/raw_results_permute_attack_success"
mkdir -p $ROOT_STATS_DIR
# for each tar file in the raw_results directory
for tar_file in ${RAW_RESULTS_DIR}/*.tar.gz; do
    # FILTER: Skip if filename does NOT contain spiking_vgg AND does NOT contain sew_resnet
    if [[ "$tar_file" != *"spiking_vgg"* ]] && [[ "$tar_file" != *"sew_resnet"* ]]; then
        continue
    fi
    echo "Processing $tar_file"
    process_single_dataset "$tar_file" &
done
wait
