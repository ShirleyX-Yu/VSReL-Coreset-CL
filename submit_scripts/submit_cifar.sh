#!/bin/bash
#SBATCH --job-name=cifar_cl
#SBATCH --output=logs/cifar_%j.out
#SBATCH --error=logs/cifar_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=cs
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=shirley.yu@princeton.edu

# Load required modules
module load cuda
module load python

# Activate your conda environment
source activate vsrel

# Print job info
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Create necessary directories
mkdir -p results/split_cifar/test1
mkdir -p data/CIFAR10
mkdir -p logs

# Run the experiment
bash scripts/run_cifar.sh

echo "Job finished at: $(date)"
