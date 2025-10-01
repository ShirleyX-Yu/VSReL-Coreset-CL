#!/bin/bash
#SBATCH --job-name=mnist_comparison
#SBATCH --output=logs/mnist_comparison_%j.out
#SBATCH --error=logs/mnist_comparison_%j.err
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
echo "=========================================="
echo "MNIST Comparison: Baseline vs Q-Vendi"
echo "=========================================="

# Create necessary directories
mkdir -p results/split_mnist_loss_diff/test1
mkdir -p results/split_mnist_qvendi/test1
mkdir -p data/MNIST
mkdir -p logs

# Run the comparison experiment
bash scripts/run_comparison_mnist.sh

echo ""
echo "Job completed at: $(date)"
echo "Results saved to:"
echo "  - results/split_mnist_loss_diff/test1/"
echo "  - results/split_mnist_qvendi/test1/"
