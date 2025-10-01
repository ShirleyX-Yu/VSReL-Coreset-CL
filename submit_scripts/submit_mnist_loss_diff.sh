#!/bin/bash
#SBATCH --job-name=mnist_loss_diff
#SBATCH --output=logs/mnist_loss_diff_%j.out
#SBATCH --error=logs/mnist_loss_diff_%j.err
#SBATCH --time=8:00:00
#SBATCH --partition=cs
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=shirley.yu@princeton.edu

module load cuda
module load python
source activate vsrel

echo "Job started at: $(date)"
mkdir -p results/split_mnist_loss_diff/test1
mkdir -p data

bash scripts/run_mnist_loss_diff.sh

echo "Job finished at: $(date)"
