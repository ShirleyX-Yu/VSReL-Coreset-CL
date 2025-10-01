#!/bin/bash
#SBATCH --job-name=cifar_loss_diff
#SBATCH --output=logs/cifar_loss_diff_%j.out
#SBATCH --error=logs/cifar_loss_diff_%j.err
#SBATCH --time=24:00:00
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
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

mkdir -p results/split_cifar_loss_diff/test1
mkdir -p data/CIFAR10
mkdir -p logs

bash scripts/run_cifar_loss_diff.sh

echo "Job finished at: $(date)"
