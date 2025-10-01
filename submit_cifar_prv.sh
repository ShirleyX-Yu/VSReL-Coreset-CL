#!/bin/bash
#SBATCH --job-name=cifar_prv
#SBATCH --output=logs/cifar_prv_%j.out
#SBATCH --error=logs/cifar_prv_%j.err
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
mkdir -p results/split_cifar_prv/test1
mkdir -p data/CIFAR10

bash scripts/run_cifar_prv.sh

echo "Job finished at: $(date)"
