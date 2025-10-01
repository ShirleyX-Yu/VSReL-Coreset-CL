#!/bin/bash
#SBATCH --job-name=mnist_prv_qvendi
#SBATCH --output=../logs/mnist_prv_qvendi_%j.out
#SBATCH --error=../logs/mnist_prv_qvendi_%j.err
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
mkdir -p ../results/split_mnist_prv_qvendi/test1
mkdir -p ../data

cd ..
bash scripts/run_mnist_prv_qvendi.sh

echo "Job finished at: $(date)"
