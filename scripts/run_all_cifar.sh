#!/bin/bash
#SBATCH --job-name=cifar_all
#SBATCH --output=logs/cifar_all_%j.out
#SBATCH --error=logs/cifar_all_%j.err
#SBATCH --time=6:00:00
#SBATCH --partition=cs
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=shirley.yu@princeton.edu

# run all split cifar-10 experiments (qvendi, loss_diff, prv_qvendi, prv_loss_diff)

module load cuda
module load python

# initialize conda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate vsrel

# create logs directory if it doesn't exist
mkdir -p logs

echo "=========================================="
echo "running all split cifar-10 experiments"
echo "=========================================="
echo ""

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

echo "running cifar_qvendi..."
bash scripts/run_cifar_qvendi.sh > ../logs/cifar_qvendi.out 2>&1
echo "✓ completed cifar_qvendi"
echo ""

echo "running cifar_loss_diff..."
bash scripts/run_cifar_loss_diff.sh > ../logs/cifar_loss_diff.out 2>&1
echo "✓ completed cifar_loss_diff"
echo ""

echo "running cifar_prv_qvendi..."
bash scripts/run_cifar_prv_qvendi.sh > ../logs/cifar_prv_qvendi.out 2>&1
echo "✓ completed cifar_prv_qvendi"
echo ""

echo "running cifar_prv_loss_diff..."
bash scripts/run_cifar_prv_loss_diff.sh > ../logs/cifar_prv_loss_diff.out 2>&1
echo "✓ completed cifar_prv_loss_diff"
echo ""

echo "=========================================="
echo "all split cifar-10 experiments completed!"
echo "logs saved to logs/cifar_*.out"
echo "=========================================="

echo "Job finished at: $(date)"
