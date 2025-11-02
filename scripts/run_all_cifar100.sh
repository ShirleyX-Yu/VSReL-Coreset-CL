#!/bin/bash
#SBATCH --job-name=cifar100_all
#SBATCH --output=logs/cifar100_all_%j.out
#SBATCH --error=logs/cifar100_all_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=cs
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=shirley.yu@princeton.edu

# run all split cifar-100 experiments (qvendi, loss_diff, prv_qvendi, prv_loss_diff)

# get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

# create logs directory if it doesn't exist
mkdir -p logs

module load cuda
module load python

# initialize conda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate vsrel

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

echo "=========================================="
echo "running all split cifar-100 experiments"
echo "=========================================="
echo ""

echo "running cifar100_qvendi..."
bash scripts/run_cifar100_qvendi.sh > logs/cifar100_qvendi.out 2>&1
echo "✓ completed cifar100_qvendi"
echo ""

echo "running cifar100_loss_diff..."
bash scripts/run_cifar100_loss_diff.sh > logs/cifar100_loss_diff.out 2>&1
echo "✓ completed cifar100_loss_diff"
echo ""

echo "running cifar100_prv_qvendi..."
bash scripts/run_cifar100_prv_qvendi.sh > logs/cifar100_prv_qvendi.out 2>&1
echo "✓ completed cifar100_prv_qvendi"
echo ""

echo "running cifar100_prv_loss_diff..."
bash scripts/run_cifar100_prv_loss_diff.sh > logs/cifar100_prv_loss_diff.out 2>&1
echo "✓ completed cifar100_prv_loss_diff"
echo ""

echo "=========================================="
echo "all split cifar-100 experiments completed!"
echo "logs saved to logs/cifar100_*.out"
echo "=========================================="

echo "Job finished at: $(date)"
