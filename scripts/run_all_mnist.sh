#!/bin/bash
#SBATCH --job-name=mnist_all
#SBATCH --output=../logs/mnist_all_%j.out
#SBATCH --error=../logs/mnist_all_%j.err
#SBATCH --time=1:00:00
#SBATCH --partition=cs
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=shirley.yu@princeton.edu

# run all split mnist experiments (qvendi, loss_diff, prv_qvendi, prv_loss_diff)

# get project root from submit directory
if [[ "${SLURM_SUBMIT_DIR}" == */scripts ]]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR%/scripts}"
else
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
fi
cd "$PROJECT_ROOT"

# initialize conda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate vsrel

echo "=========================================="
echo "running all split mnist experiments"
echo "=========================================="
echo ""

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

echo "running mnist_qvendi..."
bash scripts/run_mnist_qvendi.sh > "${PROJECT_ROOT}/logs/mnist_qvendi.out" 2>&1
echo "✓ completed mnist_qvendi"
echo ""

echo "running mnist_loss_diff..."
bash scripts/run_mnist_loss_diff.sh > "${PROJECT_ROOT}/logs/mnist_loss_diff.out" 2>&1
echo "✓ completed mnist_loss_diff"
echo ""

echo "running mnist_prv_qvendi..."
bash scripts/run_mnist_prv_qvendi.sh > "${PROJECT_ROOT}/logs/mnist_prv_qvendi.out" 2>&1
echo "✓ completed mnist_prv_qvendi"
echo ""

echo "running mnist_prv_loss_diff..."
bash scripts/run_mnist_prv_loss_diff.sh > "${PROJECT_ROOT}/logs/mnist_prv_loss_diff.out" 2>&1
echo "✓ completed mnist_prv_loss_diff"
echo ""

echo "=========================================="
echo "all split mnist experiments completed!"
echo "logs saved to logs/mnist_*.out"
echo "=========================================="

echo "Job finished at: $(date)"
