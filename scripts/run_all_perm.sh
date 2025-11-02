#!/bin/bash
#SBATCH --job-name=perm_all
#SBATCH --output=../logs/perm_all_%j.out
#SBATCH --error=../logs/perm_all_%j.err
#SBATCH --time=1:00:00
#SBATCH --partition=cs
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=shirley.yu@princeton.edu

# run all perm mnist experiments (qvendi, loss_diff, prv_qvendi, prv_loss_diff)

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

echo "=========================================="
echo "running all perm mnist experiments"
echo "=========================================="
echo ""

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

echo "running perm_qvendi..."
bash scripts/run_perm_qvendi.sh > logs/perm_qvendi.out 2>&1
echo "✓ completed perm_qvendi"
echo ""

echo "running perm_loss_diff..."
bash scripts/run_perm_loss_diff.sh > logs/perm_loss_diff.out 2>&1
echo "✓ completed perm_loss_diff"
echo ""

echo "running perm_prv_qvendi..."
bash scripts/run_perm_prv_qvendi.sh > logs/perm_prv_qvendi.out 2>&1
echo "✓ completed perm_prv_qvendi"
echo ""

echo "running perm_prv_loss_diff..."
bash scripts/run_perm_prv_loss_diff.sh > logs/perm_prv_loss_diff.out 2>&1
echo "✓ completed perm_prv_loss_diff"
echo ""

echo "=========================================="
echo "all perm mnist experiments completed!"
echo "logs saved to logs/perm_*.out"
echo "=========================================="

echo "Job finished at: $(date)"
