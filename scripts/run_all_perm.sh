#!/bin/bash
#SBATCH --job-name=perm_all
#SBATCH --output=logs/perm_all_%j.out
#SBATCH --error=logs/perm_all_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# run all perm mnist experiments (qvendi, loss_diff, prv_qvendi, prv_loss_diff)

# create logs directory if it doesn't exist
mkdir -p logs

echo "=========================================="
echo "running all perm mnist experiments"
echo "=========================================="
echo ""

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
