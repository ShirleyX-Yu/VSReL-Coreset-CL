#!/bin/bash
#SBATCH --job-name=cifar100_all
#SBATCH --output=logs/cifar100_all_%j.out
#SBATCH --error=logs/cifar100_all_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# run all split cifar-100 experiments (qvendi, loss_diff, prv_qvendi, prv_loss_diff)

# create logs directory if it doesn't exist
mkdir -p logs

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
