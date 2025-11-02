#!/bin/bash
# run all split mnist experiments (qvendi, loss_diff, prv_qvendi, prv_loss_diff)

# create logs directory if it doesn't exist
mkdir -p logs

echo "=========================================="
echo "running all split mnist experiments"
echo "=========================================="
echo ""

echo "running mnist_qvendi..."
bash scripts/run_mnist_qvendi.sh > logs/mnist_qvendi.out 2>&1
echo "✓ completed mnist_qvendi"
echo ""

echo "running mnist_loss_diff..."
bash scripts/run_mnist_loss_diff.sh > logs/mnist_loss_diff.out 2>&1
echo "✓ completed mnist_loss_diff"
echo ""

echo "running mnist_prv_qvendi..."
bash scripts/run_mnist_prv_qvendi.sh > logs/mnist_prv_qvendi.out 2>&1
echo "✓ completed mnist_prv_qvendi"
echo ""

echo "running mnist_prv_loss_diff..."
bash scripts/run_mnist_prv_loss_diff.sh > logs/mnist_prv_loss_diff.out 2>&1
echo "✓ completed mnist_prv_loss_diff"
echo ""

echo "=========================================="
echo "all split mnist experiments completed!"
echo "logs saved to logs/mnist_*.out"
echo "=========================================="
