#!/bin/bash
# run all split mnist experiments (qvendi, loss_diff, prv_qvendi, prv_loss_diff)

echo "=========================================="
echo "running all split mnist experiments"
echo "=========================================="
echo ""

bash scripts/run_mnist_qvendi.sh
echo "✓ completed mnist_qvendi"
echo ""

bash scripts/run_mnist_loss_diff.sh
echo "✓ completed mnist_loss_diff"
echo ""

bash scripts/run_mnist_prv_qvendi.sh
echo "✓ completed mnist_prv_qvendi"
echo ""

bash scripts/run_mnist_prv_loss_diff.sh
echo "✓ completed mnist_prv_loss_diff"
echo ""

echo "=========================================="
echo "all split mnist experiments completed!"
echo "logs saved to logs/mnist_*.out"
echo "=========================================="
