#!/bin/bash
# run all split cifar-10 experiments (qvendi, loss_diff, prv_qvendi, prv_loss_diff)

echo "=========================================="
echo "running all split cifar-10 experiments"
echo "=========================================="
echo ""

bash scripts/run_cifar_qvendi.sh
echo "✓ completed cifar_qvendi"
echo ""

bash scripts/run_cifar_loss_diff.sh
echo "✓ completed cifar_loss_diff"
echo ""

bash scripts/run_cifar_prv_qvendi.sh
echo "✓ completed cifar_prv_qvendi"
echo ""

bash scripts/run_cifar_prv_loss_diff.sh
echo "✓ completed cifar_prv_loss_diff"
echo ""

echo "=========================================="
echo "all split cifar-10 experiments completed!"
echo "logs saved to logs/cifar_*.out"
echo "=========================================="
