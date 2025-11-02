#!/bin/bash
# run all split cifar-100 experiments (qvendi, loss_diff, prv_qvendi, prv_loss_diff)

echo "=========================================="
echo "running all split cifar-100 experiments"
echo "=========================================="
echo ""

bash scripts/run_cifar100_qvendi.sh
echo "✓ completed cifar100_qvendi"
echo ""

bash scripts/run_cifar100_loss_diff.sh
echo "✓ completed cifar100_loss_diff"
echo ""

bash scripts/run_cifar100_prv_qvendi.sh
echo "✓ completed cifar100_prv_qvendi"
echo ""

bash scripts/run_cifar100_prv_loss_diff.sh
echo "✓ completed cifar100_prv_loss_diff"
echo ""

echo "=========================================="
echo "all split cifar-100 experiments completed!"
echo "logs saved to logs/cifar100_*.out"
echo "=========================================="
