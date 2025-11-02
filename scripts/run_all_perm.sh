#!/bin/bash
# run all perm mnist experiments (qvendi, loss_diff, prv_qvendi, prv_loss_diff)

echo "=========================================="
echo "running all perm mnist experiments"
echo "=========================================="
echo ""

bash scripts/run_perm_qvendi.sh
echo "✓ completed perm_qvendi"
echo ""

bash scripts/run_perm_loss_diff.sh
echo "✓ completed perm_loss_diff"
echo ""

bash scripts/run_perm_prv_qvendi.sh
echo "✓ completed perm_prv_qvendi"
echo ""

bash scripts/run_perm_prv_loss_diff.sh
echo "✓ completed perm_prv_loss_diff"
echo ""

echo "=========================================="
echo "all perm mnist experiments completed!"
echo "logs saved to logs/perm_*.out"
echo "=========================================="
