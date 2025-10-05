#!/bin/bash

# submit all Q-Vendi q-sweep experiments to the cluster
#
# required files:
# - submit_cifar_qvendi_q_sweep.sh
# - submit_cifar_prv_qvendi_q_sweep.sh
# - submit_mnist_qvendi_q_sweep.sh
# - submit_mnist_prv_qvendi_q_sweep.sh
#
# usage:
#   ./submit_all_qvendi_q_sweep.sh          # submit all jobs
#   ./submit_all_qvendi_q_sweep.sh --dry-run # list files without submitting

# parse command line arguments
DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
fi

# define all required files
REQUIRED_FILES=(
    "./submit_cifar_qvendi_q_sweep.sh"
    "./submit_cifar_prv_qvendi_q_sweep.sh"
    "./submit_mnist_qvendi_q_sweep.sh"
    "./submit_mnist_prv_qvendi_q_sweep.sh"
)

if [ "$DRY_RUN" = true ]; then
    echo "=== DRY RUN MODE ==="
    echo "The following files are required and will be submitted:"
    for file in "${REQUIRED_FILES[@]}"; do
        if [ -f "$file" ]; then
            echo "  ✓ $file (exists)"
        else
            echo "  ✗ $file (MISSING)"
        fi
    done
    echo ""
    echo "Run without --dry-run to submit jobs."
    exit 0
fi

echo "Submitting Q-Vendi q-sweep experiments..."

# submit CIFAR experiments
echo "Submitting CIFAR experiments..."
sbatch ./submit_cifar_qvendi_q_sweep.sh

# submit CIFAR PRV experiments
echo "Submitting CIFAR PRV experiments..."
sbatch ./submit_cifar_prv_qvendi_q_sweep.sh

# submit MNIST experiments
echo "Submitting MNIST experiments..."
sbatch ./submit_mnist_qvendi_q_sweep.sh

# submit MNIST PRV experiments
echo "Submitting MNIST PRV experiments..."
sbatch ./submit_mnist_prv_qvendi_q_sweep.sh

echo "All jobs submitted!"
echo "Check job status with: squeue -u \$USER"
echo "Check logs in: logs/"
