#!/bin/bash

# submit all Q-Vendi q-sweep experiments to the cluster

echo "Submitting Q-Vendi q-sweep experiments..."

# submit CIFAR experiments
echo "Submitting CIFAR experiments..."
sbatch submit_scripts/submit_cifar_qvendi_q_sweep.sh

# submit CIFAR PRV experiments
echo "Submitting CIFAR PRV experiments..."
sbatch submit_scripts/submit_cifar_prv_qvendi_q_sweep.sh

# submit MNIST experiments
echo "Submitting MNIST experiments..."
sbatch submit_scripts/submit_mnist_qvendi_q_sweep.sh

# submit MNIST PRV experiments
echo "Submitting MNIST PRV experiments..."
sbatch submit_scripts/submit_mnist_prv_qvendi_q_sweep.sh

echo "All jobs submitted!"
echo "Check job status with: squeue -u \$USER"
echo "Check logs in: logs/"
