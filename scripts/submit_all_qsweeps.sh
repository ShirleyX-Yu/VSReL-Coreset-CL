#!/bin/bash
# submit all q-sweep experiments for comparison against loss_diff baseline

echo "submitting q-sweep experiments for all datasets..."
echo "each job will run 7 array tasks (q = 0.1, 0.5, 1.0, 2.0, 10.0, inf + loss_diff baseline)"
echo ""

cd "$(dirname "$0")"

# submit mnist q-sweep
echo "submitting mnist q-sweep..."
MNIST_JOB=$(sbatch run_mnist_qvendi_qsweep.sh | awk '{print $4}')
echo "  job id: $MNIST_JOB"

# submit permuted mnist q-sweep
echo "submitting permuted mnist q-sweep..."
PERM_JOB=$(sbatch run_perm_qvendi_qsweep.sh | awk '{print $4}')
echo "  job id: $PERM_JOB"

# submit cifar-10 q-sweep
echo "submitting cifar-10 q-sweep..."
CIFAR_JOB=$(sbatch run_cifar_qvendi_qsweep.sh | awk '{print $4}')
echo "  job id: $CIFAR_JOB"

# submit cifar-100 q-sweep
echo "submitting cifar-100 q-sweep..."
CIFAR100_JOB=$(sbatch run_cifar100_qvendi_qsweep.sh | awk '{print $4}')
echo "  job id: $CIFAR100_JOB"

echo ""
echo "all jobs submitted!"
echo "total: 28 experiments (4 datasets × 7 tasks)"
echo "  - 24 q-vendi experiments (6 q values × 4 datasets)"
echo "  - 4 loss_diff baselines (1 per dataset)"
echo ""
echo "monitor with: squeue --me"
echo "check logs in: ../logs/*_q_sweep_*.out"
echo ""
echo "after completion, run:"
echo "  cd ../analysis"
echo "  python parse_logs.py --base ../logs --verbose"
echo "  python plot_qsweep_results.py"
