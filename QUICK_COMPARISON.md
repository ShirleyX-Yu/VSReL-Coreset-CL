# Quick Comparison Guide

## TL;DR - Generate Comparison Tables

### 1. Run both methods (example for MNIST):

```bash
# Baseline (loss_diff only)
python permuted_mnist_cl.py \
    --local_path ./results/mnist_loss_diff \
    --dataset mnist --setting permuted \
    --buffer_size 200 --use_qvendi 0 --seed 42

# Q-Vendi (loss_diff as quality + diversity)
python permuted_mnist_cl.py \
    --local_path ./results/mnist_qvendi \
    --dataset mnist --setting permuted \
    --buffer_size 200 --use_qvendi 1 --seed 42
```

### 2. Generate comparison tables:

```bash
python analysis/compare_methods.py \
    --baseline ./results/mnist_loss_diff \
    --qvendi ./results/mnist_qvendi
```

### 3. Output files:

- `analysis/table1_metrics_comparison.csv` - Detailed per-test results
- `analysis/table1_summary_comparison.csv` - Dataset summaries

## What Changed?

| Method | Selection Strategy |
|--------|-------------------|
| **Baseline** | Greedy selection by loss_diff (reducible loss) |
| **Q-Vendi** | Greedy selection by qVS = quality × diversity<br>• Quality = loss_diff<br>• Diversity = feature similarity |

## Key Files

- **Run experiments**: `scripts/run_comparison_mnist.sh`
- **Compare results**: `analysis/compare_methods.py`
- **Detailed guide**: `COMPARISON_GUIDE.md`
- **Implementation**: `QVENDI_INTEGRATION.md`

## Directory Naming Convention

For auto-discovery to work:
```
results/
├── dataset_name_loss_diff/   # Baseline
└── dataset_name_qvendi/      # Q-Vendi
```

Then run: `python analysis/compare_methods.py --base ./results`

## Example Output

```
Dataset: permuted_mnist
Test       Baseline             Q-Vendi              Improvement     % Change  
test1      0.7234 ± 0.0123      0.7456 ± 0.0098      +0.0222         +3.07%
AVERAGE    0.7234               0.7456               +0.0222         +3.07%
```

## CSV Format

**table1_metrics_comparison.csv**:
```csv
dataset,test,baseline_mean,baseline_std,qvendi_mean,qvendi_std,improvement,improvement_pct,mode
mnist,test1,0.723400,0.012300,0.745600,0.009800,0.022200,3.07,mean
```

**table1_summary_comparison.csv**:
```csv
dataset,baseline_mean,qvendi_mean,improvement,improvement_pct,n_tests,mode
mnist,0.723400,0.745600,0.022200,3.07,1,mean
```

## Quick Checklist

- [ ] Install scipy: `pip install scipy>=1.9.0`
- [ ] Run baseline with `--use_qvendi 0`
- [ ] Run q_vendi with `--use_qvendi 1`
- [ ] Keep all other parameters identical
- [ ] Use consistent seed for reproducibility
- [ ] Run comparison script
- [ ] Check CSV files in `analysis/` directory

## Common Issues

**"No paired experiments found"**
→ Use explicit paths: `--baseline PATH --qvendi PATH`

**"No matching test directories"**
→ Ensure both runs completed and created test*/buffer/ structure

**Results look identical**
→ Verify `--use_qvendi 1` was passed and scipy is installed
