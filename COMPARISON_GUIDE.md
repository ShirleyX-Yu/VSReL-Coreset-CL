# Guide: Comparing Loss_Diff vs Q_Vendi Methods

This guide explains how to run experiments and generate comparison tables between the baseline loss_diff method and the q_vendi method.

## Quick Start

### 1. Run Both Methods

Use the provided comparison script:

```bash
chmod +x scripts/run_comparison_mnist.sh
./scripts/run_comparison_mnist.sh
```

This will:
- Run the baseline (loss_diff only) and save to `./results/permuted_mnist_loss_diff`
- Run q_vendi method and save to `./results/permuted_mnist_qvendi`

### 2. Generate Comparison Tables

After both runs complete:

```bash
python analysis/compare_methods.py \
    --baseline ./results/permuted_mnist_loss_diff \
    --qvendi ./results/permuted_mnist_qvendi
```

This generates:
- **Console output**: Formatted comparison tables
- **`table1_metrics_comparison.csv`**: Detailed per-test metrics
- **`table1_summary_comparison.csv`**: Dataset-level summaries

## Directory Structure

Your results should be organized as:

```
results/
├── dataset_name_loss_diff/    # Baseline runs
│   ├── test1/
│   │   └── buffer/
│   │       ├── metrics_0.pkl
│   │       ├── metrics_1.pkl
│   │       └── ...
│   ├── test2/
│   └── ...
└── dataset_name_qvendi/       # Q-Vendi runs
    ├── test1/
    │   └── buffer/
    │       ├── metrics_0.pkl
    │       └── ...
    └── ...
```

## Manual Comparison Setup

### Step 1: Run Baseline (Loss_Diff)

```bash
python permuted_mnist_cl.py \
    --local_path ./results/my_dataset_loss_diff \
    --dataset mnist \
    --setting permuted \
    --buffer_size 200 \
    --use_qvendi 0 \
    --seed 42 \
    # ... other parameters
```

### Step 2: Run Q_Vendi

```bash
python permuted_mnist_cl.py \
    --local_path ./results/my_dataset_qvendi \
    --dataset mnist \
    --setting permuted \
    --buffer_size 200 \
    --use_qvendi 1 \
    --seed 42 \
    # ... other parameters (keep same as baseline!)
```

**Important**: Keep all parameters identical except `--use_qvendi` and `--local_path` for fair comparison.

### Step 3: Compare Results

```bash
python analysis/compare_methods.py \
    --baseline ./results/my_dataset_loss_diff \
    --qvendi ./results/my_dataset_qvendi
```

## Auto-Discovery Mode

If you organize your results with naming convention `*_loss_diff` and `*_qvendi`, the script can auto-discover pairs:

```bash
# Auto-discover all pairs in results/
python analysis/compare_methods.py --base ./results
```

This will find and compare all matching pairs automatically.

## Output Format

### Console Output

```
================================================================================
Dataset: permuted_mnist
  Baseline: ./results/permuted_mnist_loss_diff
  Q-Vendi:  ./results/permuted_mnist_qvendi
================================================================================

Test       Baseline             Q-Vendi              Improvement     % Change  
--------------------------------------------------------------------------------
test1      0.7234 ± 0.0123      0.7456 ± 0.0098      +0.0222         +3.07%
test2      0.7189 ± 0.0145      0.7398 ± 0.0112      +0.0209         +2.91%
--------------------------------------------------------------------------------
AVERAGE    0.7212               0.7427               +0.0215         +2.98%
```

### CSV Files

**table1_metrics_comparison.csv** - Detailed per-test results:
```csv
dataset,test,baseline_mean,baseline_std,qvendi_mean,qvendi_std,improvement,improvement_pct,mode
permuted_mnist,test1,0.723400,0.012300,0.745600,0.009800,0.022200,3.07,mean
permuted_mnist,test2,0.718900,0.014500,0.739800,0.011200,0.020900,2.91,mean
```

**table1_summary_comparison.csv** - Dataset-level summaries:
```csv
dataset,baseline_mean,qvendi_mean,improvement,improvement_pct,n_tests,mode
permuted_mnist,0.721200,0.742700,0.021500,2.98,2,mean
```

## Comparison Script Options

```bash
python analysis/compare_methods.py --help
```

Options:
- `--baseline PATH`: Path to baseline results root
- `--qvendi PATH`: Path to qvendi results root
- `--base PATH`: Base directory for auto-discovery (default: ../results)
- `--use-last`: Use last-task accuracy instead of mean over tasks
- `--csv-metrics FILE`: Output filename for detailed metrics CSV
- `--csv-accuracies FILE`: Output filename for accuracies CSV

## Multiple Datasets

To compare multiple datasets at once:

```bash
# Run experiments for multiple datasets
./scripts/run_comparison_mnist.sh
./scripts/run_comparison_cifar.sh
# ... etc

# Auto-compare all
python analysis/compare_methods.py --base ./results
```

The script will generate a grand summary across all datasets.

## Interpreting Results

### Metrics Explained

- **Baseline Mean**: Average accuracy using loss_diff selection only
- **Q-Vendi Mean**: Average accuracy using q_vendi (loss_diff as quality + diversity)
- **Improvement**: Absolute difference (Q-Vendi - Baseline)
- **% Change**: Relative improvement percentage

### What to Look For

1. **Positive Improvement**: Q-Vendi should show improvement if diversity helps
2. **Consistency**: Check if improvement is consistent across multiple test runs
3. **Statistical Significance**: Compare standard deviations to assess reliability
4. **Trade-offs**: Note any increase in computation time (q_vendi is slower)

## Example Workflow

```bash
# 1. Create comparison script for your dataset
cat > scripts/run_comparison_my_dataset.sh << 'EOF'
#!/bin/bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Common parameters
dataset='my_dataset'
buffer_size=500
seed=42
# ... other params

# Baseline
python permuted_mnist_cl.py \
    --local_path ./results/my_dataset_loss_diff \
    --use_qvendi 0 \
    --seed $seed \
    # ... other params

# Q-Vendi
python permuted_mnist_cl.py \
    --local_path ./results/my_dataset_qvendi \
    --use_qvendi 1 \
    --seed $seed \
    # ... other params
EOF

# 2. Run experiments
chmod +x scripts/run_comparison_my_dataset.sh
./scripts/run_comparison_my_dataset.sh

# 3. Generate comparison
python analysis/compare_methods.py \
    --baseline ./results/my_dataset_loss_diff \
    --qvendi ./results/my_dataset_qvendi

# 4. View results
cat analysis/table1_metrics_comparison.csv
```

## Troubleshooting

### No matching test directories found

**Problem**: The baseline and qvendi directories have different test directory names.

**Solution**: Ensure both runs create the same test directory structure (e.g., test1, test2, etc.).

### No paired experiments found

**Problem**: Directory naming doesn't follow `*_loss_diff` and `*_qvendi` convention.

**Solution**: Either:
- Rename directories to follow convention, OR
- Use explicit `--baseline` and `--qvendi` arguments

### Different number of pkl files

**Problem**: Baseline has 5 pkl files but qvendi has 3.

**Solution**: This is okay - the script computes statistics independently. However, ensure both runs completed successfully.

### Metrics look identical

**Problem**: Both methods show the same results.

**Solution**: 
- Verify `--use_qvendi 1` was actually passed
- Check logs to confirm q_vendi selection was used
- Ensure scipy is installed (`pip install scipy>=1.9.0`)

## Advanced: Custom Metrics

To extract custom metrics from pkl files, modify the `extract_accs()` function in `compare_methods.py`:

```python
def extract_accs(path):
    # Add custom keys to ACC_KEYS or AVG_KEYS
    ACC_KEYS = ["acc_per_task", "my_custom_metric", ...]
    # ... rest of function
```

## Batch Processing

To run multiple seeds and compare:

```bash
for seed in 42 123 456 789 1000; do
    # Baseline
    python permuted_mnist_cl.py \
        --local_path ./results/mnist_loss_diff/test_seed_${seed} \
        --use_qvendi 0 --seed $seed # ... other params
    
    # Q-Vendi
    python permuted_mnist_cl.py \
        --local_path ./results/mnist_qvendi/test_seed_${seed} \
        --use_qvendi 1 --seed $seed # ... other params
done

# Compare
python analysis/compare_methods.py \
    --baseline ./results/mnist_loss_diff \
    --qvendi ./results/mnist_qvendi
```

## Citation

If you use this comparison framework, please cite both:

1. The original CSReL paper (for loss_diff baseline)
2. The Q-Vendi paper: Nguyen & Dieng, "Quality-Weighted Vendi Scores And Their Application To Diverse Experimental Design", ICML 2024
