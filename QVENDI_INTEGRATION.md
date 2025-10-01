# Quality-Weighted Vendi Score Integration

This document describes the integration of Quality-Weighted Vendi Score (q_vendi) into the coreset selection process.

## Overview

The original CSReL (Coreset Selection via Reducible Loss) method selects samples by ranking them according to their **loss difference** (reducible loss):

```
loss_diff = current_loss - reference_loss
```

Samples with the highest loss difference are greedily selected for the coreset.

With the **q_vendi integration**, the selection process now considers both:
1. **Quality**: The loss difference (reducible loss) of each sample
2. **Diversity**: Feature similarity between samples

This results in a more diverse and representative coreset.

## Mathematical Formulation

The Quality-Weighted Vendi Score is defined as:

```
qVS(K, s) = (mean(s)) * exp(entropy(K))
```

Where:
- `K` is the similarity matrix computed from sample features (using cosine similarity)
- `s` is the quality score vector (loss differences)
- `entropy(K) = -sum(λ_i * log(λ_i))` where λ_i are eigenvalues of K/n

The greedy selection algorithm maximizes the qVS at each step by selecting the sample that, when added to the current coreset, produces the highest qVS.

## Implementation Details

### Files Modified/Created

1. **`coreset_selection/q_vendi.py`** (NEW)
   - Core implementation of Quality-Weighted Vendi Score
   - Functions: `score()`, `sequential_maximize_score()`, `score_from_kernel_matrix()`

2. **`coreset_selection/coreset_selection_functions.py`** (MODIFIED)
   - Added `use_qvendi` parameter to `select_by_loss_diff()`
   - Added `_select_by_qvendi()` helper function
   - Extracts features from model's penultimate layer using forward hooks
   - Computes cosine similarity matrix from features
   - Performs greedy selection to maximize qVS

3. **`coreset_selection/selection_agent.py`** (MODIFIED)
   - Added `use_qvendi` parameter to `RhoSelectionAgent.__init__()`
   - Passes parameter through to selection function

4. **`continual_learning/coreset_buffer.py`** (MODIFIED)
   - Reads `use_qvendi` from `selection_params`
   - Passes to `RhoSelectionAgent`

5. **`permuted_mnist_cl.py`** (MODIFIED)
   - Added `--use_qvendi` command line argument
   - Added to `selection_params` in `make_selection_params()`

6. **`requirements.txt`** (MODIFIED)
   - Added `scipy>=1.9.0` dependency (required for eigenvalue computation)

### Feature Extraction

When `use_qvendi=True`, the selection function:
1. Registers a forward hook on the model's pooling layer (before the final classifier)
2. Extracts feature representations for each sample during the forward pass
3. Flattens features if needed
4. Stores features in `id2features` dictionary

### Selection Algorithm

The `_select_by_qvendi()` function:
1. Builds feature matrix and quality score vector from all candidates
2. Normalizes quality scores to be positive
3. Computes cosine similarity matrix: `K = features_norm @ features_norm.T`
4. Greedily selects samples:
   - For each remaining candidate, compute qVS if added to current selection
   - Select the candidate with highest qVS
   - Respect class balance constraints if specified
5. Returns selected samples in order of selection

## Usage

### Command Line

Add the `--use_qvendi 1` flag to enable q_vendi selection:

```bash
python permuted_mnist_cl.py \
    --local_path ./results/qvendi_test \
    --dataset mnist \
    --setting permuted \
    --buffer_size 200 \
    --use_qvendi 1 \
    --seed 42 \
    # ... other arguments
```

### Programmatic

Set `use_qvendi=True` in `selection_params`:

```python
selection_params = {
    'init_size': 0,
    'class_balance': False,
    'only_new_data': True,
    'cur_train_steps': 30,
    'cur_train_lr': 2e-2,
    'selection_steps': 100,
    'use_qvendi': True,  # Enable q_vendi
    # ... other parameters
}
```

## Comparison: Original vs Q_Vendi

| Aspect | Original CSReL | With Q_Vendi |
|--------|---------------|--------------|
| **Selection Criterion** | Greedy by loss_diff (quality only) | Greedy by qVS (quality + diversity) |
| **Diversity** | Implicit (through iterative training) | Explicit (through feature similarity) |
| **Computational Cost** | Lower (simple sorting) | Higher (eigenvalue computation per candidate) |
| **Feature Extraction** | Not needed | Required (forward hook on model) |
| **Dependencies** | None extra | scipy (for eigenvalue computation) |

## Performance Considerations

- **Time Complexity**: O(n * k * d²) where n=candidates, k=selection size, d=feature dimension
  - Each iteration evaluates O(n) candidates
  - Each evaluation computes eigenvalues of a k×k matrix
- **Memory**: Stores full similarity matrix (n×n) and features (n×d)
- **Recommendation**: Use for moderate-sized candidate pools (< 10,000 samples)

## Example Script

See `scripts/run_mnist.sh` - add `use_qvendi=1`:

```bash
#!/bin/bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8

local_path='./results/qvendi_mnist'
dataset='mnist'
setting='permuted'
buffer_size=200
use_qvendi=1  # Enable q_vendi

python permuted_mnist_cl.py \
    --local_path=$local_path \
    --dataset=$dataset \
    --setting=$setting \
    --buffer_size=$buffer_size \
    --use_qvendi=$use_qvendi \
    --seed=42
```

## References

- **Quality-Weighted Vendi Score**: [https://github.com/vertaix/Quality-Weighted-Vendi-Score](https://github.com/vertaix/Quality-Weighted-Vendi-Score)
- **Paper**: Nguyen & Dieng, "Quality-Weighted Vendi Scores And Their Application To Diverse Experimental Design", ICML 2024
- **ArXiv**: [https://arxiv.org/abs/2405.02449](https://arxiv.org/abs/2405.02449)

## Troubleshooting

### Issue: "No module named 'scipy'"
**Solution**: Install scipy: `pip install scipy>=1.9.0`

### Issue: Features not extracted (empty id2features)
**Solution**: Check that your model has a recognizable pooling layer. The hook looks for layers with 'avgpool', 'pool', or 'flatten' in the name.

### Issue: Very slow selection
**Solution**: Reduce `selection_steps` or candidate pool size. Q_vendi is more computationally expensive than simple loss_diff ranking.

### Issue: NaN or Inf in qVS computation
**Solution**: Check that quality scores are positive and features are not all zeros. The implementation adds small epsilon values to prevent numerical issues.
