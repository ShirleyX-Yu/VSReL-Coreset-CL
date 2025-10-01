# -*-coding:utf8-*-
"""
Quality-Weighted Vendi Score implementation
Based on: https://github.com/vertaix/Quality-Weighted-Vendi-Score
"""

import numpy as np
from scipy.linalg import eigh


def score(samples, k, s):
    """
    Compute the Quality-Weighted Vendi Score.
    
    Args:
        samples: list of samples
        k: similarity function, should be symmetric and k(x, x) = 1
        s: score function that assigns quality to each sample
    
    Returns:
        qVS: Quality-Weighted Vendi Score
    """
    n = len(samples)
    if n == 0:
        return 0.0
    
    # Compute similarity matrix
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = k(samples[i], samples[j])
    
    # Compute quality scores
    scores = np.array([s(sample) for sample in samples])
    avg_score = np.mean(scores)
    
    # Normalize K
    K_normalized = K / n
    
    # Compute eigenvalues
    eigenvalues = eigh(K_normalized, eigvals_only=True)
    eigenvalues = np.maximum(eigenvalues, 0)  # Ensure non-negative
    
    # Compute entropy term: -sum(lambda_i * log(lambda_i))
    entropy = 0.0
    for lam in eigenvalues:
        if lam > 1e-12:  # Avoid log(0)
            entropy -= lam * np.log(lam)
    
    # qVS = avg_score * exp(entropy)
    qvs = avg_score * np.exp(entropy)
    
    return qvs


def sequential_maximize_score(samples, k, s, budget, initial_set=None):
    """
    Greedily select a subset that maximizes the Quality-Weighted Vendi Score.
    
    Args:
        samples: list of all candidate samples
        k: similarity function
        s: score function
        budget: number of samples to select
        initial_set: optional list of initial selected samples
    
    Returns:
        selected_samples: list of selected samples
        qvs: final Quality-Weighted Vendi Score
    """
    if initial_set is None:
        selected = []
    else:
        selected = list(initial_set)
    
    remaining = [sample for sample in samples if sample not in selected]
    
    while len(selected) < budget and len(remaining) > 0:
        best_sample = None
        best_score = -float('inf')
        
        for candidate in remaining:
            # Evaluate qVS with this candidate added
            test_set = selected + [candidate]
            candidate_score = score(test_set, k, s)
            
            if candidate_score > best_score:
                best_score = candidate_score
                best_sample = candidate
        
        if best_sample is not None:
            selected.append(best_sample)
            remaining.remove(best_sample)
        else:
            break
    
    final_qvs = score(selected, k, s)
    return selected, final_qvs


def score_from_kernel_matrix(K, quality_scores):
    """
    Compute qVS directly from a precomputed kernel matrix and quality scores.
    
    Args:
        K: (n, n) similarity/kernel matrix
        quality_scores: (n,) array of quality scores for each sample
    
    Returns:
        qVS: Quality-Weighted Vendi Score
    """
    n = K.shape[0]
    if n == 0:
        return 0.0
    
    avg_score = np.mean(quality_scores)
    
    # Normalize K
    K_normalized = K / n
    
    # Compute eigenvalues
    eigenvalues = eigh(K_normalized, eigvals_only=True)
    eigenvalues = np.maximum(eigenvalues, 0)
    
    # Compute entropy
    entropy = 0.0
    for lam in eigenvalues:
        if lam > 1e-12:
            entropy -= lam * np.log(lam)
    
    qvs = avg_score * np.exp(entropy)
    return qvs
