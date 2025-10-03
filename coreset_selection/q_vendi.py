# -*-coding:utf8-*-
"""
Quality-Weighted Vendi Score (qVS) with generalized order q.

- q = 1.0  -> Shannon/von Neumann entropy case (original Vendi)
- q < 1.0  -> more weight to rare items (higher diversity sensitivity)
- q > 1.0  -> more weight to common items (more conservative)

Assumes k(x, x) = 1 so that trace(K) = n and trace(K/n) = 1.
"""

import numpy as np
from scipy.linalg import eigh


def _vendi_diversity_from_eigs(eigenvalues: np.ndarray, q: float, eps: float = 1e-12) -> float:
    """
    Compute the diversity term from eigenvalues of K_normalized for arbitrary q.

    Args:
        eigenvalues: 1D array of eigenvalues of K_normalized (should sum ~ 1).
        q: order parameter.
        eps: numerical floor to avoid log(0) etc.

    Returns:
        diversity: scalar diversity term (VS_q).
    """
    # clip tiny negatives due to numeric noise and (optionally) renormalize
    lam = np.clip(eigenvalues, 0.0, None)
    s = lam.sum()
    if s <= eps:
        return 0.0
    lam = lam / s  # keep sum ≈ 1 to stabilize q != 1 formula

    if abs(q - 1.0) < 1e-8:
        # shannon case: exp(-sum lambda_i * log lambda_i))
        # use where to avoid warnings for zeros
        mask = lam > eps
        entropy = -np.sum(lam[mask] * np.log(lam[mask]))
        return float(np.exp(entropy))
    else:
        # general (Hill/Rényi-style) family: (sum lambda_i^q)^(1/(1-q))
        power_sum = float(np.sum(lam ** q))
        # guard against under/overflow with eps
        power_sum = max(power_sum, eps)
        return power_sum ** (1.0 / (1.0 - q))


def score(samples, k, s, q: float = 1.0):
    """
    Compute the Quality-Weighted Vendi Score (qVS) for arbitrary order q.
    
    Args:
        samples: list of samples
        k: similarity function, symmetric with k(x, x) = 1
        s: per-sample quality scoring function
        q: Vendi order parameter (q=1 recovers the original entropy form)
    
    Returns:
        qVS: Quality-Weighted Vendi Score
    """
    n = len(samples)
    if n == 0:
        return 0.0
    
    # similarity matrix
    K = np.empty((n, n), dtype=float)
    for i in range(n):
        xi = samples[i]
        for j in range(n):
            K[i, j] = k(xi, samples[j])
    
    # quality
    scores = np.array([s(sample) for sample in samples], dtype=float)
    avg_score = float(np.mean(scores))
    
    # normalize K and get eigenvalues
    K_normalized = K / n
    eigenvalues = eigh(K_normalized, eigvals_only=True)
    
    # diversity term for chosen q
    diversity = _vendi_diversity_from_eigs(eigenvalues, q)
    
    # qVS = average quality * diversity
    return avg_score * diversity


def sequential_maximize_score(samples, k, s, budget, initial_set=None, q: float = 1.0):
    """
    Greedily select a subset that maximizes qVS for order q.
    
    Args:
        samples: list of all candidate samples
        k: similarity function
        s: quality function
        budget: number of samples to select
        initial_set: optional list of initial selected samples
        q: Vendi order parameter
    
    Returns:
        selected_samples: list of selected samples
        qvs: final qVS value for the selected set
    """
    selected = [] if initial_set is None else list(initial_set)
    
    remaining = [sample for sample in samples if sample not in selected]
    
    while len(selected) < budget and len(remaining) > 0:
        best_sample = None
        best_score = -float('inf')
        
        for candidate in remaining:
            # evaluate qVS with this candidate added
            test_set = selected + [candidate]
            candidate_score = score(test_set, k, s, q=q)
            
            if candidate_score > best_score:
                best_score = candidate_score
                best_sample = candidate
        
        if best_sample is not None:
            selected.append(best_sample)
            remaining.remove(best_sample)
        else:
            break
    
    final_qvs = score(selected, k, s, q=q)
    return selected, final_qvs


def score_from_kernel_matrix(K: np.ndarray, quality_scores: np.ndarray, q: float = 1.0):
    """
    Compute qVS directly from a precomputed kernel matrix K and quality scores.
    
    Args:
        K: (n, n) similarity/kernel matrix with K[i,i] ≈ 1
        quality_scores: (n,) array of per-item quality scores
        q: Vendi order parameter
    
    Returns:
        qVS: Quality-Weighted Vendi Score
    """
    n = K.shape[0]
    if n == 0:
        return 0.0
    
    avg_score = float(np.mean(quality_scores))
    
    # normalize K
    K_normalized = K / n
    
    # compute eigenvalues
    eigenvalues = eigh(K_normalized, eigvals_only=True)
    
    # diversity for general q
    diversity = _vendi_diversity_from_eigs(eigenvalues, q)
    
    return avg_score * diversity


# --- Example usage ---
# qvs_q05 = score(samples, k, s, q=0.5)  # more weight to rare/underrepresented
# qvs_q10 = score(samples, k, s, q=1.0)  # original entropy case
# qvs_q20 = score(samples, k, s, q=2.0)  # emphasizes common / conservative
