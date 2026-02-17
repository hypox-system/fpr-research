"""
Multiple testing corrections (analysis/multiple_testing.py).

Implements BH-FDR (Benjamini-Hochberg False Discovery Rate).
"""

import numpy as np
from typing import List, Tuple


def benjamini_hochberg(
    p_values: List[float],
    alpha: float = 0.05
) -> List[bool]:
    """
    Benjamini-Hochberg FDR correction.

    Controls the expected proportion of false positives among
    rejected hypotheses.

    Args:
        p_values: List of p-values (one per hypothesis)
        alpha: Target FDR level (default 0.05)

    Returns:
        List of booleans (True = significant after correction)
    """
    n = len(p_values)
    if n == 0:
        return []

    p_values = np.array(p_values)

    # Handle NaN values (treat as non-significant)
    nan_mask = np.isnan(p_values)
    p_values = np.where(nan_mask, 1.0, p_values)

    # Sort p-values and get original indices
    sorted_indices = np.argsort(p_values)
    sorted_pvals = p_values[sorted_indices]

    # BH critical values: (rank / n) * alpha
    ranks = np.arange(1, n + 1)
    critical_values = (ranks / n) * alpha

    # Find largest k where p[k] <= critical_value[k]
    below_threshold = sorted_pvals <= critical_values

    # Find cutoff: all hypotheses with rank <= k are significant
    if not np.any(below_threshold):
        # No significant results
        return [False] * n

    max_k = np.max(np.where(below_threshold)[0])

    # Create result array
    significant = np.zeros(n, dtype=bool)
    significant[sorted_indices[:max_k + 1]] = True

    # Mask out NaN values
    significant[nan_mask] = False

    return significant.tolist()


def benjamini_hochberg_adjusted(
    p_values: List[float]
) -> List[float]:
    """
    Compute BH-adjusted p-values.

    Adjusted p-value = smallest FDR at which the hypothesis
    would be rejected.

    Args:
        p_values: List of p-values

    Returns:
        List of adjusted p-values
    """
    n = len(p_values)
    if n == 0:
        return []

    p_values = np.array(p_values)

    # Handle NaN
    nan_mask = np.isnan(p_values)
    p_values = np.where(nan_mask, 1.0, p_values)

    # Sort
    sorted_indices = np.argsort(p_values)
    sorted_pvals = p_values[sorted_indices]

    # Compute adjusted p-values
    ranks = np.arange(1, n + 1)
    adjusted = np.minimum.accumulate((sorted_pvals * n / ranks)[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)

    # Restore original order
    result = np.zeros(n)
    result[sorted_indices] = adjusted

    # Restore NaN
    result[nan_mask] = np.nan

    return result.tolist()


def holm_bonferroni(
    p_values: List[float],
    alpha: float = 0.05
) -> List[bool]:
    """
    Holm-Bonferroni step-down correction.

    More conservative than BH but controls FWER.

    Args:
        p_values: List of p-values
        alpha: Target FWER level

    Returns:
        List of booleans (True = significant)
    """
    n = len(p_values)
    if n == 0:
        return []

    p_values = np.array(p_values)
    nan_mask = np.isnan(p_values)
    p_values = np.where(nan_mask, 1.0, p_values)

    sorted_indices = np.argsort(p_values)
    sorted_pvals = p_values[sorted_indices]

    # Holm critical values: alpha / (n - rank + 1)
    critical_values = alpha / (n - np.arange(n))

    # Find first non-rejected (step-down procedure)
    rejected = np.zeros(n, dtype=bool)
    for i, (p, crit) in enumerate(zip(sorted_pvals, critical_values)):
        if p > crit:
            break
        rejected[sorted_indices[i]] = True

    rejected[nan_mask] = False
    return rejected.tolist()


def bonferroni(
    p_values: List[float],
    alpha: float = 0.05
) -> List[bool]:
    """
    Simple Bonferroni correction.

    Most conservative: rejects if p < alpha/n.

    Args:
        p_values: List of p-values
        alpha: Target FWER level

    Returns:
        List of booleans (True = significant)
    """
    n = len(p_values)
    if n == 0:
        return []

    threshold = alpha / n
    return [p < threshold and not np.isnan(p) for p in p_values]
