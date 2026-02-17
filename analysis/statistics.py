"""
Permutation-based statistical tests (analysis/statistics.py).

Delta 40: Vectorized implementation, no Python loop per variant.
"""

import numpy as np
from typing import List, Optional, Tuple


def permutation_pvalue(
    trade_returns: np.ndarray,
    n_permutations: int = 10000,
    seed: Optional[int] = None
) -> float:
    """
    Sign-flip permutation test for trade returns.

    H0: trade returns have zero mean (no edge)
    Test statistic: mean(trade_returns)
    Permutation: flip signs randomly
    p = mean(null_means >= observed_mean)  # one-sided

    IMPORTANT: trade_returns MUST be NET (after fees+slippage).
    Computed via engine/trades.py compute_trade_return() - NEVER inline.

    Delta 40: Vectorized implementation using sign-flip matrix.

    Args:
        trade_returns: Array of net trade returns
        n_permutations: Number of permutations
        seed: Random seed for reproducibility

    Returns:
        One-sided p-value (probability of observing this edge by chance)
    """
    if len(trade_returns) == 0:
        return 1.0

    trade_returns = np.asarray(trade_returns)
    n_trades = len(trade_returns)

    # Observed test statistic (mean return)
    observed_mean = np.mean(trade_returns)

    # If observed is negative or zero, cannot reject H0
    if observed_mean <= 0:
        return 1.0

    # Set random seed
    rng = np.random.default_rng(seed)

    # Generate sign-flip matrix: (n_permutations, n_trades)
    # Each element is +1 or -1
    signs = rng.choice([-1, 1], size=(n_permutations, n_trades))

    # Apply sign flips to trade returns
    permuted_returns = trade_returns * signs  # Broadcasting

    # Compute null distribution of means
    null_means = np.mean(permuted_returns, axis=1)

    # One-sided p-value: proportion of null means >= observed
    p_value = np.mean(null_means >= observed_mean)

    return float(p_value)


def permutation_pvalue_batch(
    trade_returns_list: List[np.ndarray],
    n_permutations: int = 10000,
    seed: Optional[int] = None
) -> List[float]:
    """
    Batch permutation p-values for multiple variants.

    More efficient than calling permutation_pvalue individually.

    Args:
        trade_returns_list: List of trade return arrays
        n_permutations: Number of permutations per variant
        seed: Random seed

    Returns:
        List of p-values
    """
    rng = np.random.default_rng(seed)

    p_values = []
    for returns in trade_returns_list:
        if len(returns) == 0:
            p_values.append(1.0)
            continue

        returns = np.asarray(returns)
        n_trades = len(returns)
        observed_mean = np.mean(returns)

        if observed_mean <= 0:
            p_values.append(1.0)
            continue

        # Generate signs for this variant
        signs = rng.choice([-1, 1], size=(n_permutations, n_trades))
        permuted_returns = returns * signs
        null_means = np.mean(permuted_returns, axis=1)

        p_value = np.mean(null_means >= observed_mean)
        p_values.append(float(p_value))

    return p_values


def bootstrap_confidence_interval(
    trade_returns: np.ndarray,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: Optional[int] = None
) -> Tuple[float, float]:
    """
    Bootstrap confidence interval for mean return.

    Args:
        trade_returns: Array of trade returns
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default 95%)
        seed: Random seed

    Returns:
        Tuple of (lower, upper) confidence bounds
    """
    if len(trade_returns) == 0:
        return (0.0, 0.0)

    rng = np.random.default_rng(seed)
    n = len(trade_returns)

    # Bootstrap resampling
    boot_indices = rng.integers(0, n, size=(n_bootstrap, n))
    boot_samples = trade_returns[boot_indices]
    boot_means = np.mean(boot_samples, axis=1)

    # Percentile method
    alpha = 1 - confidence
    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))

    return (float(lower), float(upper))
