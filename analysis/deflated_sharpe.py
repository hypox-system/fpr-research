"""
Deflated Sharpe Ratio (analysis/deflated_sharpe.py).

Bailey & Lopez de Prado 2014: accounts for multiple testing
and non-normality in Sharpe ratio estimation.
"""

import numpy as np
from scipy import stats
from typing import Optional


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_variants_tested: int,
    n_observations: int,
    sharpe_std: Optional[float] = None,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    annualization_factor: float = np.sqrt(252)
) -> float:
    """
    Compute Deflated Sharpe Ratio (DSR) p-value.

    The DSR accounts for:
    1. Multiple testing (via n_variants_tested)
    2. Non-normality of returns (via skewness/kurtosis)
    3. Estimation error (via n_observations)

    Args:
        observed_sharpe: Observed Sharpe ratio (annualized)
        n_variants_tested: Total number of variants tested
        n_observations: Number of return observations (trades)
        sharpe_std: Standard deviation of Sharpe estimates (optional)
        skewness: Skewness of trade returns (default 0)
        kurtosis: Kurtosis of trade returns (default 3, normal)
        annualization_factor: Factor for annualization (default sqrt(252))

    Returns:
        DSR p-value (probability that observed Sharpe is due to chance)
    """
    if n_observations < 2:
        return 1.0

    if n_variants_tested < 1:
        n_variants_tested = 1

    # Expected maximum Sharpe under null (multiple testing)
    # Using approximation from Bailey & Lopez de Prado
    expected_max_sharpe = expected_max_sharpe_null(n_variants_tested, n_observations)

    # Standard error of Sharpe ratio
    # Accounts for non-normality via skewness and kurtosis
    if sharpe_std is None:
        sharpe_std = sharpe_ratio_std(
            observed_sharpe / annualization_factor,  # Convert to per-period
            n_observations,
            skewness,
            kurtosis
        )
        sharpe_std *= annualization_factor  # Annualize

    if sharpe_std <= 0:
        return 1.0

    # Test statistic: how many SDs above expected max?
    z_score = (observed_sharpe - expected_max_sharpe) / sharpe_std

    # One-sided p-value
    p_value = 1 - stats.norm.cdf(z_score)

    return float(max(0, min(1, p_value)))


def expected_max_sharpe_null(
    n_variants: int,
    n_observations: int
) -> float:
    """
    Expected maximum Sharpe ratio under null hypothesis.

    When testing n_variants, the maximum will be inflated
    even if all strategies have zero true Sharpe.

    Uses approximation: E[max] ≈ σ * sqrt(2 * log(n))
    where σ is the standard error of Sharpe.

    Args:
        n_variants: Number of variants tested
        n_observations: Number of observations per variant

    Returns:
        Expected maximum Sharpe under null
    """
    if n_variants <= 1:
        return 0.0

    if n_observations < 2:
        return 0.0

    # Standard error of Sharpe (assuming normal returns, zero true Sharpe)
    se_sharpe = 1.0 / np.sqrt(n_observations)

    # Expected maximum of n_variants standard normals
    expected_max = se_sharpe * np.sqrt(2 * np.log(n_variants))

    # Correction for finite samples (Euler-Mascheroni constant)
    gamma = 0.5772156649
    expected_max *= (1 - gamma / (2 * np.log(n_variants)))

    return float(expected_max)


def sharpe_ratio_std(
    sharpe: float,
    n: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0
) -> float:
    """
    Standard error of Sharpe ratio accounting for non-normality.

    From Lo (2002) and Mertens (2002):
    Var(SR) ≈ (1 + 0.5*SR^2 - skew*SR + (kurtosis-3)/4 * SR^2) / n

    Args:
        sharpe: Per-period Sharpe ratio
        n: Number of observations
        skewness: Return skewness
        kurtosis: Return kurtosis

    Returns:
        Standard error of Sharpe ratio
    """
    if n < 2:
        return float('inf')

    sr2 = sharpe ** 2
    excess_kurtosis = kurtosis - 3

    variance = (1 + 0.5 * sr2 - skewness * sharpe + excess_kurtosis / 4 * sr2) / n

    return float(np.sqrt(max(0, variance)))


def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    benchmark_sharpe: float,
    n_observations: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    annualization_factor: float = np.sqrt(252)
) -> float:
    """
    Probabilistic Sharpe Ratio (PSR).

    Probability that true Sharpe exceeds benchmark.

    Args:
        observed_sharpe: Observed Sharpe (annualized)
        benchmark_sharpe: Benchmark Sharpe to beat (annualized)
        n_observations: Number of observations
        skewness: Return skewness
        kurtosis: Return kurtosis
        annualization_factor: Annualization factor

    Returns:
        Probability (0-1) that true Sharpe > benchmark
    """
    if n_observations < 2:
        return 0.5

    # Convert to per-period
    obs_pp = observed_sharpe / annualization_factor
    bench_pp = benchmark_sharpe / annualization_factor

    # Standard error
    se = sharpe_ratio_std(obs_pp, n_observations, skewness, kurtosis)

    if se <= 0:
        return 1.0 if observed_sharpe > benchmark_sharpe else 0.0

    # Z-score
    z = (obs_pp - bench_pp) / se

    # Probability
    psr = stats.norm.cdf(z)

    return float(psr)
