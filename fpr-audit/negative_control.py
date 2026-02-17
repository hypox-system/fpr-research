"""
Negative control via surrogate data (analysis/negative_control.py).

IAAFT (Iterative Amplitude Adjusted Fourier Transform) surrogates
preserve the autocorrelation and distribution of the original data
while destroying predictability.

If a strategy works on surrogates, it's likely overfitting to
statistical properties rather than finding real edge.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
import warnings


@dataclass
class NegativeControlResult:
    """Result from negative control test for one variant."""
    variant_id: str
    real_sharpe: float
    surrogate_sharpes: np.ndarray
    surrogate_pctl: float  # Percentile rank of real Sharpe
    surrogate_mean: float
    surrogate_std: float
    surrogate_z: float  # Z-score
    flagged: bool  # True if real < threshold percentile


@dataclass
class NegativeControlReport:
    """Report from negative control on multiple variants."""
    results: List[NegativeControlResult]
    n_flagged: int
    threshold_pctl: float


def iaaft_surrogate(
    data: np.ndarray,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate IAAFT (Iterative Amplitude Adjusted Fourier Transform) surrogate.

    IAAFT preserves:
    - Amplitude spectrum (autocorrelation structure)
    - Marginal distribution

    But destroys:
    - Nonlinear dependencies
    - Predictable patterns

    Args:
        data: Original time series
        max_iterations: Maximum IAAFT iterations
        tolerance: Convergence tolerance
        seed: Random seed

    Returns:
        Surrogate time series
    """
    rng = np.random.default_rng(seed)
    n = len(data)

    if n < 4:
        return data.copy()

    # Store original amplitude spectrum
    original_fft = np.fft.rfft(data)
    original_amplitudes = np.abs(original_fft)

    # Store sorted values for distribution matching
    sorted_data = np.sort(data)

    # Initialize with shuffled data
    surrogate = rng.permutation(data).astype(float)

    for _ in range(max_iterations):
        old_surrogate = surrogate.copy()

        # Step 1: Match amplitude spectrum
        surrogate_fft = np.fft.rfft(surrogate)
        phases = np.angle(surrogate_fft)
        surrogate_fft = original_amplitudes * np.exp(1j * phases)
        surrogate = np.fft.irfft(surrogate_fft, n=n)

        # Step 2: Match marginal distribution
        ranks = np.argsort(np.argsort(surrogate))
        surrogate = sorted_data[ranks]

        # Check convergence
        if np.max(np.abs(surrogate - old_surrogate)) < tolerance:
            break

    return surrogate


def generate_surrogates(
    data: np.ndarray,
    n_surrogates: int,
    seed: Optional[int] = None
) -> List[np.ndarray]:
    """
    Generate multiple IAAFT surrogates.

    Args:
        data: Original time series
        n_surrogates: Number of surrogates to generate
        seed: Base random seed

    Returns:
        List of surrogate time series
    """
    surrogates = []
    rng = np.random.default_rng(seed)

    for i in range(n_surrogates):
        surrogate_seed = rng.integers(0, 2**31)
        surrogate = iaaft_surrogate(data, seed=surrogate_seed)
        surrogates.append(surrogate)

    return surrogates


def negative_control(
    variant_results: List[dict],
    price_data: np.ndarray,
    run_strategy_fn,
    n_surrogates: int = 100,
    threshold_pctl: float = 90.0,
    seed: int = 42
) -> NegativeControlReport:
    """
    Run negative control test on top variants.

    For each variant:
    1. Generate n_surrogates IAAFT surrogates of price data
    2. Run strategy on each surrogate
    3. Compute distribution of surrogate Sharpes
    4. Flag if real Sharpe < threshold_pctl of surrogate distribution

    Args:
        variant_results: List of variant dicts with 'variant_id', 'real_sharpe', 'strategy'
        price_data: Original price series (close prices)
        run_strategy_fn: Function(strategy, prices) -> sharpe
        n_surrogates: Number of surrogates per variant
        threshold_pctl: Percentile threshold for flagging (default 90)
        seed: Random seed

    Returns:
        NegativeControlReport with results for all variants
    """
    results = []
    rng = np.random.default_rng(seed)

    # Generate surrogates once (reuse across variants)
    surrogates = generate_surrogates(price_data, n_surrogates, seed=seed)

    for var in variant_results:
        variant_id = var['variant_id']
        real_sharpe = var['real_sharpe']
        strategy = var.get('strategy')

        if strategy is None:
            continue

        # Run strategy on all surrogates
        surrogate_sharpes = []
        for surrogate in surrogates:
            try:
                sharpe = run_strategy_fn(strategy, surrogate)
                surrogate_sharpes.append(sharpe)
            except Exception:
                surrogate_sharpes.append(0.0)

        surrogate_sharpes = np.array(surrogate_sharpes)

        # Compute statistics
        surrogate_mean = np.mean(surrogate_sharpes)
        surrogate_std = np.std(surrogate_sharpes)

        # Percentile rank: what % of surrogate Sharpes are <= real Sharpe?
        surrogate_pctl = 100 * np.mean(surrogate_sharpes <= real_sharpe)

        # Z-score
        if surrogate_std > 0:
            surrogate_z = (real_sharpe - surrogate_mean) / surrogate_std
        else:
            surrogate_z = 0.0

        # Flag if real Sharpe is NOT extreme (< threshold percentile)
        flagged = surrogate_pctl < threshold_pctl

        results.append(NegativeControlResult(
            variant_id=variant_id,
            real_sharpe=real_sharpe,
            surrogate_sharpes=surrogate_sharpes,
            surrogate_pctl=surrogate_pctl,
            surrogate_mean=surrogate_mean,
            surrogate_std=surrogate_std,
            surrogate_z=surrogate_z,
            flagged=flagged,
        ))

    n_flagged = sum(1 for r in results if r.flagged)

    return NegativeControlReport(
        results=results,
        n_flagged=n_flagged,
        threshold_pctl=threshold_pctl,
    )


def simple_negative_control(
    real_sharpes: List[float],
    surrogate_sharpes_list: List[np.ndarray],
    threshold_pctl: float = 90.0
) -> List[NegativeControlResult]:
    """
    Simplified negative control with pre-computed surrogate Sharpes.

    Args:
        real_sharpes: List of real Sharpe ratios
        surrogate_sharpes_list: List of surrogate Sharpe arrays (one per variant)
        threshold_pctl: Percentile threshold for flagging

    Returns:
        List of NegativeControlResult objects
    """
    results = []

    for i, (real_sharpe, surrogate_sharpes) in enumerate(zip(real_sharpes, surrogate_sharpes_list)):
        surrogate_sharpes = np.array(surrogate_sharpes)

        surrogate_mean = np.mean(surrogate_sharpes)
        surrogate_std = np.std(surrogate_sharpes)
        surrogate_pctl = 100 * np.mean(surrogate_sharpes <= real_sharpe)

        if surrogate_std > 0:
            surrogate_z = (real_sharpe - surrogate_mean) / surrogate_std
        else:
            surrogate_z = 0.0

        flagged = surrogate_pctl < threshold_pctl

        results.append(NegativeControlResult(
            variant_id=f"variant_{i}",
            real_sharpe=real_sharpe,
            surrogate_sharpes=surrogate_sharpes,
            surrogate_pctl=surrogate_pctl,
            surrogate_mean=surrogate_mean,
            surrogate_std=surrogate_std,
            surrogate_z=surrogate_z,
            flagged=flagged,
        ))

    return results
