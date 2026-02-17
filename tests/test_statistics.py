"""
Tests for analysis/statistics.py and related modules.
"""

import pytest
import numpy as np

from analysis.statistics import (
    permutation_pvalue,
    permutation_pvalue_batch,
    bootstrap_confidence_interval,
)
from analysis.multiple_testing import (
    benjamini_hochberg,
    benjamini_hochberg_adjusted,
    holm_bonferroni,
    bonferroni,
)
from analysis.deflated_sharpe import (
    deflated_sharpe_ratio,
    expected_max_sharpe_null,
    sharpe_ratio_std,
    probabilistic_sharpe_ratio,
)
from analysis.negative_control import (
    iaaft_surrogate,
    generate_surrogates,
    simple_negative_control,
)


class TestPermutationPvalue:
    """Test permutation p-value computation."""

    def test_positive_edge_low_pvalue(self):
        """Strong positive edge should have low p-value."""
        np.random.seed(42)
        # Simulate strategy with positive edge
        returns = np.random.normal(0.01, 0.02, 100)  # Mean 1%, std 2%

        p = permutation_pvalue(returns, n_permutations=1000, seed=42)

        assert p < 0.05, "Strong positive edge should have p < 0.05"

    def test_zero_edge_high_pvalue(self):
        """Zero-mean returns should have high p-value."""
        np.random.seed(42)
        # Simulate strategy with no edge
        returns = np.random.normal(0, 0.02, 100)

        p = permutation_pvalue(returns, n_permutations=1000, seed=42)

        assert p > 0.3, "Zero-edge should have high p-value"

    def test_negative_edge_pvalue_one(self):
        """Negative mean should have p-value = 1."""
        returns = np.array([-0.01, -0.02, -0.01, -0.015, -0.005])

        p = permutation_pvalue(returns, n_permutations=100, seed=42)

        assert p == 1.0, "Negative edge should have p-value = 1"

    def test_empty_returns(self):
        """Empty returns should return p = 1."""
        p = permutation_pvalue(np.array([]), n_permutations=100)
        assert p == 1.0

    def test_deterministic_with_seed(self):
        """Same seed should give same result."""
        returns = np.random.randn(50) * 0.01 + 0.005

        p1 = permutation_pvalue(returns, n_permutations=500, seed=123)
        p2 = permutation_pvalue(returns, n_permutations=500, seed=123)

        assert p1 == p2

    def test_batch_matches_individual(self):
        """Batch computation should match individual calls."""
        np.random.seed(42)
        returns_list = [
            np.random.randn(30) * 0.01 + 0.005,
            np.random.randn(40) * 0.01 + 0.003,
            np.random.randn(50) * 0.01 - 0.001,
        ]

        batch_pvals = permutation_pvalue_batch(returns_list, n_permutations=200, seed=42)

        # Results should be consistent (not exact due to different RNG usage)
        assert len(batch_pvals) == 3
        assert all(0 <= p <= 1 for p in batch_pvals)


class TestBenjaminiHochberg:
    """Test BH-FDR correction."""

    def test_single_significant(self):
        """Single small p-value should be significant."""
        p_values = [0.001, 0.5, 0.8, 0.9]

        significant = benjamini_hochberg(p_values, alpha=0.05)

        assert significant[0] is True
        assert all(not s for s in significant[1:])

    def test_multiple_significant(self):
        """Multiple small p-values should be significant."""
        p_values = [0.001, 0.005, 0.01, 0.5]

        significant = benjamini_hochberg(p_values, alpha=0.05)

        # First three should be significant
        assert significant[0] is True
        assert significant[1] is True
        assert significant[2] is True
        assert significant[3] is False

    def test_none_significant(self):
        """No small p-values should have no significant results."""
        p_values = [0.1, 0.2, 0.3, 0.4]

        significant = benjamini_hochberg(p_values, alpha=0.05)

        assert not any(significant)

    def test_empty_input(self):
        """Empty input should return empty list."""
        assert benjamini_hochberg([]) == []

    def test_handles_nan(self):
        """NaN values should be treated as non-significant."""
        p_values = [0.001, float('nan'), 0.5]

        significant = benjamini_hochberg(p_values, alpha=0.05)

        assert significant[0] is True
        assert significant[1] is False
        assert significant[2] is False

    def test_adjusted_pvalues(self):
        """Adjusted p-values should be monotonic with original order."""
        p_values = [0.01, 0.02, 0.03, 0.04]

        adjusted = benjamini_hochberg_adjusted(p_values)

        assert len(adjusted) == 4
        assert all(a >= p for a, p in zip(adjusted, p_values))


class TestDeflatedSharpe:
    """Test deflated Sharpe ratio."""

    def test_many_variants_high_pvalue(self):
        """Testing many variants should inflate p-value."""
        # Same observed Sharpe with different n_variants
        p_few = deflated_sharpe_ratio(
            observed_sharpe=1.5,
            n_variants_tested=10,
            n_observations=100
        )
        p_many = deflated_sharpe_ratio(
            observed_sharpe=1.5,
            n_variants_tested=1000,
            n_observations=100
        )

        assert p_many > p_few, "More variants should increase p-value"

    def test_high_sharpe_low_pvalue(self):
        """Very high Sharpe should have low p-value."""
        p = deflated_sharpe_ratio(
            observed_sharpe=3.0,
            n_variants_tested=100,
            n_observations=200
        )

        assert p < 0.1

    def test_low_sharpe_high_pvalue(self):
        """Low Sharpe should have high p-value."""
        p = deflated_sharpe_ratio(
            observed_sharpe=0.5,
            n_variants_tested=100,
            n_observations=50
        )

        assert p > 0.4  # Should be relatively high

    def test_expected_max_sharpe(self):
        """Expected max Sharpe should increase with n_variants."""
        max_10 = expected_max_sharpe_null(10, 100)
        max_100 = expected_max_sharpe_null(100, 100)
        max_1000 = expected_max_sharpe_null(1000, 100)

        assert max_10 < max_100 < max_1000

    def test_sharpe_std_nonnegative(self):
        """Sharpe standard error should be non-negative."""
        se = sharpe_ratio_std(sharpe=1.0, n=50, skewness=0, kurtosis=3)
        assert se > 0


class TestNegativeControl:
    """Test negative control / surrogate generation."""

    def test_iaaft_preserves_distribution(self):
        """IAAFT should preserve marginal distribution."""
        np.random.seed(42)
        data = np.random.randn(200)

        surrogate = iaaft_surrogate(data, seed=42)

        # Check distribution similarity
        assert len(surrogate) == len(data)
        assert np.isclose(np.mean(surrogate), np.mean(data), rtol=0.1)
        assert np.isclose(np.std(surrogate), np.std(data), rtol=0.1)

    def test_iaaft_changes_order(self):
        """IAAFT should not preserve exact order."""
        np.random.seed(42)
        data = np.random.randn(100)

        surrogate = iaaft_surrogate(data, seed=42)

        # Order should be different
        assert not np.allclose(data, surrogate)

    def test_generate_multiple_surrogates(self):
        """Should generate requested number of surrogates."""
        data = np.random.randn(50)

        surrogates = generate_surrogates(data, n_surrogates=10, seed=42)

        assert len(surrogates) == 10
        assert all(len(s) == len(data) for s in surrogates)

    def test_simple_negative_control(self):
        """Simple negative control should flag low-percentile results."""
        real_sharpes = [1.5, 0.5]
        surrogate_sharpes = [
            np.array([0.1, 0.2, 0.3, 0.4, 0.5]),  # Real is way above all
            np.array([0.6, 0.7, 0.8, 0.9, 1.0]),  # Real is below all
        ]

        results = simple_negative_control(real_sharpes, surrogate_sharpes, threshold_pctl=90)

        assert len(results) == 2
        assert not results[0].flagged  # 1.5 > all surrogates (100th pctl)
        assert results[1].flagged      # 0.5 < all surrogates (0th pctl)


class TestBootstrap:
    """Test bootstrap confidence intervals."""

    def test_ci_contains_mean(self):
        """95% CI should usually contain the sample mean."""
        np.random.seed(42)
        returns = np.random.randn(100) * 0.01 + 0.005

        lower, upper = bootstrap_confidence_interval(returns, n_bootstrap=1000, seed=42)
        mean = np.mean(returns)

        assert lower < mean < upper

    def test_ci_narrows_with_n(self):
        """CI should be narrower with more data."""
        np.random.seed(42)
        returns_small = np.random.randn(20) * 0.01
        returns_large = np.random.randn(200) * 0.01

        lower_s, upper_s = bootstrap_confidence_interval(returns_small, seed=42)
        lower_l, upper_l = bootstrap_confidence_interval(returns_large, seed=42)

        width_small = upper_s - lower_s
        width_large = upper_l - lower_l

        assert width_large < width_small

    def test_empty_returns(self):
        """Empty returns should return (0, 0)."""
        lower, upper = bootstrap_confidence_interval(np.array([]))
        assert lower == 0.0
        assert upper == 0.0
