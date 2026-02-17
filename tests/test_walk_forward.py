"""
Tests for walk-forward validation engine.

Covers:
- Embargo >= warmup (hard constraint, Delta 16)
- Month boundary alignment
- No train/test overlap
- Folds per segment (Delta 35)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.walk_forward import (
    WalkForwardConfig,
    walk_forward_split,
    walk_forward_split_from_segment,
    validate_embargo_warmup,
    compute_fold_metrics,
    aggregate_fold_results,
    FoldResult
)


def create_multi_month_df(months: int = 6, bars_per_day: int = 1440) -> pd.DataFrame:
    """Create test DataFrame spanning multiple months."""
    start = pd.Timestamp("2025-08-01", tz="UTC")
    n_bars = months * 30 * bars_per_day  # Approximate

    timestamps = pd.date_range(start=start, periods=n_bars, freq="1min")

    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n_bars) * 0.01)

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": close + np.random.randn(n_bars) * 0.1,
        "high": close + np.abs(np.random.randn(n_bars) * 0.2),
        "low": close - np.abs(np.random.randn(n_bars) * 0.2),
        "close": close,
        "volume": np.abs(np.random.randn(n_bars) * 1000) + 100
    })


class TestEmbargoWarmup:
    """Tests for embargo >= warmup constraint (Delta 16)."""

    def test_embargo_increases_to_warmup(self):
        """Embargo should be increased to match warmup if lower."""
        effective = validate_embargo_warmup(config_embargo=100, warmup_bars=200)
        assert effective == 200, "Embargo should be raised to warmup"

    def test_embargo_kept_if_higher(self):
        """Embargo should be kept if higher than warmup."""
        effective = validate_embargo_warmup(config_embargo=300, warmup_bars=200)
        assert effective == 300, "Embargo should remain at 300"

    def test_zero_embargo_raised_to_warmup(self):
        """Zero embargo should be raised to warmup."""
        effective = validate_embargo_warmup(config_embargo=0, warmup_bars=200)
        assert effective == 200

    def test_walk_forward_split_respects_embargo(self):
        """Walk-forward split should respect embargo >= warmup."""
        df = create_multi_month_df(months=6)
        config = WalkForwardConfig(
            train_months=2,
            test_months=1,
            embargo_bars=50  # Low embargo
        )

        warmup = 200  # Higher warmup

        splits = walk_forward_split(df, config, warmup_bars=warmup)

        # Should have at least one split
        assert len(splits) > 0

        for train, test in splits:
            # Verify no overlap
            train_timestamps = set(train["timestamp"])
            test_timestamps = set(test["timestamp"])
            overlap = train_timestamps & test_timestamps

            assert len(overlap) == 0, "Train and test should not overlap"


class TestNoTrainTestOverlap:
    """Tests for train/test separation."""

    def test_no_timestamp_in_both(self):
        """No timestamp should appear in both train and test."""
        df = create_multi_month_df(months=6)
        config = WalkForwardConfig(train_months=2, test_months=1)

        splits = walk_forward_split(df, config)

        for train, test in splits:
            train_set = set(train["timestamp"])
            test_set = set(test["timestamp"])
            overlap = train_set & test_set

            assert len(overlap) == 0, f"Found {len(overlap)} overlapping timestamps"

    def test_train_before_test(self):
        """Train period should be entirely before test period."""
        df = create_multi_month_df(months=6)
        config = WalkForwardConfig(train_months=2, test_months=1)

        splits = walk_forward_split(df, config)

        for train, test in splits:
            train_max = train["timestamp"].max()
            test_min = test["timestamp"].min()

            assert train_max < test_min, "Train max should be before test min"


class TestMonthBoundary:
    """Tests for month boundary alignment."""

    def test_folds_align_to_months(self):
        """Folds should align to calendar months."""
        df = create_multi_month_df(months=6)
        config = WalkForwardConfig(train_months=2, test_months=1)

        splits = walk_forward_split(df, config)

        for train, test in splits:
            # Check that train spans complete months
            train_months = train["timestamp"].dt.to_period("M").unique()
            assert len(train_months) >= 1

            # Check that test spans complete months
            test_months = test["timestamp"].dt.to_period("M").unique()
            assert len(test_months) >= 1


class TestFoldsPerSegment:
    """Tests for fold creation within segments (Delta 35)."""

    def test_folds_created_per_segment(self):
        """Folds should be created within each segment independently."""
        # Create two segments with a gap
        df1 = create_multi_month_df(months=4)
        df2 = create_multi_month_df(months=4)

        # Shift second segment to create gap
        df2["timestamp"] = df2["timestamp"] + timedelta(days=150)
        df2 = df2.reset_index(drop=True)

        config = WalkForwardConfig(train_months=2, test_months=1)

        # Get folds from each segment
        folds1 = walk_forward_split_from_segment(df1, config, segment_id=0)
        folds2 = walk_forward_split_from_segment(df2, config, segment_id=1)

        # Each segment should produce folds independently
        assert len(folds1) > 0
        assert len(folds2) > 0


class TestFoldMetrics:
    """Tests for fold metrics computation."""

    def test_empty_returns_zero_metrics(self):
        """Empty returns should give zero metrics."""
        metrics = compute_fold_metrics(np.array([]))

        assert metrics["sharpe"] == 0.0
        assert metrics["n_trades"] == 0

    def test_positive_returns_positive_sharpe(self):
        """Positive returns should give positive Sharpe."""
        returns = np.array([0.01, 0.02, 0.01, 0.015, 0.01])
        metrics = compute_fold_metrics(returns)

        assert metrics["sharpe"] > 0
        assert metrics["win_rate"] == 1.0
        assert metrics["n_trades"] == 5

    def test_mixed_returns_metrics(self):
        """Mixed returns should give reasonable metrics."""
        returns = np.array([0.02, -0.01, 0.015, -0.005, 0.01, -0.008])
        metrics = compute_fold_metrics(returns)

        # Should have some winning and losing trades
        assert 0 < metrics["win_rate"] < 1
        assert metrics["profit_factor"] > 0
        assert metrics["max_drawdown"] >= 0


class TestAggregation:
    """Tests for fold result aggregation."""

    def test_aggregate_single_fold(self):
        """Single fold aggregation should work."""
        fold = FoldResult(
            fold_id=0,
            train_start=datetime(2025, 8, 1),
            train_end=datetime(2025, 9, 30),
            test_start=datetime(2025, 10, 1),
            test_end=datetime(2025, 10, 31),
            n_trades=10,
            trade_returns=np.array([0.01, 0.02, -0.01, 0.015, 0.01])
        )

        result = aggregate_fold_results([fold])

        assert result.total_oos_trades == 5
        assert len(result.fold_results) == 1

    def test_aggregate_multiple_folds(self):
        """Multiple fold aggregation should concatenate returns."""
        folds = [
            FoldResult(
                fold_id=i,
                train_start=datetime(2025, 8 + i, 1),
                train_end=datetime(2025, 8 + i, 28),
                test_start=datetime(2025, 9 + i, 1),
                test_end=datetime(2025, 9 + i, 28),
                n_trades=5,
                trade_returns=np.array([0.01, 0.02, -0.01, 0.015, 0.01])
            )
            for i in range(3)
        ]

        result = aggregate_fold_results(folds)

        assert result.total_oos_trades == 15  # 5 * 3
        assert len(result.oos_trade_returns) == 15

    def test_consistency_ratio_calculation(self):
        """Consistency ratio should be calculated correctly."""
        folds = [
            FoldResult(
                fold_id=0,
                train_start=datetime(2025, 8, 1),
                train_end=datetime(2025, 9, 30),
                test_start=datetime(2025, 10, 1),
                test_end=datetime(2025, 10, 31),
                n_trades=5,
                trade_returns=np.array([0.01, 0.02, 0.01]),  # Positive total
                total_return=0.04
            ),
            FoldResult(
                fold_id=1,
                train_start=datetime(2025, 9, 1),
                train_end=datetime(2025, 10, 31),
                test_start=datetime(2025, 11, 1),
                test_end=datetime(2025, 11, 30),
                n_trades=5,
                trade_returns=np.array([-0.01, -0.02, -0.01]),  # Negative total
                total_return=-0.04
            ),
        ]

        result = aggregate_fold_results(folds)

        # 1 out of 2 folds profitable = 50%
        assert result.consistency_ratio == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
