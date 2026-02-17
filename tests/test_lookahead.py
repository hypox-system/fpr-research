"""
Test signals for look-ahead bias.

Critical Delta constraint: signals must not use future data.

Test methodology:
1. Compute signal on truncated data (first N bars)
2. Compute signal on full data (all bars)
3. Verify signal values for first N bars are identical

If a signal has look-ahead bias, adding future data would change
historical signal values.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from signals.base import SIGNAL_REGISTRY
from data.feature_store import FeatureStore


def create_test_ohlcv(n_bars: int = 500, seed: int = 42) -> pd.DataFrame:
    """Create synthetic OHLCV data for testing."""
    dates = pd.date_range(
        start=datetime(2024, 1, 1),
        periods=n_bars,
        freq='1h'
    )

    np.random.seed(seed)

    # Random walk price
    returns = np.random.randn(n_bars) * 0.01
    close = 100 * np.exp(np.cumsum(returns))

    # Generate OHLC from close
    high = close * (1 + np.abs(np.random.randn(n_bars)) * 0.005)
    low = close * (1 - np.abs(np.random.randn(n_bars)) * 0.005)
    open_price = close * (1 + np.random.randn(n_bars) * 0.002)

    # Ensure OHLC consistency
    high = np.maximum(high, np.maximum(open_price, close))
    low = np.minimum(low, np.minimum(open_price, close))

    volume = np.random.exponential(1000, n_bars) * (1 + np.abs(returns) * 10)

    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)

    return df


class TestNoLookahead:
    """Test that signals don't have look-ahead bias."""

    @pytest.fixture
    def full_data(self):
        """Create full test dataset."""
        return create_test_ohlcv(600, seed=123)

    def test_all_signals_no_lookahead(self, full_data):
        """
        Verify no look-ahead bias for all signals.

        For each signal:
        1. Compute on truncated data (first 300 bars)
        2. Compute on full data (600 bars)
        3. First 300 values must be identical

        The comparison starts after warmup period to allow for
        indicator initialization.
        """
        truncate_at = 300
        truncated_data = full_data.iloc[:truncate_at].copy()

        for key, cls in SIGNAL_REGISTRY.items():
            signal = cls()
            lookback = signal.lookback_bars()

            # Skip comparison during warmup period
            compare_start = max(lookback + 10, 50)

            if compare_start >= truncate_at:
                # Not enough data after warmup, skip
                continue

            # Compute on truncated data
            cache_truncated = FeatureStore(truncated_data, '1h')
            result_truncated = signal.compute(truncated_data, cache_truncated)

            # Compute on full data
            cache_full = FeatureStore(full_data, '1h')
            result_full = signal.compute(full_data, cache_full)

            # Compare values after warmup
            truncated_values = result_truncated.iloc[compare_start:].values
            full_values = result_full.iloc[compare_start:truncate_at].values

            # Check lengths match
            assert len(truncated_values) == len(full_values), \
                f"{key}: length mismatch in comparison range"

            # Check values match
            if signal.SIGNAL_TYPE == 'context':
                # Float comparison with tolerance
                np.testing.assert_allclose(
                    truncated_values,
                    full_values,
                    rtol=1e-10,
                    err_msg=f"{key} has look-ahead bias (context signal)"
                )
            else:
                # Exact comparison for int/bool
                np.testing.assert_array_equal(
                    truncated_values,
                    full_values,
                    err_msg=f"{key} has look-ahead bias"
                )

    def test_signals_deterministic(self, full_data):
        """
        Verify signals are deterministic.

        Running the same signal twice on the same data
        should produce identical results.
        """
        for key, cls in SIGNAL_REGISTRY.items():
            signal1 = cls()
            signal2 = cls()

            cache1 = FeatureStore(full_data, '1h')
            cache2 = FeatureStore(full_data, '1h')

            result1 = signal1.compute(full_data, cache1)
            result2 = signal2.compute(full_data, cache2)

            if signal1.SIGNAL_TYPE == 'context':
                np.testing.assert_allclose(
                    result1.values,
                    result2.values,
                    rtol=1e-10,
                    err_msg=f"{key} is not deterministic"
                )
            else:
                np.testing.assert_array_equal(
                    result1.values,
                    result2.values,
                    err_msg=f"{key} is not deterministic"
                )


class TestFeatureStoreCaching:
    """Test that FeatureStore caching doesn't cause look-ahead."""

    def test_cache_isolated_per_instance(self):
        """Each FeatureStore instance should be isolated."""
        data1 = create_test_ohlcv(200, seed=1)
        data2 = create_test_ohlcv(200, seed=2)

        cache1 = FeatureStore(data1, '1h')
        cache2 = FeatureStore(data2, '1h')

        # Get same indicator from both caches
        ema1 = cache1.get('ema', period=20)
        ema2 = cache2.get('ema', period=20)

        # Values should be different (different data)
        assert not np.allclose(ema1.values, ema2.values), \
            "Different data should produce different EMA values"

    def test_cache_consistent_within_instance(self):
        """Same request to same cache should return same data."""
        data = create_test_ohlcv(200)
        cache = FeatureStore(data, '1h')

        ema1 = cache.get('ema', period=20)
        ema2 = cache.get('ema', period=20)

        np.testing.assert_array_equal(
            ema1.values,
            ema2.values,
            err_msg="Same cache request should return identical data"
        )


class TestRollingWindowBias:
    """Test rolling windows don't use future data."""

    def test_rolling_uses_past_only(self):
        """Rolling operations should only use past data."""
        n_bars = 100
        data = create_test_ohlcv(n_bars)
        cache = FeatureStore(data, '1h')

        # Get a rolling indicator
        sma = cache.get('sma', period=10)

        # Manual calculation for last bar
        last_10_closes = data['close'].iloc[-10:].values
        expected_sma = last_10_closes.mean()

        # Should match within tolerance
        actual_sma = sma.iloc[-1]
        np.testing.assert_almost_equal(
            actual_sma,
            expected_sma,
            decimal=10,
            err_msg="SMA should use past 10 bars only"
        )

    def test_shift_direction(self):
        """Verify shift uses correct direction (past, not future)."""
        data = create_test_ohlcv(100)
        cache = FeatureStore(data, '1h')

        close = data['close']
        prev_close = close.shift(1)

        # prev_close[i] should equal close[i-1]
        for i in range(1, len(close)):
            assert prev_close.iloc[i] == close.iloc[i - 1], \
                f"shift(1) at index {i} should be previous bar's value"


class TestIndicatorWarmup:
    """Test that indicators handle warmup period correctly."""

    def test_warmup_produces_nan_or_valid(self):
        """During warmup, indicators should be NaN or valid (not garbage)."""
        data = create_test_ohlcv(100)
        cache = FeatureStore(data, '1h')

        # 50-period EMA should need warmup
        ema = cache.get('ema', period=50)

        # First value should be NaN or a valid number
        first_valid_idx = ema.first_valid_index()

        # If there are NaN values, they should be at the start
        nan_mask = ema.isna()
        if nan_mask.any():
            # Find last NaN index
            last_nan_idx = nan_mask[nan_mask].index[-1]

            # All values before first_valid should be NaN
            assert all(nan_mask.loc[:first_valid_idx]), \
                "NaN values should be contiguous at start"

    def test_signals_handle_warmup(self):
        """Signals should not error during warmup period."""
        # Very short data - many indicators won't have enough warmup
        data = create_test_ohlcv(30)

        for key, cls in SIGNAL_REGISTRY.items():
            signal = cls()
            cache = FeatureStore(data, '1h')

            # Should not raise error
            try:
                result = signal.compute(data, cache)
                assert len(result) == len(data)
            except Exception as e:
                pytest.fail(f"{key} failed on short data: {e}")
