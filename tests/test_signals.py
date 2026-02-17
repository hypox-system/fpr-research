"""
Test all signal components.

Tests:
- Registration in SIGNAL_REGISTRY
- Correct output dtypes
- compute() runs without error
- lookback_bars() returns valid int
- param_grid() returns valid dict
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from signals.base import SIGNAL_REGISTRY, SignalComponent
from data.feature_store import FeatureStore


def create_test_ohlcv(n_bars: int = 500) -> pd.DataFrame:
    """Create synthetic OHLCV data for testing."""
    dates = pd.date_range(
        start=datetime(2024, 1, 1),
        periods=n_bars,
        freq='1h'
    )

    np.random.seed(42)

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


class TestSignalRegistry:
    """Test that all signals are properly registered."""

    def test_registry_not_empty(self):
        """Registry should contain signals."""
        assert len(SIGNAL_REGISTRY) > 0

    def test_expected_signals_registered(self):
        """All expected signal keys should be in registry."""
        expected = [
            'ema_cross',
            'ema_gating',
            'ema_stop_long',
            'ema_stop_short',
            'pvsra_entry',
            'pvsra_filter',
            'price_action_long',
            'price_action_short',
            'atr_stop_long',
            'atr_stop_short',
            'time_stop',
            'stoch_rsi_entry',
            'stoch_rsi_filter',
            'volume_filter',
            'macd_phase',
        ]

        for key in expected:
            assert key in SIGNAL_REGISTRY, f"Missing signal: {key}"

    def test_all_signals_have_required_classvars(self):
        """All signals must have KEY, SIGNAL_TYPE, SUPPORTED_SIDES."""
        for key, cls in SIGNAL_REGISTRY.items():
            assert hasattr(cls, 'KEY'), f"{key} missing KEY"
            assert hasattr(cls, 'SIGNAL_TYPE'), f"{key} missing SIGNAL_TYPE"
            assert hasattr(cls, 'SUPPORTED_SIDES'), f"{key} missing SUPPORTED_SIDES"
            assert hasattr(cls, 'SIDE_SPLIT'), f"{key} missing SIDE_SPLIT"

            assert cls.KEY == key, f"{key} KEY mismatch: {cls.KEY}"
            assert cls.SIGNAL_TYPE in ('entry', 'exit', 'filter', 'context'), \
                f"{key} invalid SIGNAL_TYPE: {cls.SIGNAL_TYPE}"
            assert cls.SUPPORTED_SIDES.issubset({'long', 'short'}), \
                f"{key} invalid SUPPORTED_SIDES: {cls.SUPPORTED_SIDES}"


class TestSignalCompute:
    """Test that all signals compute correctly."""

    @pytest.fixture
    def test_data(self):
        """Create test OHLCV data."""
        return create_test_ohlcv(500)

    @pytest.fixture
    def feature_store(self, test_data):
        """Create feature store for test data."""
        return FeatureStore(test_data, '1h')

    def test_all_signals_compute(self, test_data, feature_store):
        """All signals should compute without error."""
        for key, cls in SIGNAL_REGISTRY.items():
            signal = cls()
            result = signal.compute(test_data, feature_store)

            assert isinstance(result, pd.Series), f"{key} did not return Series"
            assert len(result) == len(test_data), f"{key} length mismatch"
            assert not result.isna().all(), f"{key} all NaN"

    def test_entry_signal_dtype(self, test_data, feature_store):
        """Entry signals should return int8."""
        for key, cls in SIGNAL_REGISTRY.items():
            if cls.SIGNAL_TYPE != 'entry':
                continue

            signal = cls()
            result = signal.compute(test_data, feature_store)

            assert result.dtype == np.int8, f"{key} should be int8, got {result.dtype}"
            assert result.isin([0, 1]).all(), f"{key} values should be 0 or 1"

    def test_exit_signal_dtype(self, test_data, feature_store):
        """Exit signals should return bool."""
        for key, cls in SIGNAL_REGISTRY.items():
            if cls.SIGNAL_TYPE != 'exit':
                continue

            signal = cls()
            result = signal.compute(test_data, feature_store)

            assert result.dtype == bool, f"{key} should be bool, got {result.dtype}"

    def test_filter_signal_dtype(self, test_data, feature_store):
        """Filter signals should return bool."""
        for key, cls in SIGNAL_REGISTRY.items():
            if cls.SIGNAL_TYPE != 'filter':
                continue

            signal = cls()
            result = signal.compute(test_data, feature_store)

            assert result.dtype == bool, f"{key} should be bool, got {result.dtype}"

    def test_context_signal_dtype(self, test_data, feature_store):
        """Context signals should return float64."""
        for key, cls in SIGNAL_REGISTRY.items():
            if cls.SIGNAL_TYPE != 'context':
                continue

            signal = cls()
            result = signal.compute(test_data, feature_store)

            assert result.dtype == np.float64, f"{key} should be float64, got {result.dtype}"
            assert (result > 0).all(), f"{key} context values should be positive"


class TestSignalMethods:
    """Test signal interface methods."""

    def test_all_signals_have_lookback_bars(self):
        """All signals should have lookback_bars method returning int."""
        for key, cls in SIGNAL_REGISTRY.items():
            signal = cls()
            lookback = signal.lookback_bars()

            assert isinstance(lookback, int), f"{key} lookback_bars should return int"
            assert lookback >= 0, f"{key} lookback_bars should be >= 0"

    def test_all_signals_have_param_grid(self):
        """All signals should have param_grid method returning dict."""
        for key, cls in SIGNAL_REGISTRY.items():
            signal = cls()
            grid = signal.param_grid()

            assert isinstance(grid, dict), f"{key} param_grid should return dict"

            for param_name, values in grid.items():
                assert isinstance(values, list), \
                    f"{key} param_grid[{param_name}] should be list"
                assert len(values) > 0, \
                    f"{key} param_grid[{param_name}] should not be empty"

    def test_signals_with_custom_params(self):
        """Signals should accept custom params."""
        for key, cls in SIGNAL_REGISTRY.items():
            # Get param grid
            default_signal = cls()
            grid = default_signal.param_grid()

            if not grid:
                continue

            # Create with first value from each param
            custom_params = {k: v[0] for k, v in grid.items()}
            custom_signal = cls(params=custom_params)

            # Verify params were set
            for param_name, value in custom_params.items():
                assert custom_signal.params.get(param_name) == value, \
                    f"{key} custom param {param_name} not set"


class TestSideSplit:
    """Test SIDE_SPLIT behavior."""

    def test_side_split_signals(self):
        """Side-split signals should only support one side."""
        side_split_signals = [
            ('ema_stop_long', 'long'),
            ('ema_stop_short', 'short'),
            ('price_action_long', 'long'),
            ('price_action_short', 'short'),
            ('atr_stop_long', 'long'),
            ('atr_stop_short', 'short'),
        ]

        for key, expected_side in side_split_signals:
            cls = SIGNAL_REGISTRY[key]

            assert cls.SIDE_SPLIT is True, f"{key} should have SIDE_SPLIT=True"
            assert cls.SUPPORTED_SIDES == {expected_side}, \
                f"{key} should only support {expected_side}"

    def test_non_side_split_signals(self):
        """Non-side-split signals should support both sides."""
        non_split_signals = [
            'ema_cross',
            'ema_gating',
            'pvsra_entry',
            'pvsra_filter',
            'time_stop',
            'stoch_rsi_entry',
            'stoch_rsi_filter',
            'volume_filter',
            'macd_phase',
        ]

        for key in non_split_signals:
            cls = SIGNAL_REGISTRY[key]

            assert cls.SIDE_SPLIT is False, f"{key} should have SIDE_SPLIT=False"
            assert cls.SUPPORTED_SIDES == {'long', 'short'}, \
                f"{key} should support both sides"


class TestSignalValues:
    """Test that signal values are reasonable."""

    @pytest.fixture
    def test_data(self):
        """Create test OHLCV data."""
        return create_test_ohlcv(1000)

    @pytest.fixture
    def feature_store(self, test_data):
        """Create feature store for test data."""
        return FeatureStore(test_data, '1h')

    def test_entry_signals_sparse(self, test_data, feature_store):
        """Entry signals should not fire on every bar."""
        for key, cls in SIGNAL_REGISTRY.items():
            if cls.SIGNAL_TYPE != 'entry':
                continue

            signal = cls()
            result = signal.compute(test_data, feature_store)

            entry_rate = result.sum() / len(result)

            # Entry signals should be sparse (< 50% of bars)
            assert entry_rate < 0.5, f"{key} fires too often: {entry_rate:.1%}"

    def test_context_signal_range(self, test_data, feature_store):
        """Context signals should be in reasonable range."""
        for key, cls in SIGNAL_REGISTRY.items():
            if cls.SIGNAL_TYPE != 'context':
                continue

            signal = cls()
            result = signal.compute(test_data, feature_store)

            # Context multipliers should be reasonable (0.1 to 10)
            assert result.min() >= 0.1, f"{key} min too low: {result.min()}"
            assert result.max() <= 10.0, f"{key} max too high: {result.max()}"
