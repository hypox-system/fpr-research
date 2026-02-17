"""
Tests for combinator module.

Tests:
- Strategy composition and validation
- Side resolver with SIDE_SPLIT
- Parameter grid expansion
- Variant counting
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from combinator.composer import (
    Strategy, resolve_signal_key, context_to_size,
    create_strategy_from_config
)
from combinator.param_grid import (
    expand_param_grid, expand_component_grid,
    expand_component_list_grid, count_variants,
    generate_variant_configs
)
from signals.base import SIGNAL_REGISTRY, get_signal
from data.feature_store import FeatureStore


def create_test_ohlcv(n_bars: int = 500) -> pd.DataFrame:
    """Create synthetic OHLCV data."""
    dates = pd.date_range(start=datetime(2024, 1, 1), periods=n_bars, freq='1h')
    np.random.seed(42)
    returns = np.random.randn(n_bars) * 0.01
    close = 100 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.randn(n_bars)) * 0.005)
    low = close * (1 - np.abs(np.random.randn(n_bars)) * 0.005)
    open_price = close * (1 + np.random.randn(n_bars) * 0.002)
    high = np.maximum(high, np.maximum(open_price, close))
    low = np.minimum(low, np.minimum(open_price, close))
    volume = np.random.exponential(1000, n_bars)

    return pd.DataFrame({
        'open': open_price, 'high': high, 'low': low,
        'close': close, 'volume': volume
    }, index=dates)


class TestSideResolver:
    """Test signal key resolution with SIDE_SPLIT."""

    def test_non_split_signal_unchanged(self):
        """Non-split signals return as-is regardless of side."""
        # ema_cross has SIDE_SPLIT=False
        assert resolve_signal_key('ema_cross', 'long') == 'ema_cross'
        assert resolve_signal_key('ema_cross', 'short') == 'ema_cross'

    def test_split_signal_resolves_to_side(self):
        """Split signals resolve to side-specific version."""
        # ema_stop should resolve to ema_stop_long/short
        assert resolve_signal_key('ema_stop', 'long') == 'ema_stop_long'
        assert resolve_signal_key('ema_stop', 'short') == 'ema_stop_short'

    def test_direct_side_signal(self):
        """Direct side-specific signals work."""
        assert resolve_signal_key('ema_stop_long', 'long') == 'ema_stop_long'
        assert resolve_signal_key('ema_stop_short', 'short') == 'ema_stop_short'

    def test_unknown_signal_raises(self):
        """Unknown signals raise ValueError."""
        with pytest.raises(ValueError, match="Unknown signal"):
            resolve_signal_key('nonexistent_signal', 'long')


class TestStrategyValidation:
    """Test Strategy fail-closed validation."""

    def test_valid_strategy(self):
        """Valid strategy passes validation."""
        entry = get_signal('ema_cross')()
        exit_sig = get_signal('ema_stop_long')()

        strategy = Strategy(
            name='test',
            entry=entry,
            exits=[exit_sig],
            side='long',
            timeframe='1h'
        )

        assert strategy.side == 'long'
        assert strategy.entry.KEY == 'ema_cross'

    def test_wrong_entry_type_fails(self):
        """Using filter signal as entry fails."""
        filter_sig = get_signal('ema_gating')()

        with pytest.raises(ValueError, match="expected 'entry'"):
            Strategy(
                name='test',
                entry=filter_sig,  # This is a filter, not entry
                exits=[get_signal('time_stop')()],
                side='long'
            )

    def test_wrong_exit_type_fails(self):
        """Using entry signal as exit fails."""
        entry = get_signal('ema_cross')()

        with pytest.raises(ValueError, match="expected 'exit'"):
            Strategy(
                name='test',
                entry=entry,
                exits=[entry],  # Can't use entry as exit
                side='long'
            )

    def test_no_exits_fails(self):
        """Strategy without exits fails."""
        entry = get_signal('ema_cross')()

        with pytest.raises(ValueError, match="At least one exit"):
            Strategy(
                name='test',
                entry=entry,
                exits=[],
                side='long'
            )

    def test_incompatible_side_fails(self):
        """Using signal that doesn't support the side fails."""
        entry = get_signal('ema_cross')()
        # ema_stop_long only supports 'long'
        exit_long = get_signal('ema_stop_long')()

        with pytest.raises(ValueError, match="does not support side"):
            Strategy(
                name='test',
                entry=entry,
                exits=[exit_long],
                side='short',  # exit_long doesn't support short
            )

    def test_side_split_signal_with_matching_side(self):
        """Side-split signal works with matching side."""
        entry = get_signal('price_action_long')()  # Only supports long
        exit_sig = get_signal('ema_stop_long')()

        strategy = Strategy(
            name='test',
            entry=entry,
            exits=[exit_sig],
            side='long'
        )
        assert strategy.side == 'long'


class TestStrategySignalGeneration:
    """Test Strategy.generate_signals()."""

    @pytest.fixture
    def test_data(self):
        return create_test_ohlcv(500)

    @pytest.fixture
    def feature_store(self, test_data):
        return FeatureStore(test_data, '1h')

    def test_generate_signals_returns_dataframe(self, test_data, feature_store):
        """generate_signals returns DataFrame with expected columns."""
        entry = get_signal('ema_cross')()
        exit_sig = get_signal('ema_stop_long')()

        strategy = Strategy(
            name='test',
            entry=entry,
            exits=[exit_sig],
            side='long'
        )

        result = strategy.generate_signals(test_data, feature_store)

        assert isinstance(result, pd.DataFrame)
        assert 'entry_signal' in result.columns
        assert 'exit_signal' in result.columns
        assert 'context_mult' in result.columns
        assert len(result) == len(test_data)

    def test_entry_signal_combines_filters(self, test_data, feature_store):
        """Entry signal is ANDed with all filters."""
        entry = get_signal('ema_cross')()
        filter1 = get_signal('ema_gating')()
        filter2 = get_signal('volume_filter')()
        exit_sig = get_signal('time_stop')()

        strategy = Strategy(
            name='test',
            entry=entry,
            filters=[filter1, filter2],
            exits=[exit_sig],
            side='long'
        )

        result = strategy.generate_signals(test_data, feature_store)

        # Entry should only be True where entry AND all filters are True
        assert result['entry_signal'].dtype == bool

    def test_context_mult_default(self, test_data, feature_store):
        """Without context signals, context_mult is 1.0."""
        entry = get_signal('ema_cross')()
        exit_sig = get_signal('time_stop')()

        strategy = Strategy(
            name='test',
            entry=entry,
            exits=[exit_sig],
            side='long'
        )

        result = strategy.generate_signals(test_data, feature_store)

        assert (result['context_mult'] == 1.0).all()


class TestContextToSize:
    """Test context multiplier to size conversion."""

    def test_empty_context(self):
        """No context returns base size."""
        assert context_to_size([], base_size=1.0) == 1.0

    def test_single_context(self):
        """Single context value used directly."""
        assert context_to_size([1.2], base_size=1.0) == 1.2

    def test_multiple_context_averaged(self):
        """Multiple contexts are averaged."""
        result = context_to_size([1.0, 1.4], base_size=1.0)
        assert result == pytest.approx(1.2)

    def test_clipping_max(self):
        """Context clipped to max."""
        result = context_to_size([2.0], base_size=1.0, max_mult=1.5)
        assert result == 1.5

    def test_clipping_min(self):
        """Context clipped to min."""
        result = context_to_size([0.3], base_size=1.0, min_mult=0.5)
        assert result == 0.5


class TestParamGrid:
    """Test parameter grid expansion."""

    def test_expand_empty_grid(self):
        """Empty grid returns single empty dict."""
        result = expand_param_grid({})
        assert result == [{}]

    def test_expand_single_param(self):
        """Single param expands correctly."""
        result = expand_param_grid({'a': [1, 2, 3]})
        assert result == [{'a': 1}, {'a': 2}, {'a': 3}]

    def test_expand_multiple_params(self):
        """Multiple params produce cartesian product."""
        result = expand_param_grid({'a': [1, 2], 'b': [3, 4]})
        assert len(result) == 4
        assert {'a': 1, 'b': 3} in result
        assert {'a': 2, 'b': 4} in result

    def test_expand_component_grid(self):
        """Component config expands correctly."""
        config = {
            'type': 'ema_cross',
            'params': {'fast': [5, 8], 'slow': [21]}
        }
        result = expand_component_grid(config)

        assert len(result) == 2
        assert all(r['type'] == 'ema_cross' for r in result)
        assert result[0]['params'] == {'fast': 5, 'slow': 21}
        assert result[1]['params'] == {'fast': 8, 'slow': 21}


class TestVariantCounting:
    """Test variant count calculation."""

    def test_simple_sweep(self):
        """Simple sweep with known variant count."""
        config = {
            'entry': {'type': 'ema_cross', 'params': {'fast': [5, 8], 'slow': [21, 34]}},
            'filters': [],
            'exits': [{'type': 'time_stop', 'params': {'max_bars': [100]}}],
            'sides': ['long'],
            'timeframes': ['1h'],
        }

        # 2 fast * 2 slow * 1 exit * 1 side * 1 tf = 4
        assert count_variants(config) == 4

    def test_sweep_with_multiple_filters(self):
        """Sweep with multiple filter components."""
        config = {
            'entry': {'type': 'ema_cross', 'params': {'fast': [5], 'slow': [21]}},
            'filters': [
                {'type': 'ema_gating', 'params': {'period': [50, 100]}},
                {'type': 'volume_filter', 'params': {'rel_vol': [1.5, 2.0]}},
            ],
            'exits': [{'type': 'time_stop', 'params': {'max_bars': [100]}}],
            'sides': ['long'],
            'timeframes': ['1h'],
        }

        # 1 entry * (2 gating * 2 volume) filters * 1 exit * 1 side * 1 tf = 4
        assert count_variants(config) == 4

    def test_sweep_with_multiple_sides_and_tfs(self):
        """Sweep multiplied by sides and timeframes."""
        config = {
            'entry': {'type': 'ema_cross', 'params': {'fast': [5], 'slow': [21]}},
            'exits': [{'type': 'time_stop', 'params': {'max_bars': [100]}}],
            'sides': ['long', 'short'],
            'timeframes': ['5min', '15min', '1h'],
        }

        # 1 entry * 1 exit * 2 sides * 3 tfs = 6
        assert count_variants(config) == 6


class TestCreateStrategyFromConfig:
    """Test strategy creation from config dict."""

    def test_basic_strategy_creation(self):
        """Create strategy from basic config."""
        config = {
            'entry': {'type': 'ema_cross', 'params': {'fast': 8, 'slow': 21}},
            'filters': [{'type': 'ema_gating', 'params': {'period': 50, 'mode': 'above'}}],
            'exits': [{'type': 'ema_stop', 'params': {'period': 50}}],
            'fees': {'taker_fee_pct': 0.06, 'slippage_pct': 0.01},
        }

        strategy = create_strategy_from_config(config, side='long', timeframe='1h')

        assert strategy.entry.KEY == 'ema_cross'
        assert strategy.entry.params['fast'] == 8
        assert len(strategy.filters) == 1
        assert len(strategy.exits) == 1
        assert strategy.exits[0].KEY == 'ema_stop_long'  # Resolved!
        assert strategy.taker_fee_rate == pytest.approx(0.0006)
        assert strategy.slippage_rate == pytest.approx(0.0001)

    def test_side_resolver_in_create(self):
        """Side resolver applied during creation."""
        config = {
            'entry': {'type': 'ema_cross', 'params': {}},
            'exits': [{'type': 'ema_stop', 'params': {'period': 50}}],
            'fees': {'taker_fee_pct': 0.06, 'slippage_pct': 0.01},
        }

        strat_long = create_strategy_from_config(config, side='long', timeframe='1h')
        strat_short = create_strategy_from_config(config, side='short', timeframe='1h')

        assert strat_long.exits[0].KEY == 'ema_stop_long'
        assert strat_short.exits[0].KEY == 'ema_stop_short'


class TestVariantId:
    """Test variant ID generation."""

    def test_variant_id_deterministic(self):
        """Same strategy produces same variant ID."""
        entry = get_signal('ema_cross')(params={'fast': 8, 'slow': 21})
        exit_sig = get_signal('ema_stop_long')(params={'period': 50})

        s1 = Strategy(name='test', entry=entry, exits=[exit_sig], side='long')
        s2 = Strategy(name='test', entry=entry, exits=[exit_sig], side='long')

        assert s1.variant_id() == s2.variant_id()

    def test_different_params_different_id(self):
        """Different params produce different variant ID."""
        exit_sig = get_signal('ema_stop_long')(params={'period': 50})

        s1 = Strategy(
            name='test',
            entry=get_signal('ema_cross')(params={'fast': 8, 'slow': 21}),
            exits=[exit_sig],
            side='long'
        )
        s2 = Strategy(
            name='test',
            entry=get_signal('ema_cross')(params={'fast': 13, 'slow': 21}),
            exits=[exit_sig],
            side='long'
        )

        assert s1.variant_id() != s2.variant_id()

    def test_variant_id_length(self):
        """Variant ID is 12 characters."""
        entry = get_signal('ema_cross')()
        exit_sig = get_signal('time_stop')()

        strategy = Strategy(name='test', entry=entry, exits=[exit_sig], side='long')

        assert len(strategy.variant_id()) == 12
