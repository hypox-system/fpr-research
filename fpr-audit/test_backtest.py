"""
Tests for backtest engine.

Covers:
- Fill model (next-bar open, side-aware slippage)
- No pyramiding (entry ignored when position open)
- No same-bar entry+exit
- Canonical trade return calculation
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.trades import compute_trade_return, compute_fill_price
from engine.backtest import Backtester, run_backtest, BacktestResult


def create_test_df(n_bars: int = 100) -> pd.DataFrame:
    """Create test OHLCV DataFrame with predictable prices."""
    timestamps = pd.date_range(
        start="2025-08-01", periods=n_bars, freq="1min", tz="UTC"
    )

    # Simple price series
    opens = np.full(n_bars, 100.0)
    highs = np.full(n_bars, 101.0)
    lows = np.full(n_bars, 99.0)
    closes = np.full(n_bars, 100.0)
    volume = np.full(n_bars, 1000.0)

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volume
    })


class TestCanonicalTradeReturn:
    """Tests for canonical trade return computation (Delta 33)."""

    def test_trivial_trade_negative_return(self):
        """Trivial trade (entry=exit) should have negative return due to fees."""
        # Same price for entry and exit
        ret = compute_trade_return(
            entry_price=100.0,
            exit_price=100.0,
            side='long',
            fee_rate=0.0006,
            slippage_rate=0.0001
        )

        assert ret < 0, "Trivial trade should have negative return (fees)"

        # Same for short
        ret_short = compute_trade_return(
            entry_price=100.0,
            exit_price=100.0,
            side='short',
            fee_rate=0.0006,
            slippage_rate=0.0001
        )

        assert ret_short < 0, "Trivial short trade should have negative return"

    def test_long_profitable_trade(self):
        """Long trade with price increase should be profitable (net of fees)."""
        ret = compute_trade_return(
            entry_price=100.0,
            exit_price=101.0,  # 1% increase
            side='long',
            fee_rate=0.0006,
            slippage_rate=0.0001
        )

        # 1% gross - 0.12% fees - slippage ~= 0.87%
        assert ret > 0, "Long with +1% should be profitable"
        assert ret < 0.01, "Return should be less than 1% after fees"

    def test_short_profitable_trade(self):
        """Short trade with price decrease should be profitable."""
        ret = compute_trade_return(
            entry_price=100.0,
            exit_price=99.0,  # 1% decrease
            side='short',
            fee_rate=0.0006,
            slippage_rate=0.0001
        )

        assert ret > 0, "Short with -1% should be profitable"


class TestFillPrice:
    """Tests for side-aware fill price computation (Delta 4, 20)."""

    def test_long_entry_more_expensive(self):
        """Long entry fill should be MORE expensive (slippage hurts)."""
        fill = compute_fill_price(100.0, 'long', is_entry=True, slippage_rate=0.001)
        assert fill > 100.0, "Long entry should be above market"

    def test_long_exit_cheaper(self):
        """Long exit fill should be CHEAPER (slippage hurts)."""
        fill = compute_fill_price(100.0, 'long', is_entry=False, slippage_rate=0.001)
        assert fill < 100.0, "Long exit should be below market"

    def test_short_entry_cheaper(self):
        """Short entry fill should be CHEAPER (slippage hurts short entry)."""
        fill = compute_fill_price(100.0, 'short', is_entry=True, slippage_rate=0.001)
        assert fill < 100.0, "Short entry should be below market"

    def test_short_exit_more_expensive(self):
        """Short exit fill should be MORE expensive (slippage hurts short exit)."""
        fill = compute_fill_price(100.0, 'short', is_entry=False, slippage_rate=0.001)
        assert fill > 100.0, "Short exit should be above market"

    def test_slippage_always_hurts(self):
        """Verify slippage always makes fills worse regardless of side."""
        # Long: entry higher, exit lower
        long_entry = compute_fill_price(100.0, 'long', True, 0.001)
        long_exit = compute_fill_price(100.0, 'long', False, 0.001)
        assert long_entry > long_exit, "Long entry should cost more than exit at same price"

        # Short: entry lower, exit higher
        short_entry = compute_fill_price(100.0, 'short', True, 0.001)
        short_exit = compute_fill_price(100.0, 'short', False, 0.001)
        assert short_entry < short_exit, "Short entry should be cheaper than exit at same price"


class TestNoPyramiding:
    """Tests for no-pyramiding rule (Delta 24)."""

    def test_two_entries_only_one_trade(self):
        """Two consecutive entry signals should only open one position."""
        df = create_test_df(20)

        # Two entry signals in a row
        entry_signal = pd.Series([0]*5 + [1, 1] + [0]*13, dtype=np.int8)
        # Exit at bar 15
        exit_signal = pd.Series([False]*15 + [True] + [False]*4)

        result = run_backtest(df, entry_signal, exit_signal, 'long')

        assert result.n_trades == 1, "Should only have 1 trade despite 2 entry signals"

    def test_entry_while_in_position_ignored(self):
        """Entry signal while in position should be ignored."""
        df = create_test_df(30)

        # Entry at bar 5, another at bar 10 (while in position), exit at bar 20
        entry_signal = pd.Series([0]*5 + [1] + [0]*4 + [1] + [0]*19, dtype=np.int8)
        exit_signal = pd.Series([False]*20 + [True] + [False]*9)

        result = run_backtest(df, entry_signal, exit_signal, 'long')

        assert result.n_trades == 1, "Entry while in position should be ignored"
        # Verify entry was at bar 5 (fill at bar 6)
        assert result.trades[0].entry_fill_bar == 6


class TestNoSameBarEntryExit:
    """Tests for no same-bar entry+exit rule (Delta 34)."""

    def test_immediate_exit_delayed(self):
        """Exit signal on entry fill bar should not execute until next bar."""
        df = create_test_df(20)

        # Entry at bar 5 (fills at bar 6), immediate exit signal at bar 6
        entry_signal = pd.Series([0]*5 + [1] + [0]*14, dtype=np.int8)
        exit_signal = pd.Series([False]*6 + [True] + [False]*13)

        result = run_backtest(df, entry_signal, exit_signal, 'long')

        assert result.n_trades == 1
        trade = result.trades[0]

        # Entry fill at bar 6
        assert trade.entry_fill_bar == 6
        # Exit should be at bar 7 (next bar after entry fill)
        assert trade.exit_fill_bar == 7, "Exit should be delayed to next bar"

    def test_minimum_trade_duration(self):
        """Minimum trade duration should be 1 bar."""
        df = create_test_df(20)

        # Entry at bar 5, exit signal at bar 6 (same bar as entry fill)
        entry_signal = pd.Series([0]*5 + [1] + [0]*14, dtype=np.int8)
        exit_signal = pd.Series([False]*6 + [True] + [False]*13)

        result = run_backtest(df, entry_signal, exit_signal, 'long')

        trade = result.trades[0]
        duration = trade.exit_fill_bar - trade.entry_fill_bar

        assert duration >= 1, "Trade must last at least 1 bar"


class TestFillOnNextOpen:
    """Tests for next-bar-open fill model."""

    def test_entry_fills_on_next_bar(self):
        """Entry signal on bar N should fill on bar N+1 open."""
        df = create_test_df(20)

        entry_signal = pd.Series([0]*5 + [1] + [0]*14, dtype=np.int8)
        exit_signal = pd.Series([False]*15 + [True] + [False]*4)

        result = run_backtest(df, entry_signal, exit_signal, 'long')

        trade = result.trades[0]
        assert trade.entry_bar == 5, "Entry signal at bar 5"
        assert trade.entry_fill_bar == 6, "Fill should be at bar 6"

    def test_exit_fills_on_next_bar(self):
        """Exit signal on bar N should fill on bar N+1 open."""
        df = create_test_df(20)

        entry_signal = pd.Series([0]*5 + [1] + [0]*14, dtype=np.int8)
        exit_signal = pd.Series([False]*10 + [True] + [False]*9)

        result = run_backtest(df, entry_signal, exit_signal, 'long')

        trade = result.trades[0]
        assert trade.exit_bar == 10, "Exit signal at bar 10"
        assert trade.exit_fill_bar == 11, "Exit fill should be at bar 11"


class TestBacktestMetrics:
    """Tests for backtest metrics computation."""

    def test_all_winning_trades(self):
        """All winning trades should have 100% win rate."""
        df = create_test_df(50)
        # Make prices increase
        df["open"] = np.linspace(100, 110, 50)

        entry_signal = pd.Series([0]*5 + [1] + [0]*20 + [1] + [0]*23, dtype=np.int8)
        exit_signal = pd.Series([False]*10 + [True] + [False]*19 + [True] + [False]*19)

        result = run_backtest(df, entry_signal, exit_signal, 'long')

        assert result.n_trades == 2
        assert result.win_rate > 0.9  # Should be ~100%

    def test_trade_returns_in_result(self):
        """Trade returns should be stored in result."""
        df = create_test_df(30)

        entry_signal = pd.Series([0]*5 + [1] + [0]*24, dtype=np.int8)
        exit_signal = pd.Series([False]*15 + [True] + [False]*14)

        result = run_backtest(df, entry_signal, exit_signal, 'long')

        assert len(result.trade_returns) == result.n_trades
        assert len(result.trades) == result.n_trades


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
