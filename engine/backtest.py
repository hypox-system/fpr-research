"""
Event-driven backtester with strict fill model.

Delta 4, 24, 34:
- Fill on next-bar open, side-aware slippage
- No pyramiding (entry ignored when position open)
- No same-bar entry+exit (exit earliest on t+2 open if entry on t+1 open)
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Dict, Any, TYPE_CHECKING
import pandas as pd
import numpy as np

from .trades import compute_trade_return, compute_fill_price

if TYPE_CHECKING:
    from combinator.composer import Strategy


@dataclass
class Trade:
    """Single completed trade."""
    entry_bar: int  # Bar index where entry signal occurred
    exit_bar: int   # Bar index where exit signal occurred
    entry_fill_bar: int  # Bar index where entry was filled
    exit_fill_bar: int   # Bar index where exit was filled

    entry_price: float  # Raw open price at entry fill
    exit_price: float   # Raw open price at exit fill
    entry_fill: float   # Actual fill price (after slippage)
    exit_fill: float    # Actual fill price (after slippage)

    side: Literal['long', 'short']
    net_return: float   # Net return after fees and slippage

    entry_timestamp: Optional[pd.Timestamp] = None
    exit_timestamp: Optional[pd.Timestamp] = None


@dataclass
class BacktestResult:
    """Result from backtesting a strategy."""
    trades: List[Trade] = field(default_factory=list)
    trade_returns: np.ndarray = field(default_factory=lambda: np.array([]))

    # Summary metrics
    n_trades: int = 0
    total_return: float = 0.0
    sharpe: float = 0.0
    profit_factor: float = 0.0
    win_rate: float = 0.0
    max_drawdown: float = 0.0

    # Status
    status: str = "OK"
    reason_code: Optional[str] = None
    error_message: Optional[str] = None


class Backtester:
    """
    Event-driven backtester.

    Fill model (Delta 4, 20):
    - Signals evaluated on bar close
    - Entry/exit filled on NEXT bar's open
    - Long entry: open * (1 + slip), Long exit: open * (1 - slip)
    - Short entry: open * (1 - slip), Short exit: open * (1 + slip)
    - Slippage ALWAYS makes fill worse

    No pyramiding (Delta 24):
    - Entry signals ignored when position is open

    No same-bar entry+exit (Delta 34):
    - If entry on bar t fills on t+1 open
    - Exit can signal on t+1 close, but fills on t+2 open
    - Minimum trade duration = 1 bar
    """

    def __init__(
        self,
        fee_rate: float = 0.0006,
        slippage_rate: float = 0.0001
    ):
        """
        Initialize backtester.

        Args:
            fee_rate: Taker fee rate in decimal (0.0006 = 0.06%)
            slippage_rate: Slippage rate in decimal (0.0001 = 0.01%)
        """
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate

    def run(
        self,
        df: pd.DataFrame,
        entry_signal: pd.Series,
        exit_signal: pd.Series,
        side: Literal['long', 'short'],
        warmup_bars: int = 0
    ) -> BacktestResult:
        """
        Run backtest on signals.

        Args:
            df: OHLCV DataFrame
            entry_signal: Entry signal (int8, 0 or 1)
            exit_signal: Exit signal (bool)
            side: Trade direction ('long' or 'short')
            warmup_bars: Bars to skip at start

        Returns:
            BacktestResult with trades and metrics
        """
        result = BacktestResult()

        try:
            trades = self._execute_trades(
                df, entry_signal, exit_signal, side, warmup_bars
            )
            result.trades = trades
            result.n_trades = len(trades)

            if trades:
                result.trade_returns = np.array([t.net_return for t in trades])
                result.total_return = np.prod(1 + result.trade_returns) - 1
                result = self._compute_metrics(result)

        except Exception as e:
            result.status = "ERROR"
            result.error_message = str(e)[:200]

        return result

    def _execute_trades(
        self,
        df: pd.DataFrame,
        entry_signal: pd.Series,
        exit_signal: pd.Series,
        side: Literal['long', 'short'],
        warmup_bars: int
    ) -> List[Trade]:
        """Execute trade logic."""
        trades = []
        n = len(df)

        if n < 2:
            return trades

        # Get OHLC data
        opens = df['open'].values
        timestamps = df['timestamp'].values if 'timestamp' in df.columns else None

        # State
        in_position = False
        entry_bar = -1
        entry_fill_bar = -1
        entry_price = 0.0
        entry_fill = 0.0

        # Start after warmup
        start_bar = max(warmup_bars, 0)

        for i in range(start_bar, n - 1):  # -1 because we need next bar for fill
            # Entry logic (Delta 24: no pyramiding)
            if not in_position and entry_signal.iloc[i] == 1:
                # Entry signal on bar i close -> fill on bar i+1 open
                fill_bar = i + 1

                entry_bar = i
                entry_fill_bar = fill_bar
                entry_price = opens[fill_bar]
                entry_fill = compute_fill_price(
                    entry_price, side, is_entry=True, slippage_rate=self.slippage_rate
                )
                in_position = True

            # Exit logic (Delta 34: no same-bar entry+exit)
            elif in_position:
                # Check if we can exit (must be after entry fill bar)
                can_exit = i >= entry_fill_bar  # Can signal on same bar as fill

                if can_exit and exit_signal.iloc[i]:
                    # Exit signal on bar i close -> fill on bar i+1 open
                    fill_bar = i + 1

                    if fill_bar >= n:
                        # Can't fill, exit at last available bar
                        fill_bar = n - 1

                    exit_price = opens[fill_bar]
                    exit_fill = compute_fill_price(
                        exit_price, side, is_entry=False, slippage_rate=self.slippage_rate
                    )

                    # Compute return
                    net_return = compute_trade_return(
                        entry_price, exit_price, side,
                        self.fee_rate, self.slippage_rate
                    )

                    trade = Trade(
                        entry_bar=entry_bar,
                        exit_bar=i,
                        entry_fill_bar=entry_fill_bar,
                        exit_fill_bar=fill_bar,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        entry_fill=entry_fill,
                        exit_fill=exit_fill,
                        side=side,
                        net_return=net_return,
                        entry_timestamp=timestamps[entry_fill_bar] if timestamps is not None else None,
                        exit_timestamp=timestamps[fill_bar] if timestamps is not None else None
                    )
                    trades.append(trade)
                    in_position = False

        # Force close if still in position at end
        if in_position:
            last_bar = n - 1
            exit_price = opens[last_bar]
            exit_fill = compute_fill_price(
                exit_price, side, is_entry=False, slippage_rate=self.slippage_rate
            )

            net_return = compute_trade_return(
                entry_price, exit_price, side,
                self.fee_rate, self.slippage_rate
            )

            trade = Trade(
                entry_bar=entry_bar,
                exit_bar=last_bar - 1,
                entry_fill_bar=entry_fill_bar,
                exit_fill_bar=last_bar,
                entry_price=entry_price,
                exit_price=exit_price,
                entry_fill=entry_fill,
                exit_fill=exit_fill,
                side=side,
                net_return=net_return,
                entry_timestamp=timestamps[entry_fill_bar] if timestamps is not None else None,
                exit_timestamp=timestamps[last_bar] if timestamps is not None else None
            )
            trades.append(trade)

        return trades

    def _compute_metrics(self, result: BacktestResult) -> BacktestResult:
        """Compute summary metrics from trade returns."""
        returns = result.trade_returns

        if len(returns) == 0:
            return result

        # Sharpe (annualized for crypto: 365 trading days/year)
        # Using 365 instead of 252 (stocks) per Delta crypto specification
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        result.sharpe = mean_ret / std_ret * np.sqrt(365 * 6) if std_ret > 0 else 0.0

        # Profit factor
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        result.profit_factor = gains / losses if losses > 0 else (float('inf') if gains > 0 else 0.0)

        # Win rate
        result.win_rate = np.mean(returns > 0)

        # Max drawdown
        cum_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cum_returns)
        drawdowns = (running_max - cum_returns) / running_max
        result.max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0

        return result


def run_backtest(
    df: pd.DataFrame,
    entry_signal: pd.Series,
    exit_signal: pd.Series,
    side: Literal['long', 'short'],
    fee_rate: float = 0.0006,
    slippage_rate: float = 0.0001,
    warmup_bars: int = 0
) -> BacktestResult:
    """
    Convenience function to run a backtest.

    Args:
        df: OHLCV DataFrame
        entry_signal: Entry signal series
        exit_signal: Exit signal series
        side: Trade direction
        fee_rate: Taker fee rate
        slippage_rate: Slippage rate
        warmup_bars: Bars to skip

    Returns:
        BacktestResult
    """
    bt = Backtester(fee_rate=fee_rate, slippage_rate=slippage_rate)
    return bt.run(df, entry_signal, exit_signal, side, warmup_bars)
