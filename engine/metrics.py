"""
Trading metrics computation.

All metrics computed on NET trade returns (after fees + slippage).
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np


@dataclass
class MetricsResult:
    """Computed trading metrics."""
    # Return metrics
    total_return: float = 0.0
    mean_return: float = 0.0
    std_return: float = 0.0

    # Risk-adjusted
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0

    # Win/loss
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    win_loss_ratio: float = 0.0

    # Drawdown
    max_drawdown: float = 0.0
    avg_drawdown: float = 0.0

    # Trade stats
    n_trades: int = 0
    n_winning: int = 0
    n_losing: int = 0

    # Higher moments (for DSR)
    skewness: float = 0.0
    kurtosis: float = 0.0

    # Consistency
    best_trade: float = 0.0
    worst_trade: float = 0.0


def compute_metrics(
    trade_returns: np.ndarray,
    annualization_factor: float = 1512.0  # sqrt(252 * 6) squared
) -> MetricsResult:
    """
    Compute all trading metrics from trade returns.

    Args:
        trade_returns: Array of NET trade returns (after fees + slippage)
        annualization_factor: Factor for annualizing Sharpe (default ~252*6 trades/year)

    Returns:
        MetricsResult with all computed metrics
    """
    result = MetricsResult()

    if len(trade_returns) == 0:
        return result

    result.n_trades = len(trade_returns)

    # Basic stats
    result.mean_return = np.mean(trade_returns)
    result.std_return = np.std(trade_returns)

    # Total return (compounded)
    result.total_return = np.prod(1 + trade_returns) - 1

    # Win/loss breakdown
    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]

    result.n_winning = len(wins)
    result.n_losing = len(losses)
    result.win_rate = len(wins) / len(trade_returns) if len(trade_returns) > 0 else 0.0

    # Average win/loss
    result.avg_win = np.mean(wins) if len(wins) > 0 else 0.0
    result.avg_loss = np.mean(losses) if len(losses) > 0 else 0.0

    # Win/loss ratio
    result.win_loss_ratio = abs(result.avg_win / result.avg_loss) if result.avg_loss != 0 else float('inf')

    # Profit factor
    total_gains = wins.sum() if len(wins) > 0 else 0.0
    total_losses = abs(losses.sum()) if len(losses) > 0 else 0.0
    result.profit_factor = total_gains / total_losses if total_losses > 0 else (float('inf') if total_gains > 0 else 0.0)

    # Sharpe ratio (annualized)
    if result.std_return > 0:
        result.sharpe = result.mean_return / result.std_return * np.sqrt(annualization_factor)
    else:
        result.sharpe = 0.0

    # Sortino ratio (using downside deviation)
    downside_returns = trade_returns[trade_returns < 0]
    if len(downside_returns) > 0:
        downside_std = np.std(downside_returns)
        if downside_std > 0:
            result.sortino = result.mean_return / downside_std * np.sqrt(annualization_factor)

    # Drawdown calculation
    cum_returns = np.cumprod(1 + trade_returns)
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = (running_max - cum_returns) / running_max

    result.max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
    result.avg_drawdown = np.mean(drawdowns) if len(drawdowns) > 0 else 0.0

    # Calmar ratio
    if result.max_drawdown > 0:
        result.calmar = result.total_return / result.max_drawdown

    # Higher moments for DSR
    if len(trade_returns) >= 3:
        result.skewness = float(skewness(trade_returns))
    if len(trade_returns) >= 4:
        result.kurtosis = float(kurtosis(trade_returns))

    # Best/worst trade
    result.best_trade = np.max(trade_returns)
    result.worst_trade = np.min(trade_returns)

    return result


def skewness(x: np.ndarray) -> float:
    """Compute sample skewness."""
    n = len(x)
    if n < 3:
        return 0.0

    mean = np.mean(x)
    std = np.std(x, ddof=1)

    if std == 0:
        return 0.0

    return n / ((n-1) * (n-2)) * np.sum(((x - mean) / std) ** 3)


def kurtosis(x: np.ndarray) -> float:
    """Compute sample excess kurtosis."""
    n = len(x)
    if n < 4:
        return 0.0

    mean = np.mean(x)
    std = np.std(x, ddof=1)

    if std == 0:
        return 0.0

    m4 = np.mean((x - mean) ** 4)
    m2 = np.mean((x - mean) ** 2)

    return m4 / (m2 ** 2) - 3


def compute_equity_curve(trade_returns: np.ndarray, initial_capital: float = 1.0) -> np.ndarray:
    """
    Compute equity curve from trade returns.

    Args:
        trade_returns: Array of trade returns
        initial_capital: Starting capital

    Returns:
        Equity curve array
    """
    return initial_capital * np.cumprod(1 + trade_returns)


def compute_drawdown_series(trade_returns: np.ndarray) -> np.ndarray:
    """
    Compute drawdown series from trade returns.

    Args:
        trade_returns: Array of trade returns

    Returns:
        Drawdown series (0 to 1)
    """
    if len(trade_returns) == 0:
        return np.array([])

    cum_returns = np.cumprod(1 + trade_returns)
    running_max = np.maximum.accumulate(cum_returns)
    return (running_max - cum_returns) / running_max


def compute_rolling_sharpe(
    trade_returns: np.ndarray,
    window: int = 20,
    annualization_factor: float = 1512.0
) -> np.ndarray:
    """
    Compute rolling Sharpe ratio.

    Args:
        trade_returns: Array of trade returns
        window: Rolling window size
        annualization_factor: Annualization factor

    Returns:
        Rolling Sharpe array
    """
    n = len(trade_returns)
    if n < window:
        return np.full(n, np.nan)

    rolling_sharpe = np.full(n, np.nan)

    for i in range(window - 1, n):
        window_returns = trade_returns[i - window + 1:i + 1]
        mean_ret = np.mean(window_returns)
        std_ret = np.std(window_returns)

        if std_ret > 0:
            rolling_sharpe[i] = mean_ret / std_ret * np.sqrt(annualization_factor)

    return rolling_sharpe
