"""Engine modules for backtest, trades, and walk-forward."""

from .trades import compute_trade_return, compute_fill_price
from .metrics import compute_metrics, MetricsResult

__all__ = [
    'compute_trade_return', 'compute_fill_price',
    'compute_metrics', 'MetricsResult'
]
