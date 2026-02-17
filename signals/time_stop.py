"""
Time Stop exit signal.

Exit after holding for a maximum number of bars.
Side-agnostic.
"""

from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Any, Set
import pandas as pd
import numpy as np

from .base import SignalComponent, register_signal


@register_signal
@dataclass
class TimeStop(SignalComponent):
    """
    Time-based stop.

    This signal is special: it requires tracking from the backtest.
    Returns True on every bar - the backtest tracks holding time.

    In practice, backtest should use max_bars to force exit.
    This signal acts as a "always allow exit" signal when combined
    with holding time logic in the backtest.
    """

    KEY: ClassVar[str] = 'time_stop'
    SIGNAL_TYPE: ClassVar[str] = 'exit'
    SUPPORTED_SIDES: ClassVar[Set[str]] = {'long', 'short'}
    SIDE_SPLIT: ClassVar[bool] = False

    params: Dict[str, Any] = field(default_factory=lambda: {
        'max_bars': 100
    })

    def compute(self, df: pd.DataFrame, cache) -> pd.Series:
        # Time stop returns True always - actual timing handled by backtest
        # The backtest will check holding duration against max_bars
        # This is a placeholder that enables the time-based exit logic
        return pd.Series(True, index=df.index, dtype=bool)

    def param_grid(self) -> Dict[str, List[Any]]:
        return {
            'max_bars': [50, 100, 200]
        }

    def lookback_bars(self) -> int:
        return 0
