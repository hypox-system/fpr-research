"""
ATR Trailing Stop for LONG positions.

Exit when price drops below trailing stop level.
"""

from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Any, Set
import pandas as pd
import numpy as np

from .base import SignalComponent, register_signal


@register_signal
@dataclass
class ATRStopLong(SignalComponent):
    """
    ATR trailing stop for long positions.

    Maintains a trailing stop at close - ATR * multiplier.
    Exit when close drops below trailing stop.
    """

    KEY: ClassVar[str] = 'atr_stop_long'
    SIGNAL_TYPE: ClassVar[str] = 'exit'
    SUPPORTED_SIDES: ClassVar[Set[str]] = {'long'}
    SIDE_SPLIT: ClassVar[bool] = True

    params: Dict[str, Any] = field(default_factory=lambda: {
        'atr_mult': 2.0,
        'atr_period': 14
    })

    def compute(self, df: pd.DataFrame, cache) -> pd.Series:
        atr_mult = self.params.get('atr_mult', 2.0)
        atr_period = self.params.get('atr_period', 14)

        atr = cache.get('atr', period=atr_period)
        close = df['close']

        # Trailing stop level (below price)
        stop_distance = atr * atr_mult
        stop_level = close - stop_distance

        # Trailing: stop can only move up
        trailing_stop = stop_level.cummax()

        # Exit when close drops below trailing stop
        signal = close < trailing_stop

        return signal.fillna(False).astype(bool)

    def param_grid(self) -> Dict[str, List[Any]]:
        return {
            'atr_mult': [1.5, 2.0, 3.0],
            'atr_period': [14, 20]
        }

    def lookback_bars(self) -> int:
        return self.params.get('atr_period', 14)
