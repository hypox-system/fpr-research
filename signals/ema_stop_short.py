"""
EMA Stop exit signal for SHORT positions.

Exit when close > EMA (price breaks above resistance).
"""

from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Any, Set
import pandas as pd

from .base import SignalComponent, register_signal


@register_signal
@dataclass
class EMAStopShort(SignalComponent):
    """
    EMA stop for short positions.

    Exit signal when close > EMA.
    SIDE_SPLIT = True: resolved from 'ema_stop' + side='short'
    """

    KEY: ClassVar[str] = 'ema_stop_short'
    SIGNAL_TYPE: ClassVar[str] = 'exit'
    SUPPORTED_SIDES: ClassVar[Set[str]] = {'short'}
    SIDE_SPLIT: ClassVar[bool] = True

    params: Dict[str, Any] = field(default_factory=lambda: {
        'period': 50
    })

    def compute(self, df: pd.DataFrame, cache) -> pd.Series:
        period = self.params.get('period', 50)
        ema = cache.get('ema', period=period)

        # Exit short when price closes above EMA
        signal = df['close'] > ema

        return signal.fillna(False).astype(bool)

    def param_grid(self) -> Dict[str, List[Any]]:
        return {
            'period': [50, 100]
        }

    def lookback_bars(self) -> int:
        return self.params.get('period', 50)
