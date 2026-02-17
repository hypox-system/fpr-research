"""
PVSRA Entry signal.

Price-Volume-Support-Resistance Analysis entry.
Triggers on high volume bars with significant price movement.
"""

from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Any, Set
import pandas as pd
import numpy as np

from .base import SignalComponent, register_signal


@register_signal
@dataclass
class PVSRAEntry(SignalComponent):
    """
    PVSRA entry signal.

    Signals when volume is significantly above average AND
    the bar has a strong close (bullish/bearish body).
    Side-agnostic: returns 1 for significant volume bars.
    """

    KEY: ClassVar[str] = 'pvsra_entry'
    SIGNAL_TYPE: ClassVar[str] = 'entry'
    SUPPORTED_SIDES: ClassVar[Set[str]] = {'long', 'short'}
    SIDE_SPLIT: ClassVar[bool] = False

    params: Dict[str, Any] = field(default_factory=lambda: {
        'vol_mult': 2.0,
        'lookback': 10
    })

    def compute(self, df: pd.DataFrame, cache) -> pd.Series:
        vol_mult = self.params.get('vol_mult', 2.0)
        lookback = self.params.get('lookback', 10)

        # Volume analysis
        vol_sma = cache.get('volume_sma', period=lookback)
        high_volume = df['volume'] > vol_sma * vol_mult

        # Strong body (close far from open)
        body = abs(df['close'] - df['open'])
        range_hl = df['high'] - df['low']
        strong_body = body > range_hl * 0.5  # Body > 50% of range

        signal = (high_volume & strong_body).astype(np.int8)
        signal = signal.fillna(0).astype(np.int8)

        return signal

    def param_grid(self) -> Dict[str, List[Any]]:
        return {
            'vol_mult': [1.5, 2.0, 3.0],
            'lookback': [5, 10, 20]
        }

    def lookback_bars(self) -> int:
        return self.params.get('lookback', 10)
