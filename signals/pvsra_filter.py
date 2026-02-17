"""
PVSRA Filter signal.

Filters for high volume environment over recent bars.
"""

from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Any, Set
import pandas as pd

from .base import SignalComponent, register_signal


@register_signal
@dataclass
class PVSRAFilter(SignalComponent):
    """
    PVSRA filter.

    Returns True when recent volume is elevated.
    Indicates institutional activity.
    """

    KEY: ClassVar[str] = 'pvsra_filter'
    SIGNAL_TYPE: ClassVar[str] = 'filter'
    SUPPORTED_SIDES: ClassVar[Set[str]] = {'long', 'short'}
    SIDE_SPLIT: ClassVar[bool] = False

    params: Dict[str, Any] = field(default_factory=lambda: {
        'vol_mult': 1.5,
        'window': 5
    })

    def compute(self, df: pd.DataFrame, cache) -> pd.Series:
        vol_mult = self.params.get('vol_mult', 1.5)
        window = self.params.get('window', 5)

        # Check if ANY bar in recent window had high volume
        vol_sma = cache.get('volume_sma', period=20)
        high_vol = df['volume'] > vol_sma * vol_mult

        # Rolling any: at least one high volume bar in window
        signal = high_vol.rolling(window=window, min_periods=1).max().astype(bool)

        return signal.fillna(False).astype(bool)

    def param_grid(self) -> Dict[str, List[Any]]:
        return {
            'vol_mult': [1.5, 2.0],
            'window': [3, 5, 10]
        }

    def lookback_bars(self) -> int:
        return 20 + self.params.get('window', 5)
