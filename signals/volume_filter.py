"""
Volume Filter signal.

Filter for minimum relative volume.
"""

from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Any, Set
import pandas as pd

from .base import SignalComponent, register_signal


@register_signal
@dataclass
class VolumeFilter(SignalComponent):
    """
    Volume filter.

    Returns True when current volume is above threshold relative to average.
    """

    KEY: ClassVar[str] = 'volume_filter'
    SIGNAL_TYPE: ClassVar[str] = 'filter'
    SUPPORTED_SIDES: ClassVar[Set[str]] = {'long', 'short'}
    SIDE_SPLIT: ClassVar[bool] = False

    params: Dict[str, Any] = field(default_factory=lambda: {
        'rel_vol': 1.5,
        'lookback': 20
    })

    def compute(self, df: pd.DataFrame, cache) -> pd.Series:
        rel_vol = self.params.get('rel_vol', 1.5)
        lookback = self.params.get('lookback', 20)

        vol_sma = cache.get('volume_sma', period=lookback)
        signal = df['volume'] > vol_sma * rel_vol

        return signal.fillna(False).astype(bool)

    def param_grid(self) -> Dict[str, List[Any]]:
        return {
            'rel_vol': [1.2, 1.5, 2.0],
            'lookback': [10, 20]
        }

    def lookback_bars(self) -> int:
        return self.params.get('lookback', 20)
