"""
Stochastic RSI Filter signal.

Filter based on stoch RSI zone.
"""

from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Any, Set
import pandas as pd

from .base import SignalComponent, register_signal


@register_signal
@dataclass
class StochRSIFilter(SignalComponent):
    """
    Stochastic RSI filter.

    Returns True when stoch RSI is NOT in overbought zone.
    Prevents entries at extreme overbought levels.
    """

    KEY: ClassVar[str] = 'stoch_rsi_filter'
    SIGNAL_TYPE: ClassVar[str] = 'filter'
    SUPPORTED_SIDES: ClassVar[Set[str]] = {'long', 'short'}
    SIDE_SPLIT: ClassVar[bool] = False

    params: Dict[str, Any] = field(default_factory=lambda: {
        'rsi_len': 14,
        'ob': 80,
        'os': 20
    })

    def compute(self, df: pd.DataFrame, cache) -> pd.Series:
        rsi_len = self.params.get('rsi_len', 14)
        ob = self.params.get('ob', 80)

        k = cache.get('stoch_rsi_k', rsi_period=rsi_len, k_smooth=3)

        # Allow entry when NOT overbought
        signal = k < ob

        return signal.fillna(False).astype(bool)

    def param_grid(self) -> Dict[str, List[Any]]:
        return {
            'rsi_len': [14],
            'ob': [70, 80],
            'os': [20, 30]
        }

    def lookback_bars(self) -> int:
        return self.params.get('rsi_len', 14) * 2
