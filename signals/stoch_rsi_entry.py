"""
Stochastic RSI Entry signal.

Entry when stoch RSI crosses out of oversold/overbought zone.
"""

from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Any, Set
import pandas as pd
import numpy as np

from .base import SignalComponent, register_signal


@register_signal
@dataclass
class StochRSIEntry(SignalComponent):
    """
    Stochastic RSI entry signal.

    Signals when K line crosses above oversold (for longs) or
    below overbought (for shorts).
    Side-agnostic: returns 1 for oversold recovery.
    """

    KEY: ClassVar[str] = 'stoch_rsi_entry'
    SIGNAL_TYPE: ClassVar[str] = 'entry'
    SUPPORTED_SIDES: ClassVar[Set[str]] = {'long', 'short'}
    SIDE_SPLIT: ClassVar[bool] = False

    params: Dict[str, Any] = field(default_factory=lambda: {
        'rsi_len': 14,
        'k_smooth': 3,
        'ob': 80,
        'os': 20
    })

    def compute(self, df: pd.DataFrame, cache) -> pd.Series:
        rsi_len = self.params.get('rsi_len', 14)
        k_smooth = self.params.get('k_smooth', 3)
        os_level = self.params.get('os', 20)

        k = cache.get('stoch_rsi_k', rsi_period=rsi_len, k_smooth=k_smooth)

        # Cross above oversold
        cross_up = (k > os_level) & (k.shift(1) <= os_level)

        signal = cross_up.astype(np.int8)
        return signal.fillna(0).astype(np.int8)

    def param_grid(self) -> Dict[str, List[Any]]:
        return {
            'rsi_len': [7, 14, 21],
            'k_smooth': [3],
            'ob': [80],
            'os': [20]
        }

    def lookback_bars(self) -> int:
        return self.params.get('rsi_len', 14) * 2 + self.params.get('k_smooth', 3)
