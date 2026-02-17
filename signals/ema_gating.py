"""
EMA Gating filter signal.

Side-agnostic filter: returns True when price is above EMA (or EMAs are fanned).
"""

from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Any, Set
import pandas as pd
import numpy as np

from .base import SignalComponent, register_signal


@register_signal
@dataclass
class EMAGating(SignalComponent):
    """
    EMA gating filter.

    Modes:
    - 'above': price close > EMA
    - 'fanned': multiple EMAs in order (bullish alignment)
    """

    KEY: ClassVar[str] = 'ema_gating'
    SIGNAL_TYPE: ClassVar[str] = 'filter'
    SUPPORTED_SIDES: ClassVar[Set[str]] = {'long', 'short'}
    SIDE_SPLIT: ClassVar[bool] = False

    params: Dict[str, Any] = field(default_factory=lambda: {
        'period': 50,
        'mode': 'above'
    })

    def compute(self, df: pd.DataFrame, cache) -> pd.Series:
        period = self.params.get('period', 50)
        mode = self.params.get('mode', 'above')

        if mode == 'above':
            ema = cache.get('ema', period=period)
            signal = df['close'] > ema
        elif mode == 'fanned':
            # Check EMA alignment: 20 > 50 > 100 > 200
            ema20 = cache.get('ema', period=20)
            ema50 = cache.get('ema', period=50)
            ema100 = cache.get('ema', period=100)
            ema200 = cache.get('ema', period=200)
            signal = (ema20 > ema50) & (ema50 > ema100) & (ema100 > ema200)
        else:
            signal = pd.Series(True, index=df.index)

        return signal.fillna(False).astype(bool)

    def param_grid(self) -> Dict[str, List[Any]]:
        return {
            'period': [20, 50, 100, 200],
            'mode': ['above', 'fanned']
        }

    def lookback_bars(self) -> int:
        # EMA with adjust=False needs ~2x period for full convergence
        mode = self.params.get('mode', 'above')
        if mode == 'fanned':
            return 400  # 2x for 200-period EMA
        return 2 * self.params.get('period', 50)
