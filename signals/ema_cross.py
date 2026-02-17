"""
EMA Cross entry signal.

Side-agnostic: returns 1 when fast EMA crosses above slow EMA.
Strategy side determines trade direction.
"""

from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Any, Set
import pandas as pd
import numpy as np

from .base import SignalComponent, register_signal


@register_signal
@dataclass
class EMACross(SignalComponent):
    """
    EMA crossover entry signal.

    Signal = 1 when fast EMA crosses above slow EMA.
    Side-agnostic: works for both long and short strategies.
    """

    KEY: ClassVar[str] = 'ema_cross'
    SIGNAL_TYPE: ClassVar[str] = 'entry'
    SUPPORTED_SIDES: ClassVar[Set[str]] = {'long', 'short'}
    SIDE_SPLIT: ClassVar[bool] = False

    params: Dict[str, Any] = field(default_factory=lambda: {
        'fast': 8,
        'slow': 21
    })

    def compute(self, df: pd.DataFrame, cache) -> pd.Series:
        fast = self.params.get('fast', 8)
        slow = self.params.get('slow', 21)

        ema_fast = cache.get('ema', period=fast)
        ema_slow = cache.get('ema', period=slow)

        # Cross above: fast > slow AND fast.shift(1) <= slow.shift(1)
        cross_above = (ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))

        signal = cross_above.astype(np.int8)
        signal = signal.fillna(0).astype(np.int8)

        return signal

    def param_grid(self) -> Dict[str, List[Any]]:
        return {
            'fast': [5, 8, 13, 21],
            'slow': [21, 34, 50, 100]
        }

    def lookback_bars(self) -> int:
        # EMA with adjust=False needs ~2x period for full convergence
        return 2 * max(self.params.get('fast', 8), self.params.get('slow', 21))
