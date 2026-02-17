"""
Price Action Short entry signal.

Detects lower highs and break of structure (BOS) for short entries.
Uses right_bars confirmation to avoid lookahead.
"""

from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Any, Set
import pandas as pd
import numpy as np

from .base import SignalComponent, register_signal


@register_signal
@dataclass
class PriceActionShort(SignalComponent):
    """
    Price action entry for short positions.

    Modes:
    - 'lower_high': Detects lower high pattern
    - 'bos_short': Break of structure (price breaks below recent swing low)

    Delta 6: Pivots confirmed after right_bars. Signal triggers on confirmation bar.
    """

    KEY: ClassVar[str] = 'price_action_short'
    SIGNAL_TYPE: ClassVar[str] = 'entry'
    SUPPORTED_SIDES: ClassVar[Set[str]] = {'short'}
    SIDE_SPLIT: ClassVar[bool] = True

    params: Dict[str, Any] = field(default_factory=lambda: {
        'swing_lb': 10,
        'right_bars': 3,
        'mode': 'lower_high'
    })

    def compute(self, df: pd.DataFrame, cache) -> pd.Series:
        swing_lb = self.params.get('swing_lb', 10)
        right_bars = self.params.get('right_bars', 3)
        mode = self.params.get('mode', 'lower_high')

        if mode == 'lower_high':
            signal = self._lower_high_signal(df, swing_lb, right_bars)
        elif mode == 'bos_short':
            signal = self._bos_signal(df, swing_lb, right_bars)
        else:
            signal = pd.Series(0, index=df.index, dtype=np.int8)

        return signal.fillna(0).astype(np.int8)

    def _lower_high_signal(self, df: pd.DataFrame, swing_lb: int, right_bars: int) -> pd.Series:
        """Detect lower high pattern with confirmation."""
        n = len(df)
        highs = df['high'].values
        signal = np.zeros(n, dtype=np.int8)

        for i in range(swing_lb + right_bars, n):
            pivot_idx = i - right_bars

            # Check if pivot_idx is a swing high
            is_swing_high = True
            pivot_high = highs[pivot_idx]

            # Check left side
            for j in range(pivot_idx - swing_lb, pivot_idx):
                if j >= 0 and highs[j] > pivot_high:
                    is_swing_high = False
                    break

            # Check right side (confirmation)
            if is_swing_high:
                for j in range(pivot_idx + 1, pivot_idx + right_bars + 1):
                    if j < n and highs[j] > pivot_high:
                        is_swing_high = False
                        break

            if is_swing_high:
                # Find previous swing high
                prev_swing_high = None
                for k in range(pivot_idx - swing_lb - right_bars, 0, -1):
                    if k >= swing_lb and highs[k] > max(highs[k-swing_lb:k]) and highs[k] > max(highs[k+1:min(k+right_bars+1, pivot_idx)]):
                        prev_swing_high = highs[k]
                        break

                # Lower high confirmed
                if prev_swing_high is not None and pivot_high < prev_swing_high:
                    signal[i] = 1

        return pd.Series(signal, index=df.index)

    def _bos_signal(self, df: pd.DataFrame, swing_lb: int, right_bars: int) -> pd.Series:
        """Break of structure: price breaks below recent swing low."""
        n = len(df)
        lows = df['low'].values
        close = df['close'].values
        signal = np.zeros(n, dtype=np.int8)

        for i in range(swing_lb + right_bars, n):
            # Find confirmed swing low
            swing_low = float('inf')
            for j in range(i - right_bars - swing_lb, i - right_bars):
                if j >= swing_lb:
                    is_swing = True
                    l = lows[j]
                    for k in range(j - swing_lb, j):
                        if k >= 0 and lows[k] < l:
                            is_swing = False
                            break
                    if is_swing:
                        for k in range(j + 1, min(j + right_bars + 1, i - right_bars + 1)):
                            if lows[k] < l:
                                is_swing = False
                                break
                    if is_swing and l < swing_low:
                        swing_low = l

            # BOS: close below swing low
            if swing_low < float('inf') and close[i] < swing_low:
                signal[i] = 1

        return pd.Series(signal, index=df.index)

    def param_grid(self) -> Dict[str, List[Any]]:
        return {
            'swing_lb': [5, 10, 20, 30],
            'right_bars': [3, 5],
            'mode': ['lower_high', 'bos_short']
        }

    def lookback_bars(self) -> int:
        return self.params.get('swing_lb', 10) * 2 + self.params.get('right_bars', 3)
