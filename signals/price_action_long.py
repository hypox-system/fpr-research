"""
Price Action Long entry signal.

Detects higher lows and break of structure (BOS) for long entries.
Uses right_bars confirmation to avoid lookahead.
"""

from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Any, Set
import pandas as pd
import numpy as np

from .base import SignalComponent, register_signal


@register_signal
@dataclass
class PriceActionLong(SignalComponent):
    """
    Price action entry for long positions.

    Modes:
    - 'higher_low': Detects higher low pattern
    - 'bos': Break of structure (price breaks above recent swing high)

    Delta 6: Pivots confirmed after right_bars. Signal triggers on confirmation bar.
    """

    KEY: ClassVar[str] = 'price_action_long'
    SIGNAL_TYPE: ClassVar[str] = 'entry'
    SUPPORTED_SIDES: ClassVar[Set[str]] = {'long'}
    SIDE_SPLIT: ClassVar[bool] = True

    params: Dict[str, Any] = field(default_factory=lambda: {
        'swing_lb': 10,
        'right_bars': 3,
        'mode': 'higher_low'
    })

    def compute(self, df: pd.DataFrame, cache) -> pd.Series:
        swing_lb = self.params.get('swing_lb', 10)
        right_bars = self.params.get('right_bars', 3)
        mode = self.params.get('mode', 'higher_low')

        n = len(df)
        signal = pd.Series(0, index=df.index, dtype=np.int8)

        if mode == 'higher_low':
            signal = self._higher_low_signal(df, swing_lb, right_bars)
        elif mode == 'bos':
            signal = self._bos_signal(df, swing_lb, right_bars)

        return signal.fillna(0).astype(np.int8)

    def _higher_low_signal(self, df: pd.DataFrame, swing_lb: int, right_bars: int) -> pd.Series:
        """Detect higher low pattern with confirmation."""
        n = len(df)
        lows = df['low'].values
        signal = np.zeros(n, dtype=np.int8)

        # Find swing lows with confirmation
        for i in range(swing_lb + right_bars, n):
            # Check if bar i - right_bars was a swing low
            pivot_idx = i - right_bars

            # Is it a local minimum? Check swing_lb bars before and right_bars after
            is_swing_low = True
            pivot_low = lows[pivot_idx]

            # Check left side
            for j in range(pivot_idx - swing_lb, pivot_idx):
                if j >= 0 and lows[j] < pivot_low:
                    is_swing_low = False
                    break

            # Check right side (confirmation)
            if is_swing_low:
                for j in range(pivot_idx + 1, pivot_idx + right_bars + 1):
                    if j < n and lows[j] < pivot_low:
                        is_swing_low = False
                        break

            if is_swing_low:
                # Find previous swing low
                prev_swing_low = None
                for k in range(pivot_idx - swing_lb - right_bars, 0, -1):
                    # Simple check: was this a lower point than surrounding?
                    if k >= swing_lb and lows[k] < min(lows[k-swing_lb:k]) and lows[k] < min(lows[k+1:min(k+right_bars+1, pivot_idx)]):
                        prev_swing_low = lows[k]
                        break

                # Higher low confirmed
                if prev_swing_low is not None and pivot_low > prev_swing_low:
                    signal[i] = 1

        return pd.Series(signal, index=df.index)

    def _bos_signal(self, df: pd.DataFrame, swing_lb: int, right_bars: int) -> pd.Series:
        """Break of structure: price breaks above recent swing high."""
        n = len(df)
        highs = df['high'].values
        close = df['close'].values
        signal = np.zeros(n, dtype=np.int8)

        # Rolling swing high
        for i in range(swing_lb + right_bars, n):
            # Find confirmed swing high
            swing_high = 0.0
            for j in range(i - right_bars - swing_lb, i - right_bars):
                if j >= swing_lb:
                    # Check if j was a swing high
                    is_swing = True
                    h = highs[j]
                    for k in range(j - swing_lb, j):
                        if k >= 0 and highs[k] > h:
                            is_swing = False
                            break
                    if is_swing:
                        for k in range(j + 1, min(j + right_bars + 1, i - right_bars + 1)):
                            if highs[k] > h:
                                is_swing = False
                                break
                    if is_swing and h > swing_high:
                        swing_high = h

            # BOS: close above swing high
            if swing_high > 0 and close[i] > swing_high:
                signal[i] = 1

        return pd.Series(signal, index=df.index)

    def param_grid(self) -> Dict[str, List[Any]]:
        return {
            'swing_lb': [5, 10, 20, 30],
            'right_bars': [3, 5],
            'mode': ['higher_low', 'bos']
        }

    def lookback_bars(self) -> int:
        return self.params.get('swing_lb', 10) * 2 + self.params.get('right_bars', 3)
