"""
MACD Phase context signal.

Returns position sizing multiplier based on MACD phase.
"""

from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Any, Set
import pandas as pd
import numpy as np

from .base import SignalComponent, register_signal


@register_signal
@dataclass
class MACDPhase(SignalComponent):
    """
    MACD phase context.

    Returns a multiplier (0.5-1.5) based on MACD histogram behavior.
    - Expansion (histogram increasing): 1.2-1.5
    - Contraction (histogram decreasing): 0.5-0.8
    - Flat: 1.0
    """

    KEY: ClassVar[str] = 'macd_phase'
    SIGNAL_TYPE: ClassVar[str] = 'context'
    SUPPORTED_SIDES: ClassVar[Set[str]] = {'long', 'short'}
    SIDE_SPLIT: ClassVar[bool] = False

    params: Dict[str, Any] = field(default_factory=lambda: {
        'z_flat': 0.3,
        'confirm': 3
    })

    def compute(self, df: pd.DataFrame, cache) -> pd.Series:
        z_flat = self.params.get('z_flat', 0.3)
        confirm = self.params.get('confirm', 3)

        hist = cache.get('macd_hist')
        delta = hist.diff()

        # Z-score of delta
        delta_std = delta.rolling(window=50, min_periods=1).std()
        delta_z = delta / delta_std.replace(0, np.nan)
        delta_z = delta_z.fillna(0)

        # Confirmed direction
        expanding = (delta_z > z_flat).rolling(window=confirm, min_periods=1).min().astype(bool)
        contracting = (delta_z < -z_flat).rolling(window=confirm, min_periods=1).min().astype(bool)

        # Multiplier
        mult = pd.Series(1.0, index=df.index)
        mult[expanding] = 1.3
        mult[contracting] = 0.7

        return mult.astype(np.float64)

    def param_grid(self) -> Dict[str, List[Any]]:
        return {
            'z_flat': [0.1, 0.3, 0.5, 0.8],
            'confirm': [2, 3, 5, 8]
        }

    def lookback_bars(self) -> int:
        return 50 + self.params.get('confirm', 3)
