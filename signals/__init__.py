"""Signal component library for FPR Research Platform."""

from .base import (
    SignalComponent, SignalType,
    register_signal, get_signal, list_signals,
    validate_signal_output, SIGNAL_REGISTRY
)

# Import all signals to trigger registration
from .ema_cross import EMACross
from .ema_gating import EMAGating
from .ema_stop_long import EMAStopLong
from .ema_stop_short import EMAStopShort
from .pvsra_entry import PVSRAEntry
from .pvsra_filter import PVSRAFilter
from .price_action_long import PriceActionLong
from .price_action_short import PriceActionShort
from .atr_stop_long import ATRStopLong
from .atr_stop_short import ATRStopShort
from .time_stop import TimeStop
from .stoch_rsi_entry import StochRSIEntry
from .stoch_rsi_filter import StochRSIFilter
from .volume_filter import VolumeFilter
from .macd_phase import MACDPhase

__all__ = [
    'SignalComponent', 'SignalType',
    'register_signal', 'get_signal', 'list_signals',
    'validate_signal_output', 'SIGNAL_REGISTRY',
    # Signal classes
    'EMACross', 'EMAGating',
    'EMAStopLong', 'EMAStopShort',
    'PVSRAEntry', 'PVSRAFilter',
    'PriceActionLong', 'PriceActionShort',
    'ATRStopLong', 'ATRStopShort',
    'TimeStop',
    'StochRSIEntry', 'StochRSIFilter',
    'VolumeFilter', 'MACDPhase',
]
