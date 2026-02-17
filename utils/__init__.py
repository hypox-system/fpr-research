"""Utility modules for FPR Research Platform."""

from .canonical import canonicalize_spec
from .timeframes import TF_TO_PANDAS, TF_TO_MINUTES, parse_timeframe
from .reason_codes import (
    E_SCHEMA, E_SIGNAL_DTYPE, E_SIGNAL_NAN, E_SIGNAL_INDEX,
    E_LOOKAHEAD, E_BACKTEST, E_EXCEPTION, E_SIDE_UNSUPPORTED,
    E_NO_TRADES, E_INSUFFICIENT_TRADES, E_SEGMENT_TOO_SHORT,
    E_INVALID_PARAMS, REASON_CODES
)

__all__ = [
    'canonicalize_spec',
    'TF_TO_PANDAS', 'TF_TO_MINUTES', 'parse_timeframe',
    'E_SCHEMA', 'E_SIGNAL_DTYPE', 'E_SIGNAL_NAN', 'E_SIGNAL_INDEX',
    'E_LOOKAHEAD', 'E_BACKTEST', 'E_EXCEPTION', 'E_SIDE_UNSUPPORTED',
    'E_NO_TRADES', 'E_INSUFFICIENT_TRADES', 'E_SEGMENT_TOO_SHORT',
    'E_INVALID_PARAMS', 'REASON_CODES'
]
