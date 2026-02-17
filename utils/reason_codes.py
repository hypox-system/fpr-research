"""
Standardized reason codes for variant status.

Delta 38: All variant status codes must use these constants.
Each variant produces OK/INVALID/ERROR in variants.jsonl.
"""

# Schema and configuration errors
E_SCHEMA = "E_SCHEMA"
E_INVALID_PARAMS = "E_INVALID_PARAMS"

# Signal computation errors
E_SIGNAL_DTYPE = "E_SIGNAL_DTYPE"
E_SIGNAL_NAN = "E_SIGNAL_NAN"
E_SIGNAL_INDEX = "E_SIGNAL_INDEX"
E_LOOKAHEAD = "E_LOOKAHEAD"

# Backtest errors
E_BACKTEST = "E_BACKTEST"
E_NO_TRADES = "E_NO_TRADES"
E_INSUFFICIENT_TRADES = "E_INSUFFICIENT_TRADES"

# Side compatibility errors
E_SIDE_UNSUPPORTED = "E_SIDE_UNSUPPORTED"

# Data/segment errors
E_SEGMENT_TOO_SHORT = "E_SEGMENT_TOO_SHORT"

# Generic exception
E_EXCEPTION = "E_EXCEPTION"

# All valid reason codes
REASON_CODES = {
    E_SCHEMA,
    E_INVALID_PARAMS,
    E_SIGNAL_DTYPE,
    E_SIGNAL_NAN,
    E_SIGNAL_INDEX,
    E_LOOKAHEAD,
    E_BACKTEST,
    E_NO_TRADES,
    E_INSUFFICIENT_TRADES,
    E_SIDE_UNSUPPORTED,
    E_SEGMENT_TOO_SHORT,
    E_EXCEPTION,
}


def validate_reason_code(code: str) -> bool:
    """
    Check if a reason code is valid.

    Valid codes are either in REASON_CODES or prefixed with E_.
    """
    if code in REASON_CODES:
        return True
    if code.startswith("E_"):
        return True
    return False
