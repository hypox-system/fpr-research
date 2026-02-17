"""
Timeframe utilities and mappings.

Delta 25: Standardized timeframe string to pandas resample code mapping.
"""

from typing import Optional

# Timeframe string to pandas resample offset (Delta 25)
TF_TO_PANDAS = {
    '1min': '1min',
    '5min': '5min',
    '15min': '15min',
    '30min': '30min',
    '1h': '1h',
    '4h': '4h',
    '1d': '1D',
    # Also accept alternate formats
    '1T': '1min',
    '5T': '5min',
    '15T': '15min',
    '30T': '30min',
    '1H': '1h',
    '4H': '4h',
    '1D': '1D',
}

# Minutes per timeframe bar
TF_TO_MINUTES = {
    '1min': 1,
    '5min': 5,
    '15min': 15,
    '30min': 30,
    '1h': 60,
    '4h': 240,
    '1d': 1440,
}


def parse_timeframe(tf: str) -> str:
    """
    Normalize timeframe string to standard format.

    Args:
        tf: Timeframe string (e.g., '5min', '5T', '1H')

    Returns:
        Normalized timeframe string

    Raises:
        ValueError: If timeframe is unknown
    """
    normalized = TF_TO_PANDAS.get(tf)
    if normalized is None:
        raise ValueError(f"Unknown timeframe: {tf}. Valid: {list(TF_TO_PANDAS.keys())}")
    return normalized


def tf_to_minutes(tf: str) -> int:
    """
    Get the number of minutes per bar for a timeframe.

    Args:
        tf: Timeframe string

    Returns:
        Minutes per bar
    """
    normalized = parse_timeframe(tf)
    return TF_TO_MINUTES.get(normalized, TF_TO_MINUTES.get(tf, 1))


def bars_to_minutes(bars: int, tf: str) -> int:
    """Convert bar count to minutes for a given timeframe."""
    return bars * tf_to_minutes(tf)


def minutes_to_bars(minutes: int, tf: str) -> int:
    """Convert minutes to bar count for a given timeframe."""
    return minutes // tf_to_minutes(tf)
