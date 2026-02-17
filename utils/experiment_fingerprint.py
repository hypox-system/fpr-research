"""
Experiment fingerprint computation.

Generates a unique, stable identifier for an experiment based on:
- Sweep configuration (canonicalized)
- Data manifest (symbol, timeframe, date_range, data_fingerprint)
- Fill model (fee_rate, slippage_rate)

The fingerprint is deterministic: same inputs always produce same output.
YAML key ordering does NOT affect the result.
"""

import hashlib
from typing import Dict, Any

from utils.canonical import canonicalize_spec


def compute_experiment_id(
    sweep_config: Dict[str, Any],
    data_manifest: Dict[str, Any],
    fill_model: Dict[str, Any],
) -> str:
    """
    Compute canonical SHA-256 fingerprint for an experiment.

    Args:
        sweep_config: Full sweep configuration dict (from YAML)
        data_manifest: Data manifest with keys:
            - symbol: str (e.g., "BTCUSDT")
            - timeframe: str (e.g., "1h")
            - date_range: dict with 'start' and 'end'
            - data_fingerprint: str (hash of actual data)
        fill_model: Fill model configuration with keys:
            - fee_rate: float (e.g., 0.0006)
            - slippage_rate: float (e.g., 0.0001)

    Returns:
        First 16 characters of SHA-256 hex digest.

    Note:
        YAML key ordering does NOT affect the result because
        canonicalize_spec() sorts all keys at all levels.
    """
    # Canonicalize each component
    canonical_sweep = canonicalize_spec(sweep_config)
    canonical_data = canonicalize_spec(data_manifest)
    canonical_fill = canonicalize_spec(fill_model)

    # Combine with separator to avoid collision between components
    combined = f"{canonical_sweep}|{canonical_data}|{canonical_fill}"

    # Compute SHA-256 and return first 16 chars
    digest = hashlib.sha256(combined.encode('utf-8')).hexdigest()
    return digest[:16]


def compute_experiment_id_from_sweep(
    sweep_config: Dict[str, Any],
    data: 'pd.DataFrame',
) -> str:
    """
    Convenience function to compute experiment ID from sweep config and data.

    Extracts data_manifest and fill_model from sweep_config and data.

    Args:
        sweep_config: Full sweep configuration dict
        data: OHLCV DataFrame with timestamp column

    Returns:
        First 16 characters of SHA-256 hex digest.
    """
    import pandas as pd
    import hashlib

    # Extract data manifest
    if 'timestamp' in data.columns:
        data_start = data['timestamp'].min()
        data_end = data['timestamp'].max()
    else:
        data_start = data.index[0]
        data_end = data.index[-1]

    # Compute data fingerprint
    data_info = f"{len(data)}_{data_start}_{data_end}"
    data_fingerprint = hashlib.sha256(data_info.encode()).hexdigest()[:16]

    data_manifest = {
        'symbol': sweep_config.get('symbol', 'UNKNOWN'),
        'timeframe': sweep_config.get('timeframes', ['1h'])[0] if isinstance(sweep_config.get('timeframes'), list) else sweep_config.get('timeframe', '1h'),
        'date_range': {
            'start': str(data_start.date()) if hasattr(data_start, 'date') else str(data_start),
            'end': str(data_end.date()) if hasattr(data_end, 'date') else str(data_end),
        },
        'data_fingerprint': data_fingerprint,
    }

    # Extract fill model
    fees = sweep_config.get('fees', {})
    fill_model = {
        'fee_rate': fees.get('taker_fee_pct', 0.06) / 100,  # Convert from pct to decimal
        'slippage_rate': fees.get('slippage_pct', 0.01) / 100,
    }

    return compute_experiment_id(sweep_config, data_manifest, fill_model)
