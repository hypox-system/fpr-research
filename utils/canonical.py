"""
Canonical spec serialization for deterministic variant IDs.

Delta 22: All variant_id generation and manifest logging uses this module.
Float precision: 10 decimal places for absolute determinism.
"""

import json
from typing import Any, Dict, Union


def _serialize_value(value: Any) -> Any:
    """Serialize a single value with deterministic float formatting."""
    if isinstance(value, float):
        # 10 decimal places for absolute determinism (Delta 22)
        return f"{value:.10f}"
    elif isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in sorted(value.items())}
    elif isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    else:
        return value


def canonicalize_spec(spec: Dict[str, Any]) -> str:
    """
    Convert a spec dict to a canonical JSON string.

    Ensures:
    - Keys are sorted at all levels
    - Floats are serialized with 10 decimal places
    - Output is deterministic across Python sessions

    Args:
        spec: Dictionary to canonicalize

    Returns:
        Deterministic JSON string
    """
    canonical = _serialize_value(spec)
    return json.dumps(canonical, sort_keys=True, separators=(',', ':'))


def spec_to_variant_id(spec: Dict[str, Any]) -> str:
    """
    Generate variant ID from spec using sha256.

    Args:
        spec: Strategy specification dict

    Returns:
        First 12 characters of sha256 hash
    """
    import hashlib
    canonical = canonicalize_spec(spec)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]
