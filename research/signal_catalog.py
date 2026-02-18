"""
Signal catalog discovery and formatting.

Auto-discovers signals from the signals/ module using the registry.
"""

import logging
from typing import Any, Dict, List

from signals.base import SIGNAL_REGISTRY, get_signal_info, list_signals

logger = logging.getLogger(__name__)

# Cache for discovered signals
_CATALOG_CACHE: Dict[str, Dict[str, Any]] = {}


def discover_signals() -> Dict[str, Dict[str, Any]]:
    """
    Discover all registered signals with their metadata.

    Returns:
        Dict mapping signal key to metadata:
        {
            "ema_cross": {
                "type": "entry",
                "params": {"fast": [5, 8, 13, 21], "slow": [21, 34, 50, 100]},
                "supported_sides": ["long", "short"],
                "side_split": False,
                "lookback_bars": 42,
                "description": "EMA crossover entry signal"
            },
            ...
        }
    """
    global _CATALOG_CACHE

    if _CATALOG_CACHE:
        return _CATALOG_CACHE

    catalog = {}
    for key in list_signals():
        try:
            info = get_signal_info(key)
            cls = SIGNAL_REGISTRY[key]

            # Extract description from docstring
            description = ""
            if cls.__doc__:
                # Get first non-empty line of docstring
                lines = [
                    line.strip() for line in cls.__doc__.split("\n") if line.strip()
                ]
                if lines:
                    description = lines[0]

            catalog[key] = {
                "type": info["type"],
                "params": info["param_grid"],
                "supported_sides": info["supported_sides"],
                "side_split": info["side_split"],
                "lookback_bars": info["lookback_bars"],
                "description": description,
            }
        except Exception as e:
            logger.warning(f"Failed to discover signal {key}: {e}")

    _CATALOG_CACHE = catalog
    return catalog


def get_valid_signal_names() -> List[str]:
    """
    Get list of valid signal names.

    Returns:
        Sorted list of signal keys.
    """
    return list_signals()


def format_for_llm(catalog: Dict[str, Dict[str, Any]] = None) -> str:
    """
    Format signal catalog as human-readable text for LLM context.

    Args:
        catalog: Optional pre-computed catalog. If None, discovers signals.

    Returns:
        Formatted text description of available signals.
    """
    if catalog is None:
        catalog = discover_signals()

    if not catalog:
        return "No signals available in the catalog."

    lines = ["## Available Signals", ""]

    # Group by type
    by_type: Dict[str, List[str]] = {}
    for key, info in catalog.items():
        signal_type = info["type"]
        if signal_type not in by_type:
            by_type[signal_type] = []
        by_type[signal_type].append(key)

    type_order = ["entry", "exit", "filter", "context"]
    for signal_type in type_order:
        if signal_type not in by_type:
            continue

        lines.append(f"### {signal_type.title()} Signals")
        lines.append("")

        for key in sorted(by_type[signal_type]):
            info = catalog[key]
            desc = info.get("description", "No description")
            sides = ", ".join(info["supported_sides"])
            params = info.get("params", {})

            lines.append(f"**{key}**: {desc}")
            lines.append(f"  - Sides: {sides}")
            if params:
                param_str = ", ".join(
                    f"{k}={v}" for k, v in params.items()
                )
                lines.append(f"  - Parameters: {param_str}")
            lines.append("")

    return "\n".join(lines)


def reset_cache() -> None:
    """Reset the catalog cache. For testing."""
    global _CATALOG_CACHE
    _CATALOG_CACHE = {}
