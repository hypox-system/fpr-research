"""
Lab configuration loader.

Reads research/config/lab.yaml. Override via FPR_LAB_CONFIG env var.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path(__file__).parent / "config" / "lab.yaml"
_CONFIG: Optional[Dict[str, Any]] = None


def get_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load and cache config. Thread-safe enough for single-process use.

    Args:
        config_path: Optional path to config file. If not provided,
                     uses FPR_LAB_CONFIG env var or default path.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is invalid YAML.
    """
    global _CONFIG
    if _CONFIG is None or config_path is not None:
        path = Path(
            config_path or os.environ.get("FPR_LAB_CONFIG", str(_DEFAULT_PATH))
        )
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            _CONFIG = yaml.safe_load(f)
        logger.debug(f"Loaded config from {path}")
    return _CONFIG


def get(section: str, key: str, default: Any = None) -> Any:
    """
    Convenience accessor for nested config values.

    Args:
        section: Top-level section name (e.g., 'verifier', 'hypothesis').
        key: Key within the section.
        default: Default value if key not found.

    Returns:
        Config value or default.

    Example:
        min_refs = get('verifier', 'min_evidence_refs', 2)
    """
    return get_config().get(section, {}).get(key, default)


def reset() -> None:
    """
    Reset cached config. For testing.
    """
    global _CONFIG
    _CONFIG = None
