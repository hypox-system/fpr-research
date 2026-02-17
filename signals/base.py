"""
Signal component base class and registry.

Delta 2, 30:
- One component = one role (entry OR filter OR exit OR context)
- Registry via ClassVar KEY, not instance field
- SIDE_SPLIT ClassVar for side-dependent signals
- SUPPORTED_SIDES ClassVar for side compatibility
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Any, Literal, Optional, Set, TYPE_CHECKING
import copy
import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from data.feature_store import FeatureStore

SignalType = Literal['entry', 'exit', 'filter', 'context']


@dataclass
class SignalComponent(ABC):
    """
    Base class for all signal components.

    Contract:
    - ONE component = ONE role (entry OR filter OR exit OR context)
    - compute() takes DataFrame + FeatureStore cache
    - compute() returns Series with SAME index as input
    - No lookahead (verified with append-test)
    - Output dtype:
        entry:   int8 in {0, 1} (1 = signal active, side determined by Strategy)
        exit:    bool
        filter:  bool (strict, no float-truthy)
        context: float64 (multiplier, e.g., 0.5-1.5. Default 1.0 = no adjustment)

    ClassVars (Delta 30):
    - KEY: Unique registry key (e.g., 'ema_cross')
    - SIGNAL_TYPE: 'entry', 'exit', 'filter', or 'context'
    - SUPPORTED_SIDES: Set of supported sides {'long'}, {'short'}, or {'long', 'short'}
    - SIDE_SPLIT: True for side-split signals (ema_stop_long, etc.)
    """

    KEY: ClassVar[str]
    SIGNAL_TYPE: ClassVar[SignalType]
    SUPPORTED_SIDES: ClassVar[Set[str]] = {'long'}
    SIDE_SPLIT: ClassVar[bool] = False

    params: Dict[str, Any] = field(default_factory=dict)

    @abstractmethod
    def compute(self, df: pd.DataFrame, cache: 'FeatureStore') -> pd.Series:
        """
        Compute the signal.

        Args:
            df: OHLCV DataFrame
            cache: FeatureStore for indicator caching

        Returns:
            Series with signal values, same index as df
        """
        pass

    @abstractmethod
    def param_grid(self) -> Dict[str, List[Any]]:
        """
        Return parameter ranges for sweep.

        Returns:
            Dict mapping param names to lists of values
        """
        pass

    def lookback_bars(self) -> int:
        """
        Return maximum lookback bars needed.

        Used by walk-forward for warmup calculation.
        Override in subclasses with actual lookback requirements.
        """
        return 0

    def with_params(self, **kwargs) -> 'SignalComponent':
        """
        Create a copy with updated params.

        Args:
            **kwargs: Parameters to update

        Returns:
            New instance with updated params
        """
        new = copy.deepcopy(self)
        new.params.update(kwargs)
        return new

    def required_columns(self) -> List[str]:
        """Return required DataFrame columns."""
        return ['open', 'high', 'low', 'close', 'volume']

    def validate_output(self, output: pd.Series, df: pd.DataFrame) -> None:
        """
        Validate signal output.

        Raises ValueError if output is invalid.
        """
        # Check index match
        if not output.index.equals(df.index):
            raise ValueError(
                f"{self.KEY}: Output index does not match input index"
            )

        # Check length
        if len(output) != len(df):
            raise ValueError(
                f"{self.KEY}: Output length {len(output)} != input length {len(df)}"
            )

        # Check dtype based on signal type
        if self.SIGNAL_TYPE == 'entry':
            # Should be int8 with values 0 or 1
            if not pd.api.types.is_integer_dtype(output):
                raise ValueError(
                    f"{self.KEY}: Entry signal must be integer, got {output.dtype}"
                )
            unique_vals = set(output.dropna().unique())
            if not unique_vals.issubset({0, 1}):
                raise ValueError(
                    f"{self.KEY}: Entry signal must be 0 or 1, got {unique_vals}"
                )

        elif self.SIGNAL_TYPE in ('exit', 'filter'):
            # Should be bool
            if not pd.api.types.is_bool_dtype(output):
                raise ValueError(
                    f"{self.KEY}: {self.SIGNAL_TYPE} signal must be bool, got {output.dtype}"
                )

        elif self.SIGNAL_TYPE == 'context':
            # Should be float64
            if not pd.api.types.is_float_dtype(output):
                raise ValueError(
                    f"{self.KEY}: Context signal must be float, got {output.dtype}"
                )


# Registry
_REGISTRY: Dict[str, type] = {}
SIGNAL_REGISTRY = _REGISTRY  # Public alias for tests


def register_signal(cls: type) -> type:
    """
    Decorator to register a signal class.

    Args:
        cls: Signal class to register

    Returns:
        The same class (for decorator chaining)

    Raises:
        ValueError: If KEY is missing or duplicate
    """
    key = getattr(cls, 'KEY', None)
    if not key:
        raise ValueError(f"{cls.__name__} is missing KEY ClassVar")

    if key in _REGISTRY:
        raise ValueError(f"Duplicate signal KEY: {key}")

    _REGISTRY[key] = cls
    return cls


def get_signal(name: str) -> type:
    """
    Get a signal class by name.

    Args:
        name: Signal KEY

    Returns:
        Signal class

    Raises:
        KeyError: If signal not found
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown signal: {name}. Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]


def list_signals() -> List[str]:
    """Return sorted list of registered signal KEYs."""
    return sorted(_REGISTRY.keys())


def signal_exists(name: str) -> bool:
    """Check if a signal is registered."""
    return name in _REGISTRY


def validate_signal_output(
    signal: SignalComponent,
    output: pd.Series,
    df: pd.DataFrame
) -> Optional[str]:
    """
    Validate signal output and return error message if invalid.

    Args:
        signal: Signal component
        output: Signal output
        df: Input DataFrame

    Returns:
        Error message string or None if valid
    """
    try:
        signal.validate_output(output, df)
        return None
    except ValueError as e:
        return str(e)


def get_signal_info(name: str) -> Dict[str, Any]:
    """
    Get information about a signal.

    Args:
        name: Signal KEY

    Returns:
        Dict with signal metadata
    """
    cls = get_signal(name)
    instance = cls()

    return {
        'key': cls.KEY,
        'type': cls.SIGNAL_TYPE,
        'supported_sides': list(cls.SUPPORTED_SIDES),
        'side_split': cls.SIDE_SPLIT,
        'param_grid': instance.param_grid(),
        'lookback_bars': instance.lookback_bars(),
        'required_columns': instance.required_columns(),
    }
