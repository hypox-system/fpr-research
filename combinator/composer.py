"""
Strategy composition (combinator/composer.py).

Delta 2, 26, 30:
- Fail-closed validation
- Side-resolver with SIDE_SPLIT
- One entry, 0+ filters, 1+ exits, 0+ context
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional, TYPE_CHECKING
import hashlib
import pandas as pd
import numpy as np

from signals.base import SignalComponent, get_signal, SIGNAL_REGISTRY
from utils.canonical import canonicalize_spec

if TYPE_CHECKING:
    from data.feature_store import FeatureStore


def resolve_signal_key(base_key: str, side: str) -> str:
    """
    Resolve signal key based on side and SIDE_SPLIT.

    Delta 26, 30:
    - If base_key exists and has SIDE_SPLIT=True:
      - If it already ends with _{side}, return as-is
      - Otherwise, append _{side}
    - If base_key exists and has SIDE_SPLIT=False, return as-is
    - If base_key doesn't exist, try base_key_{side}
    - Raise if nothing found

    Examples:
        resolve_signal_key('ema_stop', 'long') -> 'ema_stop_long'
        resolve_signal_key('ema_stop_long', 'long') -> 'ema_stop_long'
        resolve_signal_key('ema_cross', 'long') -> 'ema_cross'
    """
    # Check if base_key exists directly
    if base_key in SIGNAL_REGISTRY:
        cls = SIGNAL_REGISTRY[base_key]
        if cls.SIDE_SPLIT:
            # Check if already has correct side suffix
            if base_key.endswith(f"_{side}"):
                return base_key
            # Need to find side-specific version
            side_key = f"{base_key}_{side}"
            if side_key in SIGNAL_REGISTRY:
                return side_key
            raise ValueError(
                f"Signal '{base_key}' has SIDE_SPLIT=True but "
                f"'{side_key}' not found in registry"
            )
        else:
            # Non-split signal, use as-is
            return base_key

    # base_key doesn't exist, try with side suffix
    side_key = f"{base_key}_{side}"
    if side_key in SIGNAL_REGISTRY:
        return side_key

    raise ValueError(
        f"Unknown signal: '{base_key}'. "
        f"Neither '{base_key}' nor '{side_key}' found in registry. "
        f"Available: {list(SIGNAL_REGISTRY.keys())}"
    )


def context_to_size(
    context_mults: List[float],
    base_size: float = 1.0,
    min_mult: float = 0.5,
    max_mult: float = 1.5
) -> float:
    """
    Convert context multipliers to position size.

    size = base_size * clip(mean(context_mults), min, max)
    If no context components: size = base_size (mult = 1.0)
    """
    if not context_mults:
        return base_size
    mult = sum(context_mults) / len(context_mults)
    return base_size * max(min_mult, min(max_mult, mult))


@dataclass
class Strategy:
    """
    Composed trading strategy.

    Delta 2:
    - One entry signal (SIGNAL_TYPE == 'entry')
    - Zero or more filter signals (SIGNAL_TYPE == 'filter')
    - One or more exit signals (SIGNAL_TYPE == 'exit')
    - Zero or more context signals (SIGNAL_TYPE == 'context')
    - Single side per strategy (long OR short, never both)
    """

    name: str
    entry: SignalComponent
    filters: List[SignalComponent] = field(default_factory=list)
    exits: List[SignalComponent] = field(default_factory=list)
    context: List[SignalComponent] = field(default_factory=list)
    side: Literal['long', 'short'] = 'long'
    timeframe: str = '1h'
    taker_fee_rate: float = 0.0006  # 0.06% per side
    slippage_rate: float = 0.0001  # 0.01%

    def __post_init__(self):
        """Fail-closed validation."""
        # Validate side
        if self.side not in ('long', 'short'):
            raise ValueError(f"Invalid side: {self.side}. Must be 'long' or 'short'")

        # Validate entry
        if self.entry.SIGNAL_TYPE != 'entry':
            raise ValueError(
                f"Entry signal '{self.entry.KEY}' has type '{self.entry.SIGNAL_TYPE}', "
                f"expected 'entry'"
            )

        # Validate filters
        for f in self.filters:
            if f.SIGNAL_TYPE != 'filter':
                raise ValueError(
                    f"Filter signal '{f.KEY}' has type '{f.SIGNAL_TYPE}', "
                    f"expected 'filter'"
                )

        # Validate exits
        if len(self.exits) < 1:
            raise ValueError("At least one exit signal is required")

        for e in self.exits:
            if e.SIGNAL_TYPE != 'exit':
                raise ValueError(
                    f"Exit signal '{e.KEY}' has type '{e.SIGNAL_TYPE}', "
                    f"expected 'exit'"
                )

        # Validate context
        for c in self.context:
            if c.SIGNAL_TYPE != 'context':
                raise ValueError(
                    f"Context signal '{c.KEY}' has type '{c.SIGNAL_TYPE}', "
                    f"expected 'context'"
                )

        # Validate side compatibility for all components
        all_components = [self.entry] + self.filters + self.exits + self.context
        for comp in all_components:
            if self.side not in comp.SUPPORTED_SIDES:
                raise ValueError(
                    f"Signal '{comp.KEY}' does not support side='{self.side}'. "
                    f"Supported sides: {comp.SUPPORTED_SIDES}"
                )

    def warmup_bars(self) -> int:
        """Max lookback across all components."""
        all_components = [self.entry] + self.filters + self.exits + self.context
        return max(c.lookback_bars() for c in all_components)

    def variant_id(self) -> str:
        """SHA256 of canonical spec (first 12 chars)."""
        spec = self._to_spec()
        canonical = canonicalize_spec(spec)
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]

    def _to_spec(self) -> dict:
        """Convert strategy to canonical spec dict."""
        return {
            'entry': {'key': self.entry.KEY, 'params': self.entry.params},
            'filters': [{'key': f.KEY, 'params': f.params} for f in self.filters],
            'exits': [{'key': e.KEY, 'params': e.params} for e in self.exits],
            'context': [{'key': c.KEY, 'params': c.params} for c in self.context],
            'side': self.side,
            'timeframe': self.timeframe,
            'fee_rate': self.taker_fee_rate,
            'slippage_rate': self.slippage_rate,
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        cache: 'FeatureStore'
    ) -> pd.DataFrame:
        """
        Compute all signal components.

        Returns DataFrame with columns:
        - entry_signal: bool (entry active AND all filters pass)
        - exit_signal: bool (any exit active)
        - context_mult: float64 (mean of context multipliers, clipped)

        Fail-closed: if any component returns wrong dtype, NaN,
        or wrong index -> raises ValueError.
        """
        result = pd.DataFrame(index=df.index)

        # Compute entry
        entry_raw = self.entry.compute(df, cache)
        self._validate_output(entry_raw, df, self.entry, 'entry')
        entry_signal = entry_raw == 1

        # Compute filters and combine with entry
        for f in self.filters:
            filter_raw = f.compute(df, cache)
            self._validate_output(filter_raw, df, f, 'filter')
            entry_signal = entry_signal & filter_raw

        result['entry_signal'] = entry_signal.astype(bool)

        # Compute exits (any exit triggers)
        if self.exits:
            exit_signal = pd.Series(False, index=df.index)
            for e in self.exits:
                exit_raw = e.compute(df, cache)
                self._validate_output(exit_raw, df, e, 'exit')
                exit_signal = exit_signal | exit_raw
            result['exit_signal'] = exit_signal.astype(bool)
        else:
            result['exit_signal'] = pd.Series(False, index=df.index)

        # Compute context multipliers
        if self.context:
            context_values = []
            for c in self.context:
                ctx_raw = c.compute(df, cache)
                self._validate_output(ctx_raw, df, c, 'context')
                context_values.append(ctx_raw)

            # Stack and compute mean, then clip
            context_stack = pd.concat(context_values, axis=1)
            context_mean = context_stack.mean(axis=1)
            result['context_mult'] = context_mean.clip(0.5, 1.5).astype(np.float64)
        else:
            result['context_mult'] = pd.Series(1.0, index=df.index, dtype=np.float64)

        return result

    def _validate_output(
        self,
        output: pd.Series,
        df: pd.DataFrame,
        component: SignalComponent,
        expected_type: str
    ) -> None:
        """Validate signal output. Raises ValueError on failure."""
        # Check index match
        if not output.index.equals(df.index):
            raise ValueError(
                f"Signal '{component.KEY}' output index does not match input index"
            )

        # Check length
        if len(output) != len(df):
            raise ValueError(
                f"Signal '{component.KEY}' output length {len(output)} "
                f"!= input length {len(df)}"
            )

        # Check for all NaN (allow some NaN during warmup)
        if output.isna().all():
            raise ValueError(
                f"Signal '{component.KEY}' returned all NaN values"
            )

        # Check dtype based on type
        if expected_type == 'entry':
            if not pd.api.types.is_integer_dtype(output):
                raise ValueError(
                    f"Entry signal '{component.KEY}' must be integer, "
                    f"got {output.dtype}"
                )
        elif expected_type in ('filter', 'exit'):
            if not pd.api.types.is_bool_dtype(output):
                raise ValueError(
                    f"{expected_type.title()} signal '{component.KEY}' must be bool, "
                    f"got {output.dtype}"
                )
        elif expected_type == 'context':
            if not pd.api.types.is_float_dtype(output):
                raise ValueError(
                    f"Context signal '{component.KEY}' must be float, "
                    f"got {output.dtype}"
                )


def create_strategy_from_config(
    config: dict,
    side: str,
    timeframe: str,
    name: Optional[str] = None
) -> Strategy:
    """
    Create Strategy from sweep config dict.

    Config format:
    {
        'entry': {'type': 'ema_cross', 'params': {'fast': 8, 'slow': 21}},
        'filters': [{'type': 'ema_gating', 'params': {'period': 50}}],
        'exits': [{'type': 'ema_stop', 'params': {'period': 50}}],
        'context': [{'type': 'macd_phase', 'params': {...}}],
        'fees': {'taker_fee_pct': 0.06, 'slippage_pct': 0.01}
    }

    Delta 26: Side-resolver applied to exit signals with SIDE_SPLIT=True
    """
    # Parse fees
    fees = config.get('fees', {})
    taker_fee_rate = fees.get('taker_fee_pct', 0.06) / 100  # pct to rate
    slippage_rate = fees.get('slippage_pct', 0.01) / 100

    # Create entry signal
    entry_cfg = config['entry']
    entry_key = resolve_signal_key(entry_cfg['type'], side)
    entry_cls = get_signal(entry_key)
    entry = entry_cls(params=entry_cfg.get('params', {}))

    # Create filter signals
    filters = []
    for f_cfg in config.get('filters', []):
        f_key = resolve_signal_key(f_cfg['type'], side)
        f_cls = get_signal(f_key)
        filters.append(f_cls(params=f_cfg.get('params', {})))

    # Create exit signals (side-resolver applies here)
    exits = []
    for e_cfg in config.get('exits', []):
        e_key = resolve_signal_key(e_cfg['type'], side)
        e_cls = get_signal(e_key)
        exits.append(e_cls(params=e_cfg.get('params', {})))

    # Create context signals
    context = []
    for c_cfg in config.get('context', []):
        c_key = resolve_signal_key(c_cfg['type'], side)
        c_cls = get_signal(c_key)
        context.append(c_cls(params=c_cfg.get('params', {})))

    return Strategy(
        name=name or f"strategy_{side}_{timeframe}",
        entry=entry,
        filters=filters,
        exits=exits,
        context=context,
        side=side,
        timeframe=timeframe,
        taker_fee_rate=taker_fee_rate,
        slippage_rate=slippage_rate,
    )
