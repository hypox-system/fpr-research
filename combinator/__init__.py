"""Combinator modules for strategy composition and sweep running."""

from .composer import Strategy, context_to_size, resolve_signal_key, create_strategy_from_config
from .param_grid import (
    expand_param_grid,
    expand_component_grid,
    count_variants,
    generate_variant_configs,
)
from .sweep_runner import (
    VariantResult,
    SweepResult,
    load_sweep_config,
    run_sweep,
)

__all__ = [
    'Strategy',
    'context_to_size',
    'resolve_signal_key',
    'create_strategy_from_config',
    'expand_param_grid',
    'expand_component_grid',
    'count_variants',
    'generate_variant_configs',
    'VariantResult',
    'SweepResult',
    'load_sweep_config',
    'run_sweep',
]
