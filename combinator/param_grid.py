"""
Parameter grid generation (combinator/param_grid.py).

Generates all parameter combinations from sweep YAML config.
"""

from typing import List, Dict, Any, Iterator
from itertools import product


def expand_param_grid(params: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    Expand parameter grid to list of parameter dicts.

    Args:
        params: Dict mapping param names to lists of values
                e.g., {'fast': [5, 8], 'slow': [21, 34]}

    Returns:
        List of dicts with all combinations
        e.g., [{'fast': 5, 'slow': 21}, {'fast': 5, 'slow': 34}, ...]
    """
    if not params:
        return [{}]

    keys = list(params.keys())
    values = [params[k] for k in keys]

    combinations = []
    for combo in product(*values):
        combinations.append(dict(zip(keys, combo)))

    return combinations


def expand_component_grid(component_config: dict) -> List[dict]:
    """
    Expand a single component config to all parameter variants.

    Args:
        component_config: {'type': 'ema_cross', 'params': {'fast': [5,8], 'slow': [21,34]}}

    Returns:
        List of configs with expanded params:
        [
            {'type': 'ema_cross', 'params': {'fast': 5, 'slow': 21}},
            {'type': 'ema_cross', 'params': {'fast': 5, 'slow': 34}},
            ...
        ]
    """
    comp_type = component_config['type']
    params = component_config.get('params', {})

    # Separate scalar params from list params
    scalar_params = {}
    grid_params = {}

    for key, value in params.items():
        if isinstance(value, list):
            grid_params[key] = value
        else:
            scalar_params[key] = value

    # Expand grid params
    param_combos = expand_param_grid(grid_params)

    # Combine with scalar params
    result = []
    for combo in param_combos:
        full_params = {**scalar_params, **combo}
        result.append({
            'type': comp_type,
            'params': full_params
        })

    return result


def expand_component_list_grid(components: List[dict]) -> List[List[dict]]:
    """
    Expand a list of components (filters, exits) to all combinations.

    Each component can have its own param grid. The result is the
    cartesian product of all component grids.

    Args:
        components: List of component configs

    Returns:
        List of component lists (each list is one combination)
    """
    if not components:
        return [[]]

    # Expand each component individually
    expanded_components = [expand_component_grid(c) for c in components]

    # Cartesian product across all components
    result = []
    for combo in product(*expanded_components):
        result.append(list(combo))

    return result


def count_variants(sweep_config: dict) -> int:
    """
    Count total number of variants in a sweep config.

    Formula:
    variants = (
        n_entry_params *
        n_filter_combos *
        n_exit_combos *
        n_context_combos *
        n_sides *
        n_timeframes
    )
    """
    # Entry variants
    entry_variants = len(expand_component_grid(sweep_config['entry']))

    # Filter variants (cartesian product)
    filter_combos = expand_component_list_grid(sweep_config.get('filters', []))
    n_filter_combos = len(filter_combos)

    # Exit variants
    exit_combos = expand_component_list_grid(sweep_config.get('exits', []))
    n_exit_combos = len(exit_combos)

    # Context variants
    context_combos = expand_component_list_grid(sweep_config.get('context', []))
    n_context_combos = len(context_combos)

    # Sides and timeframes
    n_sides = len(sweep_config.get('sides', ['long']))
    n_timeframes = len(sweep_config.get('timeframes', ['1h']))

    return (
        entry_variants *
        n_filter_combos *
        n_exit_combos *
        n_context_combos *
        n_sides *
        n_timeframes
    )


def generate_variant_configs(sweep_config: dict, log_constraints: bool = True) -> Iterator[dict]:
    """
    Generate all variant configurations from a sweep config.

    Yields dicts ready for create_strategy_from_config():
    {
        'entry': {'type': '...', 'params': {...}},
        'filters': [...],
        'exits': [...],
        'context': [...],
        'fees': {...},
        'side': 'long'|'short',
        'timeframe': '...',
    }

    Note: Applies constraint filtering (e.g., fast < slow for EMA cross)
    and logs how many variants were pruned by constraints.
    """
    # Expand all component grids
    entry_variants = expand_component_grid(sweep_config['entry'])
    filter_combos = expand_component_list_grid(sweep_config.get('filters', []))
    exit_combos = expand_component_list_grid(sweep_config.get('exits', []))
    context_combos = expand_component_list_grid(sweep_config.get('context', []))

    # Get sides and timeframes
    sides = sweep_config.get('sides', ['long'])
    timeframes = sweep_config.get('timeframes', ['1h'])
    fees = sweep_config.get('fees', {})

    # Track constraint pruning
    total_raw = 0
    pruned_count = 0
    pruned_reasons = {}

    # Generate all combinations with constraint filtering
    for entry in entry_variants:
        for filters in filter_combos:
            for exits in exit_combos:
                for context in context_combos:
                    for side in sides:
                        for timeframe in timeframes:
                            total_raw += 1

                            # Apply constraints
                            skip, reason = _check_constraints(entry, filters, exits)
                            if skip:
                                pruned_count += 1
                                pruned_reasons[reason] = pruned_reasons.get(reason, 0) + 1
                                continue

                            yield {
                                'entry': entry,
                                'filters': filters,
                                'exits': exits,
                                'context': context,
                                'fees': fees,
                                'side': side,
                                'timeframe': timeframe,
                            }

    # Log constraint pruning summary
    if log_constraints and pruned_count > 0:
        print(f"  Constraint pruning: {pruned_count}/{total_raw} variants removed")
        for reason, count in pruned_reasons.items():
            print(f"    - {reason}: {count}")


def _check_constraints(entry: dict, filters: list, exits: list) -> tuple:
    """
    Check if variant should be pruned due to constraints.

    Returns:
        (skip: bool, reason: str or None)
    """
    entry_type = entry.get('type', '')
    entry_params = entry.get('params', {})

    # EMA cross constraint: fast must be < slow
    if entry_type == 'ema_cross':
        fast = entry_params.get('fast', 0)
        slow = entry_params.get('slow', 0)
        if fast >= slow:
            return True, 'fast >= slow'

    # Add more constraints here as needed

    return False, None


def validate_param_ranges(sweep_config: dict) -> List[str]:
    """
    Validate parameter ranges in sweep config.

    Returns list of warnings (empty if all OK).
    """
    warnings = []

    # Check entry params
    entry = sweep_config.get('entry', {})
    entry_params = entry.get('params', {})

    # EMA cross: fast should be < slow
    if entry.get('type') == 'ema_cross':
        fast_vals = entry_params.get('fast', [])
        slow_vals = entry_params.get('slow', [])
        if isinstance(fast_vals, list) and isinstance(slow_vals, list):
            max_fast = max(fast_vals) if fast_vals else 0
            min_slow = min(slow_vals) if slow_vals else float('inf')
            if max_fast >= min_slow:
                warnings.append(
                    f"EMA cross: some fast values ({fast_vals}) >= slow values ({slow_vals})"
                )

    return warnings
