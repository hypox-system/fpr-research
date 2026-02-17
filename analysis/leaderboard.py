"""
Leaderboard ranking (analysis/leaderboard.py).

Delta 39: Rank method='average' for ties.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class LeaderboardConfig:
    """Configuration for leaderboard filtering and ranking."""
    # Minimum filters
    min_oos_sharpe: float = 1.0
    min_is_sharpe: float = 0.5
    min_trades: int = 30
    min_trades_per_fold: int = 5
    max_drawdown: float = 0.20
    max_sharpe_decay: float = 0.6
    min_consistency: float = 0.60
    min_surrogate_pctl: float = 90.0
    
    # Ranking weights
    weight_median_sharpe: float = 0.30
    weight_profit_factor: float = 0.20
    weight_max_dd: float = 0.15
    weight_consistency: float = 0.15
    weight_win_rate: float = 0.10
    weight_trade_count: float = 0.10
    
    # Stability penalty
    stability_threshold: float = 0.50  # If one fold > 50% of profit
    stability_penalty: float = 0.20


def filter_variants(
    variants: List[Dict[str, Any]],
    config: LeaderboardConfig
) -> List[Dict[str, Any]]:
    """
    Apply minimum filters to variants.
    
    Args:
        variants: List of variant result dicts
        config: Filter configuration
        
    Returns:
        Filtered list of variants
    """
    filtered = []
    
    for v in variants:
        # Skip non-OK variants
        if v.get('status') != 'OK':
            continue
            
        # Check minimum criteria
        if v.get('oos_sharpe', 0) < config.min_oos_sharpe:
            continue
        if v.get('is_sharpe', 0) < config.min_is_sharpe:
            continue
        if v.get('n_trades_oos', 0) < config.min_trades:
            continue
        if v.get('max_drawdown', 1.0) > config.max_drawdown:
            continue
        if v.get('sharpe_decay', 1.0) > config.max_sharpe_decay:
            continue
        if v.get('consistency_ratio', 0) < config.min_consistency:
            continue
        if v.get('surrogate_pctl', 0) < config.min_surrogate_pctl:
            continue
        if not v.get('bh_fdr_significant', False):
            continue
            
        filtered.append(v)
    
    return filtered


def rank_variants(
    variants: List[Dict[str, Any]],
    config: Optional[LeaderboardConfig] = None
) -> List[Dict[str, Any]]:
    """
    Rank variants using weighted normalized metrics.
    
    Delta 39: Uses method='average' for ties.
    
    Args:
        variants: List of variant result dicts
        config: Ranking configuration
        
    Returns:
        Sorted list with 'rank' and 'score' added
    """
    if config is None:
        config = LeaderboardConfig()
    
    if not variants:
        return []
    
    # Create DataFrame for ranking
    df = pd.DataFrame(variants)
    
    # Metrics to rank (higher is better for all after transformation)
    metrics = {
        'median_oos_sharpe': ('median_oos_sharpe', False),  # Higher is better
        'profit_factor': ('profit_factor', False),
        'max_drawdown': ('max_drawdown', True),  # Lower is better (invert)
        'consistency_ratio': ('consistency_ratio', False),
        'win_rate': ('win_rate', False),
        'n_trades_oos': ('n_trades_oos', False),
    }
    
    weights = {
        'median_oos_sharpe': config.weight_median_sharpe,
        'profit_factor': config.weight_profit_factor,
        'max_drawdown': config.weight_max_dd,
        'consistency_ratio': config.weight_consistency,
        'win_rate': config.weight_win_rate,
        'n_trades_oos': config.weight_trade_count,
    }
    
    # Compute rank percentiles for each metric
    rank_scores = pd.DataFrame(index=df.index)
    
    for metric_name, (col_name, invert) in metrics.items():
        if col_name not in df.columns:
            rank_scores[metric_name] = 0.5  # Default to middle
            continue
            
        values = df[col_name].fillna(0)
        
        if invert:
            values = -values
        
        # Rank with average method for ties (Delta 39)
        ranks = values.rank(method='average', ascending=False)
        
        # Normalize to 0-1 (percentile)
        rank_scores[metric_name] = ranks / len(ranks)
    
    # Compute weighted score
    total_weight = sum(weights.values())
    df['score'] = sum(
        rank_scores[m] * w / total_weight
        for m, w in weights.items()
    )
    
    # Apply stability penalty
    # (Would need fold-level data to compute properly)
    
    # Sort by score (higher is better)
    df = df.sort_values('score', ascending=False)
    
    # Add rank
    df['rank'] = range(1, len(df) + 1)
    
    return df.to_dict('records')


def generate_leaderboard_md(
    variants: List[Dict[str, Any]],
    top_n: int = 20
) -> str:
    """
    Generate markdown leaderboard.
    
    Args:
        variants: Ranked variant list
        top_n: Number of variants to show
        
    Returns:
        Markdown string
    """
    lines = [
        "# Leaderboard",
        "",
        f"Showing top {min(top_n, len(variants))} of {len(variants)} variants",
        "",
        "| Rank | ID | OOS Sharpe | Median | PF | MaxDD | Win% | Trades | Score |",
        "|------|-----|------------|--------|-----|-------|------|--------|-------|",
    ]
    
    for v in variants[:top_n]:
        lines.append(
            f"| {v.get('rank', '-')} "
            f"| {v.get('variant_id', '-')[:8]} "
            f"| {v.get('oos_sharpe', 0):.2f} "
            f"| {v.get('median_oos_sharpe', 0):.2f} "
            f"| {v.get('profit_factor', 0):.2f} "
            f"| {v.get('max_drawdown', 0):.1%} "
            f"| {v.get('win_rate', 0):.1%} "
            f"| {v.get('n_trades_oos', 0)} "
            f"| {v.get('score', 0):.3f} |"
        )
    
    return "\n".join(lines)
