"""
Report generation (analysis/reports.py).
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional


def generate_variant_report(
    variant: Dict[str, Any],
    trades: List[Dict],
    fold_results: List[Dict],
    regime_metrics: Dict[str, Dict],
    output_path: Optional[Path] = None
) -> str:
    """
    Generate full report for a single variant.
    
    Args:
        variant: Variant result dict
        trades: List of trade dicts
        fold_results: List of fold result dicts
        regime_metrics: Per-regime metrics
        output_path: Optional path to save report
        
    Returns:
        Markdown report string
    """
    lines = [
        f"# Variant Report: {variant.get('variant_id', 'unknown')}",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Summary",
        "",
        f"- **Status**: {variant.get('status', 'unknown')}",
        f"- **OOS Sharpe**: {variant.get('oos_sharpe', 0):.3f}",
        f"- **IS Sharpe**: {variant.get('is_sharpe', 0):.3f}",
        f"- **Sharpe Decay**: {variant.get('sharpe_decay', 0):.1%}",
        f"- **Median OOS Sharpe**: {variant.get('median_oos_sharpe', 0):.3f}",
        f"- **Total OOS Trades**: {variant.get('n_trades_oos', 0)}",
        f"- **Profit Factor**: {variant.get('profit_factor', 0):.2f}",
        f"- **Max Drawdown**: {variant.get('max_drawdown', 0):.1%}",
        f"- **Win Rate**: {variant.get('win_rate', 0):.1%}",
        f"- **Consistency Ratio**: {variant.get('consistency_ratio', 0):.1%}",
        "",
        "## Statistical Significance",
        "",
        f"- **Coarse p-value**: {variant.get('p_coarse', 'N/A')}",
        f"- **Refined p-value**: {variant.get('p_refined', 'N/A')}",
        f"- **BH-FDR Significant**: {variant.get('bh_fdr_significant', False)}",
        f"- **DSR p-value**: {variant.get('dsr_pvalue', 'N/A')}",
        "",
        "## Negative Control",
        "",
        f"- **Surrogate Percentile**: {variant.get('surrogate_pctl', 'N/A')}",
        f"- **Surrogate Z-score**: {variant.get('surrogate_z', 'N/A')}",
        f"- **Flagged**: {variant.get('surrogate_flagged', False)}",
        "",
    ]
    
    # Fold results
    if fold_results:
        lines.extend([
            "## Per-Fold Results",
            "",
            "| Fold | Trades | Sharpe | PF | Return |",
            "|------|--------|--------|-----|--------|",
        ])
        for f in fold_results:
            lines.append(
                f"| {f.get('fold_id', '-')} "
                f"| {f.get('n_trades', 0)} "
                f"| {f.get('sharpe', 0):.2f} "
                f"| {f.get('profit_factor', 0):.2f} "
                f"| {f.get('total_return', 0):.1%} |"
            )
        lines.append("")
    
    # Regime analysis
    if regime_metrics:
        lines.extend([
            "## Regime Analysis",
            "",
            "| Regime | Trades | Sharpe | PF | % Total |",
            "|--------|--------|--------|-----|---------|",
        ])
        for regime, metrics in regime_metrics.items():
            lines.append(
                f"| {regime} "
                f"| {metrics.get('n_trades', 0)} "
                f"| {metrics.get('sharpe', 0):.2f} "
                f"| {metrics.get('profit_factor', 0):.2f} "
                f"| {metrics.get('pct_of_total', 0):.1%} |"
            )
        lines.append("")
    
    # Strategy spec
    if 'spec' in variant:
        lines.extend([
            "## Strategy Specification",
            "",
            "```json",
            json.dumps(variant['spec'], indent=2),
            "```",
            "",
        ])
    
    report = "\n".join(lines)
    
    if output_path:
        output_path.write_text(report)
    
    return report


def generate_sweep_findings(
    sweep_name: str,
    manifest: Dict[str, Any],
    top_variants: List[Dict],
    negative_control_results: List[Dict],
    output_path: Optional[Path] = None
) -> str:
    """
    Generate findings report for a sweep.
    
    Args:
        sweep_name: Name of the sweep
        manifest: Sweep manifest
        top_variants: List of top variant dicts
        negative_control_results: Negative control results
        output_path: Optional path to save report
        
    Returns:
        Markdown report string
    """
    lines = [
        f"# Sweep Findings: {sweep_name}",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Overview",
        "",
        f"- **Total Variants**: {manifest.get('n_variants', 0)}",
        f"- **Passed Sanity**: {manifest.get('n_passed_sanity', 0)}",
        f"- **BH-FDR Survivors**: {manifest.get('n_bh_survivors', 0)}",
        f"- **Significant**: {manifest.get('n_significant', 0)}",
        f"- **Runtime**: {manifest.get('runtime_seconds', 0):.1f}s",
        "",
        "## Top Variants",
        "",
    ]
    
    if top_variants:
        lines.extend([
            "| Rank | ID | OOS Sharpe | PF | Trades | Surrogate % |",
            "|------|-----|------------|-----|--------|-------------|",
        ])
        for i, v in enumerate(top_variants[:10], 1):
            lines.append(
                f"| {i} "
                f"| {v.get('variant_id', '-')[:8]} "
                f"| {v.get('oos_sharpe', 0):.2f} "
                f"| {v.get('profit_factor', 0):.2f} "
                f"| {v.get('n_trades_oos', 0)} "
                f"| {v.get('surrogate_pctl', 0):.1f} |"
            )
        lines.append("")
    
    # Negative control summary
    n_flagged = sum(1 for r in negative_control_results if r.get('flagged', False))
    lines.extend([
        "## Negative Control Summary",
        "",
        f"- **Variants Tested**: {len(negative_control_results)}",
        f"- **Flagged (< 90th pctl)**: {n_flagged}",
        "",
    ])
    
    # Key findings
    lines.extend([
        "## Key Findings",
        "",
        "1. TODO: Add manual observations",
        "2. TODO: Note any regime dependencies",
        "3. TODO: Identify robust patterns",
        "",
        "## Next Steps",
        "",
        "- [ ] Review top variant specifications",
        "- [ ] Check for parameter clustering",
        "- [ ] Run holdout validation on candidates",
        "",
    ])
    
    report = "\n".join(lines)
    
    if output_path:
        output_path.write_text(report)
    
    return report
