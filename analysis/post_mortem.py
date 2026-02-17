"""
Post-mortem analysis engine.

Analyzes completed sweeps and generates structured reports.

Usage:
    python -m analysis.post_mortem results/sweeps/sweep_001_ema_pvsra/
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils.experiment_fingerprint import compute_experiment_id

logger = logging.getLogger(__name__)

# Failure mode thresholds (configurable defaults)
DEFAULT_THRESHOLDS = {
    'fee_share_for_fee_drag': 30.0,  # %
    'min_duration_bars_for_fee_drag': 10,
    'fold_dispersion_ratio_for_fragile': 2.0,
    'trades_per_month_for_overtrading': 500,
    'sharpe_for_low_signal': -5.0,
}

# Valid failure modes
FAILURE_MODES = [
    'FEE_DRAG',
    'OVERTRADING',
    'REGIME_FRAGILE',
    'GOOD_GROSS_DIES_NET',
    'LEAK_SUSPECT',
    'LOW_SIGNAL',
    'NO_CONVERGENCE',
]


def generate(
    sweep_dir: str,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Generate post-mortem analysis for a sweep.

    Args:
        sweep_dir: Path to sweep directory containing variants.jsonl and manifest.json.
        thresholds: Optional custom thresholds for failure mode detection.

    Returns:
        Post-mortem data dict.

    Raises:
        FileNotFoundError: If required files not found.
        ValueError: If variants.jsonl is empty or unparseable.
    """
    sweep_path = Path(sweep_dir)
    variants_path = sweep_path / 'variants.jsonl'
    manifest_path = sweep_path / 'manifest.json'

    if not variants_path.exists():
        raise FileNotFoundError(f"variants.jsonl not found in {sweep_dir}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in {sweep_dir}")

    # Load manifest
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    # Load variants
    variants = _load_variants(variants_path)

    if not variants:
        raise ValueError(f"No valid variants found in {variants_path}")

    # Use default thresholds with any overrides
    thresh = DEFAULT_THRESHOLDS.copy()
    if thresholds:
        thresh.update(thresholds)

    # Generate post-mortem
    post_mortem = _analyze_sweep(variants, manifest, thresh)

    # Save outputs
    _save_post_mortem(sweep_path, post_mortem)

    return post_mortem


def _load_variants(variants_path: Path) -> List[Dict[str, Any]]:
    """Load variants from JSONL file."""
    variants = []
    with open(variants_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                variant = json.loads(line)
                variants.append(variant)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed line {line_num}: {e}")
                continue
    return variants


def _analyze_sweep(
    variants: List[Dict[str, Any]],
    manifest: Dict[str, Any],
    thresholds: Dict[str, float],
) -> Dict[str, Any]:
    """Perform post-mortem analysis on sweep data."""
    sweep_id = manifest.get('sweep_name', 'unknown')
    now = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')

    # Get sweep created timestamp from manifest
    # Use runtime as proxy if no explicit timestamp
    created_ts = now  # Default to now
    if 'completed_ts' in manifest:
        created_ts = manifest['completed_ts']

    # Filter OK variants for analysis
    ok_variants = [v for v in variants if v.get('status') == 'OK']

    # Compute experiment_id
    experiment_id = manifest.get('experiment_id')
    if not experiment_id:
        # Generate from manifest data
        data_range = manifest.get('data_range', {})
        data_manifest = {
            'symbol': manifest.get('symbol', 'UNKNOWN'),
            'timeframe': manifest.get('timeframes', ['1h'])[0] if isinstance(manifest.get('timeframes'), list) else '1h',
            'date_range': data_range,
            'data_fingerprint': manifest.get('data_fingerprint', 'unknown'),
        }

        # Extract fee info from first OK variant
        fee_rate = 0.0006
        slippage_rate = 0.0001
        if ok_variants:
            spec = ok_variants[0].get('spec', {})
            fee_rate = spec.get('fee_rate', fee_rate)
            slippage_rate = spec.get('slippage_rate', slippage_rate)

        fill_model = {'fee_rate': fee_rate, 'slippage_rate': slippage_rate}

        # We need the original config, but we only have manifest
        # Use manifest as proxy (it contains config_hash)
        experiment_id = compute_experiment_id(manifest, data_manifest, fill_model)

    # Summary statistics
    variant_count = len(variants)
    survivor_count = sum(1 for v in ok_variants if v.get('bh_fdr_significant', False))

    # Find best variant by OOS Sharpe
    best_variant = None
    if ok_variants:
        sorted_variants = sorted(ok_variants, key=lambda v: v.get('oos_sharpe', float('-inf')), reverse=True)
        best = sorted_variants[0]
        best_variant = {
            'variant_id': best.get('variant_id'),
            'oos_sharpe': best.get('oos_sharpe'),
            'n_trades_oos': best.get('n_trades_oos'),
            'profit_factor': best.get('profit_factor'),
            'win_rate': best.get('win_rate'),
        }

    # Fee decomposition
    fee_decomposition = _compute_fee_decomposition(ok_variants, manifest)

    # Trade duration distribution
    trade_duration = _compute_trade_duration(ok_variants)

    # Fold stability
    fold_stability, fold_dispersion = _compute_fold_stability(ok_variants)

    # Determine primary failure mode
    primary_failure_mode, failure_evidence = _determine_failure_mode(
        ok_variants, fee_decomposition, trade_duration, fold_stability, fold_dispersion,
        survivor_count, best_variant, thresholds
    )

    # Find most promising region
    most_promising = _find_promising_region(ok_variants)

    # Generate next experiment constraints
    constraints = _generate_constraints(
        primary_failure_mode, fee_decomposition, trade_duration, thresholds
    )

    return {
        'sweep_id': sweep_id,
        'experiment_id': experiment_id,
        'created_ts': created_ts,
        'generated_ts': now,
        'summary': {
            'variant_count': variant_count,
            'survivor_count': survivor_count,
            'best_variant': best_variant,
        },
        'fee_decomposition': fee_decomposition,
        'trade_duration_distribution': trade_duration,
        'fold_stability': fold_stability,
        'fold_dispersion': fold_dispersion,
        'primary_failure_mode': primary_failure_mode,
        'failure_evidence': failure_evidence,
        'most_promising_region': most_promising,
        'next_experiment_constraints': constraints,
    }


def _compute_fee_decomposition(
    variants: List[Dict[str, Any]],
    manifest: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute fee decomposition analysis."""
    # Get fee/slippage from first variant's spec
    fee_rate = 0.0006
    slippage_rate = 0.0001
    if variants:
        spec = variants[0].get('spec', {})
        fee_rate = spec.get('fee_rate', fee_rate)
        slippage_rate = spec.get('slippage_rate', slippage_rate)

    # Round-trip fee drag formula: 2 * (fee_rate + slippage_rate)
    estimated_fee_drag = 2 * (fee_rate + slippage_rate)

    # Try to get trade summary data from variants
    median_net_returns = []
    median_gross_returns = []
    data_source = 'config_only'

    for v in variants:
        trade_summary = v.get('trade_summary')
        if trade_summary:
            data_source = 'trade_summary'
            if trade_summary.get('median_net_return_per_trade') is not None:
                median_net_returns.append(trade_summary['median_net_return_per_trade'])
            if trade_summary.get('median_gross_return_per_trade') is not None:
                median_gross_returns.append(trade_summary['median_gross_return_per_trade'])

    # Compute medians
    median_net = float(np.median(median_net_returns)) if median_net_returns else None
    median_gross = float(np.median(median_gross_returns)) if median_gross_returns else None

    # Compute fee share of loss
    fee_share = None
    if median_gross is not None and median_net is not None and median_net < 0:
        # Fee share = how much of the loss is due to fees
        # If gross = -0.0021 and net = -0.0033, fee impact = 0.0012
        # fee_share = 0.0012 / 0.0033 * 100 = 36.4%
        fee_impact = median_gross - median_net
        if abs(median_net) > 0:
            fee_share = (fee_impact / abs(median_net)) * 100

    return {
        'median_gross_return_per_trade': median_gross,
        'median_net_return_per_trade': median_net,
        'estimated_fee_drag_per_trade': estimated_fee_drag,
        'fee_share_of_loss_pct': fee_share,
        'data_source': data_source,
    }


def _compute_trade_duration(
    variants: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Compute trade duration distribution."""
    median_bars = []
    p25_bars = []
    p75_bars = []

    for v in variants:
        duration_summary = v.get('duration_summary')
        if duration_summary:
            if duration_summary.get('median_bars') is not None:
                median_bars.append(duration_summary['median_bars'])
            if duration_summary.get('p25_bars') is not None:
                p25_bars.append(duration_summary['p25_bars'])
            if duration_summary.get('p75_bars') is not None:
                p75_bars.append(duration_summary['p75_bars'])

    if not median_bars:
        return None

    return {
        'median_bars': int(np.median(median_bars)),
        'p25_bars': int(np.median(p25_bars)) if p25_bars else None,
        'p75_bars': int(np.median(p75_bars)) if p75_bars else None,
    }


def _compute_fold_stability(
    variants: List[Dict[str, Any]],
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[Dict[str, float]]]:
    """Compute fold stability metrics."""
    # Aggregate fold results across variants
    fold_sharpes = {}  # fold_id -> list of sharpes
    fold_trades = {}   # fold_id -> list of n_trades

    for v in variants:
        fold_results = v.get('fold_results')
        if fold_results:
            for fold in fold_results:
                fold_id = fold.get('fold')
                if fold_id is not None:
                    if fold_id not in fold_sharpes:
                        fold_sharpes[fold_id] = []
                        fold_trades[fold_id] = []
                    if fold.get('oos_sharpe') is not None:
                        fold_sharpes[fold_id].append(fold['oos_sharpe'])
                    if fold.get('n_trades') is not None:
                        fold_trades[fold_id].append(fold['n_trades'])

    if not fold_sharpes:
        return None, None

    # Compute per-fold median metrics
    stability = []
    all_sharpes = []
    for fold_id in sorted(fold_sharpes.keys()):
        sharpes = fold_sharpes[fold_id]
        trades = fold_trades[fold_id]
        median_sharpe = float(np.median(sharpes))
        median_trades = int(np.median(trades)) if trades else 0
        stability.append({
            'fold': fold_id,
            'oos_sharpe': median_sharpe,
            'n_trades': median_trades,
        })
        all_sharpes.extend(sharpes)

    # Compute dispersion
    dispersion = None
    if all_sharpes:
        dispersion = {
            'std': float(np.std(all_sharpes)),
            'iqr': float(np.percentile(all_sharpes, 75) - np.percentile(all_sharpes, 25)),
        }

    return stability, dispersion


def _determine_failure_mode(
    variants: List[Dict[str, Any]],
    fee_decomp: Dict[str, Any],
    trade_duration: Optional[Dict[str, Any]],
    fold_stability: Optional[List[Dict[str, Any]]],
    fold_dispersion: Optional[Dict[str, float]],
    survivor_count: int,
    best_variant: Optional[Dict[str, Any]],
    thresholds: Dict[str, float],
) -> Tuple[str, str]:
    """Determine primary failure mode and evidence."""
    best_sharpe = best_variant.get('oos_sharpe', 0) if best_variant else 0
    median_gross = fee_decomp.get('median_gross_return_per_trade')
    median_net = fee_decomp.get('median_net_return_per_trade')
    fee_share = fee_decomp.get('fee_share_of_loss_pct')

    # Check: GOOD_GROSS_DIES_NET
    if median_gross is not None and median_net is not None:
        if median_gross > 0 and median_net < 0:
            return (
                'GOOD_GROSS_DIES_NET',
                f"Median gross return {median_gross:.4f} is positive, but net return "
                f"{median_net:.4f} is negative. Fee+slippage is killing the edge."
            )

    # Check: FEE_DRAG
    if trade_duration is not None and fee_share is not None:
        median_bars = trade_duration.get('median_bars', 0)
        if (median_bars < thresholds['min_duration_bars_for_fee_drag'] and
                fee_share > thresholds['fee_share_for_fee_drag']):
            fee_drag = fee_decomp.get('estimated_fee_drag_per_trade', 0) * 100
            return (
                'FEE_DRAG',
                f"Median trade duration {median_bars} bars is too short. "
                f"Fee drag accounts for {fee_share:.1f}% of losses. "
                f"Round-trip cost: {fee_drag:.2f}% per trade."
            )

    # Check: REGIME_FRAGILE
    if fold_dispersion is not None and fold_stability:
        median_fold_sharpe = abs(np.median([f['oos_sharpe'] for f in fold_stability]))
        if fold_dispersion['std'] > thresholds['fold_dispersion_ratio_for_fragile'] * median_fold_sharpe:
            return (
                'REGIME_FRAGILE',
                f"Fold Sharpe std ({fold_dispersion['std']:.2f}) is more than "
                f"{thresholds['fold_dispersion_ratio_for_fragile']}x the median absolute Sharpe "
                f"({median_fold_sharpe:.2f}). Strategy performance varies wildly across market regimes."
            )

    # Check: OVERTRADING
    # Approximate trades per month from data
    if variants and best_variant:
        n_trades = best_variant.get('n_trades_oos', 0)
        # Estimate months from date range (rough)
        # Assume ~5 months based on typical sweep range
        trades_per_month = n_trades / 5.0 if n_trades else 0
        if trades_per_month > thresholds['trades_per_month_for_overtrading']:
            return (
                'OVERTRADING',
                f"Estimated {trades_per_month:.0f} trades per month exceeds threshold of "
                f"{thresholds['trades_per_month_for_overtrading']}. Signal may be triggering too frequently."
            )

    # Check: LOW_SIGNAL
    if survivor_count == 0 and best_sharpe < thresholds['sharpe_for_low_signal']:
        return (
            'LOW_SIGNAL',
            f"No BH-FDR survivors. Best OOS Sharpe is {best_sharpe:.2f}, "
            f"well below threshold of {thresholds['sharpe_for_low_signal']}. "
            "Signal does not capture meaningful edge."
        )

    # Fallback: NO_CONVERGENCE
    return (
        'NO_CONVERGENCE',
        f"No clear failure mode identified. {len(variants)} variants tested with "
        f"{survivor_count} BH-FDR survivors. Further investigation needed."
    )


def _find_promising_region(
    variants: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Find the most promising parameter region."""
    if not variants:
        return None

    # Sort by OOS Sharpe
    ok_variants = [v for v in variants if v.get('status') == 'OK' and v.get('oos_sharpe') is not None]
    if not ok_variants:
        return None

    sorted_variants = sorted(ok_variants, key=lambda v: v.get('oos_sharpe', float('-inf')), reverse=True)

    # Take top 10% or at least top 5
    top_n = max(5, len(sorted_variants) // 10)
    top_variants = sorted_variants[:top_n]

    # Extract param ranges from top variants
    param_ranges = {}
    for v in top_variants:
        spec = v.get('spec', {})
        entry = spec.get('entry', {})
        params = entry.get('params', {})
        for key, value in params.items():
            if key not in param_ranges:
                param_ranges[key] = []
            if isinstance(value, (int, float)):
                param_ranges[key].append(value)

    # Compute ranges
    params = {}
    for key, values in param_ranges.items():
        if values:
            params[key] = [min(values), max(values)]

    if not params:
        return None

    avg_sharpe = np.mean([v.get('oos_sharpe', 0) for v in top_variants])

    return {
        'params': params,
        'score': float(avg_sharpe),
        'note': f"Least negative OOS Sharpe in top {top_n} variants",
    }


def _generate_constraints(
    failure_mode: str,
    fee_decomp: Dict[str, Any],
    trade_duration: Optional[Dict[str, Any]],
    thresholds: Dict[str, float],
) -> List[str]:
    """Generate next experiment constraints based on failure mode."""
    constraints = []

    if failure_mode == 'FEE_DRAG':
        min_bars = thresholds['min_duration_bars_for_fee_drag'] * 2
        constraints.append(f"min_trade_duration_bars >= {min_bars}")
        constraints.append("exit must be volatility-scaled (not fixed EMA)")
        constraints.append("consider 15m or 4h timeframe to change trade duration profile")

    elif failure_mode == 'GOOD_GROSS_DIES_NET':
        fee_drag = fee_decomp.get('estimated_fee_drag_per_trade', 0)
        min_return = fee_drag * 1.5 * 100  # 1.5x to cover fees
        constraints.append(f"target_gross_return_per_trade >= {min_return:.2f}%")
        constraints.append("consider maker orders for reduced fees")
        constraints.append("increase holding period to improve risk/reward")

    elif failure_mode == 'REGIME_FRAGILE':
        constraints.append("add regime filter (volatility or trend)")
        constraints.append("reduce position size in unfavorable regimes")
        constraints.append("test on multiple market cycles")

    elif failure_mode == 'OVERTRADING':
        constraints.append(f"max_trades_per_day <= 3")
        constraints.append("add cooldown period between trades")
        constraints.append("require stronger entry confirmation")

    elif failure_mode == 'LOW_SIGNAL':
        constraints.append("reconsider signal hypothesis entirely")
        constraints.append("try different entry timing (earlier/later)")
        constraints.append("combine with other confirming signals")

    else:  # NO_CONVERGENCE
        constraints.append("expand parameter search space")
        constraints.append("try different exit mechanisms")
        constraints.append("validate on different time periods")

    return constraints


def _save_post_mortem(sweep_path: Path, post_mortem: Dict[str, Any]) -> None:
    """Save post-mortem JSON and markdown files."""
    # Save JSON
    json_path = sweep_path / 'post_mortem.json'
    with open(json_path, 'w') as f:
        json.dump(post_mortem, f, indent=2)

    # Generate and save markdown
    md_content = _generate_markdown(post_mortem)
    md_path = sweep_path / 'post_mortem.md'
    with open(md_path, 'w') as f:
        f.write(md_content)

    logger.info(f"Post-mortem saved to {json_path} and {md_path}")


def _generate_markdown(pm: Dict[str, Any]) -> str:
    """Generate markdown report from post-mortem data."""
    lines = []

    sweep_id = pm.get('sweep_id', 'unknown')
    summary = pm.get('summary', {})
    variant_count = summary.get('variant_count', 0)
    survivor_count = summary.get('survivor_count', 0)
    best = summary.get('best_variant', {})
    failure_mode = pm.get('primary_failure_mode', 'UNKNOWN')
    evidence = pm.get('failure_evidence', '')

    # Determine verdict
    if survivor_count > 0:
        verdict = 'MAYBE'
    elif best and best.get('oos_sharpe', 0) > -2:
        verdict = 'GROSS-GOOD-NET-DEAD' if failure_mode == 'GOOD_GROSS_DIES_NET' else 'MAYBE'
    else:
        verdict = 'DEAD'

    lines.append(f"# Post-Mortem: {sweep_id}")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"{variant_count} variants tested. {survivor_count} BH-FDR survivors. ")
    if survivor_count == 0:
        lines.append(f"All variants show strongly negative OOS Sharpe. ")
    lines.append(f"Primary failure mode: {failure_mode}.")
    lines.append("")
    lines.append(f"## Verdict: {verdict}")
    lines.append("")

    # Top 10 variants table
    lines.append("## Top 10 Variants")
    lines.append("")
    lines.append("| Variant | OOS Sharpe | Trades | PF | Win Rate |")
    lines.append("|---------|-----------|--------|------|----------|")

    if best:
        lines.append(
            f"| {best.get('variant_id', '-')[:8]} | "
            f"{best.get('oos_sharpe', 0):.2f} | "
            f"{best.get('n_trades_oos', 0)} | "
            f"{best.get('profit_factor', 0):.2f} | "
            f"{(best.get('win_rate', 0) * 100):.1f}% |"
        )

    lines.append("")
    lines.append("## Why It Failed")
    lines.append("")
    lines.append(evidence)
    lines.append("")

    # Fee decomposition
    fee_decomp = pm.get('fee_decomposition', {})
    if fee_decomp:
        lines.append("### Fee Analysis")
        lines.append("")
        median_gross = fee_decomp.get('median_gross_return_per_trade')
        median_net = fee_decomp.get('median_net_return_per_trade')
        fee_drag = fee_decomp.get('estimated_fee_drag_per_trade', 0)
        fee_share = fee_decomp.get('fee_share_of_loss_pct')

        if median_gross is not None:
            lines.append(f"- Median gross return per trade: {median_gross * 100:.2f}%")
        if median_net is not None:
            lines.append(f"- Median net return per trade: {median_net * 100:.2f}%")
        lines.append(f"- Round-trip fee+slippage: {fee_drag * 100:.2f}%")
        if fee_share is not None:
            lines.append(f"- Fee share of losses: {fee_share:.1f}%")
        lines.append("")

    # Trade duration
    duration = pm.get('trade_duration_distribution')
    if duration:
        lines.append("### Trade Duration")
        lines.append("")
        lines.append(f"- Median: {duration.get('median_bars')} bars")
        if duration.get('p25_bars'):
            lines.append(f"- P25: {duration.get('p25_bars')} bars")
        if duration.get('p75_bars'):
            lines.append(f"- P75: {duration.get('p75_bars')} bars")
        lines.append("")

    # What to try next
    lines.append("## What To Try Next")
    lines.append("")
    constraints = pm.get('next_experiment_constraints', [])
    for c in constraints:
        lines.append(f"- {c}")
    lines.append("")

    # Most promising region
    promising = pm.get('most_promising_region')
    if promising:
        lines.append("## Most Promising Region")
        lines.append("")
        lines.append(f"Score: {promising.get('score', 0):.2f}")
        lines.append("")
        params = promising.get('params', {})
        for key, range_vals in params.items():
            lines.append(f"- {key}: {range_vals}")
        lines.append("")

    lines.append("---")
    lines.append(f"*Generated: {pm.get('generated_ts')}*")

    return '\n'.join(lines)


def main():
    """CLI entry point."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Generate post-mortem analysis for a sweep')
    parser.add_argument('sweep_dir', help='Path to sweep directory')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    try:
        post_mortem = generate(args.sweep_dir)
        print(f"Post-mortem generated for {post_mortem['sweep_id']}")
        print(f"  Variants: {post_mortem['summary']['variant_count']}")
        print(f"  Survivors: {post_mortem['summary']['survivor_count']}")
        print(f"  Failure mode: {post_mortem['primary_failure_mode']}")
        print(f"  Output: {args.sweep_dir}/post_mortem.json")
        print(f"  Output: {args.sweep_dir}/post_mortem.md")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
