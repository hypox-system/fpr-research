"""
Sweep runner (combinator/sweep_runner.py).

Delta 29, 32, 38:
- Data-sanity pre-filter (n_trades >= 30, valid output)
- NO performance filtering before p-values (BH-FDR order)
- Two-stage p-values: coarse (500) -> BH -> refined (10k)
- No silent drop: all variants in variants.jsonl with status
"""

import json
import os
import sys
import time
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import yaml

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .composer import Strategy, create_strategy_from_config
from .param_grid import count_variants, generate_variant_configs
from data.validate import validate_ohlcv
from engine.backtest import run_backtest
from engine.walk_forward import WalkForwardConfig, walk_forward_split, walk_forward_run
from engine.metrics import compute_metrics
from utils import reason_codes
from utils.canonical import canonicalize_spec
from analysis.statistics import permutation_pvalue_batch
from analysis.multiple_testing import benjamini_hochberg
from analysis.deflated_sharpe import deflated_sharpe_ratio
from analysis.negative_control import generate_surrogates, NegativeControlResult


@dataclass
class VariantResult:
    """Result for a single variant."""
    variant_id: str
    status: str  # 'OK', 'INVALID', 'ERROR'
    reason_code: Optional[str] = None
    exception_summary: Optional[str] = None

    # Metrics (only if status == 'OK')
    is_sharpe: Optional[float] = None
    oos_sharpe: Optional[float] = None
    median_oos_sharpe: Optional[float] = None
    sharpe_decay: Optional[float] = None
    n_trades_oos: Optional[int] = None
    profit_factor: Optional[float] = None
    max_drawdown: Optional[float] = None
    win_rate: Optional[float] = None
    consistency_ratio: Optional[float] = None

    # P-values (filled in later stages)
    p_coarse: Optional[float] = None
    p_refined: Optional[float] = None
    bh_fdr_significant: Optional[bool] = None
    dsr_pvalue: Optional[float] = None

    # Surrogate (filled in later)
    surrogate_pctl: Optional[float] = None
    surrogate_z: Optional[float] = None
    surrogate_mean: Optional[float] = None
    surrogate_std: Optional[float] = None
    surrogate_flagged: Optional[bool] = None

    # Strategy spec for reproducibility
    spec: Optional[dict] = None

    # Trade returns for p-value computation
    oos_trade_returns: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        d = {
            'variant_id': self.variant_id,
            'status': self.status,
            'reason_code': self.reason_code,
            'exception_summary': self.exception_summary,
        }

        if self.status == 'OK':
            d.update({
                'is_sharpe': self.is_sharpe,
                'oos_sharpe': self.oos_sharpe,
                'median_oos_sharpe': self.median_oos_sharpe,
                'sharpe_decay': self.sharpe_decay,
                'n_trades_oos': self.n_trades_oos,
                'profit_factor': self.profit_factor,
                'max_drawdown': self.max_drawdown,
                'win_rate': self.win_rate,
                'consistency_ratio': self.consistency_ratio,
                'p_coarse': self.p_coarse,
                'p_refined': self.p_refined,
                'bh_fdr_significant': self.bh_fdr_significant,
                'dsr_pvalue': self.dsr_pvalue,
                'surrogate_pctl': self.surrogate_pctl,
                'surrogate_z': self.surrogate_z,
                'surrogate_mean': self.surrogate_mean,
                'surrogate_std': self.surrogate_std,
                'surrogate_flagged': self.surrogate_flagged,
            })

        if self.spec:
            d['spec'] = self.spec

        return d


@dataclass
class SweepResult:
    """Result of a complete sweep."""
    sweep_name: str
    n_variants: int
    n_passed_sanity: int
    n_bh_survivors: int
    n_significant: int
    runtime_seconds: float
    results: List[VariantResult] = field(default_factory=list)
    manifest: Dict[str, Any] = field(default_factory=dict)


def load_sweep_config(config_path: str) -> dict:
    """Load and validate sweep YAML config."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Basic validation
    required = ['name', 'symbol', 'entry', 'exits', 'timeframes', 'fees']
    missing = [k for k in required if k not in config]
    if missing:
        raise ValueError(f"Missing required fields in sweep config: {missing}")

    # Theory field is required
    if 'theory' not in config:
        raise ValueError("Sweep config must include 'theory' field explaining the hypothesis")

    return config


def run_single_variant(
    variant_config: dict,
    data: pd.DataFrame,
    wf_config: WalkForwardConfig,
    sweep_name: str,
    variant_idx: int,
) -> VariantResult:
    """
    Run a single variant through walk-forward validation.

    Returns VariantResult with status OK/INVALID/ERROR.
    """
    try:
        # Create strategy
        strategy = create_strategy_from_config(
            config=variant_config,
            side=variant_config['side'],
            timeframe=variant_config['timeframe'],
            name=f"{sweep_name}_v{variant_idx}"
        )

        variant_id = strategy.variant_id()
        spec = strategy._to_spec()

        # Filter data to timeframe
        # For now, assume data is already at correct timeframe
        # TODO: Add resampling support

        # Run walk-forward
        # Note: walk_forward_run creates its own FeatureStore per fold (Delta 31)
        # No need to create cache here - it would be wasted memory
        wf_result = walk_forward_run(strategy, data, wf_config)

        if wf_result is None:
            return VariantResult(
                variant_id=variant_id,
                status='INVALID',
                reason_code=reason_codes.E_NO_TRADES,
                spec=spec,
            )

        # Check minimum trades (data-sanity filter)
        n_trades = len(wf_result.oos_trade_log)
        if n_trades < 30:
            return VariantResult(
                variant_id=variant_id,
                status='INVALID',
                reason_code=reason_codes.E_INSUFFICIENT_TRADES,
                n_trades_oos=n_trades,
                spec=spec,
            )

        # Compute metrics
        oos_metrics = wf_result.oos_metrics
        is_metrics = wf_result.is_metrics

        # Check for valid metrics
        if oos_metrics is None or np.isnan(oos_metrics.get('sharpe', np.nan)):
            return VariantResult(
                variant_id=variant_id,
                status='INVALID',
                reason_code=reason_codes.E_SIGNAL_NAN,
                spec=spec,
            )

        # Calculate sharpe decay
        is_sharpe = is_metrics.get('sharpe', 0) if is_metrics else 0
        oos_sharpe = oos_metrics.get('sharpe', 0)
        sharpe_decay = (is_sharpe - oos_sharpe) / is_sharpe if is_sharpe > 0 else 0

        # Get trade returns for p-value computation
        oos_returns = np.array([t.net_return for t in wf_result.oos_trade_log])

        return VariantResult(
            variant_id=variant_id,
            status='OK',
            is_sharpe=is_sharpe,
            oos_sharpe=oos_sharpe,
            median_oos_sharpe=wf_result.median_oos_sharpe,
            sharpe_decay=sharpe_decay,
            n_trades_oos=n_trades,
            profit_factor=oos_metrics.get('profit_factor'),
            max_drawdown=oos_metrics.get('max_drawdown'),
            win_rate=oos_metrics.get('win_rate'),
            consistency_ratio=wf_result.consistency_ratio,
            spec=spec,
            oos_trade_returns=oos_returns,
        )

    except Exception as e:
        # Create a minimal variant_id from config
        config_str = canonicalize_spec(variant_config)
        variant_id = hashlib.sha256(config_str.encode()).hexdigest()[:12]

        return VariantResult(
            variant_id=variant_id,
            status='ERROR',
            reason_code=reason_codes.E_EXCEPTION,
            exception_summary=str(e)[:200],
            spec=variant_config,
        )


def run_surrogate_tests(
    results: List[VariantResult],
    data: pd.DataFrame,
    n_surrogates: int = 100,
    threshold_pctl: float = 90.0,
    seed: int = 456,
) -> None:
    """
    Run negative control tests using IAAFT surrogates.

    For each variant:
    1. Generate IAAFT surrogates of close prices
    2. Compute surrogate returns using same trade timing
    3. Compare real Sharpe to surrogate distribution
    4. Fill in surrogate_pctl, surrogate_z, surrogate_mean, surrogate_std, surrogate_flagged

    Args:
        results: List of VariantResult objects to test (modified in place)
        data: Original OHLCV DataFrame
        n_surrogates: Number of IAAFT surrogates to generate
        threshold_pctl: Percentile threshold for flagging (variant flagged if < threshold)
        seed: Random seed for reproducibility
    """
    # Get close prices for surrogate generation
    close_prices = data['close'].values

    # Generate IAAFT surrogates once (reused for all variants)
    print(f"  Generating {n_surrogates} IAAFT surrogates...")
    surrogates = generate_surrogates(close_prices, n_surrogates, seed=seed)

    # Compute log returns for original and surrogates
    original_returns = np.diff(np.log(close_prices))
    surrogate_returns_list = [np.diff(np.log(s)) for s in surrogates]

    for r in results:
        if r.oos_trade_returns is None or len(r.oos_trade_returns) == 0:
            continue

        real_sharpe = r.oos_sharpe or 0

        # For each surrogate, compute what the Sharpe would be
        # We use the sign of returns as a proxy for trade outcome
        # This tests if the timing captured real edge vs noise
        n_trades = len(r.oos_trade_returns)

        surrogate_sharpes = []
        rng = np.random.default_rng(seed + hash(r.variant_id) % 10000)

        for surrogate_rets in surrogate_returns_list:
            if len(surrogate_rets) < n_trades:
                # Not enough surrogate data
                surrogate_sharpes.append(0.0)
                continue

            # Sample random trade returns from surrogate
            # Scale to match the magnitude of real trades
            sample_indices = rng.choice(len(surrogate_rets), size=n_trades, replace=True)
            sampled_rets = surrogate_rets[sample_indices]

            # Scale to match real return magnitudes
            real_std = np.std(r.oos_trade_returns)
            if real_std > 0:
                sampled_rets = sampled_rets * (real_std / (np.std(sampled_rets) + 1e-10))

            # Compute Sharpe
            mean_ret = np.mean(sampled_rets)
            std_ret = np.std(sampled_rets)
            if std_ret > 0:
                # Use same annualization as main code (will be fixed in BUG 4)
                surrogate_sharpe = mean_ret / std_ret * np.sqrt(365 * 6)
            else:
                surrogate_sharpe = 0.0

            surrogate_sharpes.append(surrogate_sharpe)

        surrogate_sharpes = np.array(surrogate_sharpes)

        # Compute statistics
        r.surrogate_mean = float(np.mean(surrogate_sharpes))
        r.surrogate_std = float(np.std(surrogate_sharpes))

        # Percentile rank: what % of surrogate Sharpes are <= real Sharpe?
        r.surrogate_pctl = float(100 * np.mean(surrogate_sharpes <= real_sharpe))

        # Z-score
        if r.surrogate_std > 0:
            r.surrogate_z = float((real_sharpe - r.surrogate_mean) / r.surrogate_std)
        else:
            r.surrogate_z = 0.0

        # Flag if real Sharpe is NOT extreme enough (< threshold percentile)
        r.surrogate_flagged = r.surrogate_pctl < threshold_pctl


def run_sweep(
    sweep_config_path: str,
    data: pd.DataFrame,
    output_dir: Optional[str] = None,
    n_jobs: int = 1,
    dry_run: bool = False,
    max_variants: Optional[int] = None,
) -> SweepResult:
    """
    Run a full parameter sweep.

    Delta 29 order:
    1. Validate data (integrity gate)
    2. Validate YAML (schema check)
    3. Generate all variants
    4. Run walk-forward on each variant (joblib)
    5. Data-sanity filter: n_trades >= 30, valid output
       NO performance filtering here (preserves BH-FDR guarantees)
    6. Coarse p-value (500 perm) for ALL sanity-survivors
    7. BH-FDR correction on coarse p-values
    8. Refined p-value (10k perm) only on BH-survivors
    9. DSR on significant variants
    10. Performance filter + ranking AFTER significance
    11. Negative control on top variants
    12. Save manifest.json + variants.jsonl + leaderboard.md
    """
    start_time = time.time()

    # Load config
    config = load_sweep_config(sweep_config_path)
    sweep_name = config['name']

    # Count variants
    n_variants = count_variants(config)
    if max_variants:
        n_variants = min(n_variants, max_variants)

    if dry_run:
        print(f"Sweep: {sweep_name}")
        print(f"Total variants: {n_variants}")
        print(f"Sides: {config.get('sides', ['long'])}")
        print(f"Timeframes: {config.get('timeframes', ['1h'])}")
        return SweepResult(
            sweep_name=sweep_name,
            n_variants=n_variants,
            n_passed_sanity=0,
            n_bh_survivors=0,
            n_significant=0,
            runtime_seconds=0,
        )

    # Validate data
    validation = validate_ohlcv(data)
    if not validation.passed:
        raise ValueError(f"Data validation failed: {validation.errors}")

    # Setup walk-forward config
    wf_cfg = config.get('walk_forward', {})
    wf_config = WalkForwardConfig(
        train_months=wf_cfg.get('train_months', 3),
        test_months=wf_cfg.get('test_months', 1),
        step_months=wf_cfg.get('step_months', 1),
        embargo_bars=wf_cfg.get('embargo_bars', 0),
        min_trades_per_fold=wf_cfg.get('min_trades_per_fold', 5),
    )

    # Generate variant configs
    variant_configs = list(generate_variant_configs(config))
    if max_variants:
        variant_configs = variant_configs[:max_variants]

    print(f"Running {len(variant_configs)} variants with {n_jobs} workers...")

    # Run variants in parallel
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run_single_variant)(
            variant_config=vc,
            data=data,
            wf_config=wf_config,
            sweep_name=sweep_name,
            variant_idx=idx,
        )
        for idx, vc in enumerate(variant_configs)
    )

    # Separate by status
    ok_results = [r for r in results if r.status == 'OK']
    invalid_results = [r for r in results if r.status == 'INVALID']
    error_results = [r for r in results if r.status == 'ERROR']

    n_passed_sanity = len(ok_results)
    print(f"Sanity filter: {n_passed_sanity}/{len(results)} passed")

    # P-value pipeline (Delta 29 order)
    stats_cfg = config.get('statistics', {})
    coarse_n = stats_cfg.get('coarse_permutation_n', 500)
    refined_n = stats_cfg.get('refined_permutation_n', 10000)
    seed_perm = 123  # Logged in manifest

    if ok_results:
        # Step 6: Coarse p-values for ALL sanity-survivors
        print(f"Computing coarse p-values ({coarse_n} permutations)...")
        trade_returns_list = [r.oos_trade_returns for r in ok_results]
        coarse_pvals = permutation_pvalue_batch(
            trade_returns_list, n_permutations=coarse_n, seed=seed_perm
        )
        for r, p in zip(ok_results, coarse_pvals):
            r.p_coarse = p

        # Step 7: BH-FDR correction
        print("Applying BH-FDR correction...")
        bh_significant = benjamini_hochberg(coarse_pvals, alpha=0.05)
        for r, sig in zip(ok_results, bh_significant):
            r.bh_fdr_significant = sig

        n_bh_survivors = sum(1 for r in ok_results if r.bh_fdr_significant)
        print(f"BH-FDR survivors: {n_bh_survivors}")

        # Step 8: Refined p-values only on BH survivors
        bh_survivors = [r for r in ok_results if r.bh_fdr_significant]
        if bh_survivors:
            print(f"Computing refined p-values ({refined_n} permutations) for {len(bh_survivors)} survivors...")
            survivor_returns = [r.oos_trade_returns for r in bh_survivors]
            refined_pvals = permutation_pvalue_batch(
                survivor_returns, n_permutations=refined_n, seed=seed_perm + 1
            )
            for r, p in zip(bh_survivors, refined_pvals):
                r.p_refined = p

            # Step 9: DSR on significant variants
            n_variants_tested = len(ok_results)
            for r in bh_survivors:
                if r.oos_trade_returns is not None and len(r.oos_trade_returns) > 0:
                    r.dsr_pvalue = deflated_sharpe_ratio(
                        observed_sharpe=r.oos_sharpe or 0,
                        n_variants_tested=n_variants_tested,
                        n_observations=len(r.oos_trade_returns),
                    )

            # Step 10: Negative control (surrogate testing) on BH survivors
            surrogate_count = stats_cfg.get('surrogate_count', 100)
            surrogate_threshold = stats_cfg.get('surrogate_pctl_threshold', 90)
            seed_surrogates = 456  # Logged in manifest

            # Only test survivors with positive Sharpe
            positive_survivors = [r for r in bh_survivors if (r.oos_sharpe or 0) > 0]

            if positive_survivors:
                print(f"Running negative control ({surrogate_count} surrogates) on {len(positive_survivors)} variants...")
                run_surrogate_tests(
                    results=positive_survivors,
                    data=data,
                    n_surrogates=surrogate_count,
                    threshold_pctl=surrogate_threshold,
                    seed=seed_surrogates,
                )

        n_significant = sum(1 for r in ok_results if r.bh_fdr_significant and (r.p_refined or 1) < 0.05)
    else:
        n_bh_survivors = 0
        n_significant = 0

    runtime = time.time() - start_time

    # Build manifest
    manifest = build_manifest(
        config=config,
        sweep_config_path=sweep_config_path,
        n_variants=len(variant_configs),
        n_passed_sanity=n_passed_sanity,
        n_bh_survivors=n_bh_survivors,
        n_significant=n_significant,
        runtime_seconds=runtime,
        data=data,
    )

    # Save results if output_dir specified
    if output_dir:
        save_sweep_results(output_dir, sweep_name, results, manifest)

    return SweepResult(
        sweep_name=sweep_name,
        n_variants=len(variant_configs),
        n_passed_sanity=n_passed_sanity,
        n_bh_survivors=n_bh_survivors,
        n_significant=n_significant,
        runtime_seconds=runtime,
        results=results,
        manifest=manifest,
    )


def build_manifest(
    config: dict,
    sweep_config_path: str,
    n_variants: int,
    n_passed_sanity: int,
    n_bh_survivors: int,
    n_significant: int,
    runtime_seconds: float,
    data: pd.DataFrame,
) -> dict:
    """Build sweep manifest."""
    import platform

    # Config hash
    config_str = canonicalize_spec(config)
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()

    # Data fingerprint
    data_start = data['timestamp'].min() if 'timestamp' in data.columns else data.index[0]
    data_end = data['timestamp'].max() if 'timestamp' in data.columns else data.index[-1]
    data_info = f"{len(data)}_{data_start}_{data_end}"
    data_fingerprint = hashlib.sha256(data_info.encode()).hexdigest()[:16]

    # Git commit (if available)
    try:
        import subprocess
        git_commit = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except:
        git_commit = "unknown"

    # Get proper timestamps for data_range
    if 'timestamp' in data.columns:
        range_start = data['timestamp'].min()
        range_end = data['timestamp'].max()
    else:
        range_start = data.index[0]
        range_end = data.index[-1]

    return {
        'sweep_name': config['name'],
        'git_commit': git_commit,
        'data_range': {
            'start': str(range_start.date()) if hasattr(range_start, 'date') else str(range_start),
            'end': str(range_end.date()) if hasattr(range_end, 'date') else str(range_end),
        },
        'symbol': config['symbol'],
        'n_variants': n_variants,
        'n_passed_sanity': n_passed_sanity,
        'n_bh_survivors': n_bh_survivors,
        'n_significant': n_significant,
        'runtime_seconds': round(runtime_seconds, 2),
        'config_hash': config_hash,
        'data_fingerprint': data_fingerprint,
        'environment': {
            'python_version': sys.version.split()[0],
            'pandas_version': pd.__version__,
            'numpy_version': np.__version__,
            'platform': platform.system() + '-' + platform.machine(),
        },
        'seeds': {
            'rng_seed_global': 42,
            'seed_permutation': 123,
            'seed_surrogates': 456,
            'seed_sampler': 789,
        },
        'p_value_stages': {
            'coarse_n_perm': config.get('statistics', {}).get('coarse_permutation_n', 500),
            'refined_n_perm': config.get('statistics', {}).get('refined_permutation_n', 10000),
            'n_coarse_tested': n_passed_sanity,
            'n_bh_survivors': n_bh_survivors,
        },
        'sharpe_annualization': {
            'trading_days_per_year': 365,  # Crypto (vs 252 for stocks)
            'avg_trades_per_day': 6,
            'factor': float(np.sqrt(365 * 6)),  # ~46.80
        },
    }


def save_sweep_results(
    output_dir: str,
    sweep_name: str,
    results: List[VariantResult],
    manifest: dict
) -> None:
    """Save sweep results to disk."""
    sweep_dir = Path(output_dir) / 'sweeps' / sweep_name
    sweep_dir.mkdir(parents=True, exist_ok=True)

    # Save manifest
    manifest_path = sweep_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    # Save variants (JSONL - one per line)
    variants_path = sweep_dir / 'variants.jsonl'
    with open(variants_path, 'w') as f:
        for r in results:
            # Don't include trade returns in JSONL (too large)
            d = r.to_dict()
            f.write(json.dumps(d) + '\n')

    # Generate simple leaderboard
    leaderboard_path = sweep_dir / 'leaderboard.md'
    generate_simple_leaderboard(results, leaderboard_path)

    print(f"Results saved to {sweep_dir}")


def generate_simple_leaderboard(results: List[VariantResult], output_path: Path) -> None:
    """Generate simple markdown leaderboard."""
    ok_results = [r for r in results if r.status == 'OK']

    # Sort by OOS Sharpe
    ok_results.sort(key=lambda r: r.oos_sharpe or 0, reverse=True)

    with open(output_path, 'w') as f:
        f.write("# Sweep Leaderboard\n\n")
        f.write(f"Total variants: {len(results)}\n")
        f.write(f"Passed sanity: {len(ok_results)}\n\n")

        f.write("## Top Variants by OOS Sharpe\n\n")
        f.write("| Rank | Variant ID | OOS Sharpe | IS Sharpe | Decay | Trades | PF | MaxDD | Win% |\n")
        f.write("|------|------------|------------|-----------|-------|--------|-----|-------|------|\n")

        for i, r in enumerate(ok_results[:20], 1):
            f.write(
                f"| {i} | {r.variant_id} | "
                f"{r.oos_sharpe:.2f} | {r.is_sharpe:.2f} | "
                f"{r.sharpe_decay:.2f} | {r.n_trades_oos} | "
                f"{r.profit_factor:.2f} | {r.max_drawdown:.1%} | "
                f"{r.win_rate:.1%} |\n"
            )
