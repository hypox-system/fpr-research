#!/usr/bin/env python3
"""
FPR Research Platform - CLI Entry Point

Usage:
    python main.py fetch [--start DATE] [--end DATE]
    python main.py validate CONFIG_PATH
    python main.py sweep CONFIG_PATH [--dry-run] [--jobs N]
    python main.py signals [SIGNAL_KEY] [--grid]
    python main.py leaderboard [--min-sharpe N] [--min-trades N]
    python main.py report VARIANT_ID
    python main.py regimes VARIANT_ID
    python main.py repro VARIANT_ID
    python main.py holdout [--top N]
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def cmd_fetch(args):
    """Fetch or update market data."""
    from data.fetcher import get_data, DEFAULT_IS_START, DEFAULT_HOLDOUT_END

    start = args.start or DEFAULT_IS_START
    end = args.end or DEFAULT_HOLDOUT_END

    print(f"Fetching data from {start} to {end}...")
    df = get_data(start_date=start, end_date=end, force_fetch=args.force)

    if df.empty:
        print("No data returned")
        return 1

    print(f"Loaded {len(df)} rows")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Validate
    from data.validate import validate_ohlcv
    report = validate_ohlcv(df)
    print(f"\n{report}")

    return 0 if report.passed else 1


def cmd_validate(args):
    """Validate a sweep configuration YAML."""
    import yaml
    import jsonschema
    from combinator.param_grid import count_variants, validate_param_ranges

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1

    schema_path = PROJECT_ROOT / "config" / "schema_sweep.json"
    if not schema_path.exists():
        print(f"Error: Schema file not found: {schema_path}")
        return 1

    # Load config and schema
    with open(config_path) as f:
        config = yaml.safe_load(f)

    with open(schema_path) as f:
        schema = json.load(f)

    # Validate against schema
    try:
        jsonschema.validate(config, schema)
        print(f"Config validated successfully: {config_path}")
    except jsonschema.ValidationError as e:
        print(f"Validation error: {e.message}")
        print(f"Path: {list(e.path)}")
        return 1

    # Check required fields
    required = ['name', 'symbol', 'theory', 'entry', 'exits', 'timeframes', 'fees']
    missing = [k for k in required if k not in config]
    if missing:
        print(f"Missing required fields: {missing}")
        return 1

    # Print config summary
    print(f"\nSweep: {config.get('name')}")
    print(f"Symbol: {config.get('symbol')}")
    print(f"Timeframes: {config.get('timeframes')}")
    print(f"Sides: {config.get('sides', ['long'])}")

    # Validate param ranges
    warnings = validate_param_ranges(config)
    for w in warnings:
        print(f"Warning: {w}")

    # Count variants using param_grid module
    total_variants = count_variants(config)
    print(f"\nTotal variants: {total_variants}")

    return 0


def cmd_sweep(args):
    """Run a parameter sweep."""
    from combinator.sweep_runner import load_sweep_config, run_sweep
    from combinator.param_grid import count_variants

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1

    # Load config for summary
    config = load_sweep_config(str(config_path))
    n_variants = count_variants(config)

    print(f"Sweep: {config['name']}")
    print(f"Symbol: {config['symbol']}")
    print(f"Variants: {n_variants}")

    if args.dry_run:
        print("\n[DRY RUN] Would execute sweep with:")
        print(f"  - {n_variants} variants")
        print(f"  - {args.jobs} parallel jobs")
        print(f"  - Timeframes: {config.get('timeframes', ['1h'])}")
        print(f"  - Sides: {config.get('sides', ['long'])}")
        return 0

    # Load data
    from data.fetcher import get_data, DEFAULT_IS_START, DEFAULT_IS_END

    print(f"\nLoading data for {config['symbol']}...")
    df = get_data(start_date=DEFAULT_IS_START, end_date=DEFAULT_IS_END)

    if df.empty:
        print("Error: No data loaded")
        return 1

    print(f"Loaded {len(df)} bars")

    # Run sweep
    output_dir = PROJECT_ROOT / "results"
    result = run_sweep(
        sweep_config_path=str(config_path),
        data=df,
        output_dir=str(output_dir),
        n_jobs=args.jobs,
        dry_run=False,
    )

    print(f"\nSweep complete:")
    print(f"  Total variants: {result.n_variants}")
    print(f"  Passed sanity: {result.n_passed_sanity}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")

    return 0


def cmd_signals(args):
    """List available signals or show signal details."""
    from signals.base import list_signals, get_signal_info, signal_exists

    if args.signal:
        # Show specific signal
        if not signal_exists(args.signal):
            print(f"Unknown signal: {args.signal}")
            print(f"Available: {list_signals()}")
            return 1

        info = get_signal_info(args.signal)
        print(f"Signal: {info['key']}")
        print(f"Type: {info['type']}")
        print(f"Supported sides: {info['supported_sides']}")
        print(f"Side split: {info['side_split']}")
        print(f"Lookback bars: {info['lookback_bars']}")

        if args.grid:
            print(f"\nParameter grid:")
            for param, values in info['param_grid'].items():
                print(f"  {param}: {values}")
    else:
        # List all signals
        signals = list_signals()
        if signals:
            print(f"Registered signals ({len(signals)}):")
            for sig in signals:
                print(f"  - {sig}")
        else:
            print("No signals registered yet")

    return 0


def cmd_leaderboard(args):
    """Show leaderboard of ranked variants."""
    print("Leaderboard command")
    print(f"Filters: min_sharpe={args.min_sharpe}, min_trades={args.min_trades}")
    # TODO: Implement in Phase 4
    print("Leaderboard not yet implemented (Phase 4)")
    return 0


def cmd_report(args):
    """Generate detailed report for a variant."""
    print(f"Report for variant: {args.variant_id}")
    # TODO: Implement in Phase 4
    print("Report generation not yet implemented (Phase 4)")
    return 0


def cmd_regimes(args):
    """Show per-regime performance for a variant."""
    print(f"Regime analysis for variant: {args.variant_id}")
    # TODO: Implement in Phase 4
    print("Regime analysis not yet implemented (Phase 4)")
    return 0


def cmd_repro(args):
    """Reproduce a variant from manifest."""
    import glob

    variant_id = args.variant_id
    print(f"Reproducing variant: {variant_id}")

    # Find variant in results
    results_dir = PROJECT_ROOT / "results" / "sweeps"
    if not results_dir.exists():
        print("No sweep results found")
        return 1

    # Search all variants.jsonl files
    variant_data = None
    sweep_name = None

    for variants_file in results_dir.glob("*/variants.jsonl"):
        with open(variants_file) as f:
            for line in f:
                v = json.loads(line)
                if v.get('variant_id', '').startswith(variant_id):
                    variant_data = v
                    sweep_name = variants_file.parent.name
                    break
        if variant_data:
            break

    if not variant_data:
        print(f"Variant {variant_id} not found in results")
        return 1

    print(f"Found variant in sweep: {sweep_name}")
    print(f"Status: {variant_data.get('status')}")

    if variant_data.get('status') != 'OK':
        print(f"Cannot reproduce non-OK variant")
        return 1

    # Display saved metrics
    print(f"\nSaved metrics:")
    print(f"  OOS Sharpe: {variant_data.get('oos_sharpe', 'N/A')}")
    print(f"  IS Sharpe: {variant_data.get('is_sharpe', 'N/A')}")
    print(f"  Trades: {variant_data.get('n_trades_oos', 'N/A')}")
    print(f"  PF: {variant_data.get('profit_factor', 'N/A')}")
    print(f"  MaxDD: {variant_data.get('max_drawdown', 'N/A')}")

    # TODO: Re-run strategy and compare
    print("\n[NOTE] Full reproduction requires re-running strategy - skipped for now")

    return 0


def cmd_holdout(args):
    """Run top variants on holdout data."""
    print(f"Holdout test for top {args.top} variants")
    # TODO: Implement in Phase 5
    print("Holdout test not yet implemented (Phase 5)")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="FPR Research Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # fetch
    p_fetch = subparsers.add_parser("fetch", help="Fetch market data")
    p_fetch.add_argument("--start", help="Start date (YYYY-MM-DD)")
    p_fetch.add_argument("--end", help="End date (YYYY-MM-DD)")
    p_fetch.add_argument("--force", action="store_true", help="Force re-fetch")

    # validate
    p_validate = subparsers.add_parser("validate", help="Validate sweep config")
    p_validate.add_argument("config", help="Path to sweep YAML")

    # sweep
    p_sweep = subparsers.add_parser("sweep", help="Run parameter sweep")
    p_sweep.add_argument("config", help="Path to sweep YAML")
    p_sweep.add_argument("--dry-run", action="store_true", help="Validate only")
    p_sweep.add_argument("--jobs", type=int, default=4, help="Parallel jobs")

    # signals
    p_signals = subparsers.add_parser("signals", help="List signals")
    p_signals.add_argument("signal", nargs="?", help="Signal KEY for details")
    p_signals.add_argument("--grid", action="store_true", help="Show param grid")

    # leaderboard
    p_leader = subparsers.add_parser("leaderboard", help="Show leaderboard")
    p_leader.add_argument("--min-sharpe", type=float, default=1.0)
    p_leader.add_argument("--min-trades", type=int, default=30)

    # report
    p_report = subparsers.add_parser("report", help="Generate variant report")
    p_report.add_argument("variant_id", help="Variant ID")

    # regimes
    p_regimes = subparsers.add_parser("regimes", help="Regime analysis")
    p_regimes.add_argument("variant_id", help="Variant ID")

    # repro
    p_repro = subparsers.add_parser("repro", help="Reproduce variant")
    p_repro.add_argument("variant_id", help="Variant ID")

    # holdout
    p_holdout = subparsers.add_parser("holdout", help="Run on holdout data")
    p_holdout.add_argument("--top", type=int, default=10, help="Top N variants")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    # Dispatch to command handler
    handlers = {
        "fetch": cmd_fetch,
        "validate": cmd_validate,
        "sweep": cmd_sweep,
        "signals": cmd_signals,
        "leaderboard": cmd_leaderboard,
        "report": cmd_report,
        "regimes": cmd_regimes,
        "repro": cmd_repro,
        "holdout": cmd_holdout,
    }

    handler = handlers.get(args.command)
    if handler:
        return handler(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
