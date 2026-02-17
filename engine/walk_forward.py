"""
Walk-forward validation engine.

Delta 16, 21, 35:
- Embargo >= warmup (hard constraint)
- Embargo bars counted in strategy.timeframe, not 1m
- Folds created WITHIN each segment, not globally
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime

from data.feature_store import FeatureStore


@dataclass
class WalkForwardConfig:
    """Walk-forward configuration."""
    train_months: int = 3
    test_months: int = 1
    step_months: int = 1
    min_trades_per_fold: int = 5
    embargo_bars: int = 0  # Buffer between train/test in strategy.timeframe bars


@dataclass
class FoldResult:
    """Result from a single walk-forward fold."""
    fold_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime

    # Trade results (OOS only)
    n_trades: int = 0
    trade_returns: np.ndarray = field(default_factory=lambda: np.array([]))

    # Metrics
    sharpe: float = 0.0
    profit_factor: float = 0.0
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    total_return: float = 0.0


@dataclass
class WalkForwardResult:
    """Result from walk-forward validation."""
    # Aggregated OOS metrics
    oos_sharpe: float = 0.0
    oos_profit_factor: float = 0.0
    oos_win_rate: float = 0.0
    oos_max_drawdown: float = 0.0
    oos_total_return: float = 0.0

    # Per-fold results
    fold_results: List[FoldResult] = field(default_factory=list)

    # All OOS trade returns concatenated
    oos_trade_returns: np.ndarray = field(default_factory=lambda: np.array([]))

    # Robust metrics
    median_oos_sharpe: float = 0.0
    consistency_ratio: float = 0.0  # % of profitable folds

    # Diagnostic IS metrics (None if IS computation failed)
    is_sharpe: Optional[float] = 0.0
    sharpe_decay: Optional[float] = 0.0

    # Trade count
    total_oos_trades: int = 0

    # Trade log and metrics dicts (set by walk_forward_run)
    oos_trade_log: List[Dict[str, Any]] = field(default_factory=list)
    oos_metrics: Dict[str, float] = field(default_factory=dict)
    is_metrics: Dict[str, float] = field(default_factory=dict)


def get_month_boundaries(
    df: pd.DataFrame,
    timestamp_col: str = 'timestamp'
) -> List[Tuple[datetime, datetime]]:
    """
    Get month boundaries from DataFrame.

    Returns list of (start, end) tuples for each month.
    """
    timestamps = df[timestamp_col]
    months = timestamps.dt.to_period('M').unique()

    boundaries = []
    for month in sorted(months):
        start = timestamps[timestamps.dt.to_period('M') == month].min()
        end = timestamps[timestamps.dt.to_period('M') == month].max()
        boundaries.append((start.to_pydatetime(), end.to_pydatetime()))

    return boundaries


def walk_forward_split(
    df: pd.DataFrame,
    config: WalkForwardConfig,
    warmup_bars: int = 0,
    timestamp_col: str = 'timestamp'
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Generate train/test splits with embargo.

    Delta 16: effective_embargo = max(config.embargo_bars, warmup_bars)
    Delta 35: Folds created WITHIN each segment

    |--- train (3 mo) ---|--embargo--|--- test (1 mo) ---|
                                      |--- train (3 mo) ---|--embargo--|--- test (1 mo) ---|

    Args:
        df: DataFrame with timestamp column
        config: Walk-forward configuration
        warmup_bars: Required warmup bars (from strategy)
        timestamp_col: Name of timestamp column

    Returns:
        List of (train_df, test_df) tuples
    """
    # Delta 16: embargo >= warmup
    effective_embargo = max(config.embargo_bars, warmup_bars)

    timestamps = df[timestamp_col]
    months = timestamps.dt.to_period('M')
    unique_months = sorted(months.unique())

    if len(unique_months) < config.train_months + config.test_months:
        return []  # Not enough data

    splits = []

    # Generate folds
    step = config.step_months
    train_size = config.train_months
    test_size = config.test_months

    for start_idx in range(0, len(unique_months) - train_size - test_size + 1, step):
        train_months = unique_months[start_idx:start_idx + train_size]
        test_months = unique_months[start_idx + train_size:start_idx + train_size + test_size]

        # Get train data
        train_mask = months.isin(train_months)
        train_df = df[train_mask].copy()

        if len(train_df) == 0:
            continue

        # Apply embargo: remove last embargo_bars from train
        if effective_embargo > 0 and len(train_df) > effective_embargo:
            train_df = train_df.iloc[:-effective_embargo]

        # Get test data
        test_mask = months.isin(test_months)
        test_df = df[test_mask].copy()

        if len(test_df) == 0:
            continue

        # Verify no overlap (critical check)
        train_timestamps = set(train_df[timestamp_col])
        test_timestamps = set(test_df[timestamp_col])
        overlap = train_timestamps & test_timestamps

        if overlap:
            raise ValueError(
                f"Train/test overlap detected: {len(overlap)} timestamps. "
                "This indicates a bug in walk_forward_split."
            )

        splits.append((
            train_df.reset_index(drop=True),
            test_df.reset_index(drop=True)
        ))

    return splits


def walk_forward_split_from_segment(
    segment_df: pd.DataFrame,
    config: WalkForwardConfig,
    warmup_bars: int = 0,
    segment_id: int = 0,
    timestamp_col: str = 'timestamp'
) -> List[Tuple[pd.DataFrame, pd.DataFrame, int]]:
    """
    Create walk-forward folds from a single segment.

    Delta 35: Fold split happens per segment, not globally.

    Args:
        segment_df: Single continuous segment
        config: Walk-forward configuration
        warmup_bars: Required warmup bars
        segment_id: Segment identifier

    Returns:
        List of (train_df, test_df, fold_id) tuples
    """
    splits = walk_forward_split(segment_df, config, warmup_bars, timestamp_col)

    # Add fold_id to each split
    return [(train, test, i) for i, (train, test) in enumerate(splits)]


def compute_fold_metrics(trade_returns: np.ndarray) -> Dict[str, float]:
    """
    Compute metrics for a single fold.

    Args:
        trade_returns: Array of net trade returns

    Returns:
        Dict with sharpe, profit_factor, win_rate, max_dd, total_return
    """
    if len(trade_returns) == 0:
        return {
            'sharpe': 0.0,
            'profit_factor': 0.0,
            'win_rate': 0.0,
            'max_drawdown': 0.0,
            'total_return': 0.0,
            'n_trades': 0
        }

    # Sharpe (annualized for crypto: 365 trading days/year, ~6 trades/day avg)
    # Using 365 instead of 252 (stocks) per Delta crypto specification
    mean_ret = np.mean(trade_returns)
    std_ret = np.std(trade_returns)
    sharpe = mean_ret / std_ret * np.sqrt(365 * 6) if std_ret > 0 else 0.0

    # Profit factor
    gains = trade_returns[trade_returns > 0].sum()
    losses = abs(trade_returns[trade_returns < 0].sum())
    profit_factor = gains / losses if losses > 0 else float('inf') if gains > 0 else 0.0

    # Win rate
    win_rate = np.mean(trade_returns > 0)

    # Max drawdown (from cumulative returns)
    cum_returns = np.cumprod(1 + trade_returns)
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = (running_max - cum_returns) / running_max
    max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0

    # Total return
    total_return = np.prod(1 + trade_returns) - 1

    return {
        'sharpe': sharpe,
        'profit_factor': profit_factor,
        'win_rate': win_rate,
        'max_drawdown': max_drawdown,
        'total_return': total_return,
        'n_trades': len(trade_returns)
    }


def aggregate_fold_results(fold_results: List[FoldResult]) -> WalkForwardResult:
    """
    Aggregate results from all folds.

    Args:
        fold_results: List of FoldResult objects

    Returns:
        Aggregated WalkForwardResult
    """
    if not fold_results:
        return WalkForwardResult()

    # Concatenate all OOS trade returns
    all_returns = np.concatenate([f.trade_returns for f in fold_results if len(f.trade_returns) > 0])

    result = WalkForwardResult(
        fold_results=fold_results,
        oos_trade_returns=all_returns,
        total_oos_trades=len(all_returns)
    )

    if len(all_returns) == 0:
        return result

    # Compute aggregated OOS metrics
    metrics = compute_fold_metrics(all_returns)
    result.oos_sharpe = metrics['sharpe']
    result.oos_profit_factor = metrics['profit_factor']
    result.oos_win_rate = metrics['win_rate']
    result.oos_max_drawdown = metrics['max_drawdown']
    result.oos_total_return = metrics['total_return']

    # Median OOS Sharpe per fold (robust)
    fold_sharpes = [f.sharpe for f in fold_results if f.n_trades > 0]
    if fold_sharpes:
        result.median_oos_sharpe = np.median(fold_sharpes)

    # Consistency ratio (% of profitable folds)
    profitable_folds = sum(1 for f in fold_results if f.total_return > 0)
    result.consistency_ratio = profitable_folds / len(fold_results) if fold_results else 0.0

    return result


class WalkForwardEngine:
    """
    Walk-forward validation engine.

    Usage:
        engine = WalkForwardEngine(config)
        result = engine.run(strategy, segments, caches)
    """

    def __init__(self, config: WalkForwardConfig):
        self.config = config

    def create_folds(
        self,
        segments: List[pd.DataFrame],
        warmup_bars: int
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame, int, int]]:
        """
        Create all folds from all segments.

        Args:
            segments: List of continuous data segments
            warmup_bars: Required warmup bars

        Returns:
            List of (train_df, test_df, segment_id, fold_id) tuples
        """
        all_folds = []

        for seg_id, segment in enumerate(segments):
            segment_folds = walk_forward_split_from_segment(
                segment, self.config, warmup_bars, seg_id
            )

            for train, test, fold_id in segment_folds:
                all_folds.append((train, test, seg_id, fold_id))

        return all_folds


def validate_embargo_warmup(
    config_embargo: int,
    warmup_bars: int
) -> int:
    """
    Validate and return effective embargo.

    Delta 16: embargo >= warmup (hard constraint)

    Args:
        config_embargo: Configured embargo bars
        warmup_bars: Required warmup bars from strategy

    Returns:
        Effective embargo bars
    """
    return max(config_embargo, warmup_bars)


def walk_forward_run(
    strategy,
    df: pd.DataFrame,
    config: WalkForwardConfig,
    cache=None,  # Deprecated: Not used, kept for backwards compatibility
    timestamp_col: str = None
) -> Optional[WalkForwardResult]:
    """
    Run strategy through walk-forward validation.

    For each fold:
    1. Compute warmup_bars from strategy
    2. Run strategy signals on (warmup + test) data
    3. Log trades ONLY on test slice
    4. Compute metrics for test period

    Note: cache parameter is deprecated and ignored.
    FeatureStore instances are created per fold as required by Delta 31.

    Args:
        strategy: Strategy with generate_signals() method
        df: Full DataFrame with OHLCV data
        config: Walk-forward configuration
        cache: Deprecated. Ignored. Kept for backwards compatibility.
        timestamp_col: Timestamp column name (auto-detect if None)

    Returns:
        WalkForwardResult or None if no valid folds
    """
    from engine.backtest import run_backtest

    # Auto-detect timestamp column
    if timestamp_col is None:
        if 'timestamp' in df.columns:
            timestamp_col = 'timestamp'
        elif isinstance(df.index, pd.DatetimeIndex):
            # Create timestamp column from index
            df = df.copy()
            df['timestamp'] = df.index
            timestamp_col = 'timestamp'
        else:
            raise ValueError("Cannot detect timestamp column. Provide timestamp_col argument.")

    # Get warmup bars from strategy
    warmup_bars = strategy.warmup_bars()

    # Create folds with embargo
    splits = walk_forward_split(df, config, warmup_bars, timestamp_col)

    if not splits:
        return None

    fold_results = []
    all_oos_trades = []
    is_trade_returns = []  # In-sample for diagnostic
    is_diagnostic_failed = False  # Track if IS computation had errors

    for fold_id, (train_df, test_df) in enumerate(splits):
        # For IS diagnostic: run on train
        try:
            # Generate signals on train
            # Delta 17/31: Create NEW FeatureStore per fold, bound to fold's data
            train_with_index = train_df.set_index(timestamp_col) if timestamp_col in train_df.columns else train_df
            fold_cache = FeatureStore(train_with_index, strategy.timeframe)
            signals_train = strategy.generate_signals(train_with_index, fold_cache)

            # Run backtest on train (IS)
            train_result = run_backtest(
                df=train_with_index,
                entry_signal=signals_train['entry_signal'],
                exit_signal=signals_train['exit_signal'],
                side=strategy.side,
                fee_rate=strategy.taker_fee_rate,
                slippage_rate=strategy.slippage_rate,
            )

            if train_result.trades:
                is_trade_returns.extend([t.net_return for t in train_result.trades])
        except Exception:
            # IS diagnostic failed - flag it so we set is_sharpe to None (not 0)
            is_diagnostic_failed = True

        # OOS: run on test
        try:
            # Need warmup data before test for indicators
            # Find where test starts in original df
            test_start = test_df[timestamp_col].min()
            test_mask = df[timestamp_col] >= test_start

            # Get warmup + test data
            full_test_idx = df[test_mask].index
            if len(full_test_idx) == 0:
                continue

            # Include warmup bars before test
            start_idx = max(0, full_test_idx[0] - warmup_bars)
            warmup_test_df = df.iloc[start_idx:full_test_idx[-1] + 1].copy()

            if len(warmup_test_df) < warmup_bars:
                continue

            # Set index for signal generation
            if timestamp_col in warmup_test_df.columns:
                warmup_test_df = warmup_test_df.set_index(timestamp_col)

            # Delta 17/31: Create NEW FeatureStore per fold, bound to fold's data
            test_cache = FeatureStore(warmup_test_df, strategy.timeframe)

            # Generate signals on warmup + test
            signals = strategy.generate_signals(warmup_test_df, test_cache)

            # Only keep signals for test period (remove warmup)
            test_start_ts = test_df[timestamp_col].min()
            if test_start_ts in signals.index:
                signals_test = signals.loc[test_start_ts:]
                df_test = warmup_test_df.loc[test_start_ts:]
            else:
                # Fallback: use last N rows where N = len(test_df)
                signals_test = signals.iloc[-len(test_df):]
                df_test = warmup_test_df.iloc[-len(test_df):]

            # Run backtest on test only
            test_result = run_backtest(
                df=df_test,
                entry_signal=signals_test['entry_signal'],
                exit_signal=signals_test['exit_signal'],
                side=strategy.side,
                fee_rate=strategy.taker_fee_rate,
                slippage_rate=strategy.slippage_rate,
            )

            # Extract trade returns
            trades = test_result.trades
            trade_returns = np.array([t.net_return for t in trades]) if trades else np.array([])

            # Skip folds with insufficient trades
            if len(trade_returns) < config.min_trades_per_fold:
                continue

            # Compute fold metrics
            metrics = compute_fold_metrics(trade_returns)

            fold_result = FoldResult(
                fold_id=fold_id,
                train_start=train_df[timestamp_col].min() if timestamp_col in train_df.columns else train_df.index[0],
                train_end=train_df[timestamp_col].max() if timestamp_col in train_df.columns else train_df.index[-1],
                test_start=test_df[timestamp_col].min() if timestamp_col in test_df.columns else test_df.index[0],
                test_end=test_df[timestamp_col].max() if timestamp_col in test_df.columns else test_df.index[-1],
                n_trades=len(trade_returns),
                trade_returns=trade_returns,
                sharpe=metrics['sharpe'],
                profit_factor=metrics['profit_factor'],
                win_rate=metrics['win_rate'],
                max_drawdown=metrics['max_drawdown'],
                total_return=metrics['total_return'],
            )

            fold_results.append(fold_result)
            all_oos_trades.extend(trades)

        except Exception as e:
            # Log error but continue with other folds
            import warnings
            warnings.warn(f"Fold {fold_id} failed: {e}")
            continue

    if not fold_results:
        return None

    # Aggregate results
    result = aggregate_fold_results(fold_results)

    # Add IS metrics for diagnostic
    # If IS diagnostic failed, set is_sharpe to None (not 0) to distinguish from zero Sharpe
    if is_diagnostic_failed:
        result.is_sharpe = None
        result.sharpe_decay = None
    elif is_trade_returns:
        is_metrics = compute_fold_metrics(np.array(is_trade_returns))
        result.is_sharpe = is_metrics['sharpe']
        if result.oos_sharpe > 0 and result.is_sharpe > 0:
            result.sharpe_decay = (result.is_sharpe - result.oos_sharpe) / result.is_sharpe
    # else: keep defaults (is_sharpe=0.0, sharpe_decay=0.0) if no IS trades but no error

    # Add trade log
    result.oos_trade_log = all_oos_trades
    result.oos_metrics = {
        'sharpe': result.oos_sharpe,
        'profit_factor': result.oos_profit_factor,
        'win_rate': result.oos_win_rate,
        'max_drawdown': result.oos_max_drawdown,
    }
    result.is_metrics = {
        'sharpe': result.is_sharpe,  # May be None if IS diagnostic failed
    }

    return result
