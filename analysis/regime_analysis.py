"""
Regime analysis (analysis/regime_analysis.py).

Classifies market regimes based on daily SMA200 and ATR.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class RegimeConfig:
    """Configuration for regime classification."""
    sma_period: int = 200
    slope_window: int = 20
    slope_threshold: float = 0.01
    atr_period: int = 14
    atr_lookback: int = 252
    high_vol_pctl: float = 0.75
    low_vol_pctl: float = 0.25


def classify_regime(
    daily_df: pd.DataFrame,
    config: Optional[RegimeConfig] = None
) -> pd.DataFrame:
    """
    Classify each day into trend and volatility regimes.
    
    Regimes:
    - Trend: bull, bear, range
    - Volatility: high_vol, low_vol, normal_vol
    
    Args:
        daily_df: Daily OHLCV DataFrame with 'close', 'high', 'low'
        config: Regime configuration
        
    Returns:
        DataFrame with regime columns added
    """
    if config is None:
        config = RegimeConfig()
    
    df = daily_df.copy()
    
    # Compute SMA200
    df['sma200'] = df['close'].rolling(window=config.sma_period, min_periods=1).mean()
    
    # Compute slope of SMA200
    df['sma200_slope'] = (
        (df['sma200'] - df['sma200'].shift(config.slope_window)) 
        / df['sma200'].shift(config.slope_window)
    )
    
    # Compute ATR
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(window=config.atr_period, min_periods=1).mean()
    
    # ATR percentile (1-year rolling)
    df['atr_pctl'] = df['atr'].rolling(
        window=config.atr_lookback, min_periods=20
    ).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    
    # Trend regime
    df['regime_trend'] = 'range'
    df.loc[
        (df['close'] > df['sma200']) & (df['sma200_slope'] > config.slope_threshold),
        'regime_trend'
    ] = 'bull'
    df.loc[
        (df['close'] < df['sma200']) & (df['sma200_slope'] < -config.slope_threshold),
        'regime_trend'
    ] = 'bear'
    
    # Volatility regime
    df['regime_vol'] = 'normal_vol'
    df.loc[df['atr_pctl'] > config.high_vol_pctl, 'regime_vol'] = 'high_vol'
    df.loc[df['atr_pctl'] < config.low_vol_pctl, 'regime_vol'] = 'low_vol'
    
    return df


def map_regime_to_intraday(
    intraday_df: pd.DataFrame,
    daily_regimes: pd.DataFrame,
    timestamp_col: str = 'timestamp'
) -> pd.DataFrame:
    """
    Map daily regime classifications to intraday data.
    
    Args:
        intraday_df: Intraday DataFrame
        daily_regimes: Daily DataFrame with regime columns
        timestamp_col: Timestamp column name
        
    Returns:
        Intraday DataFrame with regime columns added
    """
    df = intraday_df.copy()
    
    # Extract date from intraday timestamps
    if timestamp_col in df.columns:
        df['_date'] = pd.to_datetime(df[timestamp_col]).dt.date
    else:
        df['_date'] = df.index.date
    
    # Prepare daily regimes
    daily = daily_regimes[['regime_trend', 'regime_vol']].copy()
    daily['_date'] = daily.index.date if isinstance(daily.index, pd.DatetimeIndex) else daily.index
    
    # Merge
    df = df.merge(daily, on='_date', how='left')
    df = df.drop(columns=['_date'])
    
    # Fill missing with 'unknown'
    df['regime_trend'] = df['regime_trend'].fillna('unknown')
    df['regime_vol'] = df['regime_vol'].fillna('unknown')
    
    return df


def compute_regime_metrics(
    trades: List[Dict],
    regime_col: str = 'regime_trend'
) -> Dict[str, Dict[str, float]]:
    """
    Compute trading metrics per regime.
    
    Args:
        trades: List of trade dicts with 'return' and regime columns
        regime_col: Which regime column to group by
        
    Returns:
        Dict mapping regime -> metrics dict
    """
    if not trades:
        return {}
    
    df = pd.DataFrame(trades)
    
    if regime_col not in df.columns:
        return {}
    
    results = {}
    
    for regime, group in df.groupby(regime_col):
        returns = group['return'].values
        
        if len(returns) == 0:
            continue
        
        # Compute metrics
        n_trades = len(returns)
        total_return = np.prod(1 + returns) - 1
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe = mean_return / std_return * np.sqrt(252 * 6) if std_return > 0 else 0
        
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        profit_factor = gains / losses if losses > 0 else float('inf') if gains > 0 else 0
        
        win_rate = np.mean(returns > 0)
        
        results[regime] = {
            'n_trades': n_trades,
            'total_return': total_return,
            'sharpe': sharpe,
            'profit_factor': profit_factor,
            'win_rate': win_rate,
            'pct_of_total': n_trades / len(df),
        }
    
    return results


def check_regime_concentration(
    regime_metrics: Dict[str, Dict[str, float]],
    threshold: float = 0.80
) -> Tuple[bool, Optional[str]]:
    """
    Check if returns are concentrated in one regime.
    
    Args:
        regime_metrics: Output from compute_regime_metrics
        threshold: Concentration threshold (default 80%)
        
    Returns:
        Tuple of (is_concentrated, dominant_regime)
    """
    if not regime_metrics:
        return False, None
    
    # Compute total PnL contribution per regime
    total_pnl = sum(m['total_return'] for m in regime_metrics.values())
    
    if total_pnl <= 0:
        return False, None
    
    for regime, metrics in regime_metrics.items():
        pnl_contribution = metrics['total_return'] / total_pnl
        if pnl_contribution > threshold:
            return True, regime
    
    return False, None
