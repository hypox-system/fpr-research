"""
FeatureStore: Memoized indicator cache.

Delta 17, 31, 37:
- Bound to (df, timeframe). New instance per (fold_id, segment_id, timeframe).
- MTF/HTF features in SEPARATE store bound to HTF-df.
- Cache key: (name, sorted_params). Timeframe implicit via store.
- Pipeline owns stores, signals never create their own.
"""

from typing import Any, Dict, Optional, Tuple
import pandas as pd
import numpy as np


class FeatureStore:
    """
    Memoized indicator cache bound to a specific DataFrame and timeframe.

    IMPORTANT (Delta 17, 31):
    - Create new FeatureStore(df, timeframe) per (fold_id, segment_id, timeframe)
    - MTF: HTF features in separate FeatureStore bound to resampled HTF-df
    - Cache key: (name, sorted_params) - timeframe implicit via store
    - Fold-aware: no global percentiles/statistics computed
    - Only deterministic functions (EMA, MACD, ATR etc) are cached

    Usage:
        cache = FeatureStore(df, '5min')
        ema50 = cache.get('ema', period=50)  # No df arg
    """

    def __init__(self, df: pd.DataFrame, timeframe: str):
        """
        Initialize FeatureStore bound to specific data.

        Args:
            df: DataFrame with OHLCV data
            timeframe: Timeframe string (e.g., '5min', '1h')
        """
        # Store reference to df (no copy - FeatureStore is read-only)
        # This saves significant memory when creating many stores
        self._df = df
        self._timeframe = timeframe
        self._cache: Dict[Tuple, pd.Series] = {}

        # Validate df has required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")

    @property
    def df(self) -> pd.DataFrame:
        """Return the bound DataFrame."""
        return self._df

    @property
    def timeframe(self) -> str:
        """Return the bound timeframe."""
        return self._timeframe

    def _make_key(self, name: str, **params) -> Tuple:
        """Create cache key from name and sorted params."""
        return (name, tuple(sorted(params.items())))

    def get(self, name: str, **params) -> pd.Series:
        """
        Get or compute a feature.

        Args:
            name: Feature name (e.g., 'ema', 'atr', 'rsi')
            **params: Feature parameters

        Returns:
            Computed feature as Series with same index as df
        """
        key = self._make_key(name, **params)

        if key not in self._cache:
            self._cache[key] = self._compute(name, **params)

        return self._cache[key]

    def _compute(self, name: str, **params) -> pd.Series:
        """Compute a feature. Override in subclass for custom features."""

        if name == 'ema':
            return self._compute_ema(**params)
        elif name == 'sma':
            return self._compute_sma(**params)
        elif name == 'atr':
            return self._compute_atr(**params)
        elif name == 'rsi':
            return self._compute_rsi(**params)
        elif name == 'macd':
            return self._compute_macd(**params)
        elif name == 'macd_hist':
            return self._compute_macd_histogram(**params)
        elif name == 'stoch_rsi':
            return self._compute_stoch_rsi(**params)
        elif name == 'stoch_rsi_k':
            return self._compute_stoch_rsi_k(**params)
        elif name == 'stoch_rsi_d':
            return self._compute_stoch_rsi_d(**params)
        elif name == 'volume_sma':
            return self._compute_volume_sma(**params)
        elif name == 'high_low_range':
            return self._df['high'] - self._df['low']
        elif name == 'close':
            return self._df['close'].copy()
        elif name == 'open':
            return self._df['open'].copy()
        elif name == 'high':
            return self._df['high'].copy()
        elif name == 'low':
            return self._df['low'].copy()
        elif name == 'volume':
            return self._df['volume'].copy()
        else:
            raise ValueError(f"Unknown feature: {name}")

    def _compute_ema(self, period: int, source: str = 'close') -> pd.Series:
        """Exponential Moving Average."""
        series = self._df[source]
        return series.ewm(span=period, adjust=False).mean()

    def _compute_sma(self, period: int, source: str = 'close') -> pd.Series:
        """Simple Moving Average."""
        series = self._df[source]
        return series.rolling(window=period, min_periods=1).mean()

    def _compute_atr(self, period: int = 14) -> pd.Series:
        """Average True Range."""
        high = self._df['high']
        low = self._df['low']
        close = self._df['close']

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()

        return atr

    def _compute_rsi(self, period: int = 14, source: str = 'close') -> pd.Series:
        """Relative Strength Index."""
        series = self._df[source]
        delta = series.diff()

        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        # Wilder's smoothing
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(50)

        return rsi

    def _compute_macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """MACD indicator (returns DataFrame with line, signal, histogram)."""
        close = self._df['close']

        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return pd.DataFrame({
            'macd_line': macd_line,
            'signal_line': signal_line,
            'histogram': histogram
        })

    def _compute_macd_histogram(self, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """MACD histogram only."""
        macd_df = self._compute_macd(fast=fast, slow=slow, signal=signal)
        return macd_df['histogram']

    def _compute_stoch_rsi(self, rsi_period: int = 14, stoch_period: int = 14,
                           k_smooth: int = 3, d_smooth: int = 3) -> pd.DataFrame:
        """Stochastic RSI."""
        rsi = self._compute_rsi(period=rsi_period)

        lowest_rsi = rsi.rolling(window=stoch_period, min_periods=1).min()
        highest_rsi = rsi.rolling(window=stoch_period, min_periods=1).max()

        rsi_range = highest_rsi - lowest_rsi
        stoch_rsi = np.where(
            rsi_range > 0,
            (rsi - lowest_rsi) / rsi_range * 100,
            50
        )
        stoch_rsi = pd.Series(stoch_rsi, index=rsi.index)

        k_line = stoch_rsi.rolling(window=k_smooth, min_periods=1).mean()
        d_line = k_line.rolling(window=d_smooth, min_periods=1).mean()

        return pd.DataFrame({'k_line': k_line, 'd_line': d_line})

    def _compute_stoch_rsi_k(self, rsi_period: int = 14, stoch_period: int = 14,
                              k_smooth: int = 3) -> pd.Series:
        """Stochastic RSI K line only."""
        stoch = self._compute_stoch_rsi(rsi_period, stoch_period, k_smooth, 3)
        return stoch['k_line']

    def _compute_stoch_rsi_d(self, rsi_period: int = 14, stoch_period: int = 14,
                              k_smooth: int = 3, d_smooth: int = 3) -> pd.Series:
        """Stochastic RSI D line only."""
        stoch = self._compute_stoch_rsi(rsi_period, stoch_period, k_smooth, d_smooth)
        return stoch['d_line']

    def _compute_volume_sma(self, period: int = 20) -> pd.Series:
        """Volume SMA."""
        return self._df['volume'].rolling(window=period, min_periods=1).mean()

    def precompute_for_sweep(self, param_grid: Dict[str, list]) -> None:
        """
        Precompute features for a sweep config.

        Args:
            param_grid: Dict mapping feature names to lists of param dicts
        """
        for feature_name, param_list in param_grid.items():
            for params in param_list:
                self.get(feature_name, **params)

    def clear(self) -> None:
        """Clear all cached features."""
        self._cache.clear()

    def cache_size(self) -> int:
        """Return number of cached features."""
        return len(self._cache)

    def cached_keys(self) -> list:
        """Return list of cached feature keys."""
        return list(self._cache.keys())


def create_htf_store(base_df: pd.DataFrame, base_tf: str, htf: str) -> FeatureStore:
    """
    Create a FeatureStore for higher timeframe data.

    Resamples base_df to htf and applies shift(1) for HTF availability rule (Delta 5).

    Args:
        base_df: Base timeframe DataFrame
        base_tf: Base timeframe string
        htf: Higher timeframe string

    Returns:
        FeatureStore bound to resampled HTF data
    """
    from utils.timeframes import TF_TO_PANDAS

    # Resample to HTF
    htf_offset = TF_TO_PANDAS.get(htf, htf)

    df_indexed = base_df.set_index('timestamp')

    htf_df = df_indexed.resample(htf_offset, label='left', closed='left').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    htf_df = htf_df.reset_index()

    return FeatureStore(htf_df, htf)
