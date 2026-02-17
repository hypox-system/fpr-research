"""
Tests for data validation and integrity.

Covers:
- OHLCV validation
- Gap detection and segment splitting
- Data fingerprinting
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.validate import validate_ohlcv, ValidationReport, get_data_fingerprint
from data.feature_store import FeatureStore


def create_test_df(n_bars: int = 100, start_date: str = "2025-08-01") -> pd.DataFrame:
    """Create test OHLCV DataFrame."""
    start = pd.Timestamp(start_date, tz="UTC")
    timestamps = pd.date_range(start=start, periods=n_bars, freq="1min")

    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n_bars) * 0.1)
    high = close + np.abs(np.random.randn(n_bars) * 0.05)
    low = close - np.abs(np.random.randn(n_bars) * 0.05)
    open_price = close + np.random.randn(n_bars) * 0.02

    # Ensure high/low relationship
    high = np.maximum(high, np.maximum(open_price, close))
    low = np.minimum(low, np.minimum(open_price, close))

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.abs(np.random.randn(n_bars) * 1000) + 100
    })


class TestValidateOHLCV:
    """Tests for OHLCV validation."""

    def test_valid_data_passes(self):
        """Valid OHLCV data should pass validation."""
        df = create_test_df(6000)  # More than min_segment_bars default
        report = validate_ohlcv(df)

        assert report.passed, f"Validation failed: {report.errors}"
        assert report.total_rows == 6000
        assert len(report.segments) == 1

    def test_empty_df_fails(self):
        """Empty DataFrame should fail."""
        df = pd.DataFrame()
        report = validate_ohlcv(df)

        assert not report.passed
        assert "empty" in report.errors[0].lower()

    def test_missing_columns_fails(self):
        """Missing required columns should fail."""
        df = create_test_df(100)
        df = df.drop(columns=["close"])
        report = validate_ohlcv(df)

        assert not report.passed
        assert any("missing" in e.lower() for e in report.errors)

    def test_invalid_high_low_fails(self):
        """Invalid high/low relationship should fail."""
        df = create_test_df(100)
        # Make high < close for some rows
        df.loc[50, "high"] = df.loc[50, "close"] - 1
        report = validate_ohlcv(df)

        assert not report.passed
        assert any("high" in e.lower() for e in report.errors)

    def test_negative_volume_fails(self):
        """Negative volume should fail."""
        df = create_test_df(100)
        df.loc[50, "volume"] = -100
        report = validate_ohlcv(df)

        assert not report.passed
        assert any("negative" in e.lower() for e in report.errors)

    def test_nan_values_fail(self):
        """NaN values should fail."""
        df = create_test_df(100)
        df.loc[50, "close"] = np.nan
        report = validate_ohlcv(df)

        assert not report.passed
        assert any("nan" in e.lower() for e in report.errors)

    def test_duplicate_timestamps_fail(self):
        """Duplicate timestamps should fail."""
        df = create_test_df(100)
        df.loc[50, "timestamp"] = df.loc[49, "timestamp"]
        report = validate_ohlcv(df)

        assert not report.passed
        assert any("duplicate" in e.lower() for e in report.errors)

    def test_gap_detection(self):
        """Gaps should be detected and counted."""
        df = create_test_df(12000)  # Large enough for two segments after split

        # Create a 10-minute gap in the middle
        df.loc[6000:, "timestamp"] = df.loc[6000:, "timestamp"] + timedelta(minutes=10)
        report = validate_ohlcv(df, gap_threshold_minutes=5, min_segment_bars=5000)

        assert report.gap_count >= 1
        assert report.missing_minutes >= 10
        assert len(report.segments) == 2

    def test_segment_splitting(self):
        """Large gaps should split into segments."""
        df = create_test_df(200)

        # Create a 30-minute gap in the middle
        df.loc[100:, "timestamp"] = df.loc[100:, "timestamp"] + timedelta(minutes=30)
        report = validate_ohlcv(df, gap_threshold_minutes=5, min_segment_bars=50)

        assert report.passed
        assert len(report.segments) == 2
        assert len(report.segments[0]) == 100
        assert len(report.segments[1]) == 100

    def test_small_segment_dropped(self):
        """Segments smaller than min_segment_bars should be dropped."""
        df = create_test_df(200)

        # Create gap leaving small segment
        df.loc[180:, "timestamp"] = df.loc[180:, "timestamp"] + timedelta(minutes=30)
        report = validate_ohlcv(df, gap_threshold_minutes=5, min_segment_bars=50)

        assert report.passed
        assert len(report.segments) == 1  # Second segment (20 bars) dropped
        assert any("dropped" in w.lower() for w in report.warnings)


class TestDataFingerprint:
    """Tests for data fingerprinting."""

    def test_same_data_same_fingerprint(self):
        """Same data should produce same fingerprint."""
        df = create_test_df(100)
        fp1 = get_data_fingerprint(df)
        fp2 = get_data_fingerprint(df)

        assert fp1 == fp2

    def test_different_data_different_fingerprint(self):
        """Different data should produce different fingerprint."""
        df1 = create_test_df(100)
        df2 = create_test_df(100)
        df2.loc[50, "close"] = df2.loc[50, "close"] + 1

        fp1 = get_data_fingerprint(df1)
        fp2 = get_data_fingerprint(df2)

        assert fp1 != fp2


class TestFeatureStore:
    """Tests for FeatureStore."""

    def test_cache_hit(self):
        """Same feature request should return cached result."""
        df = create_test_df(200)
        store = FeatureStore(df, "5min")

        ema1 = store.get("ema", period=50)
        ema2 = store.get("ema", period=50)

        assert ema1 is ema2  # Same object (cached)
        assert store.cache_size() == 1

    def test_different_params_different_cache(self):
        """Different params should produce different cache entries."""
        df = create_test_df(200)
        store = FeatureStore(df, "5min")

        ema50 = store.get("ema", period=50)
        ema100 = store.get("ema", period=100)

        assert store.cache_size() == 2
        assert not np.allclose(ema50.values, ema100.values)

    def test_stores_independent(self):
        """Different stores should have independent caches."""
        df1 = create_test_df(200)
        df2 = create_test_df(200)
        df2["close"] = df2["close"] * 2  # Different data

        store1 = FeatureStore(df1, "5min")
        store2 = FeatureStore(df2, "5min")

        ema1 = store1.get("ema", period=50)
        ema2 = store2.get("ema", period=50)

        # Different underlying data, different results
        assert not np.allclose(ema1.values, ema2.values)

    def test_ema_output_length(self):
        """EMA output should have same length as input."""
        df = create_test_df(200)
        store = FeatureStore(df, "5min")

        ema = store.get("ema", period=50)
        assert len(ema) == len(df)

    def test_atr_computation(self):
        """ATR should be computed correctly."""
        df = create_test_df(200)
        store = FeatureStore(df, "5min")

        atr = store.get("atr", period=14)
        assert len(atr) == len(df)
        assert (atr >= 0).all()  # ATR is always positive

    def test_rsi_range(self):
        """RSI should be in [0, 100] range."""
        df = create_test_df(200)
        store = FeatureStore(df, "5min")

        rsi = store.get("rsi", period=14)
        assert len(rsi) == len(df)
        assert (rsi >= 0).all()
        assert (rsi <= 100).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
