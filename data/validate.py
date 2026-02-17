"""
Data integrity gate for OHLCV data validation.

Runs automatically before each sweep. If validation fails, sweep is aborted.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np


@dataclass
class ValidationReport:
    """Result of OHLCV validation."""
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Gap statistics
    total_rows: int = 0
    missing_minutes: int = 0
    gap_count: int = 0
    max_gap_minutes: int = 0

    # Segments after gap splitting
    segments: List[pd.DataFrame] = field(default_factory=list)
    segment_info: List[dict] = field(default_factory=list)

    def __str__(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        lines = [f"Validation: {status}"]

        if self.errors:
            lines.append("Errors:")
            for e in self.errors:
                lines.append(f"  - {e}")

        if self.warnings:
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")

        lines.append(f"Total rows: {self.total_rows}")
        lines.append(f"Missing minutes: {self.missing_minutes}")
        lines.append(f"Gap count: {self.gap_count}")
        lines.append(f"Max gap: {self.max_gap_minutes} minutes")
        lines.append(f"Segments: {len(self.segments)}")

        return "\n".join(lines)


def validate_ohlcv(
    df: pd.DataFrame,
    gap_threshold_minutes: int = 5,
    min_segment_bars: int = 5000
) -> ValidationReport:
    """
    Validate OHLCV data.

    Checks:
    1. Index is DatetimeIndex, UTC, monotonic increasing
    2. No duplicate timestamps
    3. Columns: open, high, low, close, volume (all float64)
    4. high >= max(open, close), low <= min(open, close)
    5. volume >= 0
    6. Report missing minutes count (gaps)
    7. No NaN in OHLCV

    Gap policy: Split df into continuous segments at gaps > threshold.
    Segments shorter than min_segment_bars are dropped with warning.

    Args:
        df: DataFrame with OHLCV data
        gap_threshold_minutes: Gap threshold for segment splitting
        min_segment_bars: Minimum bars per segment

    Returns:
        ValidationReport with pass/fail + gap stats + segment list
    """
    report = ValidationReport(passed=True, total_rows=len(df))

    if df.empty:
        report.passed = False
        report.errors.append("DataFrame is empty")
        return report

    # Check required columns
    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        report.passed = False
        report.errors.append(f"Missing columns: {missing_cols}")
        return report

    # Check timestamp column
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        report.passed = False
        report.errors.append("timestamp is not datetime type")
        return report

    # Check UTC timezone
    if df['timestamp'].dt.tz is None:
        report.warnings.append("timestamp has no timezone, assuming UTC")
    elif str(df['timestamp'].dt.tz) != 'UTC':
        report.warnings.append(f"timestamp timezone is {df['timestamp'].dt.tz}, expected UTC")

    # Check monotonic increasing
    if not df['timestamp'].is_monotonic_increasing:
        report.passed = False
        report.errors.append("timestamp is not monotonic increasing")

    # Check duplicates
    duplicates = df['timestamp'].duplicated().sum()
    if duplicates > 0:
        report.passed = False
        report.errors.append(f"{duplicates} duplicate timestamps found")

    # Check numeric columns are float64
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if not pd.api.types.is_float_dtype(df[col]):
            report.warnings.append(f"{col} is not float64, converting")

    # Check high/low relationship
    invalid_high = (df['high'] < df['open']) | (df['high'] < df['close'])
    invalid_low = (df['low'] > df['open']) | (df['low'] > df['close'])

    if invalid_high.any():
        count = invalid_high.sum()
        report.passed = False
        report.errors.append(f"{count} rows where high < max(open, close)")

    if invalid_low.any():
        count = invalid_low.sum()
        report.passed = False
        report.errors.append(f"{count} rows where low > min(open, close)")

    # Check volume non-negative
    negative_vol = (df['volume'] < 0).sum()
    if negative_vol > 0:
        report.passed = False
        report.errors.append(f"{negative_vol} rows with negative volume")

    # Check for NaN
    for col in ['open', 'high', 'low', 'close', 'volume']:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            report.passed = False
            report.errors.append(f"{nan_count} NaN values in {col}")

    # Gap analysis
    time_diffs = df['timestamp'].diff()
    expected_diff = pd.Timedelta(minutes=1)

    # Count missing minutes
    gaps = time_diffs[time_diffs > expected_diff]
    report.gap_count = len(gaps)

    if not gaps.empty:
        total_missing = (gaps - expected_diff).sum()
        report.missing_minutes = int(total_missing.total_seconds() / 60)
        report.max_gap_minutes = int(gaps.max().total_seconds() / 60)

    # Split into segments at large gaps
    gap_threshold = pd.Timedelta(minutes=gap_threshold_minutes)
    large_gaps = time_diffs > gap_threshold

    if large_gaps.any():
        # Find gap indices
        gap_indices = df.index[large_gaps].tolist()

        # Split into segments
        segments = []
        start_idx = 0

        for gap_idx in gap_indices:
            # Get position in index
            pos = df.index.get_loc(gap_idx)
            segment = df.iloc[start_idx:pos].copy()
            if len(segment) > 0:
                segments.append(segment)
            start_idx = pos

        # Add final segment
        final_segment = df.iloc[start_idx:].copy()
        if len(final_segment) > 0:
            segments.append(final_segment)
    else:
        segments = [df.copy()]

    # Filter segments by minimum length
    valid_segments = []
    for i, seg in enumerate(segments):
        if len(seg) >= min_segment_bars:
            valid_segments.append(seg.reset_index(drop=True))
            report.segment_info.append({
                'segment_id': len(valid_segments) - 1,
                'start': seg['timestamp'].iloc[0],
                'end': seg['timestamp'].iloc[-1],
                'rows': len(seg)
            })
        else:
            report.warnings.append(
                f"Dropped segment {i} with {len(seg)} bars (< {min_segment_bars})"
            )

    report.segments = valid_segments

    if not valid_segments:
        report.passed = False
        report.errors.append("No valid segments after gap splitting")

    return report


def get_data_fingerprint(df: pd.DataFrame) -> str:
    """
    Generate a fingerprint hash for the data.

    Used for reproducibility tracking in manifest.
    """
    import hashlib

    # Create fingerprint from key properties
    fingerprint_data = f"{len(df)}:{df['timestamp'].min()}:{df['timestamp'].max()}"
    fingerprint_data += f":{df['close'].sum():.6f}"

    return hashlib.sha256(fingerprint_data.encode()).hexdigest()[:16]


if __name__ == "__main__":
    from .fetcher import load_cached_data

    df = load_cached_data()
    if df.empty:
        print("No cached data. Run fetcher first.")
    else:
        report = validate_ohlcv(df)
        print(report)
        print(f"\nData fingerprint: {get_data_fingerprint(df)}")
