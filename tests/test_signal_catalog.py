"""
Tests for research/signal_catalog.py

Verifies:
- discover_signals() finds ema_cross and ema_stop_long
- get_valid_signal_names() returns list of strings
- format_for_llm() returns text (not JSON, not code)
"""

import pytest

from research.signal_catalog import (
    discover_signals,
    get_valid_signal_names,
    format_for_llm,
    reset_cache,
)


class TestDiscoverSignals:
    """Tests for discover_signals()."""

    def setup_method(self):
        """Reset catalog cache before each test."""
        reset_cache()

    def test_finds_ema_cross(self):
        """Should find ema_cross signal."""
        catalog = discover_signals()
        assert 'ema_cross' in catalog

    def test_finds_ema_stop_long(self):
        """Should find ema_stop_long signal."""
        catalog = discover_signals()
        assert 'ema_stop_long' in catalog

    def test_ema_cross_metadata(self):
        """ema_cross should have correct metadata."""
        catalog = discover_signals()
        ema_cross = catalog['ema_cross']

        assert ema_cross['type'] == 'entry'
        assert 'long' in ema_cross['supported_sides']
        assert ema_cross['side_split'] is False
        assert 'params' in ema_cross
        assert 'fast' in ema_cross['params']
        assert 'slow' in ema_cross['params']

    def test_ema_stop_long_metadata(self):
        """ema_stop_long should have correct metadata."""
        catalog = discover_signals()
        ema_stop = catalog['ema_stop_long']

        assert ema_stop['type'] == 'exit'
        assert 'long' in ema_stop['supported_sides']
        assert ema_stop['side_split'] is True

    def test_catalog_is_cached(self):
        """Second call should return cached result."""
        catalog1 = discover_signals()
        catalog2 = discover_signals()
        # Same object reference (cached)
        assert catalog1 is catalog2

    def test_all_signals_have_required_fields(self):
        """All signals should have type, params, supported_sides."""
        catalog = discover_signals()

        for key, info in catalog.items():
            assert 'type' in info, f"{key} missing 'type'"
            assert 'params' in info, f"{key} missing 'params'"
            assert 'supported_sides' in info, f"{key} missing 'supported_sides'"
            assert 'side_split' in info, f"{key} missing 'side_split'"


class TestGetValidSignalNames:
    """Tests for get_valid_signal_names()."""

    def test_returns_list(self):
        """Should return a list."""
        names = get_valid_signal_names()
        assert isinstance(names, list)

    def test_returns_strings(self):
        """All names should be strings."""
        names = get_valid_signal_names()
        assert all(isinstance(n, str) for n in names)

    def test_contains_expected_signals(self):
        """Should contain expected signals."""
        names = get_valid_signal_names()
        assert 'ema_cross' in names
        assert 'ema_stop_long' in names

    def test_is_sorted(self):
        """Names should be sorted."""
        names = get_valid_signal_names()
        assert names == sorted(names)


class TestFormatForLLM:
    """Tests for format_for_llm()."""

    def setup_method(self):
        """Reset catalog cache before each test."""
        reset_cache()

    def test_returns_text(self):
        """Should return a string."""
        text = format_for_llm()
        assert isinstance(text, str)

    def test_not_json(self):
        """Output should not be JSON (should be human-readable text)."""
        text = format_for_llm()
        # Should not start with { or [
        assert not text.strip().startswith('{')
        assert not text.strip().startswith('[')

    def test_not_code(self):
        """Output should not be code."""
        text = format_for_llm()
        # Should not contain Python code patterns
        assert 'def ' not in text
        assert 'import ' not in text
        assert 'class ' not in text

    def test_contains_signal_names(self):
        """Output should mention signal names."""
        text = format_for_llm()
        assert 'ema_cross' in text
        assert 'ema_stop_long' in text

    def test_contains_type_info(self):
        """Output should describe signal types."""
        text = format_for_llm()
        # Should have section headers for types
        assert 'Entry' in text or 'entry' in text
        assert 'Exit' in text or 'exit' in text

    def test_contains_parameters(self):
        """Output should include parameter info."""
        text = format_for_llm()
        # Should mention parameter names
        assert 'fast' in text
        assert 'slow' in text

    def test_handles_empty_catalog(self):
        """Should handle empty catalog gracefully."""
        text = format_for_llm({})
        assert 'No signals' in text or 'no signals' in text.lower()

    def test_custom_catalog(self):
        """Should accept custom catalog."""
        custom = {
            'test_signal': {
                'type': 'entry',
                'params': {'param1': [1, 2, 3]},
                'supported_sides': ['long'],
                'side_split': False,
                'description': 'Test signal',
            }
        }
        text = format_for_llm(custom)
        assert 'test_signal' in text
