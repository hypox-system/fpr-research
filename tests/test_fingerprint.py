"""
Tests for utils/experiment_fingerprint.py

Verifies:
- Same config with keys in different order -> same experiment_id
- Different param -> different experiment_id
- Different fee -> different experiment_id
- Different symbol -> different experiment_id
"""

import pytest
from utils.experiment_fingerprint import compute_experiment_id


class TestExperimentFingerprint:
    """Tests for compute_experiment_id()."""

    def test_same_config_same_id(self):
        """Same config produces same experiment_id."""
        config = {
            'name': 'test_sweep',
            'entry': {'type': 'ema_cross', 'params': {'fast': 5, 'slow': 21}},
        }
        data_manifest = {
            'symbol': 'BTCUSDT',
            'timeframe': '1h',
            'date_range': {'start': '2025-01-01', 'end': '2025-06-01'},
            'data_fingerprint': 'abc123',
        }
        fill_model = {'fee_rate': 0.0006, 'slippage_rate': 0.0001}

        id1 = compute_experiment_id(config, data_manifest, fill_model)
        id2 = compute_experiment_id(config, data_manifest, fill_model)

        assert id1 == id2
        assert len(id1) == 16

    def test_key_order_invariant(self):
        """YAML key order does NOT affect experiment_id."""
        config1 = {
            'name': 'test_sweep',
            'entry': {'type': 'ema_cross', 'params': {'fast': 5, 'slow': 21}},
            'symbol': 'BTCUSDT',
        }
        config2 = {
            'symbol': 'BTCUSDT',
            'name': 'test_sweep',
            'entry': {'params': {'slow': 21, 'fast': 5}, 'type': 'ema_cross'},
        }
        data_manifest = {
            'symbol': 'BTCUSDT',
            'timeframe': '1h',
            'date_range': {'start': '2025-01-01', 'end': '2025-06-01'},
            'data_fingerprint': 'abc123',
        }
        fill_model = {'fee_rate': 0.0006, 'slippage_rate': 0.0001}

        id1 = compute_experiment_id(config1, data_manifest, fill_model)
        id2 = compute_experiment_id(config2, data_manifest, fill_model)

        assert id1 == id2

    def test_different_param_different_id(self):
        """Changing a param produces different experiment_id."""
        config1 = {
            'name': 'test_sweep',
            'entry': {'type': 'ema_cross', 'params': {'fast': 5, 'slow': 21}},
        }
        config2 = {
            'name': 'test_sweep',
            'entry': {'type': 'ema_cross', 'params': {'fast': 8, 'slow': 21}},  # Changed fast
        }
        data_manifest = {
            'symbol': 'BTCUSDT',
            'timeframe': '1h',
            'date_range': {'start': '2025-01-01', 'end': '2025-06-01'},
            'data_fingerprint': 'abc123',
        }
        fill_model = {'fee_rate': 0.0006, 'slippage_rate': 0.0001}

        id1 = compute_experiment_id(config1, data_manifest, fill_model)
        id2 = compute_experiment_id(config2, data_manifest, fill_model)

        assert id1 != id2

    def test_different_fee_different_id(self):
        """Changing fee produces different experiment_id."""
        config = {
            'name': 'test_sweep',
            'entry': {'type': 'ema_cross', 'params': {'fast': 5, 'slow': 21}},
        }
        data_manifest = {
            'symbol': 'BTCUSDT',
            'timeframe': '1h',
            'date_range': {'start': '2025-01-01', 'end': '2025-06-01'},
            'data_fingerprint': 'abc123',
        }
        fill_model1 = {'fee_rate': 0.0006, 'slippage_rate': 0.0001}
        fill_model2 = {'fee_rate': 0.001, 'slippage_rate': 0.0001}  # Changed fee

        id1 = compute_experiment_id(config, data_manifest, fill_model1)
        id2 = compute_experiment_id(config, data_manifest, fill_model2)

        assert id1 != id2

    def test_different_symbol_different_id(self):
        """Changing symbol produces different experiment_id."""
        config = {
            'name': 'test_sweep',
            'entry': {'type': 'ema_cross', 'params': {'fast': 5, 'slow': 21}},
        }
        data_manifest1 = {
            'symbol': 'BTCUSDT',
            'timeframe': '1h',
            'date_range': {'start': '2025-01-01', 'end': '2025-06-01'},
            'data_fingerprint': 'abc123',
        }
        data_manifest2 = {
            'symbol': 'ETHUSDT',  # Changed symbol
            'timeframe': '1h',
            'date_range': {'start': '2025-01-01', 'end': '2025-06-01'},
            'data_fingerprint': 'abc123',
        }
        fill_model = {'fee_rate': 0.0006, 'slippage_rate': 0.0001}

        id1 = compute_experiment_id(config, data_manifest1, fill_model)
        id2 = compute_experiment_id(config, data_manifest2, fill_model)

        assert id1 != id2

    def test_different_date_range_different_id(self):
        """Changing date range produces different experiment_id."""
        config = {
            'name': 'test_sweep',
            'entry': {'type': 'ema_cross', 'params': {'fast': 5, 'slow': 21}},
        }
        data_manifest1 = {
            'symbol': 'BTCUSDT',
            'timeframe': '1h',
            'date_range': {'start': '2025-01-01', 'end': '2025-06-01'},
            'data_fingerprint': 'abc123',
        }
        data_manifest2 = {
            'symbol': 'BTCUSDT',
            'timeframe': '1h',
            'date_range': {'start': '2025-01-01', 'end': '2025-12-31'},  # Changed end
            'data_fingerprint': 'abc123',
        }
        fill_model = {'fee_rate': 0.0006, 'slippage_rate': 0.0001}

        id1 = compute_experiment_id(config, data_manifest1, fill_model)
        id2 = compute_experiment_id(config, data_manifest2, fill_model)

        assert id1 != id2

    def test_float_precision_determinism(self):
        """Float precision is consistent (10 decimals via canonicalize_spec)."""
        config = {
            'name': 'test_sweep',
            'threshold': 0.123456789012345,  # More than 10 decimals
        }
        data_manifest = {
            'symbol': 'BTCUSDT',
            'timeframe': '1h',
            'date_range': {'start': '2025-01-01', 'end': '2025-06-01'},
            'data_fingerprint': 'abc123',
        }
        fill_model = {'fee_rate': 0.0006, 'slippage_rate': 0.0001}

        id1 = compute_experiment_id(config, data_manifest, fill_model)
        id2 = compute_experiment_id(config, data_manifest, fill_model)

        assert id1 == id2

    def test_nested_key_order_invariant(self):
        """Nested dict key order does NOT affect experiment_id."""
        config1 = {
            'entry': {
                'type': 'ema_cross',
                'params': {'fast': 5, 'slow': 21, 'threshold': 0.5}
            },
            'filters': [
                {'type': 'pvsra', 'params': {'vol_mult': 1.5, 'window': 3}}
            ]
        }
        config2 = {
            'filters': [
                {'params': {'window': 3, 'vol_mult': 1.5}, 'type': 'pvsra'}
            ],
            'entry': {
                'params': {'threshold': 0.5, 'slow': 21, 'fast': 5},
                'type': 'ema_cross'
            }
        }
        data_manifest = {
            'symbol': 'BTCUSDT',
            'timeframe': '1h',
            'date_range': {'start': '2025-01-01', 'end': '2025-06-01'},
            'data_fingerprint': 'abc123',
        }
        fill_model = {'fee_rate': 0.0006, 'slippage_rate': 0.0001}

        id1 = compute_experiment_id(config1, data_manifest, fill_model)
        id2 = compute_experiment_id(config2, data_manifest, fill_model)

        assert id1 == id2
