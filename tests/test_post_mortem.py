"""
Tests for analysis/post_mortem.py

Verifies:
- Parsing sweep_001 variants.jsonl produces valid post_mortem.json
- fee_decomposition.data_source is correct
- fee_decomposition.estimated_fee_drag_per_trade formula is correct
- primary_failure_mode is a valid enum
- next_experiment_constraints is non-empty
- Empty variants.jsonl raises exception
- Idempotent: two runs produce identical JSON (excluding generated_ts)
"""

import json
import os
import tempfile
import pytest
from pathlib import Path

from analysis.post_mortem import generate, FAILURE_MODES


class TestPostMortemParsing:
    """Tests for basic post-mortem parsing."""

    def test_parse_sweep_001(self):
        """Parse existing sweep_001 and verify output."""
        sweep_dir = 'results/sweeps/sweep_001_ema_pvsra'
        if not os.path.exists(sweep_dir):
            pytest.skip("sweep_001 not available")

        pm = generate(sweep_dir)

        # Required top-level fields
        assert 'sweep_id' in pm
        assert 'experiment_id' in pm
        assert 'created_ts' in pm
        assert 'generated_ts' in pm
        assert 'summary' in pm
        assert 'fee_decomposition' in pm
        assert 'primary_failure_mode' in pm
        assert 'failure_evidence' in pm
        assert 'next_experiment_constraints' in pm

        # Summary fields
        summary = pm['summary']
        assert 'variant_count' in summary
        assert 'survivor_count' in summary
        assert 'best_variant' in summary
        assert summary['variant_count'] > 0

    def test_fee_decomposition_data_source(self):
        """Verify data_source is one of expected values."""
        sweep_dir = 'results/sweeps/sweep_001_ema_pvsra'
        if not os.path.exists(sweep_dir):
            pytest.skip("sweep_001 not available")

        pm = generate(sweep_dir)
        data_source = pm['fee_decomposition']['data_source']
        assert data_source in ('npz', 'trade_summary', 'config_only')

    def test_fee_drag_formula(self):
        """Verify estimated_fee_drag_per_trade = 2 * (fee_rate + slippage_rate)."""
        sweep_dir = 'results/sweeps/sweep_001_ema_pvsra'
        if not os.path.exists(sweep_dir):
            pytest.skip("sweep_001 not available")

        pm = generate(sweep_dir)
        fee_drag = pm['fee_decomposition']['estimated_fee_drag_per_trade']

        # From sweep_001: fee_rate=0.0006, slippage_rate=0.0001
        expected = 2 * (0.0006 + 0.0001)  # 0.0014
        assert abs(fee_drag - expected) < 1e-10

    def test_primary_failure_mode_valid_enum(self):
        """Verify primary_failure_mode is a valid enum."""
        sweep_dir = 'results/sweeps/sweep_001_ema_pvsra'
        if not os.path.exists(sweep_dir):
            pytest.skip("sweep_001 not available")

        pm = generate(sweep_dir)
        assert pm['primary_failure_mode'] in FAILURE_MODES

    def test_next_experiment_constraints_non_empty(self):
        """Verify next_experiment_constraints is non-empty list."""
        sweep_dir = 'results/sweeps/sweep_001_ema_pvsra'
        if not os.path.exists(sweep_dir):
            pytest.skip("sweep_001 not available")

        pm = generate(sweep_dir)
        constraints = pm['next_experiment_constraints']
        assert isinstance(constraints, list)
        assert len(constraints) > 0


class TestPostMortemEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_variants_raises_exception(self):
        """Empty variants.jsonl should raise exception."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create empty variants.jsonl
            variants_path = Path(tmpdir) / 'variants.jsonl'
            variants_path.write_text('')

            # Create minimal manifest.json
            manifest_path = Path(tmpdir) / 'manifest.json'
            manifest_path.write_text(json.dumps({
                'sweep_name': 'test_empty',
                'symbol': 'BTCUSDT',
                'data_range': {'start': '2025-01-01', 'end': '2025-06-01'},
                'data_fingerprint': 'abc123',
            }))

            with pytest.raises(ValueError, match="No valid variants"):
                generate(tmpdir)

    def test_missing_variants_raises_exception(self):
        """Missing variants.jsonl should raise FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError, match="variants.jsonl"):
                generate(tmpdir)

    def test_malformed_variant_skipped(self):
        """Malformed variant lines should be skipped, not crash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create variants.jsonl with one good and one bad line
            variants_path = Path(tmpdir) / 'variants.jsonl'
            good_variant = {
                'variant_id': 'abc123',
                'status': 'OK',
                'oos_sharpe': -5.0,
                'n_trades_oos': 100,
                'profit_factor': 0.5,
                'win_rate': 0.1,
                'spec': {
                    'entry': {'key': 'ema_cross', 'params': {'fast': 5, 'slow': 21}},
                    'fee_rate': 0.0006,
                    'slippage_rate': 0.0001,
                }
            }
            variants_path.write_text(
                json.dumps(good_variant) + '\n' +
                'this is not valid json\n'
            )

            manifest_path = Path(tmpdir) / 'manifest.json'
            manifest_path.write_text(json.dumps({
                'sweep_name': 'test_malformed',
                'symbol': 'BTCUSDT',
                'data_range': {'start': '2025-01-01', 'end': '2025-06-01'},
                'data_fingerprint': 'abc123',
            }))

            # Should not crash
            pm = generate(tmpdir)
            assert pm['summary']['variant_count'] == 1


class TestPostMortemIdempotence:
    """Tests for idempotence."""

    def test_idempotent_json_output(self):
        """Two runs should produce identical JSON (excluding timestamps)."""
        sweep_dir = 'results/sweeps/sweep_001_ema_pvsra'
        if not os.path.exists(sweep_dir):
            pytest.skip("sweep_001 not available")

        pm1 = generate(sweep_dir)
        pm2 = generate(sweep_dir)

        # Remove timestamps that change
        del pm1['generated_ts']
        del pm2['generated_ts']
        del pm1['created_ts']
        del pm2['created_ts']

        assert pm1 == pm2


class TestPostMortemWithTradeSummary:
    """Tests for post-mortem with trade_summary data."""

    def test_with_trade_summary_data(self):
        """Post-mortem with trade_summary data should have non-null gross/net."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create variant with trade_summary
            variant = {
                'variant_id': 'abc123',
                'status': 'OK',
                'oos_sharpe': -5.0,
                'n_trades_oos': 100,
                'profit_factor': 0.5,
                'win_rate': 0.1,
                'bh_fdr_significant': False,
                'trade_summary': {
                    'median_net_return_per_trade': -0.003,
                    'median_gross_return_per_trade': -0.002,
                    'mean_net_return_per_trade': -0.0025,
                    'std_net_return_per_trade': 0.01,
                    'n_positive_trades': 10,
                    'n_negative_trades': 90,
                },
                'spec': {
                    'entry': {'key': 'ema_cross', 'params': {'fast': 5, 'slow': 21}},
                    'fee_rate': 0.0006,
                    'slippage_rate': 0.0001,
                }
            }

            variants_path = Path(tmpdir) / 'variants.jsonl'
            variants_path.write_text(json.dumps(variant) + '\n')

            manifest_path = Path(tmpdir) / 'manifest.json'
            manifest_path.write_text(json.dumps({
                'sweep_name': 'test_with_summary',
                'symbol': 'BTCUSDT',
                'data_range': {'start': '2025-01-01', 'end': '2025-06-01'},
                'data_fingerprint': 'abc123',
            }))

            pm = generate(tmpdir)
            fee_decomp = pm['fee_decomposition']

            assert fee_decomp['data_source'] == 'trade_summary'
            assert fee_decomp['median_net_return_per_trade'] is not None
            assert fee_decomp['median_gross_return_per_trade'] is not None

    def test_without_trade_summary_data(self):
        """Post-mortem without trade_summary should have config_only source."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create variant without trade_summary
            variant = {
                'variant_id': 'abc123',
                'status': 'OK',
                'oos_sharpe': -5.0,
                'n_trades_oos': 100,
                'profit_factor': 0.5,
                'win_rate': 0.1,
                'bh_fdr_significant': False,
                'spec': {
                    'entry': {'key': 'ema_cross', 'params': {'fast': 5, 'slow': 21}},
                    'fee_rate': 0.0006,
                    'slippage_rate': 0.0001,
                }
            }

            variants_path = Path(tmpdir) / 'variants.jsonl'
            variants_path.write_text(json.dumps(variant) + '\n')

            manifest_path = Path(tmpdir) / 'manifest.json'
            manifest_path.write_text(json.dumps({
                'sweep_name': 'test_without_summary',
                'symbol': 'BTCUSDT',
                'data_range': {'start': '2025-01-01', 'end': '2025-06-01'},
                'data_fingerprint': 'abc123',
            }))

            pm = generate(tmpdir)
            fee_decomp = pm['fee_decomposition']

            assert fee_decomp['data_source'] == 'config_only'
            assert fee_decomp['median_net_return_per_trade'] is None
            assert fee_decomp['median_gross_return_per_trade'] is None
