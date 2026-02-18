"""
Tests for prediction audit functionality in combinator/sweep_runner.py

Verifies:
- Proposal + post_mortem → audit JSON written
- No proposal → skip (no error)
- Audit failure → warning, not blocking
"""

import json
import pytest
from pathlib import Path

from research.config import reset as reset_config
from research.knowledge_base import (
    init_db,
    ingest_sweep,
    write_proposal,
    get_proposal,
)
from combinator.sweep_runner import run_prediction_audit


@pytest.fixture
def populated_db_with_proposal(tmp_path):
    """Create a KB with test sweep and proposal."""
    db_path = str(tmp_path / 'test.db')
    db = init_db(db_path)

    # Create test sweep directory
    sweep_dir = tmp_path / 'sweep_audit_test'
    sweep_dir.mkdir()

    manifest = {
        'sweep_name': 'sweep_audit_test',
        'symbol': 'BTCUSDT',
        'timeframes': ['4h'],
        'data_range': {'start': '2024-01-01', 'end': '2024-06-01'},
        'experiment_id': 'exp_audit_test',
    }
    (sweep_dir / 'manifest.json').write_text(json.dumps(manifest))

    post_mortem = {
        'sweep_id': 'sweep_audit_test',
        'experiment_id': 'exp_audit_test',
        'created_ts': '2024-01-01T00:00:00Z',
        'primary_failure_mode': 'LOW_SIGNAL',
        'summary': {
            'variant_count': 100,
            'survivor_count': 0,
            'best_variant': {'oos_sharpe': -5.0},
            'trades_per_day': 0.5,
        },
        'fee_decomposition': {
            'estimated_fee_drag_per_trade': 0.0012,
        },
        'trade_duration_distribution': {
            'median_bars': 15,
        },
    }
    (sweep_dir / 'post_mortem.json').write_text(json.dumps(post_mortem))

    variant = {
        'variant_id': 'abc123',
        'status': 'OK',
        'oos_sharpe': -5.0,
        'spec': {
            'entry': {'key': 'ema_cross', 'params': {'fast': 8, 'slow': 21}},
            'fee_rate': 0.0006,
        }
    }
    (sweep_dir / 'variants.jsonl').write_text(json.dumps(variant) + '\n')

    ingest_sweep(db, str(sweep_dir), emit_events=False)

    # Create proposal with predictions
    write_proposal(
        db,
        proposal_id='proposal_audit_test',
        experiment_intent='failure_mitigation',
        proposed_config={'entry': {'key': 'ema_cross'}},
        evidence_refs=['sweep_audit_test'],
        novelty_claim={'coverage_diff': ['longer_ema']},
        predictions={
            'trade_duration_median_bars': 20,
            'trades_per_day': 0.6,
            'gross_minus_net_gap': 0.0015,
        },
        expected_mechanism='Test mechanism explanation',
        kill_criteria=['OOS Sharpe < -3'],
        compute_budget={'max_variants': 100, 'max_runtime_minutes': 60},
        experiment_id='exp_audit_test',
    )
    db.close()

    return db_path, str(sweep_dir)


@pytest.fixture
def populated_db_no_proposal(tmp_path):
    """Create a KB with test sweep but no proposal."""
    db_path = str(tmp_path / 'test.db')
    db = init_db(db_path)

    # Create test sweep directory
    sweep_dir = tmp_path / 'sweep_no_proposal'
    sweep_dir.mkdir()

    manifest = {
        'sweep_name': 'sweep_no_proposal',
        'symbol': 'BTCUSDT',
        'timeframes': ['4h'],
        'data_range': {'start': '2024-01-01', 'end': '2024-06-01'},
        'experiment_id': 'exp_no_proposal',
    }
    (sweep_dir / 'manifest.json').write_text(json.dumps(manifest))

    post_mortem = {
        'sweep_id': 'sweep_no_proposal',
        'experiment_id': 'exp_no_proposal',
        'created_ts': '2024-01-01T00:00:00Z',
        'primary_failure_mode': 'LOW_SIGNAL',
        'summary': {'variant_count': 100, 'survivor_count': 0, 'trades_per_day': 1.0},
        'trade_duration_distribution': {'median_bars': 10},
        'fee_decomposition': {'estimated_fee_drag_per_trade': 0.001},
    }
    (sweep_dir / 'post_mortem.json').write_text(json.dumps(post_mortem))

    variant = {
        'variant_id': 'xyz789',
        'status': 'OK',
        'spec': {'entry': {'key': 'ema_cross'}, 'fee_rate': 0.0006},
    }
    (sweep_dir / 'variants.jsonl').write_text(json.dumps(variant) + '\n')

    ingest_sweep(db, str(sweep_dir), emit_events=False)
    db.close()

    return db_path, str(sweep_dir)


class TestPredictionAudit:
    """Tests for run_prediction_audit()."""

    def test_audit_with_proposal(self, populated_db_with_proposal):
        """Proposal + post_mortem → audit JSON written."""
        reset_config()
        db_path, sweep_dir = populated_db_with_proposal

        # Run audit
        result = run_prediction_audit(
            sweep_dir=sweep_dir,
            db_path=db_path,
            sweep_id='sweep_audit_test',
        )

        # Should return audit result (dict of comparisons)
        assert result is not None
        assert isinstance(result, dict)
        assert len(result) > 0

        # Check DB has audit
        db = init_db(db_path)
        proposal = get_proposal(db, 'proposal_audit_test')
        db.close()

        assert proposal['prediction_audit'] is not None
        audit = proposal['prediction_audit']
        assert len(audit) > 0

    def test_audit_comparisons_structure(self, populated_db_with_proposal):
        """Audit should compare predicted vs actual values."""
        reset_config()
        db_path, sweep_dir = populated_db_with_proposal

        result = run_prediction_audit(
            sweep_dir=sweep_dir,
            db_path=db_path,
            sweep_id='sweep_audit_test',
        )

        # Should have comparisons for each prediction field
        assert 'trade_duration_median_bars' in result
        assert 'trades_per_day' in result
        assert 'gross_minus_net_gap' in result

        # Each comparison should have predicted, actual, diff_pct
        duration_cmp = result['trade_duration_median_bars']
        assert 'predicted' in duration_cmp
        assert 'actual' in duration_cmp
        assert 'diff_pct' in duration_cmp

    def test_no_proposal_skips_gracefully(self, populated_db_no_proposal):
        """No proposal → skip (no error)."""
        reset_config()
        db_path, sweep_dir = populated_db_no_proposal

        # Should not raise, should return None
        result = run_prediction_audit(
            sweep_dir=sweep_dir,
            db_path=db_path,
            sweep_id='sweep_no_proposal',
        )

        assert result is None

    def test_audit_does_not_block_on_missing_data(self, tmp_path):
        """Audit with missing data → handles gracefully, not blocking."""
        reset_config()
        db_path = str(tmp_path / 'test.db')
        db = init_db(db_path)

        # Create sweep without post_mortem initially
        sweep_dir = tmp_path / 'sweep_missing'
        sweep_dir.mkdir()

        manifest = {
            'sweep_name': 'sweep_missing',
            'experiment_id': 'exp_missing',
        }
        (sweep_dir / 'manifest.json').write_text(json.dumps(manifest))

        # No post_mortem.json - should return None gracefully
        db.close()

        result = run_prediction_audit(
            sweep_dir=str(sweep_dir),
            db_path=db_path,
            sweep_id='sweep_missing',
        )

        # Should return None since no matching proposal
        assert result is None


class TestPredictionAuditCalculations:
    """Tests for audit calculation accuracy."""

    def test_diff_pct_calculation(self, populated_db_with_proposal):
        """diff_pct should be (actual - predicted) / |predicted| * 100."""
        reset_config()
        db_path, sweep_dir = populated_db_with_proposal

        result = run_prediction_audit(
            sweep_dir=sweep_dir,
            db_path=db_path,
            sweep_id='sweep_audit_test',
        )

        # trade_duration_median_bars: predicted=20, actual=15
        # diff_pct = (15 - 20) / 20 * 100 = -25%
        duration = result['trade_duration_median_bars']
        assert duration['predicted'] == 20
        assert duration['actual'] == 15
        assert abs(duration['diff_pct'] - (-25.0)) < 0.001

        # trades_per_day: predicted=0.6, actual=0.5
        # diff_pct = (0.5 - 0.6) / 0.6 * 100 = -16.67%
        trades = result['trades_per_day']
        assert trades['predicted'] == 0.6
        assert trades['actual'] == 0.5
        assert abs(trades['diff_pct'] - (-16.666666666)) < 0.01


class TestPredictionAuditEdgeCases:
    """Tests for edge cases in prediction audit."""

    def test_missing_manifest_returns_none(self, tmp_path):
        """Should return None if no manifest.json."""
        db_path = str(tmp_path / 'test.db')
        db = init_db(db_path)
        db.close()

        sweep_dir = tmp_path / 'sweep_no_manifest'
        sweep_dir.mkdir()
        # No manifest.json

        result = run_prediction_audit(
            sweep_dir=str(sweep_dir),
            db_path=db_path,
            sweep_id='sweep_no_manifest',
        )

        assert result is None

    def test_missing_experiment_id_returns_none(self, tmp_path):
        """Should return None if manifest has no experiment_id."""
        db_path = str(tmp_path / 'test.db')
        db = init_db(db_path)
        db.close()

        sweep_dir = tmp_path / 'sweep_no_expid'
        sweep_dir.mkdir()

        manifest = {'sweep_name': 'test'}  # No experiment_id
        (sweep_dir / 'manifest.json').write_text(json.dumps(manifest))

        result = run_prediction_audit(
            sweep_dir=str(sweep_dir),
            db_path=db_path,
            sweep_id='sweep_no_expid',
        )

        assert result is None

    def test_zero_predicted_value_handling(self, tmp_path):
        """Should handle zero predicted value without division error."""
        reset_config()
        db_path = str(tmp_path / 'test.db')
        db = init_db(db_path)

        # Create sweep
        sweep_dir = tmp_path / 'sweep_zero'
        sweep_dir.mkdir()

        manifest = {
            'sweep_name': 'sweep_zero',
            'symbol': 'BTCUSDT',
            'experiment_id': 'exp_zero',
        }
        (sweep_dir / 'manifest.json').write_text(json.dumps(manifest))

        post_mortem = {
            'sweep_id': 'sweep_zero',
            'experiment_id': 'exp_zero',
            'primary_failure_mode': 'LOW_SIGNAL',
            'summary': {'trades_per_day': 1.0},
            'trade_duration_distribution': {'median_bars': 10},
            'fee_decomposition': {'estimated_fee_drag_per_trade': 0.001},
        }
        (sweep_dir / 'post_mortem.json').write_text(json.dumps(post_mortem))

        variant = {'variant_id': 'v1', 'status': 'OK', 'spec': {'entry': {'key': 'ema_cross'}}}
        (sweep_dir / 'variants.jsonl').write_text(json.dumps(variant) + '\n')

        ingest_sweep(db, str(sweep_dir), emit_events=False)

        # Create proposal with zero prediction
        write_proposal(
            db,
            proposal_id='proposal_zero',
            experiment_intent='failure_mitigation',
            proposed_config={},
            evidence_refs=['sweep_zero'],
            novelty_claim={'coverage_diff': ['x']},
            predictions={
                'trade_duration_median_bars': 0,  # Zero predicted
                'trades_per_day': 1,
                'gross_minus_net_gap': 0.001,
            },
            expected_mechanism='x' * 60,
            kill_criteria=['x'],
            compute_budget={'max_variants': 100, 'max_runtime_minutes': 60},
            experiment_id='exp_zero',
        )
        db.close()

        # Should not raise ZeroDivisionError
        result = run_prediction_audit(
            sweep_dir=str(sweep_dir),
            db_path=db_path,
            sweep_id='sweep_zero',
        )

        assert result is not None
        # diff_pct should be None when predicted is 0
        duration = result['trade_duration_median_bars']
        assert duration['diff_pct'] is None
