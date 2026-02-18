"""
Tests for research/proposal_verifier.py

Verifies:
- Valid proposal passes
- Missing required field fails
- Duplicate experiment_id fails
- Budget over limit fails
- Evidence ref not in KB fails
- Bootstrap mode: 1 ref godkänns when KB has < 3 sweeps
- Invalid sweep config YAML fails
- Unknown signal fails
"""

import json
import os
import tempfile
import pytest
from pathlib import Path

from research.config import reset as reset_config
from research.knowledge_base import init_db, ingest_sweep, write_proposal
from research.proposal_verifier import verify_proposal, VerificationResult
from research.signal_catalog import reset_cache as reset_signal_cache


def make_valid_proposal(evidence_refs=None):
    """Create a valid proposal dict."""
    return {
        'experiment_intent': 'failure_mitigation',
        'evidence_refs': evidence_refs if evidence_refs is not None else ['sweep_test'],
        'novelty_claim': {
            'coverage_diff': ['longer_ema_periods'],
            'near_dup_score': 0.3,
        },
        'expected_mechanism': 'This is a detailed explanation of the expected mechanism that should capture market inefficiencies through longer-term trend following.',
        'predictions': {
            'trade_duration_median_bars': 25,
            'trades_per_day': 0.8,
            'gross_minus_net_gap': 0.0014,
        },
        'expected_failure_mode': 'LOW_SIGNAL',
        'kill_criteria': ['OOS Sharpe < -3', 'Win rate < 25%'],
        'compute_budget': {
            'max_variants': 100,
            'max_runtime_minutes': 60,
        },
        'sweep_config_yaml': '''name: sweep_002
symbol: BTCUSDT
entry:
  key: ema_cross
  params:
    fast: [13, 21]
    slow: [55, 89]
exit:
  key: ema_stop_long
fee_rate: 0.0006
''',
    }


@pytest.fixture
def populated_db(tmp_path):
    """Create a KB with test sweep."""
    db_path = str(tmp_path / 'test.db')
    db = init_db(db_path)

    # Create test sweep directory
    sweep_dir = tmp_path / 'sweep_test'
    sweep_dir.mkdir()

    manifest = {
        'sweep_name': 'sweep_test',
        'symbol': 'BTCUSDT',
        'data_range': {'start': '2024-01-01', 'end': '2024-06-01'},
    }
    (sweep_dir / 'manifest.json').write_text(json.dumps(manifest))

    post_mortem = {
        'sweep_id': 'sweep_test',
        'experiment_id': 'exp_test_123',
        'created_ts': '2024-01-01T00:00:00Z',
        'primary_failure_mode': 'LOW_SIGNAL',
        'summary': {'variant_count': 100, 'survivor_count': 0},
    }
    (sweep_dir / 'post_mortem.json').write_text(json.dumps(post_mortem))

    variant = {
        'variant_id': 'abc',
        'status': 'OK',
        'spec': {'entry': {'key': 'ema_cross'}, 'fee_rate': 0.0006},
    }
    (sweep_dir / 'variants.jsonl').write_text(json.dumps(variant) + '\n')

    ingest_sweep(db, str(sweep_dir), emit_events=False)
    db.close()

    return db_path


class TestValidProposal:
    """Tests for valid proposals."""

    def test_valid_proposal_passes(self, populated_db):
        """Valid proposal should pass all checks."""
        reset_signal_cache()
        proposal = make_valid_proposal()
        result = verify_proposal(proposal, populated_db)

        assert result.passed, f"Failed checks: {[c.name for c in result.checks if not c.passed]}"

    def test_all_checks_run(self, populated_db):
        """All checks should be run."""
        reset_signal_cache()
        proposal = make_valid_proposal()
        result = verify_proposal(proposal, populated_db)

        check_names = {c.name for c in result.checks}
        expected = {
            'schema_valid',
            'intent_valid',
            'evidence_refs_exist',
            'evidence_refs_minimum',
            'predictions_complete',
            'sweep_config_valid',
            'experiment_not_duplicate',
            'budget_within_limits',
            'kill_criteria_present',
            'novelty_claim_non_empty',
            'signals_exist',
        }
        assert expected.issubset(check_names)


class TestSchemaValidation:
    """Tests for schema validation."""

    def test_missing_required_field_fails(self, populated_db):
        """Missing required field should fail."""
        proposal = make_valid_proposal()
        del proposal['predictions']

        result = verify_proposal(proposal, populated_db)

        assert not result.passed
        schema_check = next(c for c in result.checks if c.name == 'schema_valid')
        assert not schema_check.passed

    def test_invalid_intent_fails(self, populated_db):
        """Invalid experiment_intent should fail."""
        proposal = make_valid_proposal()
        proposal['experiment_intent'] = 'invalid_intent'

        result = verify_proposal(proposal, populated_db)

        intent_check = next(c for c in result.checks if c.name == 'intent_valid')
        assert not intent_check.passed


class TestEvidenceRefs:
    """Tests for evidence_refs validation."""

    def test_nonexistent_ref_fails(self, populated_db):
        """Reference to non-existent sweep should fail."""
        proposal = make_valid_proposal(evidence_refs=['nonexistent_sweep'])

        result = verify_proposal(proposal, populated_db)

        refs_check = next(c for c in result.checks if c.name == 'evidence_refs_exist')
        assert not refs_check.passed
        assert 'not found' in refs_check.reason.lower()

    def test_valid_ref_passes(self, populated_db):
        """Reference to existing sweep should pass."""
        proposal = make_valid_proposal(evidence_refs=['sweep_test'])

        result = verify_proposal(proposal, populated_db)

        refs_check = next(c for c in result.checks if c.name == 'evidence_refs_exist')
        assert refs_check.passed


class TestBootstrapMode:
    """Tests for bootstrap mode (< 3 sweeps in KB)."""

    def test_one_ref_accepted_in_bootstrap(self, populated_db):
        """1 ref should be accepted when KB has < 3 sweeps."""
        # KB has 1 sweep, which is < 3 (bootstrap_threshold)
        proposal = make_valid_proposal(evidence_refs=['sweep_test'])

        reset_signal_cache()
        result = verify_proposal(proposal, populated_db)

        min_check = next(c for c in result.checks if c.name == 'evidence_refs_minimum')
        assert min_check.passed

    def test_zero_refs_fails_even_in_bootstrap(self, populated_db):
        """0 refs should fail even in bootstrap mode."""
        proposal = make_valid_proposal(evidence_refs=[])

        result = verify_proposal(proposal, populated_db)

        min_check = next(c for c in result.checks if c.name == 'evidence_refs_minimum')
        assert not min_check.passed


class TestBudgetLimits:
    """Tests for compute budget validation."""

    def test_over_max_variants_fails(self, populated_db):
        """Budget with too many variants should fail."""
        proposal = make_valid_proposal()
        proposal['compute_budget']['max_variants'] = 1000  # Over 500 limit

        result = verify_proposal(proposal, populated_db)

        budget_check = next(c for c in result.checks if c.name == 'budget_within_limits')
        assert not budget_check.passed
        assert '1000' in budget_check.reason

    def test_over_runtime_fails(self, populated_db):
        """Budget with too much runtime should fail."""
        proposal = make_valid_proposal()
        proposal['compute_budget']['max_runtime_minutes'] = 200  # Over 120 limit

        result = verify_proposal(proposal, populated_db)

        budget_check = next(c for c in result.checks if c.name == 'budget_within_limits')
        assert not budget_check.passed


class TestSweepConfigValidation:
    """Tests for sweep_config_yaml validation."""

    def test_invalid_yaml_fails(self, populated_db):
        """Invalid YAML should fail."""
        proposal = make_valid_proposal()
        proposal['sweep_config_yaml'] = 'invalid: yaml: content: ['

        result = verify_proposal(proposal, populated_db)

        config_check = next(c for c in result.checks if c.name == 'sweep_config_valid')
        assert not config_check.passed

    def test_empty_yaml_fails(self, populated_db):
        """Empty YAML should fail."""
        proposal = make_valid_proposal()
        proposal['sweep_config_yaml'] = ''

        result = verify_proposal(proposal, populated_db)

        config_check = next(c for c in result.checks if c.name == 'sweep_config_valid')
        assert not config_check.passed


class TestSignalValidation:
    """Tests for signal existence validation."""

    def test_unknown_signal_fails(self, populated_db):
        """Unknown signal in config should fail."""
        reset_signal_cache()
        proposal = make_valid_proposal()
        proposal['sweep_config_yaml'] = '''name: test
symbol: BTCUSDT
entry:
  key: nonexistent_signal
'''
        result = verify_proposal(proposal, populated_db)

        signals_check = next(c for c in result.checks if c.name == 'signals_exist')
        assert not signals_check.passed
        assert 'nonexistent_signal' in signals_check.reason


class TestDuplicateExperiment:
    """Tests for duplicate experiment detection."""

    def test_duplicate_experiment_id_fails(self, populated_db, tmp_path):
        """Duplicate experiment_id should fail."""
        # Add a proposal with specific experiment_id
        db = init_db(populated_db)
        write_proposal(
            db,
            proposal_id='existing_proposal',
            experiment_intent='gap_fill',
            proposed_config={},
            evidence_refs=['sweep_test'],
            novelty_claim={'coverage_diff': ['x']},
            predictions={'trade_duration_median_bars': 10, 'trades_per_day': 1, 'gross_minus_net_gap': 0.001},
            expected_mechanism='x' * 60,
            kill_criteria=['x'],
            compute_budget={'max_variants': 100, 'max_runtime_minutes': 60},
            experiment_id='duplicate_exp_id',
        )
        db.close()

        # Try to verify a proposal with same experiment_id
        proposal = make_valid_proposal()
        proposal['experiment_id'] = 'duplicate_exp_id'

        reset_signal_cache()
        result = verify_proposal(proposal, populated_db)

        dup_check = next(c for c in result.checks if c.name == 'experiment_not_duplicate')
        assert not dup_check.passed


class TestNoveltyValidation:
    """Tests for novelty_claim validation."""

    def test_empty_coverage_diff_fails(self, populated_db):
        """Empty coverage_diff should fail."""
        proposal = make_valid_proposal()
        proposal['novelty_claim']['coverage_diff'] = []

        result = verify_proposal(proposal, populated_db)

        novelty_check = next(c for c in result.checks if c.name == 'novelty_claim_non_empty')
        assert not novelty_check.passed


class TestKillCriteria:
    """Tests for kill_criteria validation."""

    def test_no_kill_criteria_fails(self, populated_db):
        """Empty kill_criteria should fail."""
        proposal = make_valid_proposal()
        proposal['kill_criteria'] = []

        result = verify_proposal(proposal, populated_db)

        kill_check = next(c for c in result.checks if c.name == 'kill_criteria_present')
        assert not kill_check.passed


class TestVerificationResult:
    """Tests for VerificationResult class."""

    def test_to_dict(self, populated_db):
        """to_dict() should return serializable dict."""
        reset_signal_cache()
        proposal = make_valid_proposal()
        result = verify_proposal(proposal, populated_db)

        d = result.to_dict()
        assert 'passed' in d
        assert 'checks' in d
        assert isinstance(d['checks'], list)

        # Should be JSON serializable
        json.dumps(d)
