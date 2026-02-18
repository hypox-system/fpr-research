"""
Tests for research/hypothesis_gen.py

Verifies:
- Mock LLM produces PENDING proposal in DB
- Invalid LLM response → events logged, no crash
- --dry-run → no DB write
- E2E: sweep in KB → retrieval → gen → verify → proposal in proposals table
"""

import json
import os
import tempfile
import pytest
from pathlib import Path

from research.config import reset as reset_config
from research.knowledge_base import init_db, ingest_sweep, get_proposals, export_events
from research.hypothesis_gen import run, verify_file, _parse_llm_response
from research.signal_catalog import reset_cache as reset_signal_cache


@pytest.fixture
def populated_db(tmp_path):
    """Create a KB with test sweep."""
    db_path = str(tmp_path / 'test.db')
    db = init_db(db_path)

    # Create test sweep directory
    sweep_dir = tmp_path / 'sweep_001_ema_pvsra'
    sweep_dir.mkdir()

    manifest = {
        'sweep_name': 'sweep_001_ema_pvsra',
        'symbol': 'BTCUSDT',
        'timeframes': ['4h'],
        'data_range': {'start': '2024-01-01', 'end': '2024-06-01'},
        'experiment_id': 'exp_test_001',
    }
    (sweep_dir / 'manifest.json').write_text(json.dumps(manifest))

    post_mortem = {
        'sweep_id': 'sweep_001_ema_pvsra',
        'experiment_id': 'exp_test_001',
        'created_ts': '2024-01-01T00:00:00Z',
        'primary_failure_mode': 'LOW_SIGNAL',
        'summary': {'variant_count': 270, 'survivor_count': 0, 'best_variant': {'oos_sharpe': -8.0}},
        'fee_decomposition': {'estimated_fee_drag_per_trade': 0.0014},
        'failure_evidence': 'No survivors',
        'next_experiment_constraints': ['Try longer EMA periods', 'Consider different timeframes'],
    }
    (sweep_dir / 'post_mortem.json').write_text(json.dumps(post_mortem))

    variant = {
        'variant_id': 'abc123',
        'status': 'OK',
        'oos_sharpe': -8.0,
        'spec': {
            'entry': {'key': 'ema_cross', 'params': {'fast': 8, 'slow': 21}},
            'fee_rate': 0.0006,
            'slippage_rate': 0.0001,
        }
    }
    (sweep_dir / 'variants.jsonl').write_text(json.dumps(variant) + '\n')

    ingest_sweep(db, str(sweep_dir), emit_events=False)
    db.close()

    return db_path


class TestMockLLMIntegration:
    """Tests with mock LLM provider."""

    def test_mock_produces_pending_proposal(self, populated_db):
        """Mock LLM should produce a PENDING proposal."""
        reset_signal_cache()
        reset_config()

        result = run(db_path=populated_db, provider='mock')

        assert result['status'] == 'accepted'
        assert result['proposal_id'] is not None

        # Check DB
        db = init_db(populated_db)
        proposals = get_proposals(db, status='PENDING')
        db.close()

        assert len(proposals) >= 1
        # Find our proposal
        our_proposal = next((p for p in proposals if p['proposal_id'] == result['proposal_id']), None)
        assert our_proposal is not None
        assert our_proposal['status'] == 'PENDING'

    def test_proposal_has_all_fields(self, populated_db):
        """Proposal should have all 9 HypothesisFX fields."""
        reset_signal_cache()
        reset_config()

        result = run(db_path=populated_db, provider='mock')

        db = init_db(populated_db)
        proposals = get_proposals(db, status='PENDING')
        db.close()

        proposal = proposals[0]
        assert 'experiment_intent' in proposal
        assert 'evidence_refs' in proposal
        assert 'novelty_claim' in proposal
        assert 'expected_mechanism' in proposal
        assert 'predictions' in proposal
        assert 'expected_failure_mode' in proposal
        assert 'kill_criteria' in proposal
        assert 'compute_budget' in proposal
        assert 'proposed_config' in proposal

    def test_proposal_references_sweep(self, populated_db):
        """Proposal should reference the sweep in evidence_refs."""
        reset_signal_cache()
        reset_config()

        result = run(db_path=populated_db, provider='mock')

        db = init_db(populated_db)
        proposals = get_proposals(db, status='PENDING')
        db.close()

        proposal = proposals[0]
        assert 'sweep_001_ema_pvsra' in proposal['evidence_refs']


class TestDryRun:
    """Tests for dry-run mode."""

    def test_dry_run_no_db_write(self, populated_db):
        """Dry-run should not write to DB."""
        reset_signal_cache()
        reset_config()

        # Get initial proposal count
        db = init_db(populated_db)
        initial_count = len(get_proposals(db))
        db.close()

        # Run with dry-run
        result = run(db_path=populated_db, provider='mock', dry_run=True)

        # Check DB unchanged
        db = init_db(populated_db)
        final_count = len(get_proposals(db))
        db.close()

        assert final_count == initial_count
        assert result['status'] == 'accepted'


class TestEventLogging:
    """Tests for event logging."""

    def test_events_logged_on_success(self, populated_db):
        """Success should log HYPOTHESIS_GEN_* and PROPOSAL_CREATED events."""
        reset_signal_cache()
        reset_config()

        run(db_path=populated_db, provider='mock')

        db = init_db(populated_db)
        events = export_events(db, last_n=50)
        db.close()

        event_types = [e['event_type'] for e in events]
        assert 'HYPOTHESIS_GEN_STARTED' in event_types
        assert 'HYPOTHESIS_GEN_COMPLETED' in event_types
        assert 'PROPOSAL_CREATED' in event_types


class TestParseResponse:
    """Tests for _parse_llm_response()."""

    def test_parse_clean_json(self):
        """Should parse clean JSON."""
        raw = '{"key": "value"}'
        result = _parse_llm_response(raw)
        assert result == {'key': 'value'}

    def test_parse_json_in_code_block(self):
        """Should extract JSON from markdown code block."""
        raw = '''```json
{"key": "value"}
```'''
        result = _parse_llm_response(raw)
        assert result == {'key': 'value'}

    def test_parse_json_with_text_before(self):
        """Should extract JSON after text."""
        raw = '''Here is my response:
{"key": "value"}'''
        result = _parse_llm_response(raw)
        assert result == {'key': 'value'}

    def test_parse_json_with_text_after(self):
        """Should extract JSON before text."""
        raw = '''{"key": "value"}
That's my response.'''
        result = _parse_llm_response(raw)
        assert result == {'key': 'value'}

    def test_parse_invalid_json_raises(self):
        """Invalid JSON should raise ValueError."""
        raw = 'not json at all'
        with pytest.raises(ValueError, match='Failed to parse JSON'):
            _parse_llm_response(raw)


class TestVerifyFile:
    """Tests for verify_file()."""

    def test_verify_valid_file(self, populated_db, tmp_path):
        """Should verify a valid proposal file."""
        reset_signal_cache()

        proposal = {
            'experiment_intent': 'failure_mitigation',
            'evidence_refs': ['sweep_001_ema_pvsra'],
            'novelty_claim': {'coverage_diff': ['test']},
            'expected_mechanism': 'x' * 60,
            'predictions': {
                'trade_duration_median_bars': 10,
                'trades_per_day': 1,
                'gross_minus_net_gap': 0.001,
            },
            'expected_failure_mode': 'LOW_SIGNAL',
            'kill_criteria': ['OOS Sharpe < -3'],
            'compute_budget': {'max_variants': 100, 'max_runtime_minutes': 60},
            'sweep_config_yaml': '''name: test
symbol: BTCUSDT
entry:
  key: ema_cross
''',
        }

        file_path = tmp_path / 'proposal.json'
        file_path.write_text(json.dumps(proposal))

        result = verify_file(str(file_path), populated_db)

        assert result['passed']

    def test_verify_invalid_file(self, populated_db, tmp_path):
        """Should reject invalid proposal file."""
        proposal = {
            'experiment_intent': 'failure_mitigation',
            # Missing required fields
        }

        file_path = tmp_path / 'bad_proposal.json'
        file_path.write_text(json.dumps(proposal))

        result = verify_file(str(file_path), populated_db)

        assert not result['passed']


class TestE2EIntegration:
    """End-to-end integration tests."""

    def test_full_pipeline(self, populated_db):
        """Full E2E: KB with sweep → retrieval → gen → verify → proposal."""
        reset_signal_cache()
        reset_config()

        # Run pipeline
        result = run(db_path=populated_db, provider='mock')

        # Verify success
        assert result['status'] == 'accepted'
        assert result['proposal_id'] is not None

        # Verify proposal in DB
        db = init_db(populated_db)
        proposals = get_proposals(db, status='PENDING')
        db.close()

        assert len(proposals) >= 1

        # Verify proposal content
        proposal = proposals[0]
        assert proposal['experiment_intent'] in ['gap_fill', 'failure_mitigation', 'robustness', 'regime_test']
        assert len(proposal['evidence_refs']) >= 1
        assert 'coverage_diff' in proposal['novelty_claim']
        assert 'trade_duration_median_bars' in proposal['predictions']
        assert len(proposal['kill_criteria']) >= 1
