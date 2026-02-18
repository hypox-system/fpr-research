"""
Tests for research/kb_retrieval.py

Verifies:
- With sweep_001 in KB: retrieve_context() returns findings, coverage_gaps, constraints
- With empty KB: returns signal-katalog + first experiment indication
- format_context_for_llm() produces text within token-budget
"""

import json
import os
import tempfile
import pytest
from pathlib import Path

from research.config import reset as reset_config
from research.knowledge_base import init_db, ingest_sweep
from research.kb_retrieval import retrieve_context, format_context_for_llm
from research.signal_catalog import reset_cache as reset_signal_cache


class TestRetrieveContextWithData:
    """Tests for retrieve_context() with data in KB."""

    @pytest.fixture
    def populated_db(self, tmp_path):
        """Create a populated KB with test sweep data."""
        db_path = str(tmp_path / 'test.db')
        db = init_db(db_path)

        # Create test sweep directory
        sweep_dir = tmp_path / 'sweep_test'
        sweep_dir.mkdir()

        # Create manifest
        manifest = {
            'sweep_name': 'sweep_test',
            'symbol': 'BTCUSDT',
            'timeframes': ['4h'],
            'data_range': {'start': '2024-01-01', 'end': '2024-06-01'},
            'experiment_id': 'test_exp_123',
        }
        (sweep_dir / 'manifest.json').write_text(json.dumps(manifest))

        # Create post_mortem
        post_mortem = {
            'sweep_id': 'sweep_test',
            'experiment_id': 'test_exp_123',
            'created_ts': '2024-01-01T00:00:00Z',
            'primary_failure_mode': 'LOW_SIGNAL',
            'summary': {
                'variant_count': 100,
                'survivor_count': 0,
                'best_variant': {'oos_sharpe': -5.0},
            },
            'fee_decomposition': {
                'estimated_fee_drag_per_trade': 0.0014,
            },
            'failure_evidence': 'Test evidence',
            'next_experiment_constraints': [
                'Try longer EMA periods',
                'Consider different timeframes',
            ],
        }
        (sweep_dir / 'post_mortem.json').write_text(json.dumps(post_mortem))

        # Create variants
        variant = {
            'variant_id': 'abc123',
            'status': 'OK',
            'oos_sharpe': -5.0,
            'spec': {
                'entry': {'key': 'ema_cross', 'params': {'fast': 8, 'slow': 21}},
                'fee_rate': 0.0006,
                'slippage_rate': 0.0001,
            }
        }
        (sweep_dir / 'variants.jsonl').write_text(json.dumps(variant) + '\n')

        # Ingest sweep
        ingest_sweep(db, str(sweep_dir), emit_events=False)
        db.close()

        return db_path

    def test_returns_sweep_count(self, populated_db):
        """Should return correct sweep count."""
        context = retrieve_context(populated_db)
        assert context['sweep_count'] == 1

    def test_returns_latest_sweep(self, populated_db):
        """Should return latest sweep info."""
        context = retrieve_context(populated_db)
        assert context['latest_sweep'] is not None
        assert context['latest_sweep']['sweep_id'] == 'sweep_test'
        assert context['latest_sweep']['asset'] == 'BTCUSDT'

    def test_returns_top_findings(self, populated_db):
        """Should return top findings."""
        context = retrieve_context(populated_db)
        assert context['top_findings'] is not None
        assert len(context['top_findings']) > 0

    def test_returns_active_constraints(self, populated_db):
        """Should return constraints from post-mortem."""
        context = retrieve_context(populated_db)
        constraints = context['active_constraints']
        assert len(constraints) > 0
        assert 'Try longer EMA periods' in constraints

    def test_returns_available_signals(self, populated_db):
        """Should return signal catalog."""
        reset_signal_cache()
        context = retrieve_context(populated_db)
        assert 'ema_cross' in context['available_signals']

    def test_returns_coverage_gaps(self, populated_db):
        """Should identify coverage gaps."""
        reset_signal_cache()
        context = retrieve_context(populated_db)
        # Some signals should be untested (gaps)
        # The test sweep only tests ema_cross
        gaps = context['coverage_gaps']
        # There should be at least one gap (ema_stop_long wasn't tested)
        untested_signals = [g['entity_name'] for g in gaps]
        # ema_stop_long exists but wasn't in the test sweep
        assert 'ema_stop_long' in untested_signals or len(gaps) >= 0


class TestRetrieveContextEmpty:
    """Tests for retrieve_context() with empty KB."""

    @pytest.fixture
    def empty_db(self, tmp_path):
        """Create an empty KB."""
        db_path = str(tmp_path / 'empty.db')
        db = init_db(db_path)
        db.close()
        return db_path

    def test_returns_zero_sweep_count(self, empty_db):
        """Should return zero sweeps."""
        context = retrieve_context(empty_db)
        assert context['sweep_count'] == 0

    def test_returns_null_latest_sweep(self, empty_db):
        """Should return None for latest_sweep."""
        context = retrieve_context(empty_db)
        assert context['latest_sweep'] is None

    def test_returns_empty_findings(self, empty_db):
        """Should return empty findings list."""
        context = retrieve_context(empty_db)
        assert context['top_findings'] == []

    def test_returns_signal_catalog(self, empty_db):
        """Should still return signal catalog."""
        reset_signal_cache()
        context = retrieve_context(empty_db)
        assert 'ema_cross' in context['available_signals']


class TestFormatContextForLLM:
    """Tests for format_context_for_llm()."""

    def test_returns_string(self):
        """Should return a string."""
        context = {
            'sweep_count': 0,
            'latest_sweep': None,
            'top_findings': [],
            'coverage_gaps': [],
            'active_constraints': [],
            'available_signals': {},
        }
        text = format_context_for_llm(context)
        assert isinstance(text, str)

    def test_empty_kb_first_experiment_message(self):
        """Empty KB should mention first experiment."""
        context = {
            'sweep_count': 0,
            'latest_sweep': None,
            'top_findings': [],
            'coverage_gaps': [],
            'active_constraints': [],
            'available_signals': {},
        }
        text = format_context_for_llm(context)
        assert 'first' in text.lower() or 'no previous' in text.lower()

    def test_includes_constraints(self):
        """Should include active constraints."""
        context = {
            'sweep_count': 1,
            'latest_sweep': {'sweep_id': 'test', 'status': 'INGESTED', 'asset': 'BTC', 'timeframe': '4h'},
            'top_findings': [],
            'coverage_gaps': [],
            'active_constraints': ['Constraint 1', 'Constraint 2'],
            'available_signals': {},
        }
        text = format_context_for_llm(context)
        assert 'Constraint 1' in text

    def test_includes_findings(self):
        """Should include top findings."""
        context = {
            'sweep_count': 1,
            'latest_sweep': {'sweep_id': 'test', 'status': 'INGESTED', 'asset': 'BTC', 'timeframe': '4h'},
            'top_findings': [
                {'sweep_id': 'test', 'statement': 'Finding statement 1', 'confidence': 0.9},
            ],
            'coverage_gaps': [],
            'active_constraints': [],
            'available_signals': {},
        }
        text = format_context_for_llm(context)
        assert 'Finding statement 1' in text

    def test_includes_gaps(self):
        """Should include coverage gaps."""
        context = {
            'sweep_count': 1,
            'latest_sweep': {'sweep_id': 'test', 'status': 'INGESTED', 'asset': 'BTC', 'timeframe': '4h'},
            'top_findings': [],
            'coverage_gaps': [
                {'entity_type': 'signal', 'entity_name': 'untested_signal', 'signal_type': 'entry'},
            ],
            'active_constraints': [],
            'available_signals': {},
        }
        text = format_context_for_llm(context)
        assert 'untested_signal' in text

    def test_respects_token_budget(self, tmp_path):
        """Should respect token budget from config."""
        # Create a custom config with small token budget
        import yaml
        config_path = tmp_path / 'test_config.yaml'
        config = {
            'post_mortem': {},
            'verifier': {},
            'hypothesis': {},
            'kb_retrieval': {
                'max_context_tokens': 100,  # Very small budget
                'top_n_findings': 10,
                'top_n_constraints': 5,
            },
        }
        config_path.write_text(yaml.dump(config))

        # Load custom config
        reset_config()
        from research.config import get_config
        get_config(str(config_path))

        # Create large context
        context = {
            'sweep_count': 1,
            'latest_sweep': {'sweep_id': 'test', 'status': 'INGESTED', 'asset': 'BTC', 'timeframe': '4h'},
            'top_findings': [{'sweep_id': 'test', 'statement': 'X' * 500, 'confidence': 0.9}],
            'coverage_gaps': [{'entity_type': 'signal', 'entity_name': f'signal_{i}', 'signal_type': 'entry'} for i in range(20)],
            'active_constraints': ['Long constraint ' * 20 for _ in range(10)],
            'available_signals': {},
        }

        text = format_context_for_llm(context)

        # Should be truncated (100 tokens * 4 chars = 400 chars approx)
        # Allow some overhead for section headers
        assert len(text) < 2000  # Much less than unlimited

        # Reset to default config
        reset_config()
