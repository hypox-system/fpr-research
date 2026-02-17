"""
Tests for research/knowledge_base.py

Verifies:
- Init DB creates all tables
- Ingest sweep creates sweeps, findings, coverage, artifacts rows
- Query finds matching sweeps
- Stats returns coverage summary
- Duplicate experiment_id is handled (upsert, not crash)
- Events are written correctly at ingest
- Export-events returns chronological order
"""

import json
import os
import sqlite3
import tempfile
import pytest
from pathlib import Path

from research.knowledge_base import (
    init_db,
    write_event,
    ingest_sweep,
    query,
    stats,
    export_events,
    get_sweep_status,
    check_experiment_exists,
)


class TestDatabaseInit:
    """Tests for database initialization."""

    def test_init_db_creates_tables(self):
        """Init DB should create all required tables."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            db = init_db(db_path)

            # Check tables exist
            tables = db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            table_names = {t['name'] for t in tables}

            assert 'events' in table_names
            assert 'sweeps' in table_names
            assert 'findings' in table_names
            assert 'coverage' in table_names
            assert 'artifacts' in table_names

            db.close()
        finally:
            os.unlink(db_path)

    def test_init_db_idempotent(self):
        """Init DB can be called multiple times without error."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            db1 = init_db(db_path)
            db1.close()

            # Should not raise
            db2 = init_db(db_path)
            db2.close()
        finally:
            os.unlink(db_path)


class TestEventWriting:
    """Tests for event writing."""

    def test_write_event(self):
        """Write event should create event row."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            db = init_db(db_path)
            event_id = write_event(
                db, 'SWEEP_STARTED', 'sweep_001', 'exp_123',
                status='RUNNING', payload={'foo': 'bar'}
            )

            # Verify event was written
            row = db.execute(
                "SELECT * FROM events WHERE event_id = ?", (event_id,)
            ).fetchone()

            assert row is not None
            assert row['event_type'] == 'SWEEP_STARTED'
            assert row['sweep_id'] == 'sweep_001'
            assert row['experiment_id'] == 'exp_123'
            assert row['status'] == 'RUNNING'
            assert json.loads(row['payload_json']) == {'foo': 'bar'}

            db.close()
        finally:
            os.unlink(db_path)


class TestIngestSweep:
    """Tests for sweep ingestion."""

    @pytest.fixture
    def temp_sweep_dir(self, tmp_path):
        """Create a temporary sweep directory with test data."""
        sweep_dir = tmp_path / 'sweep_test'
        sweep_dir.mkdir()

        # Create manifest
        manifest = {
            'sweep_name': 'sweep_test',
            'symbol': 'BTCUSDT',
            'data_range': {'start': '2025-01-01', 'end': '2025-06-01'},
            'data_fingerprint': 'abc123',
            'n_variants': 10,
            'n_passed_sanity': 8,
            'n_bh_survivors': 0,
        }
        (sweep_dir / 'manifest.json').write_text(json.dumps(manifest))

        # Create post_mortem
        post_mortem = {
            'sweep_id': 'sweep_test',
            'experiment_id': 'exp_test_123',
            'created_ts': '2025-01-01T00:00:00Z',
            'generated_ts': '2025-01-01T00:00:00Z',
            'summary': {
                'variant_count': 10,
                'survivor_count': 0,
                'best_variant': {
                    'variant_id': 'abc123',
                    'oos_sharpe': -5.0,
                }
            },
            'fee_decomposition': {
                'median_gross_return_per_trade': -0.002,
                'median_net_return_per_trade': -0.003,
                'estimated_fee_drag_per_trade': 0.0014,
                'fee_share_of_loss_pct': 35.0,
                'data_source': 'trade_summary',
            },
            'primary_failure_mode': 'FEE_DRAG',
            'failure_evidence': 'Test evidence',
            'next_experiment_constraints': ['constraint 1'],
        }
        (sweep_dir / 'post_mortem.json').write_text(json.dumps(post_mortem))

        # Create variants.jsonl
        variant = {
            'variant_id': 'abc123',
            'status': 'OK',
            'oos_sharpe': -5.0,
            'n_trades_oos': 100,
            'spec': {
                'entry': {'key': 'ema_cross', 'params': {'fast': 5}},
                'fee_rate': 0.0006,
                'slippage_rate': 0.0001,
            }
        }
        (sweep_dir / 'variants.jsonl').write_text(json.dumps(variant) + '\n')

        return sweep_dir

    def test_ingest_creates_sweep_row(self, temp_sweep_dir):
        """Ingest should create a row in sweeps table."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            db = init_db(db_path)
            ingest_sweep(db, str(temp_sweep_dir))

            row = db.execute(
                "SELECT * FROM sweeps WHERE sweep_id = 'sweep_test'"
            ).fetchone()

            assert row is not None
            assert row['experiment_id'] == 'exp_test_123'
            assert row['status'] == 'INGESTED'
            assert row['asset'] == 'BTCUSDT'
            assert row['primary_failure_mode'] == 'FEE_DRAG'

            db.close()
        finally:
            os.unlink(db_path)

    def test_ingest_creates_findings(self, temp_sweep_dir):
        """Ingest should create findings rows."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            db = init_db(db_path)
            ingest_sweep(db, str(temp_sweep_dir))

            rows = db.execute(
                "SELECT * FROM findings WHERE sweep_id = 'sweep_test'"
            ).fetchall()

            assert len(rows) >= 1

            db.close()
        finally:
            os.unlink(db_path)

    def test_ingest_creates_coverage(self, temp_sweep_dir):
        """Ingest should create coverage rows."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            db = init_db(db_path)
            ingest_sweep(db, str(temp_sweep_dir))

            rows = db.execute(
                "SELECT * FROM coverage WHERE sweep_id = 'sweep_test'"
            ).fetchall()

            # Should have asset, signal, fee_regime coverage
            entity_types = {r['entity_type'] for r in rows}
            assert 'asset' in entity_types
            assert 'signal' in entity_types

            db.close()
        finally:
            os.unlink(db_path)

    def test_ingest_creates_artifacts(self, temp_sweep_dir):
        """Ingest should create artifacts rows."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            db = init_db(db_path)
            ingest_sweep(db, str(temp_sweep_dir))

            rows = db.execute(
                "SELECT * FROM artifacts WHERE sweep_id = 'sweep_test'"
            ).fetchall()

            # Should have manifest, variants, post_mortem artifacts
            artifact_types = {r['artifact_type'] for r in rows}
            assert 'manifest' in artifact_types
            assert 'variants' in artifact_types
            assert 'post_mortem_json' in artifact_types

            db.close()
        finally:
            os.unlink(db_path)

    def test_ingest_idempotent(self, temp_sweep_dir):
        """Ingest twice should not crash (upsert)."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            db = init_db(db_path)
            ingest_sweep(db, str(temp_sweep_dir))
            ingest_sweep(db, str(temp_sweep_dir))  # Should not crash

            # Should still have only one sweep
            rows = db.execute(
                "SELECT * FROM sweeps WHERE sweep_id = 'sweep_test'"
            ).fetchall()
            assert len(rows) == 1

            db.close()
        finally:
            os.unlink(db_path)

    def test_ingest_writes_events(self, temp_sweep_dir):
        """Ingest should write KB_INGEST events."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            db = init_db(db_path)
            ingest_sweep(db, str(temp_sweep_dir), emit_events=True)

            rows = db.execute(
                "SELECT * FROM events WHERE sweep_id = 'sweep_test'"
            ).fetchall()

            event_types = {r['event_type'] for r in rows}
            assert 'KB_INGEST_STARTED' in event_types
            assert 'KB_INGEST_COMPLETED' in event_types

            db.close()
        finally:
            os.unlink(db_path)


class TestQuery:
    """Tests for query function."""

    def test_query_finds_sweep(self, tmp_path):
        """Query should find matching sweeps."""
        sweep_dir = tmp_path / 'sweep_ema'
        sweep_dir.mkdir()

        manifest = {
            'sweep_name': 'sweep_ema',
            'symbol': 'BTCUSDT',
            'data_range': {'start': '2025-01-01', 'end': '2025-06-01'},
        }
        (sweep_dir / 'manifest.json').write_text(json.dumps(manifest))

        post_mortem = {
            'sweep_id': 'sweep_ema',
            'experiment_id': 'exp_ema',
            'created_ts': '2025-01-01T00:00:00Z',
            'primary_failure_mode': 'LOW_SIGNAL',
            'summary': {'variant_count': 10, 'survivor_count': 0},
        }
        (sweep_dir / 'post_mortem.json').write_text(json.dumps(post_mortem))

        variant = {
            'variant_id': 'abc',
            'status': 'OK',
            'spec': {'entry': {'key': 'ema_cross'}, 'fee_rate': 0.0006}
        }
        (sweep_dir / 'variants.jsonl').write_text(json.dumps(variant) + '\n')

        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            db = init_db(db_path)
            ingest_sweep(db, str(sweep_dir))

            results = query(db, 'ema_cross')
            assert len(results) == 1
            assert results[0]['sweep_id'] == 'sweep_ema'

            db.close()
        finally:
            os.unlink(db_path)


class TestStats:
    """Tests for stats function."""

    def test_stats_returns_coverage(self, tmp_path):
        """Stats should return coverage summary."""
        sweep_dir = tmp_path / 'sweep_stats'
        sweep_dir.mkdir()

        manifest = {
            'sweep_name': 'sweep_stats',
            'symbol': 'ETHUSDT',
            'timeframes': ['4h'],
            'data_range': {'start': '2025-01-01', 'end': '2025-06-01'},
        }
        (sweep_dir / 'manifest.json').write_text(json.dumps(manifest))

        post_mortem = {
            'sweep_id': 'sweep_stats',
            'experiment_id': 'exp_stats',
            'created_ts': '2025-01-01T00:00:00Z',
            'primary_failure_mode': 'NO_CONVERGENCE',
            'summary': {'variant_count': 5, 'survivor_count': 0},
        }
        (sweep_dir / 'post_mortem.json').write_text(json.dumps(post_mortem))

        variant = {
            'variant_id': 'xyz',
            'status': 'OK',
            'spec': {'entry': {'key': 'rsi_entry'}, 'fee_rate': 0.001}
        }
        (sweep_dir / 'variants.jsonl').write_text(json.dumps(variant) + '\n')

        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            db = init_db(db_path)
            ingest_sweep(db, str(sweep_dir))

            s = stats(db)

            assert s['total_sweeps'] == 1
            assert any(a['name'] == 'ETHUSDT' for a in s['assets'])

            db.close()
        finally:
            os.unlink(db_path)


class TestExportEvents:
    """Tests for export_events function."""

    def test_export_events_chronological(self):
        """Export events should return chronological order."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            db = init_db(db_path)

            # Write events
            write_event(db, 'SWEEP_STARTED', 'sweep_1')
            write_event(db, 'SWEEP_COMPLETED', 'sweep_1')
            write_event(db, 'POST_MORTEM_STARTED', 'sweep_1')

            events = export_events(db, last_n=3)

            # Should be in chronological order (oldest first)
            assert events[0]['event_type'] == 'SWEEP_STARTED'
            assert events[1]['event_type'] == 'SWEEP_COMPLETED'
            assert events[2]['event_type'] == 'POST_MORTEM_STARTED'

            db.close()
        finally:
            os.unlink(db_path)


class TestIntegration:
    """Integration tests."""

    def test_full_pipeline(self, tmp_path):
        """Test full ingest -> query -> verify chain."""
        # Create sweep directory
        sweep_dir = tmp_path / 'sweep_full'
        sweep_dir.mkdir()

        manifest = {
            'sweep_name': 'sweep_full',
            'symbol': 'BTCUSDT',
            'data_range': {'start': '2025-01-01', 'end': '2025-06-01'},
        }
        (sweep_dir / 'manifest.json').write_text(json.dumps(manifest))

        post_mortem = {
            'sweep_id': 'sweep_full',
            'experiment_id': 'exp_full_123',
            'created_ts': '2025-01-01T00:00:00Z',
            'primary_failure_mode': 'FEE_DRAG',
            'summary': {'variant_count': 10, 'survivor_count': 0, 'best_variant': {'oos_sharpe': -5}},
            'fee_decomposition': {'fee_share_of_loss_pct': 40, 'data_source': 'config_only'},
        }
        (sweep_dir / 'post_mortem.json').write_text(json.dumps(post_mortem))

        variant = {
            'variant_id': 'full123',
            'status': 'OK',
            'spec': {'entry': {'key': 'macd_cross'}, 'fee_rate': 0.0006}
        }
        (sweep_dir / 'variants.jsonl').write_text(json.dumps(variant) + '\n')

        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            # Ingest
            db = init_db(db_path)
            ingest_sweep(db, str(sweep_dir))

            # Query
            results = query(db, 'macd_cross')
            assert len(results) == 1
            assert results[0]['sweep_id'] == 'sweep_full'
            assert results[0]['primary_failure_mode'] == 'FEE_DRAG'

            # Verify artifacts
            assert 'manifest' in results[0]['artifacts']

            # Stats
            s = stats(db)
            assert s['total_sweeps'] == 1

            db.close()
        finally:
            os.unlink(db_path)
