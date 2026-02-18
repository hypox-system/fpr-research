"""
Knowledge Base for FPR Research Platform.

CRUD + query interface for research.db.

Usage:
    python -m research.knowledge_base ingest results/sweeps/sweep_001_ema_pvsra/
    python -m research.knowledge_base query "ema_cross"
    python -m research.knowledge_base stats
    python -m research.knowledge_base export-events --last 20
"""

import hashlib
import json
import logging
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def get_schema_path() -> Path:
    """Get path to kb_schema.sql."""
    return Path(__file__).parent / 'kb_schema.sql'


def init_db(db_path: str) -> sqlite3.Connection:
    """
    Initialize database with schema.

    Args:
        db_path: Path to SQLite database file.

    Returns:
        SQLite connection.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Read and execute schema
    schema_path = get_schema_path()
    with open(schema_path, 'r') as f:
        schema_sql = f.read()

    conn.executescript(schema_sql)
    conn.commit()

    return conn


def write_event(
    db: sqlite3.Connection,
    event_type: str,
    sweep_id: Optional[str] = None,
    experiment_id: Optional[str] = None,
    status: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Write an event to the events table (append-only).

    Args:
        db: SQLite connection.
        event_type: Type of event (e.g., 'SWEEP_STARTED').
        sweep_id: Associated sweep ID.
        experiment_id: Associated experiment ID.
        status: Status string.
        payload: Additional JSON payload.

    Returns:
        Generated event_id.
    """
    event_id = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    payload_json = json.dumps(payload) if payload else None

    db.execute(
        """
        INSERT INTO events (event_id, ts, event_type, sweep_id, experiment_id, status, payload_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (event_id, ts, event_type, sweep_id, experiment_id, status, payload_json)
    )
    db.commit()

    logger.debug(f"Event written: {event_type} for sweep {sweep_id}")
    return event_id


def _compute_file_hash(file_path: str) -> Tuple[str, int]:
    """Compute SHA-256 hash and size of a file."""
    sha256 = hashlib.sha256()
    size = 0
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
            size += len(chunk)
    return sha256.hexdigest(), size


def ingest_sweep(
    db: sqlite3.Connection,
    sweep_dir_path: str,
    emit_events: bool = True,
) -> str:
    """
    Ingest a sweep from disk into the knowledge base.

    Reads manifest.json + post_mortem.json, creates/updates:
    - sweeps row (upsert)
    - findings rows
    - coverage rows
    - artifacts rows

    Idempotent: upsert on sweep_id.

    Args:
        db: SQLite connection.
        sweep_dir_path: Path to sweep directory.
        emit_events: Whether to emit KB_INGEST events (default True).
                    Set False when called from sweep_runner (which emits its own).

    Returns:
        sweep_id

    Raises:
        FileNotFoundError: If manifest.json not found.
        ValueError: If experiment_id already exists with COMPLETED/ANALYZED/INGESTED status.
    """
    sweep_dir = Path(sweep_dir_path)
    manifest_path = sweep_dir / 'manifest.json'
    post_mortem_path = sweep_dir / 'post_mortem.json'
    variants_path = sweep_dir / 'variants.jsonl'

    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in {sweep_dir}")

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    sweep_id = manifest.get('sweep_name', sweep_dir.name)

    # Load post_mortem if exists
    post_mortem = None
    if post_mortem_path.exists():
        with open(post_mortem_path, 'r') as f:
            post_mortem = json.load(f)

    experiment_id = post_mortem.get('experiment_id') if post_mortem else manifest.get('experiment_id')
    if not experiment_id:
        # Generate from config_hash if not present
        experiment_id = manifest.get('config_hash', '')[:16]

    # Check for duplicate experiment_id
    existing = db.execute(
        "SELECT sweep_id, status FROM sweeps WHERE experiment_id = ?",
        (experiment_id,)
    ).fetchone()

    if existing and existing['sweep_id'] != sweep_id:
        if existing['status'] in ('COMPLETED', 'ANALYZED', 'INGESTED'):
            logger.warning(
                f"Experiment {experiment_id} already exists as sweep {existing['sweep_id']} "
                f"with status {existing['status']}. Skipping ingest."
            )
            return sweep_id

    if emit_events:
        write_event(db, 'KB_INGEST_STARTED', sweep_id, experiment_id)

    # Extract data from manifest
    data_range = manifest.get('data_range', {})
    symbol = manifest.get('symbol', 'UNKNOWN')

    # Get timeframe from config
    timeframe = '1h'  # Default
    if isinstance(manifest.get('timeframes'), list) and manifest['timeframes']:
        timeframe = manifest['timeframes'][0]

    # Get fees (manifest stores as decimal, we store as bps)
    # fees might be in sweep config, check for fee_rate in various places
    fee_rate = 0.0006  # Default
    slippage_rate = 0.0001  # Default

    # Check sharpe_annualization for hints
    # Or check first variant for fee_rate
    if variants_path.exists():
        with open(variants_path, 'r') as f:
            first_line = f.readline()
            if first_line:
                first_variant = json.loads(first_line)
                if 'spec' in first_variant:
                    fee_rate = first_variant['spec'].get('fee_rate', fee_rate)
                    slippage_rate = first_variant['spec'].get('slippage_rate', slippage_rate)

    # Prepare sweep data
    status = 'INGESTED'
    # Get created_ts from post_mortem, or fallback to now
    if post_mortem and post_mortem.get('created_ts'):
        created_ts = post_mortem['created_ts']
    else:
        created_ts = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')

    best_metrics = None
    primary_failure_mode = None
    if post_mortem:
        best_variant = post_mortem.get('summary', {}).get('best_variant')
        if best_variant:
            best_metrics = json.dumps(best_variant)
        primary_failure_mode = post_mortem.get('primary_failure_mode')

    # Upsert sweep
    db.execute(
        """
        INSERT INTO sweeps (
            sweep_id, experiment_id, status, created_ts, asset, timeframe,
            date_range_start, date_range_end, fee_bps, slippage_bps,
            config_json, best_metrics_json, primary_failure_mode
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(sweep_id) DO UPDATE SET
            status = excluded.status,
            best_metrics_json = excluded.best_metrics_json,
            primary_failure_mode = excluded.primary_failure_mode
        """,
        (
            sweep_id,
            experiment_id,
            status,
            created_ts,
            symbol,
            timeframe,
            data_range.get('start'),
            data_range.get('end'),
            fee_rate * 10000,  # Convert to bps
            slippage_rate * 10000,  # Convert to bps
            json.dumps(manifest),
            best_metrics,
            primary_failure_mode,
        )
    )

    # Clear existing findings/coverage/artifacts for this sweep (for idempotent re-ingest)
    db.execute("DELETE FROM findings WHERE sweep_id = ?", (sweep_id,))
    db.execute("DELETE FROM coverage WHERE sweep_id = ?", (sweep_id,))
    db.execute("DELETE FROM artifacts WHERE sweep_id = ?", (sweep_id,))

    # Create findings from post_mortem
    if post_mortem:
        _create_findings_from_post_mortem(db, sweep_id, post_mortem)

    # Create coverage entries
    _create_coverage_entries(db, sweep_id, manifest, variants_path)

    # Create artifact entries
    _create_artifact_entries(db, sweep_id, sweep_dir)

    db.commit()

    if emit_events:
        write_event(db, 'KB_INGEST_COMPLETED', sweep_id, experiment_id, status='INGESTED')

    logger.info(f"Ingested sweep {sweep_id} (experiment: {experiment_id})")
    return sweep_id


def _create_findings_from_post_mortem(
    db: sqlite3.Connection,
    sweep_id: str,
    post_mortem: Dict[str, Any],
) -> None:
    """Create findings from post_mortem data."""
    findings = []

    # Finding 1: Primary failure mode
    failure_mode = post_mortem.get('primary_failure_mode')
    if failure_mode:
        findings.append({
            'statement': f"Primary failure mode: {failure_mode}",
            'tags': ['failure_mode', failure_mode.lower()],
            'confidence': 0.9,
            'evidence': post_mortem.get('failure_evidence'),
        })

    # Finding 2: Summary verdict
    summary = post_mortem.get('summary', {})
    survivor_count = summary.get('survivor_count', 0)
    variant_count = summary.get('variant_count', 0)
    if survivor_count == 0 and variant_count > 0:
        findings.append({
            'statement': f"No BH-FDR survivors from {variant_count} variants tested",
            'tags': ['dead', 'no_survivors'],
            'confidence': 1.0,
            'evidence': None,
        })

    # Finding 3: Fee decomposition insight
    fee_decomp = post_mortem.get('fee_decomposition', {})
    fee_share = fee_decomp.get('fee_share_of_loss_pct')
    if fee_share is not None and fee_share > 30:
        findings.append({
            'statement': f"Fee drag accounts for {fee_share:.1f}% of losses",
            'tags': ['fee_drag', 'cost_analysis'],
            'confidence': 0.85,
            'evidence': json.dumps(fee_decomp),
        })

    # Insert findings
    for finding in findings:
        finding_id = str(uuid.uuid4())
        db.execute(
            """
            INSERT INTO findings (finding_id, sweep_id, statement, tags_json, confidence, evidence_refs_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                finding_id,
                sweep_id,
                finding['statement'],
                json.dumps(finding['tags']),
                finding['confidence'],
                finding['evidence'],
            )
        )


def _create_coverage_entries(
    db: sqlite3.Connection,
    sweep_id: str,
    manifest: Dict[str, Any],
    variants_path: Path,
) -> None:
    """Create coverage entries from manifest and variants."""
    coverage_entries = []

    # Asset coverage
    symbol = manifest.get('symbol')
    if symbol:
        coverage_entries.append(('asset', symbol))

    # Timeframe coverage
    timeframes = manifest.get('timeframes', [])
    if isinstance(timeframes, str):
        timeframes = [timeframes]
    for tf in timeframes:
        coverage_entries.append(('timeframe', tf))

    # Fee regime coverage
    # Try to get fee from sharpe_annualization or default
    fee_rate = 0.0006
    if variants_path.exists():
        with open(variants_path, 'r') as f:
            first_line = f.readline()
            if first_line:
                first_variant = json.loads(first_line)
                if 'spec' in first_variant:
                    fee_rate = first_variant['spec'].get('fee_rate', fee_rate)
    coverage_entries.append(('fee_regime', f"{fee_rate * 100:.2f}%"))

    # Signal coverage (from variants)
    signals_seen = set()
    if variants_path.exists():
        with open(variants_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    variant = json.loads(line)
                    spec = variant.get('spec', {})
                    # Entry signal
                    entry = spec.get('entry', {})
                    if entry.get('key'):
                        signals_seen.add(('signal', entry['key']))
                    # Filter signals
                    for filt in spec.get('filters', []):
                        if filt.get('key'):
                            signals_seen.add(('signal', filt['key']))
                    # Exit signals
                    for exit_sig in spec.get('exits', []):
                        if exit_sig.get('key'):
                            signals_seen.add(('signal', exit_sig['key']))
                except json.JSONDecodeError:
                    continue

    coverage_entries.extend(signals_seen)

    # Insert coverage
    for entity_type, entity_name in coverage_entries:
        db.execute(
            "INSERT INTO coverage (entity_type, entity_name, sweep_id) VALUES (?, ?, ?)",
            (entity_type, entity_name, sweep_id)
        )


def _create_artifact_entries(
    db: sqlite3.Connection,
    sweep_id: str,
    sweep_dir: Path,
) -> None:
    """Create artifact entries for files in sweep directory."""
    artifact_files = [
        ('manifest', 'manifest.json'),
        ('variants', 'variants.jsonl'),
        ('post_mortem_json', 'post_mortem.json'),
        ('post_mortem_md', 'post_mortem.md'),
        ('leaderboard', 'leaderboard.md'),
        ('trade_data', 'trade_data.npz'),
    ]

    for artifact_type, filename in artifact_files:
        file_path = sweep_dir / filename
        if file_path.exists():
            sha256, size = _compute_file_hash(str(file_path))
            created_ts = datetime.fromtimestamp(file_path.stat().st_mtime).isoformat() + 'Z'

            db.execute(
                """
                INSERT INTO artifacts (sweep_id, artifact_type, path, sha256, bytes, created_ts)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (sweep_id, artifact_type, str(file_path), sha256, size, created_ts)
            )


def query(
    db: sqlite3.Connection,
    search_term: str,
) -> List[Dict[str, Any]]:
    """
    Search findings, tags, and coverage for matches.

    Args:
        db: SQLite connection.
        search_term: Term to search for.

    Returns:
        List of matching sweeps with status, failure_mode, best metric, artifact paths.
    """
    search_pattern = f"%{search_term}%"

    # Find matching sweep IDs
    rows = db.execute(
        """
        SELECT DISTINCT s.sweep_id, s.status, s.primary_failure_mode, s.best_metrics_json,
               s.asset, s.timeframe, s.date_range_start, s.date_range_end
        FROM sweeps s
        LEFT JOIN findings f ON s.sweep_id = f.sweep_id
        LEFT JOIN coverage c ON s.sweep_id = c.sweep_id
        WHERE f.statement LIKE ? OR f.tags_json LIKE ? OR c.entity_name LIKE ?
        ORDER BY s.created_ts DESC
        """,
        (search_pattern, search_pattern, search_pattern)
    ).fetchall()

    results = []
    for row in rows:
        sweep_id = row['sweep_id']

        # Get artifacts
        artifacts = db.execute(
            "SELECT artifact_type, path FROM artifacts WHERE sweep_id = ?",
            (sweep_id,)
        ).fetchall()
        artifact_paths = {a['artifact_type']: a['path'] for a in artifacts}

        # Get findings
        findings = db.execute(
            "SELECT statement FROM findings WHERE sweep_id = ?",
            (sweep_id,)
        ).fetchall()

        results.append({
            'sweep_id': sweep_id,
            'status': row['status'],
            'primary_failure_mode': row['primary_failure_mode'],
            'best_metrics': json.loads(row['best_metrics_json']) if row['best_metrics_json'] else None,
            'asset': row['asset'],
            'timeframe': row['timeframe'],
            'date_range': f"{row['date_range_start']} to {row['date_range_end']}",
            'artifacts': artifact_paths,
            'findings': [f['statement'] for f in findings],
        })

    return results


def stats(db: sqlite3.Connection) -> Dict[str, Any]:
    """
    Get coverage summary.

    Args:
        db: SQLite connection.

    Returns:
        Dict with timeframes, fee_regimes, signals, assets tested.
    """
    result = {
        'timeframes': [],
        'fee_regimes': [],
        'signals': [],
        'assets': [],
        'total_sweeps': 0,
        'total_findings': 0,
    }

    # Get counts
    row = db.execute("SELECT COUNT(*) as count FROM sweeps").fetchone()
    result['total_sweeps'] = row['count']

    row = db.execute("SELECT COUNT(*) as count FROM findings").fetchone()
    result['total_findings'] = row['count']

    # Get coverage by entity type
    for entity_type in ['timeframe', 'fee_regime', 'signal', 'asset']:
        rows = db.execute(
            """
            SELECT DISTINCT entity_name, COUNT(*) as count
            FROM coverage
            WHERE entity_type = ?
            GROUP BY entity_name
            ORDER BY count DESC
            """,
            (entity_type,)
        ).fetchall()

        key = entity_type + 's' if not entity_type.endswith('e') else entity_type + 's'
        if entity_type == 'fee_regime':
            key = 'fee_regimes'
        result[key] = [{'name': r['entity_name'], 'count': r['count']} for r in rows]

    return result


def export_events(
    db: sqlite3.Connection,
    last_n: int = 20,
) -> List[Dict[str, Any]]:
    """
    Export recent events.

    Args:
        db: SQLite connection.
        last_n: Number of events to return.

    Returns:
        List of events in chronological order.
    """
    rows = db.execute(
        """
        SELECT event_id, ts, event_type, sweep_id, experiment_id, status, payload_json
        FROM events
        ORDER BY ts DESC
        LIMIT ?
        """,
        (last_n,)
    ).fetchall()

    events = []
    for row in rows:
        events.append({
            'event_id': row['event_id'],
            'ts': row['ts'],
            'event_type': row['event_type'],
            'sweep_id': row['sweep_id'],
            'experiment_id': row['experiment_id'],
            'status': row['status'],
            'payload': json.loads(row['payload_json']) if row['payload_json'] else None,
        })

    # Return in chronological order
    return list(reversed(events))


def get_sweep_status(
    db: sqlite3.Connection,
    sweep_id: str,
) -> Optional[str]:
    """
    Get status for a sweep, derived from events table.

    Args:
        db: SQLite connection.
        sweep_id: Sweep ID.

    Returns:
        Status string or None if not found.
    """
    # Get the most recent event for this sweep
    row = db.execute(
        """
        SELECT event_type, status
        FROM events
        WHERE sweep_id = ?
        ORDER BY ts DESC
        LIMIT 1
        """,
        (sweep_id,)
    ).fetchone()

    if not row:
        return None

    # Map event type to status
    event_type = row['event_type']
    if row['status']:
        return row['status']

    status_map = {
        'SWEEP_STARTED': 'RUNNING',
        'SWEEP_COMPLETED': 'COMPLETED',
        'POST_MORTEM_STARTED': 'COMPLETED',
        'POST_MORTEM_COMPLETED': 'ANALYZED',
        'KB_INGEST_STARTED': 'ANALYZED',
        'KB_INGEST_COMPLETED': 'INGESTED',
        'STEP_FAILED': 'NEEDS_REVIEW',
    }

    return status_map.get(event_type, 'UNKNOWN')


def check_experiment_exists(
    db: sqlite3.Connection,
    experiment_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Check if an experiment_id already exists.

    Args:
        db: SQLite connection.
        experiment_id: Experiment ID to check.

    Returns:
        Dict with sweep_id and status if exists, None otherwise.
    """
    row = db.execute(
        "SELECT sweep_id, status FROM sweeps WHERE experiment_id = ?",
        (experiment_id,)
    ).fetchone()

    if row:
        return {'sweep_id': row['sweep_id'], 'status': row['status']}
    return None


# ============================================================================
# Proposal CRUD (Fas 2)
# ============================================================================


def write_proposal(
    db: sqlite3.Connection,
    proposal_id: str,
    experiment_intent: str,
    proposed_config: Dict[str, Any],
    evidence_refs: List[str],
    novelty_claim: Dict[str, Any],
    predictions: Dict[str, Any],
    expected_mechanism: str,
    kill_criteria: List[str],
    compute_budget: Dict[str, Any],
    experiment_id: Optional[str] = None,
    rationale_md: Optional[str] = None,
    expected_failure_mode: Optional[str] = None,
    status: str = 'PENDING',
) -> str:
    """
    Write a proposal to the proposals table.

    Args:
        db: SQLite connection.
        proposal_id: Unique ID for the proposal.
        experiment_intent: Type of experiment (gap_fill, failure_mitigation, etc.).
        proposed_config: YAML config as dict.
        evidence_refs: List of sweep_ids that inform this proposal.
        novelty_claim: Dict with coverage_diff, near_dup_score.
        predictions: Dict with predicted metrics.
        expected_mechanism: Explanation of expected mechanism.
        kill_criteria: List of conditions that would invalidate the hypothesis.
        compute_budget: Dict with max_variants, max_runtime_minutes.
        experiment_id: Pre-computed experiment fingerprint.
        rationale_md: Optional markdown rationale.
        expected_failure_mode: Most likely failure mode.
        status: Proposal status (default PENDING).

    Returns:
        proposal_id
    """
    created_ts = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')

    db.execute(
        """
        INSERT INTO proposals (
            proposal_id, created_ts, status, experiment_intent,
            proposed_config_json, experiment_id, rationale_md,
            evidence_refs_json, novelty_claim_json, predictions_json,
            expected_mechanism, expected_failure_mode, kill_criteria_json,
            compute_budget_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            proposal_id,
            created_ts,
            status,
            experiment_intent,
            json.dumps(proposed_config),
            experiment_id,
            rationale_md,
            json.dumps(evidence_refs),
            json.dumps(novelty_claim),
            json.dumps(predictions),
            expected_mechanism,
            expected_failure_mode,
            json.dumps(kill_criteria),
            json.dumps(compute_budget),
        )
    )
    db.commit()

    logger.info(f"Written proposal {proposal_id} with status {status}")
    return proposal_id


def get_proposals(
    db: sqlite3.Connection,
    status: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """
    Get proposals, optionally filtered by status.

    Args:
        db: SQLite connection.
        status: Optional status filter (PENDING, APPROVED, REJECTED, EXPIRED).
        limit: Maximum number of proposals to return.

    Returns:
        List of proposal dicts.
    """
    if status:
        rows = db.execute(
            """
            SELECT * FROM proposals
            WHERE status = ?
            ORDER BY created_ts DESC
            LIMIT ?
            """,
            (status, limit)
        ).fetchall()
    else:
        rows = db.execute(
            """
            SELECT * FROM proposals
            ORDER BY created_ts DESC
            LIMIT ?
            """,
            (limit,)
        ).fetchall()

    proposals = []
    for row in rows:
        proposals.append(_row_to_proposal(row))

    return proposals


def get_proposal(
    db: sqlite3.Connection,
    proposal_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Get a single proposal by ID.

    Args:
        db: SQLite connection.
        proposal_id: Proposal ID.

    Returns:
        Proposal dict or None if not found.
    """
    row = db.execute(
        "SELECT * FROM proposals WHERE proposal_id = ?",
        (proposal_id,)
    ).fetchone()

    if row:
        return _row_to_proposal(row)
    return None


def get_proposal_by_experiment_id(
    db: sqlite3.Connection,
    experiment_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Get a proposal by experiment_id.

    Args:
        db: SQLite connection.
        experiment_id: Experiment fingerprint.

    Returns:
        Proposal dict or None if not found.
    """
    row = db.execute(
        "SELECT * FROM proposals WHERE experiment_id = ?",
        (experiment_id,)
    ).fetchone()

    if row:
        return _row_to_proposal(row)
    return None


def _row_to_proposal(row: sqlite3.Row) -> Dict[str, Any]:
    """Convert a database row to a proposal dict."""
    return {
        'proposal_id': row['proposal_id'],
        'created_ts': row['created_ts'],
        'status': row['status'],
        'experiment_intent': row['experiment_intent'],
        'proposed_config': json.loads(row['proposed_config_json']),
        'experiment_id': row['experiment_id'],
        'rationale_md': row['rationale_md'],
        'evidence_refs': json.loads(row['evidence_refs_json']),
        'novelty_claim': json.loads(row['novelty_claim_json']),
        'predictions': json.loads(row['predictions_json']),
        'expected_mechanism': row['expected_mechanism'],
        'expected_failure_mode': row['expected_failure_mode'],
        'kill_criteria': json.loads(row['kill_criteria_json']),
        'compute_budget': json.loads(row['compute_budget_json']),
        'prediction_audit': json.loads(row['prediction_audit_json']) if row['prediction_audit_json'] else None,
    }


def update_proposal_status(
    db: sqlite3.Connection,
    proposal_id: str,
    status: str,
    reason: Optional[str] = None,
) -> bool:
    """
    Update proposal status.

    Args:
        db: SQLite connection.
        proposal_id: Proposal ID.
        status: New status (PENDING, APPROVED, REJECTED, EXPIRED).
        reason: Optional reason for status change.

    Returns:
        True if updated, False if proposal not found.
    """
    result = db.execute(
        "UPDATE proposals SET status = ? WHERE proposal_id = ?",
        (status, proposal_id)
    )
    db.commit()

    if result.rowcount > 0:
        # Log event
        write_event(
            db,
            f'PROPOSAL_{status}',
            experiment_id=proposal_id,
            status=status,
            payload={'reason': reason} if reason else None,
        )
        logger.info(f"Updated proposal {proposal_id} to status {status}")
        return True

    return False


def write_prediction_audit(
    db: sqlite3.Connection,
    proposal_id: str,
    audit_data: Dict[str, Any],
) -> bool:
    """
    Write prediction audit results to a proposal.

    Args:
        db: SQLite connection.
        proposal_id: Proposal ID.
        audit_data: Audit results dict.

    Returns:
        True if updated, False if proposal not found.
    """
    result = db.execute(
        "UPDATE proposals SET prediction_audit_json = ? WHERE proposal_id = ?",
        (json.dumps(audit_data), proposal_id)
    )
    db.commit()

    if result.rowcount > 0:
        logger.info(f"Written prediction audit for proposal {proposal_id}")
        return True

    return False


def count_sweeps(db: sqlite3.Connection) -> int:
    """
    Count total number of sweeps in KB.

    Args:
        db: SQLite connection.

    Returns:
        Number of sweeps.
    """
    row = db.execute("SELECT COUNT(*) as count FROM sweeps").fetchone()
    return row['count']


def get_latest_sweep(db: sqlite3.Connection) -> Optional[Dict[str, Any]]:
    """
    Get the most recent sweep.

    Args:
        db: SQLite connection.

    Returns:
        Sweep dict or None if no sweeps exist.
    """
    row = db.execute(
        """
        SELECT * FROM sweeps
        ORDER BY created_ts DESC
        LIMIT 1
        """
    ).fetchone()

    if not row:
        return None

    return {
        'sweep_id': row['sweep_id'],
        'experiment_id': row['experiment_id'],
        'status': row['status'],
        'created_ts': row['created_ts'],
        'asset': row['asset'],
        'timeframe': row['timeframe'],
        'date_range_start': row['date_range_start'],
        'date_range_end': row['date_range_end'],
        'primary_failure_mode': row['primary_failure_mode'],
        'config': json.loads(row['config_json']),
        'best_metrics': json.loads(row['best_metrics_json']) if row['best_metrics_json'] else None,
    }


def get_top_findings(
    db: sqlite3.Connection,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """
    Get top findings by confidence.

    Args:
        db: SQLite connection.
        limit: Maximum number of findings.

    Returns:
        List of finding dicts.
    """
    rows = db.execute(
        """
        SELECT f.*, s.sweep_id as sweep_ref
        FROM findings f
        JOIN sweeps s ON f.sweep_id = s.sweep_id
        ORDER BY f.confidence DESC, s.created_ts DESC
        LIMIT ?
        """,
        (limit,)
    ).fetchall()

    findings = []
    for row in rows:
        findings.append({
            'finding_id': row['finding_id'],
            'sweep_id': row['sweep_id'],
            'statement': row['statement'],
            'tags': json.loads(row['tags_json']) if row['tags_json'] else [],
            'confidence': row['confidence'],
            'evidence': row['evidence_refs_json'],
        })

    return findings


def get_active_constraints(db: sqlite3.Connection) -> List[str]:
    """
    Get active constraints from the most recent sweep's post-mortem.

    Args:
        db: SQLite connection.

    Returns:
        List of constraint strings.
    """
    # Get latest sweep directory from artifacts
    row = db.execute(
        """
        SELECT a.path
        FROM artifacts a
        JOIN sweeps s ON a.sweep_id = s.sweep_id
        WHERE a.artifact_type = 'post_mortem_json'
        ORDER BY s.created_ts DESC
        LIMIT 1
        """
    ).fetchone()

    if not row:
        return []

    try:
        with open(row['path'], 'r') as f:
            post_mortem = json.load(f)
        return post_mortem.get('next_experiment_constraints', [])
    except Exception as e:
        logger.warning(f"Failed to load constraints from post_mortem: {e}")
        return []


def get_coverage_entities(db: sqlite3.Connection) -> Dict[str, List[str]]:
    """
    Get all covered entities grouped by type.

    Args:
        db: SQLite connection.

    Returns:
        Dict mapping entity_type to list of entity_names.
    """
    rows = db.execute(
        """
        SELECT DISTINCT entity_type, entity_name
        FROM coverage
        ORDER BY entity_type, entity_name
        """
    ).fetchall()

    coverage = {}
    for row in rows:
        entity_type = row['entity_type']
        if entity_type not in coverage:
            coverage[entity_type] = []
        coverage[entity_type].append(row['entity_name'])

    return coverage


# CLI
def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='FPR Research Knowledge Base')
    parser.add_argument('--db', default='research.db', help='Database path')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest a sweep')
    ingest_parser.add_argument('sweep_dir', help='Path to sweep directory')

    # query command
    query_parser = subparsers.add_parser('query', help='Search knowledge base')
    query_parser.add_argument('search_term', help='Search term')

    # stats command
    subparsers.add_parser('stats', help='Show coverage statistics')

    # export-events command
    events_parser = subparsers.add_parser('export-events', help='Export recent events')
    events_parser.add_argument('--last', type=int, default=20, help='Number of events')

    args = parser.parse_args()

    # Initialize DB
    db = init_db(args.db)

    if args.command == 'ingest':
        try:
            sweep_id = ingest_sweep(db, args.sweep_dir)
            print(f"Ingested: {sweep_id}")
        except Exception as e:
            print(f"Error: {e}")
            raise

    elif args.command == 'query':
        results = query(db, args.search_term)
        if not results:
            print(f"No results for '{args.search_term}'")
        else:
            for r in results:
                print(f"\n=== {r['sweep_id']} ===")
                print(f"Status: {r['status']}")
                print(f"Failure Mode: {r['primary_failure_mode']}")
                print(f"Asset: {r['asset']} | Timeframe: {r['timeframe']}")
                print(f"Date Range: {r['date_range']}")
                if r['best_metrics']:
                    print(f"Best Metrics: {json.dumps(r['best_metrics'], indent=2)}")
                if r['findings']:
                    print("Findings:")
                    for f in r['findings']:
                        print(f"  - {f}")
                if r['artifacts']:
                    print("Artifacts:")
                    for atype, path in r['artifacts'].items():
                        print(f"  - {atype}: {path}")

    elif args.command == 'stats':
        s = stats(db)
        print(f"\n=== Knowledge Base Statistics ===")
        print(f"Total Sweeps: {s['total_sweeps']}")
        print(f"Total Findings: {s['total_findings']}")
        print(f"\nTimeframes tested:")
        for t in s['timeframes']:
            print(f"  - {t['name']}: {t['count']} sweeps")
        print(f"\nFee regimes tested:")
        for f in s['fee_regimes']:
            print(f"  - {f['name']}: {f['count']} sweeps")
        print(f"\nSignals tested:")
        for sig in s['signals']:
            print(f"  - {sig['name']}: {sig['count']} sweeps")
        print(f"\nAssets tested:")
        for a in s['assets']:
            print(f"  - {a['name']}: {a['count']} sweeps")

    elif args.command == 'export-events':
        events = export_events(db, args.last)
        if not events:
            print("No events found")
        else:
            print(f"\n=== Last {len(events)} Events ===")
            for e in events:
                status_str = f" [{e['status']}]" if e['status'] else ""
                print(f"{e['ts']} | {e['event_type']}{status_str} | {e['sweep_id'] or '-'}")
                if e['payload']:
                    print(f"    Payload: {json.dumps(e['payload'])}")

    db.close()


if __name__ == '__main__':
    main()
