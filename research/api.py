"""
Internal Python API for FPR Research Platform.

TUI/web interfaces use this API, never DB directly.

Fas 1: Basic wrappers around knowledge_base.py.
Fas 2 will add proposal management.
"""

from typing import Any, Dict, List, Optional

from research.knowledge_base import (
    init_db,
    query as kb_query,
    stats as kb_stats,
    export_events as kb_export_events,
    get_sweep_status,
)


def get_feed(
    db_path: str,
    limit: int = 50,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """
    Get recent activity feed.

    Args:
        db_path: Path to research.db.
        limit: Max items to return.
        offset: Number of items to skip.

    Returns:
        List of events/activities.
    """
    db = init_db(db_path)
    try:
        # Get recent events
        rows = db.execute(
            """
            SELECT event_id, ts, event_type, sweep_id, experiment_id, status, payload_json
            FROM events
            ORDER BY ts DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset)
        ).fetchall()

        import json
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
        return events
    finally:
        db.close()


def get_sweep(
    db_path: str,
    sweep_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Get details for a single sweep.

    Args:
        db_path: Path to research.db.
        sweep_id: Sweep ID.

    Returns:
        Sweep details dict or None if not found.
    """
    db = init_db(db_path)
    try:
        import json

        row = db.execute(
            """
            SELECT sweep_id, experiment_id, status, created_ts, asset, timeframe,
                   date_range_start, date_range_end, fee_bps, slippage_bps,
                   config_json, best_metrics_json, primary_failure_mode
            FROM sweeps
            WHERE sweep_id = ?
            """,
            (sweep_id,)
        ).fetchone()

        if not row:
            return None

        # Get findings
        findings = db.execute(
            "SELECT finding_id, statement, tags_json, confidence FROM findings WHERE sweep_id = ?",
            (sweep_id,)
        ).fetchall()

        # Get artifacts
        artifacts = db.execute(
            "SELECT artifact_type, path, sha256, bytes FROM artifacts WHERE sweep_id = ?",
            (sweep_id,)
        ).fetchall()

        # Get coverage
        coverage = db.execute(
            "SELECT entity_type, entity_name FROM coverage WHERE sweep_id = ?",
            (sweep_id,)
        ).fetchall()

        return {
            'sweep_id': row['sweep_id'],
            'experiment_id': row['experiment_id'],
            'status': row['status'],
            'created_ts': row['created_ts'],
            'asset': row['asset'],
            'timeframe': row['timeframe'],
            'date_range': {
                'start': row['date_range_start'],
                'end': row['date_range_end'],
            },
            'fee_bps': row['fee_bps'],
            'slippage_bps': row['slippage_bps'],
            'config': json.loads(row['config_json']) if row['config_json'] else None,
            'best_metrics': json.loads(row['best_metrics_json']) if row['best_metrics_json'] else None,
            'primary_failure_mode': row['primary_failure_mode'],
            'findings': [
                {
                    'finding_id': f['finding_id'],
                    'statement': f['statement'],
                    'tags': json.loads(f['tags_json']) if f['tags_json'] else [],
                    'confidence': f['confidence'],
                }
                for f in findings
            ],
            'artifacts': [
                {
                    'type': a['artifact_type'],
                    'path': a['path'],
                    'sha256': a['sha256'],
                    'bytes': a['bytes'],
                }
                for a in artifacts
            ],
            'coverage': [
                {'type': c['entity_type'], 'name': c['entity_name']}
                for c in coverage
            ],
        }
    finally:
        db.close()


def get_coverage(
    db_path: str,
    entity_type: str,
    entity_name: str,
) -> List[Dict[str, Any]]:
    """
    Get sweeps that cover a specific entity.

    Args:
        db_path: Path to research.db.
        entity_type: Type of entity (e.g., 'signal', 'asset').
        entity_name: Name of entity (e.g., 'ema_cross', 'BTCUSDT').

    Returns:
        List of sweep summaries.
    """
    db = init_db(db_path)
    try:
        rows = db.execute(
            """
            SELECT DISTINCT s.sweep_id, s.status, s.primary_failure_mode, s.created_ts
            FROM sweeps s
            JOIN coverage c ON s.sweep_id = c.sweep_id
            WHERE c.entity_type = ? AND c.entity_name = ?
            ORDER BY s.created_ts DESC
            """,
            (entity_type, entity_name)
        ).fetchall()

        return [
            {
                'sweep_id': r['sweep_id'],
                'status': r['status'],
                'primary_failure_mode': r['primary_failure_mode'],
                'created_ts': r['created_ts'],
            }
            for r in rows
        ]
    finally:
        db.close()


def get_findings(
    db_path: str,
    tags: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Get findings, optionally filtered by tags.

    Args:
        db_path: Path to research.db.
        tags: Optional list of tags to filter by.

    Returns:
        List of findings.
    """
    db = init_db(db_path)
    try:
        import json

        if tags:
            # Search for any matching tag
            placeholders = ','.join(['?' for _ in tags])
            tag_patterns = [f'%"{t}"%' for t in tags]

            # Build OR condition for each tag
            conditions = ' OR '.join(['f.tags_json LIKE ?' for _ in tags])

            rows = db.execute(
                f"""
                SELECT f.finding_id, f.sweep_id, f.statement, f.tags_json, f.confidence,
                       s.status, s.primary_failure_mode
                FROM findings f
                JOIN sweeps s ON f.sweep_id = s.sweep_id
                WHERE {conditions}
                ORDER BY s.created_ts DESC
                """,
                tag_patterns
            ).fetchall()
        else:
            rows = db.execute(
                """
                SELECT f.finding_id, f.sweep_id, f.statement, f.tags_json, f.confidence,
                       s.status, s.primary_failure_mode
                FROM findings f
                JOIN sweeps s ON f.sweep_id = s.sweep_id
                ORDER BY s.created_ts DESC
                """
            ).fetchall()

        return [
            {
                'finding_id': r['finding_id'],
                'sweep_id': r['sweep_id'],
                'statement': r['statement'],
                'tags': json.loads(r['tags_json']) if r['tags_json'] else [],
                'confidence': r['confidence'],
                'sweep_status': r['status'],
                'sweep_failure_mode': r['primary_failure_mode'],
            }
            for r in rows
        ]
    finally:
        db.close()


def get_proposals(
    db_path: str,
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Get proposals (Fas 2 feature).

    Args:
        db_path: Path to research.db.
        status: Optional status filter.

    Returns:
        Empty list in Fas 1.
    """
    # Fas 1: No proposals table yet
    return []


def export_events(
    db_path: str,
    last_n: int = 20,
) -> List[Dict[str, Any]]:
    """
    Export recent events.

    Args:
        db_path: Path to research.db.
        last_n: Number of events to return.

    Returns:
        List of events in chronological order.
    """
    db = init_db(db_path)
    try:
        return kb_export_events(db, last_n)
    finally:
        db.close()
