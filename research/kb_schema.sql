-- Knowledge Base Schema for FPR Research Platform
-- Version: Fas 1

-- Events table: append-only event log
CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    ts TEXT NOT NULL,
    event_type TEXT NOT NULL,
    sweep_id TEXT,
    experiment_id TEXT,
    status TEXT,
    payload_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_sweep ON events(sweep_id);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_experiment ON events(experiment_id);

-- Sweeps table: one row per sweep
CREATE TABLE IF NOT EXISTS sweeps (
    sweep_id TEXT PRIMARY KEY,
    experiment_id TEXT UNIQUE NOT NULL,
    status TEXT NOT NULL,
    created_ts TEXT NOT NULL,
    asset TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    date_range_start TEXT,
    date_range_end TEXT,
    fee_bps REAL,
    slippage_bps REAL,
    config_json TEXT NOT NULL,
    best_metrics_json TEXT,
    primary_failure_mode TEXT
);
CREATE INDEX IF NOT EXISTS idx_sweeps_experiment ON sweeps(experiment_id);
CREATE INDEX IF NOT EXISTS idx_sweeps_status ON sweeps(status);

-- Findings table: extracted insights from sweeps
CREATE TABLE IF NOT EXISTS findings (
    finding_id TEXT PRIMARY KEY,
    sweep_id TEXT NOT NULL REFERENCES sweeps(sweep_id),
    statement TEXT NOT NULL,
    tags_json TEXT,
    confidence REAL,
    evidence_refs_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_findings_sweep ON findings(sweep_id);

-- Coverage table: what has been tested
CREATE TABLE IF NOT EXISTS coverage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type TEXT NOT NULL,
    entity_name TEXT NOT NULL,
    sweep_id TEXT NOT NULL REFERENCES sweeps(sweep_id)
);
CREATE INDEX IF NOT EXISTS idx_coverage_entity ON coverage(entity_type, entity_name);
CREATE INDEX IF NOT EXISTS idx_coverage_sweep ON coverage(sweep_id);

-- Artifacts table: files associated with sweeps
CREATE TABLE IF NOT EXISTS artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sweep_id TEXT NOT NULL REFERENCES sweeps(sweep_id),
    artifact_type TEXT NOT NULL,
    path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    bytes INTEGER,
    created_ts TEXT
);
CREATE INDEX IF NOT EXISTS idx_artifacts_sweep ON artifacts(sweep_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_type ON artifacts(artifact_type);
