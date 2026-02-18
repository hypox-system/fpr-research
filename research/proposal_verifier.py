"""
Proposal verifier for hypothesis validation.

Deterministic validation of HypothesisFX proposals.
All thresholds read from config - no hardcoded values.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import jsonschema
import yaml

from research.config import get
from research.knowledge_base import (
    check_experiment_exists,
    count_sweeps,
    get_proposal_by_experiment_id,
    init_db,
)
from research.signal_catalog import get_valid_signal_names

logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    """Result of a single verification check."""

    name: str
    passed: bool
    reason: Optional[str] = None


@dataclass
class VerificationResult:
    """Result of full proposal verification."""

    passed: bool
    checks: List[CheckResult] = field(default_factory=list)

    def add_check(self, name: str, passed: bool, reason: Optional[str] = None) -> None:
        """Add a check result."""
        self.checks.append(CheckResult(name=name, passed=passed, reason=reason))
        if not passed:
            self.passed = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization."""
        return {
            "passed": self.passed,
            "checks": [
                {"name": c.name, "passed": c.passed, "reason": c.reason}
                for c in self.checks
            ],
        }


def verify_proposal(proposal: Dict[str, Any], db_path: str) -> VerificationResult:
    """
    Verify a hypothesis proposal against all rules.

    Args:
        proposal: HypothesisFX proposal dict.
        db_path: Path to knowledge base.

    Returns:
        VerificationResult with pass/fail and check details.
    """
    result = VerificationResult(passed=True)
    db = init_db(db_path)

    try:
        # Run all checks
        _check_schema_valid(proposal, result)
        _check_intent_valid(proposal, result)
        _check_evidence_refs_exist(proposal, db, result)
        _check_evidence_refs_minimum(proposal, db, result)
        _check_predictions_complete(proposal, result)
        _check_sweep_config_valid(proposal, result)
        _check_experiment_not_duplicate(proposal, db, result)
        _check_budget_within_limits(proposal, result)
        _check_kill_criteria_present(proposal, result)
        _check_novelty_claim_non_empty(proposal, result)
        _check_signals_exist(proposal, result)
    finally:
        db.close()

    return result


def _get_schema() -> Dict[str, Any]:
    """Load HypothesisFX JSON schema."""
    schema_path = Path(__file__).parent / "schemas" / "hypothesis_fx.json"
    with open(schema_path) as f:
        return json.load(f)


def _check_schema_valid(proposal: Dict[str, Any], result: VerificationResult) -> None:
    """Check proposal against JSON schema."""
    try:
        schema = _get_schema()
        jsonschema.validate(proposal, schema)
        result.add_check("schema_valid", True)
    except jsonschema.ValidationError as e:
        result.add_check("schema_valid", False, str(e.message))
    except Exception as e:
        result.add_check("schema_valid", False, f"Schema error: {e}")


def _check_intent_valid(proposal: Dict[str, Any], result: VerificationResult) -> None:
    """Check experiment_intent is valid enum."""
    valid_intents = {"gap_fill", "failure_mitigation", "robustness", "regime_test"}
    intent = proposal.get("experiment_intent")

    if intent in valid_intents:
        result.add_check("intent_valid", True)
    else:
        result.add_check(
            "intent_valid",
            False,
            f"Invalid intent '{intent}'. Must be one of: {valid_intents}",
        )


def _check_evidence_refs_exist(
    proposal: Dict[str, Any],
    db: Any,
    result: VerificationResult,
) -> None:
    """Check all evidence_refs exist in KB."""
    evidence_refs = proposal.get("evidence_refs", [])

    if not evidence_refs:
        result.add_check("evidence_refs_exist", False, "No evidence_refs provided")
        return

    missing = []
    for ref in evidence_refs:
        # Check if sweep exists
        row = db.execute(
            "SELECT sweep_id FROM sweeps WHERE sweep_id = ?", (ref,)
        ).fetchone()
        if not row:
            missing.append(ref)

    if missing:
        result.add_check(
            "evidence_refs_exist",
            False,
            f"Sweep(s) not found in KB: {missing}",
        )
    else:
        result.add_check("evidence_refs_exist", True)


def _check_evidence_refs_minimum(
    proposal: Dict[str, Any],
    db: Any,
    result: VerificationResult,
) -> None:
    """Check minimum evidence refs based on bootstrap mode."""
    sweep_count = count_sweeps(db)
    bootstrap_threshold = get("verifier", "bootstrap_threshold", 3)

    if sweep_count < bootstrap_threshold:
        min_refs = get("verifier", "min_evidence_refs_bootstrap", 1)
    else:
        min_refs = get("verifier", "min_evidence_refs", 2)

    evidence_refs = proposal.get("evidence_refs", [])

    if len(evidence_refs) >= min_refs:
        result.add_check("evidence_refs_minimum", True)
    else:
        result.add_check(
            "evidence_refs_minimum",
            False,
            f"Need at least {min_refs} evidence refs, got {len(evidence_refs)}",
        )


def _check_predictions_complete(
    proposal: Dict[str, Any],
    result: VerificationResult,
) -> None:
    """Check predictions has all required numeric values."""
    predictions = proposal.get("predictions", {})
    required_fields = [
        "trade_duration_median_bars",
        "trades_per_day",
        "gross_minus_net_gap",
    ]

    missing = []
    invalid = []

    for field in required_fields:
        if field not in predictions:
            missing.append(field)
        elif not isinstance(predictions[field], (int, float)):
            invalid.append(field)

    if missing:
        result.add_check(
            "predictions_complete",
            False,
            f"Missing prediction fields: {missing}",
        )
    elif invalid:
        result.add_check(
            "predictions_complete",
            False,
            f"Non-numeric prediction fields: {invalid}",
        )
    else:
        result.add_check("predictions_complete", True)


def _check_sweep_config_valid(
    proposal: Dict[str, Any],
    result: VerificationResult,
) -> None:
    """Check sweep_config_yaml is valid YAML."""
    config_yaml = proposal.get("sweep_config_yaml", "")

    if not config_yaml:
        result.add_check("sweep_config_valid", False, "Empty sweep_config_yaml")
        return

    try:
        config = yaml.safe_load(config_yaml)
        if not isinstance(config, dict):
            result.add_check(
                "sweep_config_valid",
                False,
                "sweep_config_yaml must be a YAML dict",
            )
            return

        # Basic validation: should have at least name and symbol
        if "name" not in config and "symbol" not in config:
            result.add_check(
                "sweep_config_valid",
                False,
                "sweep_config_yaml missing 'name' or 'symbol'",
            )
            return

        result.add_check("sweep_config_valid", True)
    except yaml.YAMLError as e:
        result.add_check("sweep_config_valid", False, f"Invalid YAML: {e}")


def _check_experiment_not_duplicate(
    proposal: Dict[str, Any],
    db: Any,
    result: VerificationResult,
) -> None:
    """Check experiment_id doesn't already exist in KB."""
    # Try to compute experiment_id from proposal config
    experiment_id = proposal.get("experiment_id")

    if not experiment_id:
        # No pre-computed ID, can't check for duplicates yet
        result.add_check("experiment_not_duplicate", True)
        return

    # Check sweeps table
    existing_sweep = check_experiment_exists(db, experiment_id)
    if existing_sweep:
        result.add_check(
            "experiment_not_duplicate",
            False,
            f"Experiment {experiment_id} already exists as sweep {existing_sweep['sweep_id']}",
        )
        return

    # Check proposals table
    existing_proposal = get_proposal_by_experiment_id(db, experiment_id)
    if existing_proposal:
        result.add_check(
            "experiment_not_duplicate",
            False,
            f"Experiment {experiment_id} already has proposal {existing_proposal['proposal_id']}",
        )
        return

    result.add_check("experiment_not_duplicate", True)


def _check_budget_within_limits(
    proposal: Dict[str, Any],
    result: VerificationResult,
) -> None:
    """Check compute_budget is within configured limits."""
    budget = proposal.get("compute_budget", {})

    max_variants_limit = get("verifier", "max_variants", 500)
    max_runtime_limit = get("verifier", "max_runtime_minutes", 120)

    variants = budget.get("max_variants", 0)
    runtime = budget.get("max_runtime_minutes", 0)

    errors = []
    if variants > max_variants_limit:
        errors.append(f"max_variants {variants} > limit {max_variants_limit}")
    if runtime > max_runtime_limit:
        errors.append(f"max_runtime_minutes {runtime} > limit {max_runtime_limit}")

    if errors:
        result.add_check("budget_within_limits", False, "; ".join(errors))
    else:
        result.add_check("budget_within_limits", True)


def _check_kill_criteria_present(
    proposal: Dict[str, Any],
    result: VerificationResult,
) -> None:
    """Check minimum kill criteria are present."""
    kill_criteria = proposal.get("kill_criteria", [])
    min_criteria = get("verifier", "min_kill_criteria", 1)

    if len(kill_criteria) >= min_criteria:
        result.add_check("kill_criteria_present", True)
    else:
        result.add_check(
            "kill_criteria_present",
            False,
            f"Need at least {min_criteria} kill criteria, got {len(kill_criteria)}",
        )


def _check_novelty_claim_non_empty(
    proposal: Dict[str, Any],
    result: VerificationResult,
) -> None:
    """Check novelty_claim.coverage_diff is non-empty."""
    novelty_claim = proposal.get("novelty_claim", {})
    coverage_diff = novelty_claim.get("coverage_diff", [])

    if coverage_diff:
        result.add_check("novelty_claim_non_empty", True)
    else:
        result.add_check(
            "novelty_claim_non_empty",
            False,
            "novelty_claim.coverage_diff is empty",
        )


def _check_signals_exist(
    proposal: Dict[str, Any],
    result: VerificationResult,
) -> None:
    """Check all signals in sweep_config exist in catalog."""
    config_yaml = proposal.get("sweep_config_yaml", "")

    if not config_yaml:
        result.add_check("signals_exist", True)  # Already failed in config check
        return

    try:
        config = yaml.safe_load(config_yaml)
    except yaml.YAMLError:
        result.add_check("signals_exist", True)  # Already failed in config check
        return

    if not isinstance(config, dict):
        result.add_check("signals_exist", True)
        return

    valid_signals = set(get_valid_signal_names())
    unknown_signals = []

    # Check entry signal
    entry = config.get("entry", {})
    if isinstance(entry, dict) and entry.get("key"):
        if entry["key"] not in valid_signals:
            unknown_signals.append(entry["key"])

    # Check exit signal
    exit_config = config.get("exit", {})
    if isinstance(exit_config, dict) and exit_config.get("key"):
        if exit_config["key"] not in valid_signals:
            unknown_signals.append(exit_config["key"])

    # Check filters
    filters = config.get("filters", [])
    if isinstance(filters, list):
        for f in filters:
            if isinstance(f, dict) and f.get("key"):
                if f["key"] not in valid_signals:
                    unknown_signals.append(f["key"])

    if unknown_signals:
        result.add_check(
            "signals_exist",
            False,
            f"Unknown signals: {unknown_signals}. Available: {sorted(valid_signals)}",
        )
    else:
        result.add_check("signals_exist", True)
