"""
Knowledge Base retrieval for hypothesis generation.

Extracts focused context from KB for LLM prompts.
"""

import logging
from typing import Any, Dict, List, Optional

from research.config import get
from research.knowledge_base import (
    count_sweeps,
    get_active_constraints,
    get_coverage_entities,
    get_latest_sweep,
    get_top_findings,
    init_db,
)
from research.signal_catalog import discover_signals, format_for_llm

logger = logging.getLogger(__name__)


def retrieve_context(db_path: str) -> Dict[str, Any]:
    """
    Retrieve relevant context from KB for hypothesis generation.

    Args:
        db_path: Path to the knowledge base SQLite file.

    Returns:
        Context dict with:
        - sweep_count: Number of sweeps in KB
        - latest_sweep: Most recent sweep info
        - top_findings: List of top findings
        - coverage_gaps: Entities not yet tested
        - active_constraints: Constraints from latest post-mortem
        - available_signals: Signal catalog
    """
    db = init_db(db_path)

    try:
        sweep_count = count_sweeps(db)
        latest_sweep = get_latest_sweep(db)

        # Get config values
        top_n_findings = get("kb_retrieval", "top_n_findings", 10)
        top_n_constraints = get("kb_retrieval", "top_n_constraints", 5)

        top_findings = get_top_findings(db, limit=top_n_findings)
        active_constraints = get_active_constraints(db)[:top_n_constraints]
        coverage = get_coverage_entities(db)

        # Get available signals
        signals = discover_signals()

        # Compute coverage gaps
        coverage_gaps = _compute_coverage_gaps(coverage, signals)

        return {
            "sweep_count": sweep_count,
            "latest_sweep": latest_sweep,
            "top_findings": top_findings,
            "coverage_gaps": coverage_gaps,
            "active_constraints": active_constraints,
            "available_signals": signals,
        }
    finally:
        db.close()


def _compute_coverage_gaps(
    coverage: Dict[str, List[str]],
    signals: Dict[str, Dict[str, Any]],
) -> List[Dict[str, str]]:
    """
    Identify coverage gaps - signals not yet tested.

    Args:
        coverage: Current coverage from KB.
        signals: Available signals from catalog.

    Returns:
        List of gaps, each with entity_type and entity_name.
    """
    gaps = []

    # Get tested signals
    tested_signals = set(coverage.get("signal", []))

    # Find untested signals
    for signal_key, signal_info in signals.items():
        if signal_key not in tested_signals:
            gaps.append({
                "entity_type": "signal",
                "entity_name": signal_key,
                "signal_type": signal_info.get("type", "unknown"),
            })

    return gaps


def format_context_for_llm(context: Dict[str, Any]) -> str:
    """
    Format context as human-readable text for LLM prompt.

    Respects token budget from config. Prioritizes:
    1. Active constraints (from latest post-mortem)
    2. Coverage gaps
    3. Top findings
    4. Available signals
    5. Latest sweep summary

    Args:
        context: Context dict from retrieve_context().

    Returns:
        Formatted text for LLM.
    """
    max_tokens = get("kb_retrieval", "max_context_tokens", 2000)
    # Estimate: 1 token ~= 4 characters
    max_chars = max_tokens * 4

    lines = []

    # Handle empty KB case
    if context["sweep_count"] == 0:
        lines.append("## Research Status")
        lines.append("")
        lines.append("**No previous sweeps.** This is the first experiment.")
        lines.append("")
        lines.append(format_for_llm(context["available_signals"]))
        return "\n".join(lines)

    # Build context with priority ordering
    sections = []

    # 1. Active constraints (highest priority)
    if context["active_constraints"]:
        section = ["## Active Constraints", ""]
        section.append("From the most recent sweep analysis:")
        for i, constraint in enumerate(context["active_constraints"], 1):
            section.append(f"{i}. {constraint}")
        section.append("")
        sections.append(("\n".join(section), 1))

    # 2. Coverage gaps
    if context["coverage_gaps"]:
        section = ["## Coverage Gaps", ""]
        section.append("Signals not yet tested:")
        for gap in context["coverage_gaps"][:5]:  # Limit to 5 gaps
            section.append(f"- {gap['entity_name']} ({gap['signal_type']})")
        section.append("")
        sections.append(("\n".join(section), 2))

    # 3. Top findings
    if context["top_findings"]:
        section = ["## Key Findings", ""]
        for finding in context["top_findings"][:5]:  # Limit to 5 findings
            sweep = finding.get("sweep_id", "unknown")
            statement = finding["statement"]
            section.append(f"- [{sweep}] {statement}")
        section.append("")
        sections.append(("\n".join(section), 3))

    # 4. Available signals
    signal_text = format_for_llm(context["available_signals"])
    sections.append((signal_text, 4))

    # 5. Latest sweep summary
    if context["latest_sweep"]:
        sweep = context["latest_sweep"]
        section = ["## Latest Sweep Summary", ""]
        section.append(f"- **Sweep ID**: {sweep['sweep_id']}")
        section.append(f"- **Status**: {sweep['status']}")
        section.append(f"- **Asset**: {sweep['asset']}")
        section.append(f"- **Timeframe**: {sweep['timeframe']}")
        if sweep.get("primary_failure_mode"):
            section.append(f"- **Failure Mode**: {sweep['primary_failure_mode']}")
        section.append("")
        sections.append(("\n".join(section), 5))

    # Assemble within budget
    # Sort by priority
    sections.sort(key=lambda x: x[1])

    current_chars = 0
    for section_text, _ in sections:
        if current_chars + len(section_text) <= max_chars:
            lines.append(section_text)
            current_chars += len(section_text)
        else:
            # Truncate if over budget
            remaining = max_chars - current_chars
            if remaining > 100:  # Only add if meaningful
                lines.append(section_text[:remaining] + "\n[truncated]")
            break

    return "\n".join(lines)
