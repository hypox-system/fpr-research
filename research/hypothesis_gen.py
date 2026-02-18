"""
Hypothesis generation pipeline.

Orchestrates: KB retrieval → LLM → parsing → verification → storage.

Usage:
    python -m research.hypothesis_gen run [--dry-run] [--provider mock]
    python -m research.hypothesis_gen verify <file>
    python -m research.hypothesis_gen show [--status PENDING]
"""

import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from research.config import get, get_config
from research.kb_retrieval import format_context_for_llm, retrieve_context
from research.knowledge_base import (
    get_proposals,
    init_db,
    write_event,
    write_proposal,
)
from research.llm_client import LLMClient, LLMError
from research.proposal_verifier import verify_proposal
from utils.experiment_fingerprint import compute_experiment_id

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "research.db"


def run(
    db_path: str = DEFAULT_DB_PATH,
    dry_run: bool = False,
    provider: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the hypothesis generation pipeline.

    Flow:
    1. Retrieve context from KB
    2. Load system prompt
    3. Build user prompt with context
    4. Call LLM
    5. Parse JSON response
    6. Compute experiment_id
    7. Verify proposal
    8. Store if valid (unless dry_run)

    Args:
        db_path: Path to knowledge base.
        dry_run: If True, don't write to DB.
        provider: Override LLM provider (default from config).

    Returns:
        Dict with status, proposal_id, reason.
    """
    db = init_db(db_path)

    try:
        # Log start
        write_event(db, "HYPOTHESIS_GEN_STARTED")

        # 1. Retrieve context
        logger.info("Retrieving KB context...")
        context = retrieve_context(db_path)

        # 2. Load system prompt
        prompt_path = get("hypothesis", "prompt_path", "research/prompts/hypothesis_system.md")
        system_prompt = _load_prompt(prompt_path)

        # 3. Build user prompt
        user_prompt = format_context_for_llm(context)

        # 4. Call LLM
        logger.info("Calling LLM...")
        try:
            client = LLMClient(provider=provider)
            raw_response = client.generate(system_prompt, user_prompt)
        except LLMError as e:
            logger.error(f"LLM error: {e}")
            write_event(
                db,
                "STEP_FAILED",
                payload={"step": "llm_call", "error": str(e)},
            )
            return {
                "status": "failed",
                "proposal_id": None,
                "reason": f"LLM error: {e}",
            }

        # 5. Parse JSON response
        logger.info("Parsing response...")
        try:
            proposal = _parse_llm_response(raw_response)
        except ValueError as e:
            logger.error(f"Parse error: {e}")
            write_event(
                db,
                "STEP_FAILED",
                payload={
                    "step": "parse_response",
                    "error": str(e),
                    "raw_response": raw_response[:1000],
                },
            )
            return {
                "status": "failed",
                "proposal_id": None,
                "reason": f"Parse error: {e}",
            }

        # 6. Compute experiment_id
        experiment_id = _compute_proposal_experiment_id(proposal)
        proposal["experiment_id"] = experiment_id

        # 7. Verify proposal
        logger.info("Verifying proposal...")
        verification = verify_proposal(proposal, db_path)

        if not verification.passed:
            failed_checks = [
                c for c in verification.checks if not c.passed
            ]
            reasons = [f"{c.name}: {c.reason}" for c in failed_checks]
            reason_str = "; ".join(reasons)

            logger.warning(f"Proposal rejected: {reason_str}")

            if not dry_run:
                # Store rejected proposal for audit
                proposal_id = str(uuid.uuid4())[:12]
                write_proposal(
                    db,
                    proposal_id=proposal_id,
                    experiment_intent=proposal.get("experiment_intent", "unknown"),
                    proposed_config=_extract_config(proposal),
                    evidence_refs=proposal.get("evidence_refs", []),
                    novelty_claim=proposal.get("novelty_claim", {}),
                    predictions=proposal.get("predictions", {}),
                    expected_mechanism=proposal.get("expected_mechanism", ""),
                    kill_criteria=proposal.get("kill_criteria", []),
                    compute_budget=proposal.get("compute_budget", {}),
                    experiment_id=experiment_id,
                    expected_failure_mode=proposal.get("expected_failure_mode"),
                    status="REJECTED",
                )
                write_event(
                    db,
                    "PROPOSAL_REJECTED",
                    experiment_id=experiment_id,
                    payload={"reason": reason_str, "checks": verification.to_dict()},
                )

            write_event(db, "HYPOTHESIS_GEN_COMPLETED", status="rejected")
            return {
                "status": "rejected",
                "proposal_id": None,
                "reason": reason_str,
            }

        # 8. Store proposal
        proposal_id = str(uuid.uuid4())[:12]

        if not dry_run:
            write_proposal(
                db,
                proposal_id=proposal_id,
                experiment_intent=proposal["experiment_intent"],
                proposed_config=_extract_config(proposal),
                evidence_refs=proposal["evidence_refs"],
                novelty_claim=proposal["novelty_claim"],
                predictions=proposal["predictions"],
                expected_mechanism=proposal["expected_mechanism"],
                kill_criteria=proposal["kill_criteria"],
                compute_budget=proposal["compute_budget"],
                experiment_id=experiment_id,
                expected_failure_mode=proposal.get("expected_failure_mode"),
                status="PENDING",
            )
            write_event(
                db,
                "PROPOSAL_CREATED",
                experiment_id=experiment_id,
                payload={"proposal_id": proposal_id},
            )
            logger.info(f"Proposal {proposal_id} created with status PENDING")
        else:
            logger.info(f"[DRY-RUN] Would create proposal {proposal_id}")

        write_event(db, "HYPOTHESIS_GEN_COMPLETED", status="accepted")
        return {
            "status": "accepted",
            "proposal_id": proposal_id,
            "reason": None,
        }

    finally:
        db.close()


def verify_file(file_path: str, db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """
    Verify a proposal from a JSON file.

    Args:
        file_path: Path to proposal JSON file.
        db_path: Path to knowledge base.

    Returns:
        Verification result dict.
    """
    with open(file_path) as f:
        proposal = json.load(f)

    result = verify_proposal(proposal, db_path)
    return result.to_dict()


def show_proposals(
    db_path: str = DEFAULT_DB_PATH,
    status: Optional[str] = None,
    limit: int = 10,
) -> None:
    """
    Display proposals from DB.

    Args:
        db_path: Path to knowledge base.
        status: Filter by status.
        limit: Maximum proposals to show.
    """
    db = init_db(db_path)
    try:
        proposals = get_proposals(db, status=status, limit=limit)

        if not proposals:
            print(f"No proposals found" + (f" with status {status}" if status else ""))
            return

        for p in proposals:
            print(f"\n{'='*60}")
            print(f"Proposal: {p['proposal_id']}")
            print(f"Status: {p['status']}")
            print(f"Created: {p['created_ts']}")
            print(f"Intent: {p['experiment_intent']}")
            print(f"Experiment ID: {p['experiment_id']}")
            print(f"Evidence Refs: {p['evidence_refs']}")
            print(f"Novelty: {p['novelty_claim'].get('coverage_diff', [])}")
            print(f"Predictions:")
            for k, v in p['predictions'].items():
                print(f"  - {k}: {v}")
            print(f"Kill Criteria: {p['kill_criteria']}")
            print(f"Budget: {p['compute_budget']}")
    finally:
        db.close()


def _load_prompt(prompt_path: str) -> str:
    """Load system prompt from file."""
    path = Path(prompt_path)
    if not path.exists():
        # Try relative to project root
        path = Path(__file__).parent.parent / prompt_path
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return path.read_text()


def _parse_llm_response(raw_response: str) -> Dict[str, Any]:
    """
    Parse LLM response to extract JSON.

    Handles:
    - Clean JSON
    - JSON wrapped in ```json ... ```
    - Text before/after JSON

    Args:
        raw_response: Raw text from LLM.

    Returns:
        Parsed proposal dict.

    Raises:
        ValueError: If JSON cannot be extracted or parsed.
    """
    text = raw_response.strip()

    # Try to extract JSON from markdown code block
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if json_match:
        text = json_match.group(1).strip()

    # Try to find JSON object boundaries
    if not text.startswith("{"):
        start = text.find("{")
        if start >= 0:
            text = text[start:]

    if not text.endswith("}"):
        end = text.rfind("}")
        if end >= 0:
            text = text[: end + 1]

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}")


def _extract_config(proposal: Dict[str, Any]) -> Dict[str, Any]:
    """Extract sweep config from proposal."""
    config_yaml = proposal.get("sweep_config_yaml", "")
    try:
        return yaml.safe_load(config_yaml) or {}
    except yaml.YAMLError:
        return {"raw_yaml": config_yaml}


def _compute_proposal_experiment_id(proposal: Dict[str, Any]) -> str:
    """Compute experiment_id from proposal config."""
    config = _extract_config(proposal)

    # Build minimal sweep config for fingerprint
    sweep_config = {
        "name": config.get("name", "unnamed"),
        "entry": config.get("entry", {}),
        "exit": config.get("exit", {}),
        "filters": config.get("filters", []),
    }

    # Build minimal data manifest
    data_manifest = {
        "symbol": config.get("symbol", "UNKNOWN"),
        "timeframes": config.get("timeframes", []),
        "date_range": config.get("date_range", {}),
    }

    # Build fill model
    fill_model = {
        "fee_rate": config.get("fee_rate", 0.0006),
        "slippage_rate": config.get("slippage_rate", 0.0001),
    }

    return compute_experiment_id(sweep_config, data_manifest, fill_model)


# CLI
def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="FPR Hypothesis Generator")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="Database path")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run command
    run_parser = subparsers.add_parser("run", help="Run hypothesis generation")
    run_parser.add_argument(
        "--dry-run", action="store_true", help="Don't write to DB"
    )
    run_parser.add_argument(
        "--provider", help="Override LLM provider (mock, anthropic, ollama)"
    )

    # verify command
    verify_parser = subparsers.add_parser("verify", help="Verify a proposal file")
    verify_parser.add_argument("file", help="Path to proposal JSON file")

    # show command
    show_parser = subparsers.add_parser("show", help="Show proposals")
    show_parser.add_argument("--status", help="Filter by status")
    show_parser.add_argument("--limit", type=int, default=10, help="Max proposals")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if args.command == "run":
        result = run(
            db_path=args.db,
            dry_run=args.dry_run,
            provider=args.provider,
        )
        print(f"\nResult: {result['status']}")
        if result["proposal_id"]:
            print(f"Proposal ID: {result['proposal_id']}")
        if result["reason"]:
            print(f"Reason: {result['reason']}")

    elif args.command == "verify":
        result = verify_file(args.file, args.db)
        print(f"\nVerification: {'PASSED' if result['passed'] else 'FAILED'}")
        for check in result["checks"]:
            status = "✓" if check["passed"] else "✗"
            print(f"  {status} {check['name']}", end="")
            if check["reason"]:
                print(f": {check['reason']}")
            else:
                print()

    elif args.command == "show":
        show_proposals(db_path=args.db, status=args.status, limit=args.limit)


if __name__ == "__main__":
    main()
