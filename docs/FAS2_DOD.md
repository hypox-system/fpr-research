# Fas 2 — Definition of Done

**Status:** ACTIVE  
**Scope:** Fas 2 — Hypothesis Pipeline (Config System + Signal Catalog + KB Retrieval + LLM Generator + Verifier + Prediction Audit)  
**Senast uppdaterad:** 2026-02-18

---

## A. Globala principer (gäller fortfarande från fas 1)

**Fail-closed:** Om ett steg fallerar loggas det med event + anledning. Inga tysta fel.  
**Determinism:** Givet samma KB-state + samma LLM-svar → samma resultat (exkl timestamps).  
**Infra vs Config:** All beteendestyrning lever i `research/config/lab.yaml` och `research/prompts/`. Koden är mekanik.

---

## B. Config System

### B1) Config loader

**Done när:**

- `research/config.py` finns med `get_config()`, `get()`, `reset()`
- `research/config/lab.yaml` innehåller alla sektioner: `post_mortem`, `verifier`, `hypothesis`, `kb_retrieval`
- Override via `FPR_LAB_CONFIG` env var fungerar
- Alla fas 2-moduler läser config härifrån — inga hårdkodade tröskelvärden i Python

### B2) Post-mortem config-migration

**Done när:**

- Om `analysis/post_mortem.py` har hårdkodade tröskelvärden: refaktorera till `config.get('post_mortem', ...)`
- Befintliga tester passerar fortfarande

---

## C. Signal Catalog

### C1) Discovery

**Done när:**

- `research/signal_catalog.py` finns
- `discover_signals()` returnerar alla registrerade signaler med namn, typ, params
- `get_valid_signal_names()` returnerar lista med strängar
- `format_for_llm()` returnerar human-readable text

---

## D. KB Migration

### D1) Proposals-tabell

**Done när:**

- `proposals` CREATE TABLE finns i `kb_schema.sql`
- `init_db()` skapar tabellen utan att bryta existerande tabeller
- CRUD: `write_proposal()`, `get_proposals()`, `update_proposal_status()`, `write_prediction_audit()`
- `api.py` wrappers: `get_proposals()`, `get_proposal()`, `approve_proposal()`, `reject_proposal()`

---

## E. KB Retrieval

### E1) Context extraction

**Done när:**

- `research/kb_retrieval.py` finns
- `retrieve_context(db_path)` returnerar: sweep_count, latest_sweep, top_findings, coverage_gaps, active_constraints, available_signals
- `format_context_for_llm(context)` returnerar text inom token-budget (config)
- Med tom KB: returnerar signal-katalog + first-experiment-indikation
- Med sweep_001: returnerar findings, constraints, gaps

---

## F. LLM Client

### F1) Provider abstraction

**Done när:**

- `research/llm_client.py` finns med `LLMClient` class
- Providers: `anthropic`, `ollama`, `mock`
- `mock` returnerar valid HypothesisFX JSON som passerar verifiering
- API-fel → `LLMError` exception (inte crash, inte retry)
- Provider + model konfigureras via `lab.yaml`

---

## G. Hypothesis Generator

### G1) Pipeline controller

**Done när:**

- `research/hypothesis_gen.py` finns
- `run(db_path, dry_run, provider)` kör hela flödet: retrieval → LLM → parse → verify → store
- JSON-parsing hanterar ```json wrapper, text före/efter, malformaterat svar
- Events loggas: `HYPOTHESIS_GEN_STARTED`, `HYPOTHESIS_GEN_COMPLETED`, `PROPOSAL_CREATED` / `PROPOSAL_REJECTED`
- LLM-fel → event `STEP_FAILED`, ingen crash

### G2) CLI

**Done när:**

- `python -m research.hypothesis_gen run [--dry-run] [--provider mock]` fungerar
- `python -m research.hypothesis_gen verify <file>` validerar en proposal-fil
- `python -m research.hypothesis_gen show [--status PENDING]` visar proposals

---

## H. Proposal Verifier

### H1) Deterministisk validering

**Done när:**

- `research/proposal_verifier.py` finns
- `verify_proposal(proposal, db_path)` returnerar `VerificationResult` med pass/fail + checks
- Alla checks i spec implementerade (schema, intent, evidence, predictions, config, dedup, budget, kill_criteria, novelty, signals)
- Thresholds läses från config

### H2) Bootstrap-logik

**Done när:**

- Om KB har < `bootstrap_threshold` sweeps: acceptera `min_evidence_refs_bootstrap` refs
- Om KB har ≥ threshold: kräv `min_evidence_refs` refs
- Tröskelvärden konfigurerbara

---

## I. JSON Schema

### I1) HypothesisFX

**Done när:**

- `research/schemas/hypothesis_fx.json` finns och är valid JSON Schema (draft-07)
- Verifier använder schemat via `jsonschema.validate()`
- Alla 9 fält definierade med rätt typer och krav

---

## J. LLM Prompt

### J1) System prompt

**Done när:**

- `research/prompts/hypothesis_system.md` finns
- Innehåller: rollbeskrivning, kontraktskrav, output-format, regler
- Laddas av `hypothesis_gen.py` via config-path
- Funkar med mock LLM (prompten behöver inte vara perfekt, men måste finnas)

---

## K. Prediction Audit

### K1) Hook i finalize_sweep

**Done när:**

- `run_prediction_audit(sweep_dir, db_path, sweep_id)` finns
- Körs mellan post_mortem och KB ingest i `finalize_sweep()`
- Om proposal finns: beräknar diff, skriver prediction_audit_json, skapar finding
- Om ingen proposal finns: skip med event `PREDICTION_AUDIT_SKIPPED`
- Audit-failure blockerar INTE ingest (logga varning)

---

## L. Testkrav

**Done när alla testkategorier passerar:**

**Config:**

- Default config laddar alla sektioner
- Override via custom path
- Reset rensar cache

**Signal Catalog:**

- Hittar befintliga signaler (ema_cross, ema_stop_long)
- Format-for-LLM producerar text

**KB Retrieval:**

- Med data: returnerar findings, gaps, constraints
- Utan data: returnerar signal-katalog + first-experiment
- Token-budget respekteras

**Hypothesis Gen:**

- Mock LLM → PENDING proposal i DB
- Invalid LLM response → event, ingen crash
- Dry-run → ingen DB-write
- E2E integration

**Verifier:**

- Valid → pass
- Saknat fält → fail
- Duplicate experiment_id → fail
- Over-budget → fail
- Bad evidence ref → fail
- Bootstrap mode → 1 ref godkänns
- Invalid signal → fail

**Prediction Audit:**

- Proposal + post_mortem → audit JSON
- Ingen proposal → skip
- Audit failure → varning, inte block

**Coverage:** ≥80% på nya kärnmoduler.

---

## M. Manuell acceptans (beviskedja)

**H1:**

```bash
python -m research.hypothesis_gen run --provider mock
```

→ Skapar PENDING proposal i DB med alla 9 HypothesisFX-fält

**H2:**
Skapa en malformad proposal-JSON (ta bort `predictions`). Kör:

```bash
python -m research.hypothesis_gen verify path/to/bad_proposal.json
```

→ Visar specifik rejection-anledning

**H3:**
Ändra `verifier.max_variants: 10` i `lab.yaml`. Kör:

```bash
python -m research.hypothesis_gen run --provider mock
```

→ Proposal rejectas pga budget (mock föreslår >10 variants)

Återställ config.

**H4:**

```bash
python -c "from research.signal_catalog import discover_signals; print(discover_signals())"
```

→ Visar ema_cross + ema_stop_long med params

**H5:**

```bash
# Full E2E (sweep_001 redan i KB):
python -m research.hypothesis_gen run --provider mock
python -m research.hypothesis_gen show --status PENDING
python -m research.knowledge_base export-events --last 30
```

→ Events visar: HYPOTHESIS_GEN_STARTED → HYPOTHESIS_GEN_COMPLETED → PROPOSAL_CREATED
→ Proposal refererar sweep_001 i evidence_refs

**H6:**

```bash
# Prediction audit (kräver mock-data):
# Se test_prediction_audit.py för fixture-setup
pytest tests/test_prediction_audit.py -v
```

→ Alla audit-tester passerar

---

## N. Inte Done om...

- Config-värden är hårdkodade i Python (utöver `get()` default-parameter)
- LLM-prompt är en sträng i Python-koden (inte en separat fil)
- Signal-katalogen är en manuell lista (inte introspection)
- Verifier använder LLM för validering
- Pipeline crashar vid dåligt LLM-svar
- Proposals saknas i DB efter lyckad run
- Events saknar `HYPOTHESIS_GEN_*` eller `PROPOSAL_*` typer
- Bootstrap-logik saknas (pipeline kan aldrig generera första förslaget)
- Prediction audit blockerar ingest vid fel
