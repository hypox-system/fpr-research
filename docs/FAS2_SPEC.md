# Fas 2 — SPEC (CC Build Prompt)

**Status:** READY  
**Target:** Claude Code  
**Repo:** `github.com/hypox-system/fpr-research.git`  
**Branch:** Skapa `feat/lab-fas2-hypothesis` från `main`  
**Föregående:** Fas 1 levererad och mergad. 157 tester passerar. Tag: `v3.0.0`  
**DoD:** Se sibling-page "Fas 2 — Definition of Done"

---

## Arbetssätt (VIKTIGT — läs först)

Du är autonom. Jobba själv tills allt är klart. Fråga INTE användaren.

**Regler:**

1. **Kör alla kommandon själv.** `pytest`, `python -m`, `git`, etc. Fråga aldrig "ska jag köra X?". Kör det.
2. **Fixa fel själv.** Om ett test failar: läs output, fixa, kör igen. Iterera tills grönt. Fråga inte.
3. **Specen är din auktoritet.** Om svaret står i denna prompt eller DoD: följ den. Fråga inte för bekräftelse.
4. **Vid osäkerhet: välj enklare alternativet.** Om två approaches är rimliga, välj den med färre dependencies och mindre scope. Dokumentera valet i en kommentar.
5. **Säg aldrig "ska jag fortsätta?".** Du fortsätter tills DoD är uppfylld.
6. **Kör `pytest` efter varje ny fil.** Inte efter du är "klar" — efter varje fil. Fånga regressioner tidigt.
7. **Kör befintliga tester först** (`pytest tests/ -x`). Om något redan är trasigt, rapportera det — men fortsätt med ditt arbete.
8. **Inspektera innan du antar.** Läs faktisk kod innan du beslutar om implementation. Gissa inte.
9. **Leverera allt i en session.** Brancha, bygg, testa, rapportera resultat. En commit-kedja.
10. **När du är klar:** kör alla H-steg (H1–H6 i DoD) och visa output. Det är ditt "proof of done".

**Enda gången du får fråga:** Om befintlig kod är trasig på ett sätt som blockerar ditt arbete OCH du inte kan fixa det utan att bryta scope.

---

## Designprincip: Infrastruktur vs Konfiguration

**Denna princip genomsyrar hela fas 2.**

Koden du bygger är **infrastruktur** — pipelines, validering, LLM-klient, schema-laddning. Den ska vara stabil och generell.

Allt som styr _beteende_ är **konfiguration** — tröskelvärden, prompts, regler, modellval. Det ska leva i filer som byts utan kodändring.

**Konkret:**

- Verifieringsregler → `research/config/lab.yaml`, inte hårdkodade i Python
- LLM system-prompt → `research/prompts/hypothesis_system.md`, inte en sträng i koden
- Signal-katalog → auto-genererad från `signals/` mappen via introspection, inte en manuell lista
- Tröskelvärden (min evidence_refs, max variants, etc.) → config med defaults
- Failure mode thresholds i post-mortem → bör redan vara config (verifiera att fas 1 implementerade det)

**Regeln:** Om du tvekar om något ska vara kod eller config → gör det till config. Kod ska vara mekanik. Config ska vara knobs.

---

## Kontext: Vad som finns efter fas 1

### Existerande moduler

```
fpr-research/
├── utils/experiment_fingerprint.py    # SHA-256 experiment ID
├── research/
│   ├── kb_schema.sql                  # DDL: events, sweeps, findings, coverage, artifacts
│   ├── knowledge_base.py              # CRUD + CLI (ingest, query, stats, export-events)
│   └── api.py                         # Internal Python API (tunna wrappers)
├── analysis/post_mortem.py            # Post-mortem generator + structured decisions
├── combinator/sweep_runner.py         # Sweep engine + finalize_sweep
├── signals/
│   ├── base.py                        # SignalComponent registry
│   ├── ema_cross.py
│   └── ema_stop_long.py
├── config/sweeps/                     # Sweep YAML configs
├── schema_sweep.json                  # JSON Schema för sweep configs
└── results/sweeps/sweep_001_ema_pvsra/
    ├── manifest.json
    ├── variants.jsonl
    ├── post_mortem.json
    ├── post_mortem.md
    └── leaderboard.md
```

### KB-status

- 1 sweep (sweep_001_ema_pvsra) ingested
- Verdict: DEAD, failure mode: LOW_SIGNAL
- Events: RUNNING → COMPLETED → ANALYZED → INGESTED
- research.db finns med alla fas 1-tabeller

### Vad som SAKNAS i KB-schemat

`proposals`-tabellen — definierad i projektplanen men explicit exkluderad från fas 1 ("Bygg inte proposals-tabell (fas 2)"). Du behöver skapa den.

---

## Vad du ska bygga (exakt)

### 1. Config System: `research/config.py` + `research/config/lab.yaml`

**Vad:** Centraliserad konfigurationsladdning. En YAML-fil, en loader.

**`research/config.py`:**

```python
"""
Lab configuration loader.
Reads research/config/lab.yaml. Override via FPR_LAB_CONFIG env var.
"""
import os, yaml
from pathlib import Path

_DEFAULT_PATH = Path(__file__).parent / "config" / "lab.yaml"
_CONFIG = None

def get_config(config_path: str = None) -> dict:
    """Load and cache config. Thread-safe enough for single-process use."""
    global _CONFIG
    if _CONFIG is None or config_path is not None:
        path = Path(config_path or os.environ.get("FPR_LAB_CONFIG", str(_DEFAULT_PATH)))
        with open(path) as f:
            _CONFIG = yaml.safe_load(f)
    return _CONFIG

def get(section: str, key: str, default=None):
    """Convenience: get('verifier', 'min_evidence_refs', 2)"""
    return get_config().get(section, {}).get(key, default)

def reset():
    """Reset cached config. For testing."""
    global _CONFIG
    _CONFIG = None
```

**`research/config/lab.yaml`:**

```yaml
# FPR Lab Configuration
# Change values here, not in code.

post_mortem:
  thresholds:
    fee_share_pct: 30
    fold_dispersion_multiplier: 2.0
    overtrading_trades_per_month: 500
    low_signal_sharpe: -5
    short_trade_bars: 10

verifier:
  min_evidence_refs: 2
  min_evidence_refs_bootstrap: 1 # used when KB has < bootstrap_threshold sweeps
  bootstrap_threshold: 3 # KB sweep count below which bootstrap rules apply
  max_variants: 500
  max_runtime_minutes: 120
  min_kill_criteria: 1

hypothesis:
  llm_provider: "anthropic"
  model: "claude-sonnet-4-20250514"
  temperature: 0.7
  max_output_tokens: 4000
  prompt_path: "research/prompts/hypothesis_system.md"

kb_retrieval:
  max_context_tokens: 2000
  top_n_findings: 10
  top_n_constraints: 5
  include_coverage_gaps: true
```

**Krav:**

- `get_config()` och `get()` är de enda publika funktionerna
- YAML-filen innehåller alla defaults — koden har inga fallback-värden utöver `get(section, key, default)`
- Alla moduler i fas 2 läser config via detta system
- `reset()` för testning (rensa cache)
- Verifiera: om post_mortem.py i fas 1 har hårdkodade thresholds, refaktorera dem till config (minor, men gör det)

---

### 2. LLM Client: `research/llm_client.py`

**Vad:** Provider-agnostisk LLM-klient. Infrastruktur — inte bunden till en specifik modell eller prompt.

```python
"""
LLM client abstraction.
Supports: anthropic, ollama, mock (for tests).
"""

class LLMClient:
    def __init__(self, provider: str = None, model: str = None, **kwargs):
        """Init from explicit args or config."""
        cfg = get_config().get('hypothesis', {})
        self.provider = provider or cfg.get('llm_provider', 'mock')
        self.model = model or cfg.get('model', '')
        self.temperature = kwargs.get('temperature', cfg.get('temperature', 0.7))
        self.max_tokens = kwargs.get('max_output_tokens', cfg.get('max_output_tokens', 4000))

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Send prompt, return raw text response."""
        if self.provider == 'anthropic':
            return self._call_anthropic(system_prompt, user_prompt)
        elif self.provider == 'ollama':
            return self._call_ollama(system_prompt, user_prompt)
        elif self.provider == 'mock':
            return self._mock_response(system_prompt, user_prompt)
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    def _call_anthropic(self, system, user) -> str:
        """Anthropic API via SDK."""
        # import anthropic
        # client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        # response = client.messages.create(...)
        # return response text
        ...

    def _call_ollama(self, system, user) -> str:
        """Ollama local API."""
        # requests.post("http://localhost:11434/api/generate", ...)
        ...

    def _mock_response(self, system, user) -> str:
        """Return a valid HypothesisFX JSON for testing."""
        # Return a hardcoded valid proposal JSON
        # Used by tests so they don't need API keys
        ...
```

**Krav:**

- `mock` provider returnerar valid HypothesisFX JSON — detta gör att hela pipelinen kan testas utan API-nyckel
- Mock-responsen ska vara realistisk: referera sweep_001, ha rimliga predictions, valid YAML
- `anthropic` provider använder `anthropic` SDK (redan i requirements eller lägg till)
- `ollama` provider använder `requests` till localhost
- Alla providers returnerar rå text (parsing sker i hypothesis_gen.py)
- Vid API-fel (rate limit, timeout, invalid response): raise `LLMError` med detaljer. INTE retry-logik i klienten — det tillhör hypothesis_gen.

---

### 3. Signal Catalog: `research/signal_catalog.py`

**Vad:** Auto-upptäcker tillgängliga signaler från `signals/` mappen. Ingen manuell lista.

**Inspektera först:** Läs `signals/base.py` och förstå hur `SignalComponent` registry fungerar. Titta på `ema_cross.py` och `ema_stop_long.py` för att se hur signaler registreras och vilka params de har.

**Bygg sedan:**

```python
def discover_signals() -> dict:
    """
    Introspect signals/ directory. Return catalog:
    {
        "ema_cross": {
            "type": "entry",
            "params": {"fast": {"type": "int"}, "slow": {"type": "int"}},
            "module": "signals.ema_cross",
            "description": "..."
        },
        ...
    }
    """
```

```python
def get_valid_signal_names() -> list[str]:
    """Just the names. For verifier to check against."""
```

```python
def format_for_llm(catalog: dict) -> str:
    """Format catalog as readable text for LLM context. Keep it concise."""
```

**Krav:**

- Bygger på befintlig registry-mekanism, inte filsystem-scanning
- Om `signals/base.py` inte har tillräcklig info (t.ex. param-ranges): extrahera vad som finns, markera resten som "unknown"
- `format_for_llm()` ska ge LLM tillräcklig info utan att vara ett kodblock — skriv det som koncis text
- Cachebart (signaler ändras inte under körning)

---

### 4. KB Migration: Proposals-tabell

**Vad:** Lägg till `proposals`-tabellen i `research/kb_schema.sql`.

**DDL:**

```sql
CREATE TABLE IF NOT EXISTS proposals (
    proposal_id TEXT PRIMARY KEY,
    created_ts TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'PENDING',   -- PENDING/APPROVED/REJECTED/EXPIRED
    experiment_intent TEXT NOT NULL,           -- gap_fill/failure_mitigation/robustness/regime_test
    proposed_config_json TEXT NOT NULL,
    experiment_id TEXT,                        -- pre-computed fingerprint
    rationale_md TEXT,
    evidence_refs_json TEXT NOT NULL,
    novelty_claim_json TEXT NOT NULL,
    predictions_json TEXT NOT NULL,
    expected_mechanism TEXT NOT NULL,
    expected_failure_mode TEXT,
    kill_criteria_json TEXT NOT NULL,
    compute_budget_json TEXT NOT NULL,
    prediction_audit_json TEXT                -- populated post-sweep
);
CREATE INDEX IF NOT EXISTS idx_proposals_status ON proposals(status);
CREATE INDEX IF NOT EXISTS idx_proposals_experiment ON proposals(experiment_id);
```

**Krav:**

- Lägg till i befintlig `kb_schema.sql` (inte ny fil)
- `init_db()` i `knowledge_base.py` skapar tabellen automatiskt (CREATE IF NOT EXISTS)
- Migration: existerande `research.db` ska inte förlora data — bara lägga till tabellen
- Lägg till CRUD i `knowledge_base.py`: `write_proposal()`, `get_proposals()`, `update_proposal_status()`, `write_prediction_audit()`
- Lägg till i `api.py`: `get_proposals(db_path, status=None)` (uppdatera befintlig stub som returnerar [])

---

### 5. KB Retrieval: `research/kb_retrieval.py`

**Vad:** Hämtar fokuserad kontext från KB för LLM-prompten. Inte hela KB-dumpen.

```python
def retrieve_context(db_path: str) -> dict:
    """
    Hämtar relevant kontext för hypothesis generator.

    Returns:
        {
            "sweep_count": int,
            "latest_sweep": {...},
            "top_findings": [...],
            "coverage_gaps": [...],
            "active_constraints": [...],
            "available_signals": [...],
        }
    """
```

```python
def format_context_for_llm(context: dict) -> str:
    """
    Formatera kontext som text för LLM system/user prompt.
    Respektera token-budget från config.
    Prioritetsordning om budget tar slut:
    1. Active constraints (från senaste post-mortem)
    2. Coverage gaps
    3. Top findings
    4. Available signals (från signal_catalog)
    5. Latest sweep summary
    """
```

**Krav:**

- `retrieve_context()` anropar `knowledge_base.py` och `signal_catalog.py`
- `format_context_for_llm()` producerar text, inte JSON — LLM förstår text bättre
- Token-budget är config (`kb_retrieval.max_context_tokens`). Estimera tokens som `len(text) / 4`
- Coverage gaps = entity_types/names som INTE finns i coverage-tabellen men som finns i signal-katalogen
- Active constraints = `next_experiment_constraints` från senaste sweep's post-mortem
- Om KB är tom (0 sweeps): returnera bara signal-katalog + "No previous sweeps. This is the first experiment."

---

### 6. Hypothesis Generator: `research/hypothesis_gen.py`

**Vad:** Orkestrerare som kopplar ihop retrieval → LLM → parsing → verifiering. Det är pipeline-controllern.

**Flöde:**

```
1. retrieve_context(db_path)
2. Ladda system-prompt från config path
3. Bygg user-prompt med kontext
4. Anropa LLM via llm_client
5. Parsa JSON-svar till HypothesisFX dict
6. Beräkna experiment_id (fingerprint)
7. Kör proposal_verifier
8. Om PASS: skriv till proposals-tabell + events
9. Om FAIL: logga reject-event + skriv rejected proposal med anledning
10. Returnera resultat
```

**Parsing:**

- LLM-svaret ska vara JSON (be om det i prompten)
- Extrahera JSON från svar (hantera att LLM kan wrappa i `json ... ` eller ha text före/efter)
- Om parsing misslyckas: logga rå-svaret, returnera error — ingen retry

**CLI:**

```bash
# Kör hela pipelinen (retrieval → gen → verify → store)
python -m research.hypothesis_gen run

# Verifiera en befintlig proposal-fil
python -m research.hypothesis_gen verify path/to/proposal.json

# Visa senaste förslag
python -m research.hypothesis_gen show --status PENDING
```

**Krav:**

- `run()` returnerar `{"status": "accepted"|"rejected", "proposal_id": str, "reason": str|None}`
- Hela flödet loggar events: `HYPOTHESIS_GEN_STARTED`, `HYPOTHESIS_GEN_COMPLETED`, `PROPOSAL_CREATED` eller `PROPOSAL_REJECTED`
- Om LLM-anrop misslyckas → event `STEP_FAILED` med detaljer, INTE crash
- `--dry-run` flag: kör allt utom att skriva till DB (för testning)
- `--provider mock` flag: tvinga mock LLM (för testning utan API-nyckel)

---

### 7. Proposal Verifier: `research/proposal_verifier.py`

**Vad:** Deterministisk validering. Kod, INTE LLM. Kontrollerar att ett HypothesisFX-objekt uppfyller alla krav.

```python
def verify_proposal(proposal: dict, db_path: str) -> VerificationResult:
    """
    Kör alla checks. Returnerar pass/fail med detaljer.

    Returns:
        VerificationResult(
            passed: bool,
            checks: [
                {"name": "schema_valid", "passed": True},
                {"name": "evidence_refs_exist", "passed": False, "reason": "sweep_999 not in KB"},
                ...
            ]
        )
    """
```

**Checks (läses från config, inte hårdkodade):**

| Check                    | Config-nyckel                                                | Default   | Logik                                                                         |
| ------------------------ | ------------------------------------------------------------ | --------- | ----------------------------------------------------------------------------- |
| Schema valid             | —                                                            | —         | Validera mot `research/schemas/hypothesis_fx.json`                            |
| Intent valid             | —                                                            | —         | `experiment_intent` ∈ {gap_fill, failure_mitigation, robustness, regime_test} |
| Evidence refs exist      | `verifier.min_evidence_refs`                                 | 2         | Alla sweep_ids finns i KB                                                     |
| Evidence refs minimum    | `verifier.min_evidence_refs` / `min_evidence_refs_bootstrap` | 2 / 1     | Dynamiskt baserat på KB sweep count vs `bootstrap_threshold`                  |
| Predictions complete     | —                                                            | —         | Alla obligatoriska fält har numeriska värden                                  |
| Sweep config valid       | —                                                            | —         | YAML valideras mot `schema_sweep.json`                                        |
| Experiment not duplicate | —                                                            | —         | `experiment_id` finns INTE redan i KB                                         |
| Budget within limits     | `verifier.max_variants` / `max_runtime_minutes`              | 500 / 120 | Kontrollera compute_budget                                                    |
| Kill criteria present    | `verifier.min_kill_criteria`                                 | 1         | Minst N villkor                                                               |
| Novelty claim non-empty  | —                                                            | —         | `coverage_diff` är icke-tom                                                   |
| Signals exist            | —                                                            | —         | Alla signaler i sweep_config finns i signal-katalogen                         |

**Bootstrap-logik:**

```python
sweep_count = count_sweeps(db)
if sweep_count < config.get('verifier', 'bootstrap_threshold', 3):
    min_refs = config.get('verifier', 'min_evidence_refs_bootstrap', 1)
else:
    min_refs = config.get('verifier', 'min_evidence_refs', 2)
```

**Krav:**

- Varje check är en separat funktion (testbar isolerat)
- Alla thresholds läses via `research/config.py`
- `VerificationResult` har tillräcklig info för att logga _varför_ ett förslag rejectades
- Reject = tyst i pipeline-mening (inga retries). Men logga allt.

---

### 8. HypothesisFX JSON Schema: `research/schemas/hypothesis_fx.json`

**Vad:** Formell JSON Schema som definierar kontraktet.

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "HypothesisFX",
  "type": "object",
  "required": [
    "experiment_intent",
    "evidence_refs",
    "novelty_claim",
    "expected_mechanism",
    "predictions",
    "expected_failure_mode",
    "kill_criteria",
    "compute_budget",
    "sweep_config_yaml"
  ],
  "properties": {
    "experiment_intent": {
      "type": "string",
      "enum": ["gap_fill", "failure_mitigation", "robustness", "regime_test"]
    },
    "evidence_refs": {
      "type": "array",
      "items": { "type": "string" },
      "minItems": 1
    },
    "novelty_claim": {
      "type": "object",
      "required": ["coverage_diff"],
      "properties": {
        "coverage_diff": {
          "type": "array",
          "items": { "type": "string" },
          "minItems": 1
        },
        "near_dup_score": { "type": "number" }
      }
    },
    "expected_mechanism": {
      "type": "string",
      "minLength": 50
    },
    "predictions": {
      "type": "object",
      "required": [
        "trade_duration_median_bars",
        "trades_per_day",
        "gross_minus_net_gap"
      ],
      "properties": {
        "trade_duration_median_bars": { "type": "number" },
        "trades_per_day": { "type": "number" },
        "gross_minus_net_gap": { "type": "number" }
      }
    },
    "expected_failure_mode": { "type": "string" },
    "kill_criteria": {
      "type": "array",
      "items": { "type": "string" },
      "minItems": 1
    },
    "compute_budget": {
      "type": "object",
      "required": ["max_variants", "max_runtime_minutes"],
      "properties": {
        "max_variants": { "type": "integer", "maximum": 500 },
        "max_runtime_minutes": { "type": "integer", "maximum": 120 }
      }
    },
    "sweep_config_yaml": {
      "type": "string",
      "minLength": 10
    }
  }
}
```

**Krav:**

- Schema valideras med `jsonschema` Python-paket (lägg till i requirements om saknas)
- `evidence_refs.minItems` i schemat sätts till 1 (bootstrap-logiken hanterar det faktiska minimumet i verifiern, inte i schemat)

---

### 9. LLM Prompt: `research/prompts/hypothesis_system.md`

**Vad:** System-prompten som styr LLM:s beteende. Config-fil, inte kod.

**Skriv en rimlig första version.** Du kan inte veta exakt vad som funkar — vi itererar. Men inkludera:

- Rollbeskrivning: "Du är en kvantitativ forskningsassistent som föreslår nästa experiment"
- HypothesisFX-kontraktet (alla 9 fält, typer, krav)
- Exempel på output-format (valid JSON)
- Regler: citera evidence, förklara mekanism, gör mätbara predictions
- Explicit: "Svara ENBART med JSON. Ingen text före eller efter."

**Krav:**

- Filen ska vara ren markdown, inte Python
- `hypothesis_gen.py` läser den via config-path
- Prompten ska vara generell nog att fungera oavsett vilka signaler/sweeps som finns i KB

---

### 10. Prediction Audit: hook i `combinator/sweep_runner.py`

**Vad:** Efter att en sweep kör klart, jämför proposal-predictions med faktiskt utfall.

**Var i pipeline:**

```
sweep completes
  → post_mortem generates
  → prediction_audit runs (NYT STEG)
  → KB ingest
```

**Logik:**

```python
def run_prediction_audit(sweep_dir: str, db_path: str, sweep_id: str) -> dict | None:
    """
    Om det finns en APPROVED proposal för denna sweep:
    1. Läs proposal.predictions_json
    2. Läs post_mortem.json summary + trade data
    3. Beräkna diff per prediction
    4. Skriv prediction_audit_json på proposal-raden
    5. Skapa finding i KB med auditresultatet

    Om ingen proposal finns: returnera None (sweep kördes manuellt).
    """
```

**Diff-beräkning (enkel):**

```python
audit = {}
for key in predictions:
    predicted = predictions[key]
    actual = extract_actual(post_mortem, key)  # map key → post-mortem field
    if actual is not None:
        audit[key] = {
            "predicted": predicted,
            "actual": actual,
            "diff_pct": (actual - predicted) / abs(predicted) * 100 if predicted != 0 else None
        }
```

**Krav:**

- Hookas in i `finalize_sweep()` mellan post_mortem och KB ingest
- Om audit misslyckas → logga varning, men blockera INTE ingest (detta är icke-kritiskt)
- Skriv event `PREDICTION_AUDIT_COMPLETED` eller `PREDICTION_AUDIT_SKIPPED` (om ingen proposal)
- Om ingen proposal matchar sweep_id: skip silently (manuella sweeps har inga predictions)

---

### 11. Uppdatera `research/api.py`

**Utöka befintligt API med nya funktioner:**

```python
# Befintliga (från fas 1):
def get_feed(db_path, limit=50, offset=0): ...
def get_sweep(db_path, sweep_id): ...
def get_coverage(db_path, entity_type, entity_name): ...
def get_findings(db_path, tags=None): ...
def export_events(db_path, last_n=20): ...

# Nya (fas 2):
def get_proposals(db_path, status=None): ...
def get_proposal(db_path, proposal_id): ...
def approve_proposal(db_path, proposal_id): ...   # ändrar status PENDING → APPROVED
def reject_proposal(db_path, proposal_id, reason): ...
def get_signal_catalog(): ...
def get_coverage_gaps(db_path): ...
def run_hypothesis_pipeline(db_path, dry_run=False, provider=None): ...
```

**Krav:**

- `approve_proposal()` ändrar bara status + loggar event. Startar INTE sweep (det är fas 4).
- `run_hypothesis_pipeline()` wrapprar `hypothesis_gen.run()` — API-lagret gör att framtida UI/CLI inte behöver importera hypothesis_gen direkt
- Alla nya funktioner har type hints + docstrings

---

## Nya filer du skapar

```
fpr-research/
├── research/
│   ├── config.py                        # NY — config loader
│   ├── config/
│   │   └── lab.yaml                     # NY — alla config-värden
│   ├── prompts/
│   │   └── hypothesis_system.md         # NY — LLM system prompt
│   ├── schemas/
│   │   └── hypothesis_fx.json           # NY — JSON Schema
│   ├── signal_catalog.py                # NY — signal introspection
│   ├── kb_retrieval.py                  # NY — kontext-hämtning
│   ├── hypothesis_gen.py                # NY — pipeline controller
│   ├── proposal_verifier.py             # NY — deterministisk validering
│   ├── llm_client.py                    # NY — LLM abstraction
│   ├── kb_schema.sql                    # UPPDATERAD — proposals tabell
│   ├── knowledge_base.py                # UPPDATERAD — proposal CRUD
│   └── api.py                           # UPPDATERAD — nya endpoints
├── combinator/
│   └── sweep_runner.py                  # UPPDATERAD — prediction audit hook
└── tests/
    ├── test_config.py                   # NY
    ├── test_signal_catalog.py           # NY
    ├── test_kb_retrieval.py             # NY
    ├── test_hypothesis_gen.py           # NY
    ├── test_proposal_verifier.py        # NY
    └── test_prediction_audit.py         # NY
```

---

## Testkrav

### `tests/test_config.py`

- Ladda default config → alla sektioner finns
- Override via custom path → nya värden används
- `reset()` → config laddas om
- Saknad fil → tydligt felmeddelande

### `tests/test_signal_catalog.py`

- `discover_signals()` hittar ema_cross och ema_stop_long
- `get_valid_signal_names()` returnerar lista med strängar
- `format_for_llm()` returnerar text (inte JSON, inte kod)

### `tests/test_kb_retrieval.py`

- Med sweep_001 i KB: `retrieve_context()` returnerar findings, coverage_gaps, constraints
- Med tom KB: returnerar signal-katalog + "first experiment" indikation
- `format_context_for_llm()` producerar text inom token-budget

### `tests/test_hypothesis_gen.py`

- Med mock LLM: `run()` producerar en PENDING proposal i DB
- Mock LLM med invalid JSON → events loggas, ingen crash
- `--dry-run` → ingen DB-write
- E2E: sweep_001 i KB → retrieval → gen → verify → proposal i proposals-tabell

### `tests/test_proposal_verifier.py`

- Valid proposal → passes
- Saknat fält → fails med specifik anledning
- Duplicate experiment_id → fails
- Budget over limit → fails
- Evidence ref som inte finns i KB → fails
- Bootstrap mode: 1 ref godkänns när KB har < 3 sweeps
- Invalid sweep config YAML → fails
- Signal som inte finns i katalogen → fails

### `tests/test_prediction_audit.py`

- Proposal med predictions + mock post_mortem → audit JSON populeras
- Ingen proposal för sweep → audit skippas (returnerar None)
- Audit failure → varning loggas, ingest blockeras INTE

---

## Backfill / E2E-verifiering

Efter allt är byggt, kör hela kedjan:

```bash
# 1. Verifiera att befintliga tester fortfarande passerar
pytest tests/ -x

# 2. Kör hypothesis pipeline med mock LLM
python -m research.hypothesis_gen run --provider mock

# 3. Visa genererat förslag
python -m research.hypothesis_gen show --status PENDING

# 4. Verifiera att proposal finns i DB
python -m research.knowledge_base stats

# 5. Verifiera events
python -m research.knowledge_base export-events --last 20
```

---

## Vad du INTE ska göra

- Bygg inte UI/TUI (fas 3)
- Starta inte sweeps automatiskt — bara generera och lagra förslag
- Implementera inte retry-logik i LLM-klienten
- Använd inte LangChain, AutoGen, eller andra LLM-ramverk
- Lägg inte till vektor-embeddings, ML-modeller, eller Celery
- Ändra inte befintliga tester
- Ändra inte sweep_runner-flödet utöver prediction audit hook
- Gör INTE prompten perfekt — skriv en rimlig första version, vi itererar

---

## Kvalitetskrav

- Alla befintliga tester (157) måste fortfarande passera
- Nya tester måste passera
- Typing: alla publika funktioner har type hints
- Docstrings: alla publika funktioner
- Inga `print()` i library-kod (använd `logging`)
- CLI-kommandon får printa
- Config-värden som kan ändras ska ALDRIG vara hårdkodade i Python

---

## Definition of Done (sammanfattning)

Se fullständig DoD i sibling-page. Kortversion:

✅ **H1:** `python -m research.hypothesis_gen run --provider mock` producerar ett PENDING proposal i DB med alla 9 HypothesisFX-fält  
✅ **H2:** Verifier rejectar proposal med saknade fält / ogiltigt schema / duplicate experiment_id  
✅ **H3:** Config override fungerar (ändra lab.yaml → beteende ändras utan kodändring)  
✅ **H4:** Signal-katalog upptäcker alla registrerade signaler  
✅ **H5:** E2E: sweep_001 i KB → retrieval → gen (mock) → verify → proposal i proposals-tabell → events loggade  
✅ **H6:** Prediction audit: proposal med predictions + simulerad completion → audit JSON populeras  
✅ Alla tester passerar (gamla + nya)
