# Fas 1 — CC Build Prompt

**Status:** READY
**Target:** Claude Code
**Repo:** `github.com/hypox-system/fpr-research.git`
**Lokal mapp:** `/Users/danielwarberg/fpr-research`
**Branch:** Skapa `feat/research-station-fas1` från `main`
**DoD:** Se `docs/FAS1_DOD.md`

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
8. **Inspektera innan du antar.** Läs faktisk kod (`backtest.py`, `walk_forward.py`, `VariantResult`) för att avgöra om durations/fold-data finns. Gissa inte.
9. **Leverera allt i en session.** Brancha, bygg, testa, kör backfill, rapportera resultat. En commit-kedja.
10. **När du är klar:** kör alla H-steg (H1–H4 i DoD) och visa output. Det är ditt "proof of done".

**Enda gången du får fråga:** Om befintlig kod är trasig på ett sätt som blockerar ditt arbete OCH du inte kan fixa det utan att bryta scope.

---

## Kritiska klarifik (läs innan du börjar)

1. **Packages:** Skapa `utils/__init__.py` (tom) och `analysis/__init__.py` (tom, om saknas). Behövs för `python -m` imports.
2. **Event ownership:** `ingest_sweep(db, sweep_dir_path, emit_events=True)`. När `sweep_runner` anropar: `emit_events=False` (sweep_runner skriver events själv runt stegen). När CLI anropar: default `True`.
3. **Round-trip fee-formel:** `estimated_fee_drag_per_trade = 2 * (fee_rate + slippage_rate)` (decimal, t.ex. 0.0014 för 0.06%+0.01%). Använd denna explicit i post-mortem. Gissa inte.
4. **Mapping:** `manifest.symbol` → `sweeps.asset`. Kolumnnamnet i DB är `asset`.
5. **Failure-path:** Bygg en testbar `finalize_sweep(sweep_dir, db_path)` helper i `sweep_runner` som wrapprar post-mortem + ingest med try/except → `STEP_FAILED` + `NEEDS_REVIEW`.
6. **Metrics-tillgänglighet:** Se sektion "Trade Data" nedan. Om en metric saknas i JSONL: skriv `null` i JSON + notera `"missing: <metric>"` i `failure_evidence`. Behåll ALLTID schema-nycklar.

---

## Kontext

FPR Research Platform v2.1.1 är en backtesting-pipeline för crypto-strategier. Den kör parameter-sweeps, walk-forward-validering, och statistisk filtrering (BH-FDR, surrogat, DSR). Plattformen fungerar — 164 tester passerar, audit klar.

Problem: plattformen har inget minne mellan sweeps. Varje sweep är isolerad. Ingen automatisk analys. Ingen kunskapsackumulering. Samma döda experiment kan köras igen utan varning.

**Du ska bygga fas 1 av Research Station: post-mortem-generator + knowledge base + event log + experiment fingerprint.**

Detta ger plattformen minne och gör att en framtida hypothesis-generator (fas 2) kan resonera över ackumulerad kunskap.

---

## Vad du ska bygga (exakt)

### 1. `utils/experiment_fingerprint.py`

En ren funktion som beräknar unik identitet per experiment.

```python
def compute_experiment_id(
    sweep_config: dict,
    data_manifest: dict,  # symbol, timeframe, date_range, data_fingerprint
    fill_model: dict,     # fee_rate, slippage_rate
) -> str:
    """
    Canonical SHA-256 för hela experimentet.
    
    1. Canonicalisera sweep_config (använd canonical.py som redan finns)
    2. Canonicalisera data_manifest
    3. Canonicalisera fill_model
    4. Concat + SHA-256
    5. Returnera hex (första 16 tecken)
    
    KRAV: YAML-ordning får INTE påverka resultatet.
    """
```

Använd `combinator/canonical.py:canonicalize_spec()` som redan finns i repot. Den sorterar keys, normaliserar floats till 10 decimaler.

Dependencies: `hashlib` (stdlib). Inget mer.

---

### 2. `research/kb_schema.sql`

SQLite DDL för alla tabeller. Se DoD E2 för exakt schema:

```sql
CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    ts TEXT NOT NULL,
    event_type TEXT NOT NULL,
    sweep_id TEXT,
    experiment_id TEXT,
    status TEXT,
    payload_json TEXT
);
CREATE INDEX idx_events_sweep ON events(sweep_id);
CREATE INDEX idx_events_ts ON events(ts);

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

CREATE TABLE IF NOT EXISTS findings (
    finding_id TEXT PRIMARY KEY,
    sweep_id TEXT NOT NULL REFERENCES sweeps(sweep_id),
    statement TEXT NOT NULL,
    tags_json TEXT,
    confidence REAL,
    evidence_refs_json TEXT
);

CREATE TABLE IF NOT EXISTS coverage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type TEXT NOT NULL,
    entity_name TEXT NOT NULL,
    sweep_id TEXT NOT NULL REFERENCES sweeps(sweep_id)
);
CREATE INDEX idx_coverage_entity ON coverage(entity_type, entity_name);

CREATE TABLE IF NOT EXISTS artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sweep_id TEXT NOT NULL REFERENCES sweeps(sweep_id),
    artifact_type TEXT NOT NULL,
    path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    bytes INTEGER,
    created_ts TEXT
);
```

---

### 3. `research/knowledge_base.py`

CRUD + query-interface för `research.db`.

**Init:** `init_db(db_path)` — kör `kb_schema.sql`, skapar tabeller om saknas.

**Write:**
- `write_event(db, event_type, sweep_id, experiment_id, status, payload)` — append-only
- `ingest_sweep(db, sweep_dir_path, emit_events=True)` — läser `manifest.json` + `post_mortem.json`, skapar/uppdaterar sweeps + findings + coverage + artifacts. Idempotent (upsert på sweep_id).

**Read:**
- `query(db, search_term)` — söker i findings.statement + findings.tags_json + coverage.entity_name. Returnerar matchande sweeps med status, failure_mode, best metric, artifact paths.
- `stats(db)` — coverage-summering: timeframes, fee-regimer, signals, assets.
- `export_events(db, last_n)` — senaste N events formaterade för stdout.
- `get_sweep_status(db, sweep_id)` — härledd från events-tabellen.

**CLI (via `__main__`):**

```bash
python -m research.knowledge_base ingest results/sweeps/sweep_001_ema_pvsra/
python -m research.knowledge_base query "ema_cross"
python -m research.knowledge_base stats
python -m research.knowledge_base export-events --last 20
```

**Dedup-gate:** Vid ingest, kontrollera `experiment_id` UNIQUE. Om redan finns och status är COMPLETED|ANALYZED|INGESTED → logga varning + skippa (inte krascha).

---

### 4. `research/api.py`

Internt Python-API. TUI/webb anropar detta, aldrig DB direkt.

Fas 1 = grundstruktur med funktioner som wrapprar `knowledge_base.py`:

```python
def get_feed(db_path, limit=50, offset=0): ...
def get_sweep(db_path, sweep_id): ...
def get_coverage(db_path, entity_type, entity_name): ...
def get_findings(db_path, tags=None): ...
def get_proposals(db_path, status=None): ...  # returns [] i fas 1
def export_events(db_path, last_n=20): ...
```

Detta är tunna wrappers. Ingen logik här, bara interface.

---

### 5. `analysis/post_mortem.py`

Själva analysmotorn. Den här är den viktigaste filen.

**Input:** sweep directory (innehåller `variants.jsonl` + `manifest.json`)

**Output:** `post_mortem.json` + `post_mortem.md` i samma directory

**JSON-schema (se DoD D2 för fullständigt krav):**

```json
{
    "sweep_id": "sweep_001_ema_pvsra",
    "experiment_id": "a1b2c3d4e5f67890",
    "created_ts": "2025-12-31T00:00:00Z",
    "generated_ts": "2026-02-17T14:30:00Z",
    
    "summary": {
        "variant_count": 270,
        "survivor_count": 0,
        "best_variant": {
            "variant_id": "973d0aca8a02",
            "oos_sharpe": -6.71,
            "n_trades_oos": 1109,
            "profit_factor": 0.23
        }
    },
    
    "fee_decomposition": {
        "median_gross_return_per_trade": -0.0021,
        "median_net_return_per_trade": -0.0033,
        "estimated_fee_drag_per_trade": 0.0014,
        "fee_share_of_loss_pct": 36.4,
        "data_source": "trade_summary"
    },
    
    "trade_duration_distribution": {
        "median_bars": 8,
        "p25_bars": 4,
        "p75_bars": 18
    },
    
    "fold_stability": [
        {"fold": 0, "oos_sharpe": -5.2, "n_trades": 180},
        {"fold": 1, "oos_sharpe": -8.1, "n_trades": 210}
    ],
    "fold_dispersion": {"std": 2.3, "iqr": 3.1},
    
    "primary_failure_mode": "FEE_DRAG",
    "failure_evidence": "Median gross return -0.21% per trade. Fee+slippage cost 0.12% per trade. 36% of loss is pure fee drag. Median trade duration 8 bars = too short for 0.06% fee regime.",
    
    "most_promising_region": {
        "params": {"fast": [5, 8], "slow": [50, 100]},
        "score": -3.2,
        "note": "Least negative OOS Sharpe in fast=5-8, slow=50-100 region"
    },
    
    "next_experiment_constraints": [
        "min_trade_duration_bars >= 20",
        "exit must be volatility-scaled (not fixed EMA)",
        "consider 15m or 4h timeframe to change trade duration profile"
    ]
}
```

**Data source hierarchy:**
- `fee_decomposition.median_gross_return_per_trade` — från trade_summary (net + roundtrip_cost)
- `fee_decomposition.median_net_return_per_trade` — från trade_summary
- `fee_decomposition.estimated_fee_drag_per_trade` — `2*(fee_rate+slippage_rate)`, ALLTID beräkningsbart
- `fee_decomposition.fee_share_of_loss_pct` — null om gross saknas
- `fee_decomposition.data_source` — `"npz"` | `"trade_summary"` | `"config_only"`
- `trade_duration_distribution` — null om duration_summary saknas i JSONL
- `fold_stability` — null om fold_results saknas i JSONL

**Hur `primary_failure_mode` bestäms (logik):**

```
// Steg 1: beräkna vad som finns (alla checks som kräver null-data → skip)
OM median_gross_return (från trade_summary) > 0 OCH median_net_return < 0:
    → GOOD_GROSS_DIES_NET
OM trade_duration (från duration_summary) median < 10 bars OCH fee_share > 30%:
    → FEE_DRAG  
OM fold_dispersion (från fold_results) std > 2x median abs(oos_sharpe):
    → REGIME_FRAGILE
OM n_trades_oos median > 500 per månad:
    → OVERTRADING
OM survivor_count == 0 OCH best oos_sharpe < -5:
    → LOW_SIGNAL
// Steg 2: fallback om data saknas för specifika modes
OM trade_summary saknas OCH duration_summary saknas:
    → LOW_SIGNAL (om best sharpe < -5) eller NO_CONVERGENCE
ANNARS:
    → NO_CONVERGENCE
```

Dessa trösklar är startvärden. Hardkoda inte — gör dem konfigurerbara (dict med defaults).

**Markdown-rapport (`.md`):**

```markdown
# Post-Mortem: sweep_001_ema_pvsra

## Executive Summary
270 variants tested. 0 BH-FDR survivors. All variants show strongly negative 
OOS Sharpe. Primary failure mode: FEE_DRAG.

## Verdict: DEAD

## Top 10 Variants
| Variant | OOS Sharpe | Trades | PF | Win Rate |
|---------|-----------|--------|------|----------|
| 973d0a  | -6.71     | 1109   | 0.23 | 10.4%    |
| ...     | ...       | ...    | ...  | ...      |

## Why It Failed
Median gross return per trade: -0.21%. Fee+slippage cost: 0.12% per trade.
36% of total loss is pure fee drag. Median trade duration of 8 bars is too 
short for 0.06% taker fee regime.

## What To Try Next
- Increase minimum trade duration to 20+ bars
- Use volatility-scaled exit instead of fixed EMA
- Test on 15m or 4h timeframe
```

**Fail-closed:** Om `variants.jsonl` är trasig/otolkbar → raise exception. Anropande kod (sweep_runner) ansvarar för att sätta NEEDS_REVIEW.

**CLI:**

```bash
python -m analysis.post_mortem results/sweeps/sweep_001_ema_pvsra/
```

---

### 6. Integration i `combinator/sweep_runner.py`

**Ändra `run_sweep()`** så den:

1. Beräknar `experiment_id` **innan sweep startar** (från config + data manifest + fill model)
2. Checkar om `experiment_id` redan finns i KB → om ja, abort med meddelande (om inte `--force`)
3. Skriver `SWEEP_STARTED` event
4. Kör sweep som vanligt
5. Skriver `SWEEP_COMPLETED` event
6. Anropar `finalize_sweep()` (se nedan)

**Bygg `finalize_sweep(sweep_dir, db_path)` som testbar helper:**

```python
def finalize_sweep(sweep_dir: str, db_path: str) -> str:
    """
    Post-sweep pipeline: post-mortem + KB-ingest.
    Returnerar slutstatus: 'INGESTED' eller 'NEEDS_REVIEW'.
    """
    db = init_db(db_path)
    try:
        write_event(db, 'POST_MORTEM_STARTED', sweep_id, ...)
        post_mortem.generate(sweep_dir)
        write_event(db, 'POST_MORTEM_COMPLETED', sweep_id, ...)
    except Exception as e:
        write_event(db, 'STEP_FAILED', sweep_id, status='NEEDS_REVIEW',
                    payload={'step': 'post_mortem', 'error': str(e)})
        return 'NEEDS_REVIEW'  # STOPPA HÄR
    
    try:
        write_event(db, 'KB_INGEST_STARTED', sweep_id, ...)
        ingest_sweep(db, sweep_dir, emit_events=False)
        write_event(db, 'KB_INGEST_COMPLETED', sweep_id, ...)
    except Exception as e:
        write_event(db, 'STEP_FAILED', sweep_id, status='NEEDS_REVIEW',
                    payload={'step': 'kb_ingest', 'error': str(e)})
        return 'NEEDS_REVIEW'
    
    return 'INGESTED'
```

Detta gör failure-path testbart utan att köra hela sweep-motorn.

**Ändra `save_sweep_results()`:** Lägg till `trade_summary`, `duration_summary`, `fold_results` i `VariantResult.to_dict()` (se sektion 7).

**Ändra `build_manifest()`:** Lägg till `experiment_id` + `trade_data_present: bool`.

**Lägg till `--save-trade-data` flag** för att spara `trade_data.npz`.

**Lägg till `--force` flag** för re-run av existerande experiment (loggar `SWEEP_FORCE_RERUN`).

**KB path:** Default `research.db` i repo-root. Konfigurerbar via env eller arg.

---

### 7. Trade Data: summary-scalars + optional NPZ

**Problem:** Nuvarande `VariantResult.to_dict()` exkluderar `oos_trade_returns` (kommentar: "too large"). Det gör att post-mortem inte kan beräkna per-trade metrics från disk.

**Lösning (två delar):**

#### A1: Summary-scalars i JSONL (obligatoriskt)

Lägg till följande fält i `VariantResult.to_dict()` (beräknas in-memory under sweep, sparas som scalars):

```python
# I to_dict(), under if self.status == 'OK':
'trade_summary': {
    'median_net_return_per_trade': float,      # median av oos_trade_returns
    'median_gross_return_per_trade': float,     # net + roundtrip_cost (2*(fee+slip))
    'mean_net_return_per_trade': float,
    'std_net_return_per_trade': float,
    'n_positive_trades': int,
    'n_negative_trades': int,
}
```

**Trade durations:** Inspektera `backtest.py` / `walk_forward.py` för om `entry_bar_idx` / `exit_bar_idx` eller liknande finns per trade. Om ja, lägg till:

```python
'duration_summary': {
    'median_bars': int,
    'p25_bars': int,
    'p75_bars': int,
}
```

Om durations inte finns i minnet: hoppa duration-analytics, skriv `null` + note i post-mortem.

**Fold-level data:** Om per-fold OOS Sharpe + n_trades finns i minnet (från walk_forward), lägg till:

```python
'fold_results': [
    {'fold': 0, 'oos_sharpe': float, 'n_trades': int},
    ...
]
```

Om inte: skriv `null` i post-mortem fold_stability.

**Regel:** Behåll kommentaren "Don't include trade returns in JSONL" — du inkluderar inte arrays, bara summaries. JSONL-storleken ökar marginellt.

#### A2: `trade_data.npz` bakom flagga (valfritt)

**När:** `--save-trade-data` flag på sweep eller `statistics.save_trade_data: true` i sweep YAML.

**Vad:** Komprimerad NPZ med per-variant arrays: `oos_trade_returns`, `oos_trade_durations` (om finns).

**Var:** `results/sweeps/<sweep_name>/trade_data.npz`

**Manifest:** Lägg till `trade_data_present: bool` + `trade_data_path` + `trade_data_sha256` (om fil skapas).

**Post-mortem:** Använd NPZ om den finns för exakt analys. Annars: fall tillbaka till summary-scalars från JSONL. Annars: `null` + note.

**Artifacts-tabell:** Om NPZ skapas, registrera i artifacts med `artifact_type: "trade_data"`.

---

## Existerande kod du måste förstå

### Filstruktur (relevant del)

```
fpr-research/
├── combinator/
│   ├── sweep_runner.py    # Huvudloop — du hookar in här
│   ├── canonical.py       # canonicalize_spec() — använd för fingerprint
│   └── composer.py        # Variant-generering
├── analysis/
│   ├── statistics.py      # Permutation test, BH-FDR
│   ├── multiple_testing.py
│   └── negative_control.py
├── signals/
│   ├── base.py            # SignalComponent registry
│   ├── ema_cross.py
│   └── ema_stop_long.py
├── engine/
│   ├── backtest.py
│   └── walk_forward.py
├── config/
│   └── sweeps/
│       └── sweep_001_ema_pvsra.yaml
├── results/
│   └── sweeps/
│       └── sweep_001_ema_pvsra/
│           ├── manifest.json
│           ├── variants.jsonl
│           └── leaderboard.md
├── tests/
└── schema_sweep.json
```

### Nya filer du skapar

```
fpr-research/
├── utils/
│   ├── __init__.py                # NY (tom)
│   └── experiment_fingerprint.py   # NY
├── analysis/
│   ├── __init__.py                # NY (tom, om saknas)
│   └── post_mortem.py             # NY
├── research/
│   ├── __init__.py                # NY
│   ├── kb_schema.sql              # NY
│   ├── knowledge_base.py          # NY
│   └── api.py                     # NY
├── tests/
│   ├── test_fingerprint.py        # NY
│   ├── test_post_mortem.py        # NY
│   └── test_knowledge_base.py     # NY
└── research.db                    # Auto-skapad vid första init
```

### `canonical.py` (redan finns)

```python
def canonicalize_spec(spec: dict) -> str:
    """Sorterar keys, normaliserar floats (10 decimaler), returnerar JSON."""

def spec_to_variant_id(spec: dict) -> str:
    """SHA-256[:12] av canonicaliserad spec."""
```

Använd `canonicalize_spec()` för experiment fingerprint. Importera från `combinator.canonical`.

### `VariantResult` fält (från `variants.jsonl`)

Varje rad i JSONL har (befintliga fält):

```
variant_id, status, reason_code, exception_summary,
is_sharpe, oos_sharpe, median_oos_sharpe, sharpe_decay,
n_trades_oos, profit_factor, max_drawdown, win_rate, consistency_ratio,
p_coarse, p_refined, bh_fdr_significant, dsr_pvalue,
surrogate_pctl, surrogate_z, surrogate_mean, surrogate_std, surrogate_flagged,
spec: { entry, filters, exits, context, side, timeframe, fee_rate, slippage_rate }
```

**Nya fält du lägger till (se sektion 7: Trade Data):**

```
trade_summary: { median_net_return_per_trade, median_gross_return_per_trade,
                 mean_net_return_per_trade, std_net_return_per_trade,
                 n_positive_trades, n_negative_trades }
duration_summary: { median_bars, p25_bars, p75_bars }  // null om data saknas
fold_results: [ {fold, oos_sharpe, n_trades}, ... ]     // null om data saknas
```

Viktigt: `spec` innehåller enskild variant-config (inte sweep-grid). Post-mortem måste aggregera över alla varianter.

### `manifest.json` (nuvarande format)

```json
{
  "sweep_name": "sweep_001_ema_pvsra",
  "git_commit": "1606cc2",
  "data_range": {"start": "2025-08-01", "end": "2025-12-31"},
  "symbol": "BTCUSDT",
  "n_variants": 288,
  "n_passed_sanity": 270,
  "n_bh_survivors": 0,
  "n_significant": 0,
  "runtime_seconds": 471.8,
  "config_hash": "662aaf40...",
  "data_fingerprint": "a6f49dba7611ea15",
  "environment": {},
  "seeds": {},
  "p_value_stages": {}
}
```

Du lägger till `experiment_id` här.

### Sweep config YAML (format)

```yaml
name: "sweep_001_ema_pvsra"
symbol: "BTCUSDT"
theory: |
  EMA crossovers capture momentum shifts...
sides: [long]
entry:
  type: ema_cross
  params:
    fast: [5, 8, 13, 21]
    slow: [21, 34, 50, 100]
filters:
  - type: ema_gating
    params: {period: [50, 100, 200], mode: ["above"]}
  - type: pvsra_filter
    params: {vol_mult: [1.5, 2.0], window: [3, 5, 10]}
exits:
  - type: ema_stop
    params: {period: [50]}
timeframes: ["1h"]
walk_forward:
  train_months: 3
  test_months: 1
  step_months: 1
  embargo_bars: 200
  min_trades_per_fold: 5
fees:
  taker_fee_pct: 0.06
  slippage_pct: 0.01
statistics:
  coarse_permutation_n: 500
  refined_permutation_n: 10000
  surrogate_count: 100
```

---

## Testkrav

### `tests/test_fingerprint.py`

- Samma config med keys i annan ordning → samma `experiment_id`
- Ändra en param → annan `experiment_id`
- Ändra fee → annan `experiment_id`
- Ändra symbol → annan `experiment_id`

### `tests/test_post_mortem.py`

- Parse sweep_001 variants.jsonl → valid `post_mortem.json` med alla fält
- `fee_decomposition.data_source` är en av `"npz"` / `"trade_summary"` / `"config_only"`
- `fee_decomposition.estimated_fee_drag_per_trade` == `2 * (fee_rate + slippage_rate)` från config
- Om `trade_summary` finns i JSONL: `median_gross_return_per_trade` och `median_net_return_per_trade` är icke-null
- Om `trade_summary` saknas: fälten är `null`, `data_source` == `"config_only"`
- `primary_failure_mode` är en valid enum
- `next_experiment_constraints` är icke-tom lista
- Tom variants.jsonl → exception (fail-closed)
- Variant med saknade fält → hanteras gracefully (skip, logga)
- Idempotens: kör två gånger → identisk JSON (exkl `generated_ts`)

### `tests/test_knowledge_base.py`

- Init DB → alla tabeller existerar
- Ingest sweep_001 → sweeps rad + findings + coverage + artifacts
- Query "ema_cross" → returnerar sweep_001
- Stats → visar 1h, 0.06% fee, ema_cross signal
- Duplicate experiment_id → upsert (inte crash)
- Events skrivs korrekt vid ingest
- Export-events → kronologisk ordning per sweep_id

### Integration test

- Fixture: minimal `variants.jsonl` (5 rader) + `manifest.json`
- Kör: post_mortem → ingest → query → verifierar hela kedjan

### Failure path test

- Trasig JSONL → exception från post_mortem → `STEP_FAILED` event → status `NEEDS_REVIEW`

**Coverage:** Minst 80% på `post_mortem.py` + `knowledge_base.py` + `experiment_fingerprint.py`.

---

## Backfill

Efter allt är byggt, kör backfill på sweep_001:

```bash
python -m analysis.post_mortem results/sweeps/sweep_001_ema_pvsra/
python -m research.knowledge_base ingest results/sweeps/sweep_001_ema_pvsra/
python -m research.knowledge_base query "ema_cross"
python -m research.knowledge_base stats
python -m research.knowledge_base export-events --last 20
```

Alla fem kommandon ska fungera och producera meningsfullt output.

---

## Vad du INTE ska göra

- Rör inte befintliga tester (de ska fortfarande passera)
- Bygg inte UI (fas 3)
- Bygg inte hypothesis generator (fas 2)
- Bygg inte proposals-tabell (fas 2)
- Använd inte externa dependencies utöver stdlib + pandas + numpy (som redan finns)
- Lägg inte till vector embeddings, ML, eller Celery

---

## Kvalitetskrav

- Alla befintliga tester måste fortfarande passera
- Nya tester måste passera
- Typing: alla publika funktioner har type hints
- Docstrings: alla publika funktioner
- Inga `print()` i library-kod (använd `logging`)
- CLI-kommandon får printa

---

## Definition of Done (sammanfattning)

Se fullständig DoD i `docs/FAS1_DOD.md`. Kortversion:

- ✅ `python -m analysis.post_mortem results/sweeps/sweep_001_ema_pvsra/` producerar `post_mortem.json` med alla structured fields + `.md`
- ✅ `python -m research.knowledge_base ingest ...` uppdaterar research.db + skriver events
- ✅ `python -m research.knowledge_base query "ema_cross"` visar sweep_001 med artifacts + failure mode
- ✅ `python -m research.knowledge_base export-events --last 20` visar kronologisk eventsekvens
- ✅ `experiment_id` stabil oavsett YAML-ordning
- ✅ `sweep_runner` blockerar duplicate `experiment_id` (fail-closed)
- ✅ Trasig JSONL → `STEP_FAILED` + `NEEDS_REVIEW` (aldrig tyst COMPLETE)
- ✅ Alla tester passerar (gamla + nya)
- ✅ 80% coverage på nya kärnmoduler
