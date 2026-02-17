# Fas 1 — Definition of Done (v3.2)

**Status:** ACTIVE
**Scope:** Fas 1 — Post-Mortem + Knowledge Base + Events (SQLite) + Experiment Fingerprint + Trade Data
**Senast uppdaterad:** 2026-02-17 (v3.2: trade data, finalize_sweep, fee-formel, CC-autonomi-regler)

---

## A. Globala principer (måste uppfyllas överallt)

**Fail-closed**
Om ett steg fallerar får sweepen inte markeras som färdig. Status hamnar i tydligt läge: `NEEDS_REVIEW` eller motsvarande.

**Determinism & spårbarhet**
Outputs ska vara reproducerbara givet samma inputs (data + config + kodversion). Allt ska kunna härledas via `experiment_id`, `sweep_id`, `event_id`.

**Inga silent drops**
Om något ignoreras/filtreras måste det loggas med orsak (event + reason_code).

**Packages**
`utils/__init__.py` och `analysis/__init__.py` (tomma) måste finnas för `python -m` imports.

**Namnmappning**
`manifest.symbol` → `sweeps.asset`. DB-kolumnen heter `asset`.

---

## B. Identiteter & kontrakt

### B1) Experiment Fingerprint

**Done när:**

`compute_experiment_id(...)` → `str` finns och:
- Tar canonicaliserad sweep-config + signal_spec version/hash + dataset manifest + fill-model/fee/slippage-config
- Returnerar stabil SHA-256 (hex)

`experiment_id` sparas i:
- `results/sweeps/{sweep_id}/manifest.json`
- `research.db` — `sweeps.experiment_id`
- `research.db` — `events.experiment_id` på relevanta events

**Fail-closed krav:**

`sweep_runner` vägrar köra om `experiment_id` redan finns i KB med status COMPLETED|ANALYZED|INGESTED. Default = block.

Override via `--force` måste logga event `SWEEP_FORCE_RERUN` med motivering.

---

## C. Events & State Machine (SQLite = single source of truth)

### C1) Events-tabell (primär)

**Done när:**

`events` finns som tabell i `research.db`, append-only (ingen UPDATE/DELETE i normal drift).

Varje eventrad har minst:
- `event_id` (uuid)
- `ts` (ISO8601)
- `event_type` (enum)
- `sweep_id`
- `experiment_id`
- `status`
- `payload_json` (TEXT)

Optional: spegla till `research/events.jsonl` som backup/observability. SQLite = sanning.

CLI-funktion för export/visning (se H4).

### C2) Minsta eventtyper

- `SWEEP_STARTED`
- `SWEEP_COMPLETED`
- `SWEEP_FORCE_RERUN` (vid `--force` override)
- `POST_MORTEM_STARTED`
- `POST_MORTEM_COMPLETED`
- `KB_INGEST_STARTED`
- `KB_INGEST_COMPLETED`
- `STEP_FAILED` (payload: `failed_step`, `error_type`, `error_msg`, `stacktrace_trunc`)

### C3) State transitions

**Done när:**

Statusmodell (i `sweeps.status`):

`RUNNING` → `COMPLETED` → `ANALYZED` → `INGESTED`

Post-mortem fail ⇒ `COMPLETED` + `NEEDS_REVIEW`
Ingest fail ⇒ `ANALYZED` + `NEEDS_REVIEW`

Status kan alltid härledas från events + sweeps-tabellen. Ingen implicit "vi antar det gick bra".

---

## D. Post-Mortem Generator

### D1) Inputs/Outputs

**Done när:**

Modul: `analysis/post_mortem.py` (importerbar + körbar som `python -m`)

Input:
- `results/sweeps/{sweep_id}/variants.jsonl`
- `results/sweeps/{sweep_id}/manifest.json`
- Sweep-config path (via manifest eller CLI-arg)

Output:
- `results/sweeps/{sweep_id}/post_mortem.json`
- `results/sweeps/{sweep_id}/post_mortem.md`

### D2) Post-mortem JSON (minimikrav)

**Done när `post_mortem.json` innehåller:**

`sweep_id`, `experiment_id`

`created_ts` — **idempotensregel:** sätts till `sweep_completed_ts` från manifest (stabil vid re-run). Separat `generated_ts` får ändras vid re-run men får inte bryta determinism för övriga fält.

`summary`:
- `variant_count`, `survivor_count`
- `best_variant` (id + metrics)

`fee_decomposition`:
- `median_gross_return_per_trade` (från trade_summary, null om saknas)
- `median_net_return_per_trade` (från trade_summary, null om saknas)
- `estimated_fee_drag_per_trade` = `2 * (fee_rate + slippage_rate)` (ALLTID beräkningsbart från config)
- `fee_share_of_loss_pct` (null om gross saknas)
- `data_source`: `"npz"` | `"trade_summary"` | `"config_only"` (använd bästa tillgängliga)

`trade_duration_distribution`:
- median, p25, p75 (bars) — från `duration_summary` i JSONL. `null` om duration-data saknas.

`fold_stability`:
- per fold: oos_sharpe + n_trades — från `fold_results` i JSONL. `null` om fold-data saknas.
- dispersionmått: std + IQR

`primary_failure_mode` (enum) + `evidence`

Enums: `FEE_DRAG` | `OVERTRADING` | `REGIME_FRAGILE` | `GOOD_GROSS_DIES_NET` | `LEAK_SUSPECT` | `LOW_SIGNAL` | `NO_CONVERGENCE`

`most_promising_region`:
- param-ranges för "least bad" + score

`next_experiment_constraints`:
- Maskinläsbara constraints, ex:
  - `min_trade_duration_bars >= X`
  - `max_trades_per_day <= Y`
  - `must_use_volatility_scaled_exit = true`

### D3) Markdown-rapport

**Done när `.md` innehåller:**
- Kort Executive summary
- Tydlig slutsats: `dead` / `maybe` / `gross-good-net-dead`
- Tabell: top 10 varianter (id + nyckelmetrics)
- "Why it failed" (primary failure mode + stöd)
- "What to try next" (constraints + 1–3 riktade förslag)

### D4) Fail-closed

**Done när:**

Trasig/otolkbar `variants.jsonl` ⇒
- `STEP_FAILED` event loggas
- Sweep status = `NEEDS_REVIEW`
- Inga COMPLETED→ANALYZED transitions

---

## E. Knowledge Base (SQLite)

### E1) Schema & init

**Done när:**

`research/kb_schema.sql` definierar tabeller + index.
`research/knowledge_base.py` kan initiera DB, skriva data, köra queries.

### E2) Minsta tabeller

`sweeps` — sweep_id, experiment_id (UNIQUE), status, created_ts, asset, timeframe, date_range_start, date_range_end, fee_bps, slippage_bps, config_json, best_metrics_json, primary_failure_mode

`events` — event_id, ts, event_type, sweep_id, experiment_id, status, payload_json

`findings` — finding_id, sweep_id, statement, tags_json, confidence, evidence_refs_json

`coverage` — id, entity_type, entity_name, sweep_id

`artifacts` — sweep_id, artifact_type, path, sha256, bytes, created_ts

### E3) Ingest-kontrakt

**Done när ingestion skapar/uppdaterar:**
- `sweeps` rad (upsert)
- Minst 1–3 `findings` rader
- `coverage` för signals, timeframe, fee-regime, asset
- `artifacts` för manifest + variants.jsonl + post_mortem.json/md (med sha256 + bytes)

### E4) Query CLI

**Done när:**

`python -m research.knowledge_base query "ema_cross"` returnerar:
- Sweeps + status + failure_mode + best metric + artifact paths

`python -m research.knowledge_base stats` visar:
- Timeframes testade, fee-regimer testade, signals testade, assets testade

---

## F. Integration i sweep_runner

### F1) Auto-hook + finalize_sweep

**Done när `sweep_runner.py` kör i ordning:**

1. Beräkna `experiment_id` + dedup-check
2. Skriv `SWEEP_STARTED` event
3. Kör sweep (variants.jsonl + manifest)
4. Skriv `SWEEP_COMPLETED` event
5. Anropa `finalize_sweep(sweep_dir, db_path)` som:
   - Kör post-mortem (events start/complete)
   - Kör KB-ingest med `emit_events=False` (sweep_runner skriver events)
   - Vid fel: `STEP_FAILED` + `NEEDS_REVIEW`, stoppar kedjan

`finalize_sweep()` är testbar separat (utan att köra hela sweep-motorn).

`ingest_sweep(db, path, emit_events=True)` — CLI använder default True, sweep_runner använder False.

### F2) Idempotens

**Done när:**
- Post-mortem re-run ger identiska analytiska outputs (exkl. `generated_ts`)
- Ingest är idempotent (upsert på `sweep_id`/`experiment_id`; artifacts dedup på sha256/path)

---

## F½. Trade Data (ny i v3.2)

### F½.1) Summary-scalars i JSONL (obligatoriskt)

**Done när:**

`VariantResult.to_dict()` inkluderar (för varianter med status OK):

- `trade_summary`: `median_net_return_per_trade`, `median_gross_return_per_trade` (net + 2*(fee+slip)), `mean_net_return_per_trade`, `std_net_return_per_trade`, `n_positive_trades`, `n_negative_trades`
- `duration_summary`: `median_bars`, `p25_bars`, `p75_bars` — OM durations finns i minnet. Annars: utelämna fältet.
- `fold_results`: `[{fold, oos_sharpe, n_trades}, ...]` — OM fold-data finns i minnet. Annars: utelämna fältet.

Befintlig kommentar "Don't include trade returns" kvarstår — detta är summaries, inte arrays.

### F½.2) trade_data.npz bakom flagga (valfritt)

**Done när:**
- `--save-trade-data` flag eller `statistics.save_trade_data: true` i YAML skriver `trade_data.npz`
- `manifest.json` innehåller `trade_data_present: bool` + `trade_data_sha256` (om fil skapas)
- Post-mortem använder NPZ om den finns, annars trade_summary, annars config_only
- Artifacts-tabell registrerar NPZ med `artifact_type: "trade_data"`

### F½.3) Graceful degradation

**Done när:**
- Post-mortem producerar valid JSON oavsett vilken data-nivå som finns
- Alla schema-nycklar finns ALLTID (värde = null + note om data saknas)
- `fee_decomposition.data_source` visar vilken källa som användes
- `primary_failure_mode` logik hanterar null-fält utan crash (fallback-kedja)

---

## G. Testkrav (pytest)

**Done när följande testkategorier passerar:**

**Unit:**
- Canonicalisering + experiment_id stabilitet (YAML-ordning påverkar ej hash)
- Post-mortem parser klarar edge cases + fail-closed
- `fee_decomposition.estimated_fee_drag_per_trade` == `2 * (fee_rate + slippage_rate)` från config
- `fee_decomposition.data_source` är en av `"npz"` / `"trade_summary"` / `"config_only"`
- Post-mortem med trade_summary i JSONL: gross/net är icke-null
- Post-mortem utan trade_summary: gross/net är null, data_source == `"config_only"`

**Integration:**
- Fixture sweep → post_mortem → ingest → query fungerar
- `finalize_sweep()` testbar separat med fixture-data

**Failure path:**
- Trasig JSONL → `STEP_FAILED` + status `NEEDS_REVIEW` + ingen ANALYZED/INGESTED
- `finalize_sweep()` med trasig data → returnerar `NEEDS_REVIEW` + korrekt event

**Coverage:**
- Minst 80% på `post_mortem.py` + `knowledge_base.py` + `experiment_fingerprint.py`
- CLI wrappers/boilerplate undantagna (ska vara tunna)

---

## H. Manuell acceptans (beviskedja)

**H1:**
```bash
python -m analysis.post_mortem results/sweeps/sweep_001_ema_pvsra/
```
→ Skapar `post_mortem.json` (alla D2-fält) + `post_mortem.md`

**H2:**
```bash
python -m research.knowledge_base ingest results/sweeps/sweep_001_ema_pvsra/
```
→ Uppdaterar `research.db` + skriver events

**H3:**
```bash
python -m research.knowledge_base query "ema_cross"
```
→ Visar sweep_001 + artifacts + failure mode

**H4:**
```bash
python -m research.knowledge_base export-events --last 20
```
→ Visar korrekt eventsekvens utan luckor (läser från SQLite)

---

## I. Inte Done om…

- Post-mortem saknar `primary_failure_mode` eller `next_experiment_constraints`
- KB saknar coverage så du inte kan svara "vad har testats?"
- Systemet kan hamna i "klar" trots post-mortem/ingest-fel
- `experiment_id` ändras av YAML-omordning
- Events saknar `STEP_FAILED` för något failure scenario
- Artifacts saknar sha256-hash
- Events går inte att läsa ut kronologiskt per sweep (audit trail trasig)

---

*Denna DoD säkerställer att Fas 2 (Hypothesis Generator) har en riktig kunskapsbas att resonera över — inte bara data, utan strukturerade beslutspunkter och bevisade slutsatser att citera.*
