# FPR Research Station v3.2 — Project Plan

**v3.2** — Uppdaterad 2026-02-17 efter hypothesis-generator-review (GPT expansion + kritik inkorporerad)

---

## Vision

**Mål:** En autonom forskningsstation som systematiskt producerar statistiskt validerade handelsstrategier — inte en backtest-motor som kräver manuell orkestrering.

Dagsläget: FPR v2.1.1 är en korrekt backtesting-pipeline. Den kan testa en hypotes och svara ja/nej. Men varje iteration kräver en människa som formulerar hypotes, promptar CC, granskar resultat och formulerar nästa hypotes. Varje cykel tar timmar.

Målet: En miljö där agenter (CC, Claude, GPT) driver forskningsloopen och du övervakar, godkänner och styr riktning.

---

## Arkitekturskiss

```
┌───────────────────────────────────────────────────────────────┐
│                    FPR RESEARCH STATION v3.2                   │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌────────────┐   ┌────────────┐   ┌────────────┐            │
│  │ KNOWLEDGE   │◄──│ POST-      │◄──│ SWEEP      │            │
│  │ BASE        │   │ MORTEM     │   │ ENGINE     │            │
│  │ (SQLite +   │   │ (actionable│   │ (v2.1.1)   │            │
│  │  events)    │   │  decisions)│   │            │            │
│  └──────┬─────┘   └────┬───────┘   └─────▲──────┘            │
│         │            │                    │                    │
│         │            │ PREDICTION AUDIT   │                    │
│         │            │ (predictions vs    │                    │
│         │            │  utfall → KB)      │                    │
│         │            │                    │                    │
│         │    ┌────────────┐              │                    │
│         │    │ EXPERIMENT │──── dedup ────┘                    │
│         │    │ FINGERPRINT│                                    │
│         │    └────────────┘                                    │
│         │                                                      │
│         │  ┌──────────────────────────────────────────┐        │
│         │  │          HYPOTHESIS PIPELINE              │        │
│         │  │                                          │        │
│         │  │  ┌──────────┐  ┌──────────┐  ┌─────────┐│        │
│         └─►│  KB       │─►│ HYPO-    │─►│ VERIFIER││        │
│            │  RETRIEVAL│  │ THESIS   │  │ (kod,   ││        │
│            │  (top-N   │  │ GEN (LLM)│  │ ej LLM) ││        │
│            │  findings │  └──────────┘  └────┬────┘│        │
│            │  +coverage│                     │     │        │
│            │  gaps)    │  reject ◄───────────┤     │        │
│            └──────────┘              pass    │     │        │
│                                              │     │        │
│  └──────────────────────────────────────────┘      │        │
│                                     │                       │
│                                     ▼                       │
│                              ┌────────────┐                │
│                              │ APPROVAL   │                │
│                              │ QUEUE      │─ approved ─► SWEEP │
│                              └──────┬─────┘                │
│                                     │                       │
│  research/api.py  ┌────────────┐    │                       │
│  ▲ alla vyer      │    UI      │    │ ◄─ Du sitter här      │
│  │ anropar API    │  (Feed,    │    │                       │
│  │ inte DB direkt │   KB, Q)   │    │                       │
│                   └────────────┘                            │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ EVENTS TABLE (append-only i research.db)               │   │
│  │ PROPOSED → APPROVED → RUNNING → COMPLETED →            │   │
│  │ ANALYZED → INGESTED → ARCHIVED                         │   │
│  └──────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────┘
```

---

## Systemkomponenter

### 1. Experiment Fingerprint + Dedup

Unik identitet per experiment. Canonicaliserad sweep-config + signal_spec + dataset manifest + fill model → SHA-256[:16].

KB blockerar duplicates. Hypothesis Generator får error vid duplicate förslag.

### 2. Post-Mortem Generator

Automatisk analys efter varje sweep. Producerar actionable beslutspunkter.

**Analytics:** Fee-decomposition, parameter-heatmap, trade-duration-distribution, gross vs net return, fold-stabilitet.

**Structured decisions:** `primary_failure_mode`, `most_promising_region`, `regime_map`, `sensitivity`, `next_experiment_constraints`.

### 3. Knowledge Base (SQLite)

Maskinläsbar kunskapsbas: events, sweeps, findings, coverage, proposals.

Tre frågor: Vad har vi testat? Vad lärde vi oss? Vad är nästa bästa experiment?

### 4. Hypothesis Pipeline (tre steg, v3.2)

1. **KB Retrieval** — top-N findings + coverage-gaps + constraints (~2000 tokens)
2. **Hypothesis Generator** — LLM producerar HypothesisFX-objekt (9 fält: experiment_intent, evidence_refs, novelty_claim, expected_mechanism, predictions, expected_failure_mode, kill_criteria, compute_budget, sweep_config_yaml)
3. **Proposal Verifier** — deterministisk kod (ej LLM): schema + dedup + KB-refs + budget + predictions-format

**Prediction Audit:** Post-sweep jämförelse predictions vs utfall, diff som finding i KB.

### 5. Approval Queue (dubbla köer)

- **Sweep Proposal Queue** — configs med existerande signaler
- **Signal Proposal Queue** — nya signal-idéer (kräver mänsklig gate)

### 6. Backend API

`research/api.py` — all data-access. UI anropar API, aldrig DB direkt.

### 7. UI

TUI (Textual) i fas 3. Vyer: Feed, KB Browser, Sweep Detail, Approval Queue, Live View.

---

## Fas-plan

### Fas 1: Post-Mortem + Knowledge Base + Fingerprint + Events

Automatisk analys, minne, dedup, audit trail. Se `docs/FAS1_BUILD_PROMPT.md` och `docs/FAS1_DOD.md`.

### Fas 2: Hypothesis Pipeline

Tre-stegs pipeline med HypothesisFX-kontrakt + prediction audit.

### Fas 3: UI Shell (TUI)

Textual-baserad TUI via `research/api.py`.

### Fas 4: Live Orchestration

Approve → start sweep från UI. Live progress. State machine enforced.

### Fas 5: Paper Trading Bridge

Bybit testnet. Signal → order. Live Sharpe tracking. Auto kill-switch.

### Fas 6: Multi-Asset Validation

BTC-survivors testas på ETH + SOL. Cross-asset = filter, inte sökutrymme.

---

## Design-principer

- **Fail-closed genomgående** — inget steg kan "glida förbi"
- **Agenten föreslår, du godkänner** — ingen automatisk exekvering
- **Agenten måste citera** — evidence refs överallt
- **Allt loggas, inget förloras** — append-only events
- **Kunskap ackumuleras, arbete upprepas inte** — fingerprint + dedup
- **Signaler och sweeps är separata köer**
- **Kod verifierar, LLM kreerar** (v3.2)
- **Predictions mäts, inte bara loggas** (v3.2)
- **Backend API före UI**

---

## Framtida refactors (inte i scope, men noterade)

### Trade Ledger (post-fas-1)

Nuvarande `backtest.py` producerar numpy-arrays av returns. Post-mortem måste reverse-engineera per-trade metrics från aggregerade scalars (A1) eller sparade arrays (A2). Rätt lösning: strukturerat `Trade`-objekt (`entry_bar`, `exit_bar`, `entry_price`, `exit_price`, `gross_pnl`, `net_pnl`, `fees_paid`, `duration_bars`). Gör A2 (NPZ) överflödig.

Inkluderar: explicit schema för `VariantResult` (JSON Schema eller dataclass med `to_json()`/`from_json()`).

Trigger: efter fas 1, innan fas 2 om tid finns.

---

## Changelog

- **v3.2:** Hypothesis Pipeline review. Tre-stegs pipeline, HypothesisFX 9-fält, prediction audit, information gain-heuristik, deterministisk verifier.
- **v3.1:** Extern review. Fingerprint, structured decisions, anti-curve-fit-kontrakt, dubbla köer, backend API, events + state machine.
- **v3.0:** Initial projektplan.
