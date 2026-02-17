# FPR Research Platform — Claude Code Build Prompt v2.1

> **Mål:** Bygg allt (fas 1–5), gröna tester, kör sweep_001 på IS, generera alla artifacts.
> **Checkpoint-regel:** fail → fix → rerun samma checkpoint → diff/commit → vidare.
> **Aldrig:** silent drop, performance-filter före BH, same-bar entry+exit, HTF utan shift(1), pivots utan right_bars confirmation, FeatureStore delad mellan folds.
> **Output:** `results/sweeps/`, `results/checkpoints/`, `results/reports/`, alla tester gröna.

_2026-02-16. För Claude Code. Refererar till: v2.0 Spec, H1 Post-Mortem, v1.1 Backtest Spec._

---

## v2.1 Delta — Kritiska fixar från peer review

### ⚠️ DELTA INDEX — Top Constraints (läs dessa först)

Följande deltas är de mest load-bearing. Om CC bara internaliserar 12 regler, låt det vara dessa:

- **Δ4** — Fill: next-bar open, side-aware. Slippage gör ALLTID fill sämre.
- **Δ8** — Tvåstegs p-value: coarse (500 perm) → BH-FDR → refined (10k perm).
- **Δ17** — FeatureStore låst till (df, timeframe). Ny instans per (fold, segment, TF).
- **Δ28** — Fas-checkpoint: pytest + smoke → fixa → kör om → commit → nästa fas.
- **Δ29** — BH-FDR-ordning HÅRD: sanity-filter (ej performance) före p-values.
- **Δ30** — Side-resolver: SIDE_SPLIT=True krävs. Bas i registry → aldrig resolve.
- **Δ32** — Ingen silent drop: varje variant → OK/INVALID/ERROR i variants.jsonl.
- **Δ33** — Canonical trade return i `engine/trades.py`. Alla stats använder den.
- **Δ34** — No same-bar entry+exit. Entry t+1 open → exit tidigast t+2 open.
- **Δ36** — Primary index = strategy.timeframe. HTF shift(1) efter resample.
- **Δ38** — Standardiserade reason_codes (E_SCHEMA, E_SIGNAL_NAN, etc.).
- **Δ40** — Permutation test vektoriserad. Ingen Python-loop per variant.

Följande designbeslut är **låsta** och får inte ändras under build:

1. **Fee/slippage i decimal (rate), inte procent.** `taker_fee_rate = 0.0006` (0.06%), `slippage_rate = 0.0001` (0.01%). YAML får skriva procent men konverteras vid parsing.

2. **En komponent = en roll, en side.** Ingen `entry+filter`. PVSRA: `pvsra_entry.py` + `pvsra_filter.py`. Stoch RSI: `stoch_rsi_entry.py` + `stoch_rsi_filter.py`. Riktade signaler: `price_action_long.py` + `price_action_short.py`, `ema_stop_long.py` + `ema_stop_short.py` etc.

3. **Registry via ClassVar KEY**, inte instansfält. Se interface nedan.

4. **Fill-modell: nästa bars open, side-aware.** Signaler evalueras på bar close. Long: entry=open*(1+slip), exit=open*(1-slip). Short: entry=open*(1-slip), exit=open*(1+slip). Slippage gör ALLTID fill sämre. Fees på båda sidor. Ingen same-bar fill.

5. **HTF-data availability-regel.** Vid MTF: bara senaste _stängda_ HTF-bar används. Implementera med `shift(1)` på HTF-features efter resampling. Enhetstest obligatoriskt.

6. **Pivot-confirmation-regel.** `price_action` och `rsi_divergence`: pivots confirmas först efter `right_bars` barer. Signal triggar på confirmation bar, inte pivot bar. Leak-test: beräkna signal på `df[:N]`, append slumpdata `df[:N+M]`, assert identiska värden för index ≤ N.

7. **IS/Holdout utan överlapp.** IS: `2025-08-01` → `2025-12-31` (inklusiv). Holdout: `2026-01-01` → `2026-02-15` (inklusiv). Ingen data i båda.

8. **Tvåstegs p-value för BH-FDR.** Steg 1: coarse p-value (500 permutationer) för ALLA varianter — möjliggör korrekt BH-FDR. Steg 2: refined p-value (10 000 perm) bara på varianter som klarar BH vid coarse. Teststat: OOS mean netto trade return (efter fees+slippage). Null: sign-flip permutation (one-sided). DSR skew/kurt beräknas på OOS trade returns. Manifest loggar båda: `p_coarse`, `p_refined`.

9. **FeatureStore / IndicatorCache.** Precomputa alla indikatorer en gång per (feature, params, timeframe). Signalernas `compute(df, cache)` tar både df och cache. Cache är bunden till (df, timeframe) — `cache.get('ema', period=50)` utan df-arg. Fold-aware: inga globala percentiler/statistics som läcker.

10. **Regim på daily med shift(1).** Regim-klassificering sker på daglig data (daily SMA200, daily ATR). Mappas till intraday via date-join MED `shift(1)` — intraday-bars under 2025-09-10 använder regim från 2025-09-09 (du vet inte dagens close intraday). Enhetstest obligatoriskt.

11. **Variant-ID = sha256(canonical_json(spec))[:12].** Spec inkluderar: components+params, side, timeframe, fee/slippage, WF-config, symbol. Sort keys. Floats serialiseras med 10 decimaler (`f'{v:.10f}'`) för absolut determinism. Alternativt: konvertera fees/slippage till int ppm internt (0.0006 → 600 ppm).

12. **Short-capable motor, side-separerade sweeps.** Motorn (backtest/portfolio) stödjer `direction` fullt ut (long + short). Strategy har fält `side: 'long' | 'short'`. Sweeps körs som separata experiment per side — aldrig blandat i samma variant-ranking. Varje signal deklarerar `SUPPORTED_SIDES: ClassVar[set] = {'long'}` eller `{'long','short'}`. Om en signal saknar short-stöd → varianten är INVALID när `side='short'` (fail-closed, ingen tyst fallback). Signaler börjar med `{'long'}` och får `'short'` adderat först när explicit short-logik är implementerad och testad.

13. **Negativ kontroll per sweep (distributional).** Kör topp-10 varianter på N IAAFT-surrogater. Bygg distribution av surrogat-Sharpe per variant. Rapportera `surrogate_pctl` = percentile rank där real Sharpe hamnar i surrogatdistributionen. Flagga om real Sharpe < 90:e percentilen av surrogat-distribution.

14. **Seeds i manifest.** Alla stokastiska operationer (permutationer, surrogater, random search) måste använda reproducerbara seeds. Manifest loggar: `rng_seed_global`, `seed_permutation`, `seed_surrogates`, `seed_sampler`. `python main.py repro` använder exakt dessa seeds.

15. **Gap-policy: split segments + aggregering.** Om data har gap > 5 min: splittra df i kontinuerliga segment. WF/backtest körs per segment. Aggregering: concat trade_returns över segment (trade-level). Equity: concat med reset gap (position stängs vid segmentslut, ingen holding över gap). Per-segment metrics loggas separat + total. Flagga om >80% av trades eller PnL från ett enda segment.

16. **Embargo ≥ warmup (hård).** `effective_embargo = max(config.embargo_bars, strategy.warmup_bars())`. YAML kan inte sätta lägre. Enhetstest obligatoriskt.

17. **FeatureStore: bunden till (df, timeframe) — LÅST.** Konstruktor: `FeatureStore(df: pd.DataFrame, timeframe: str)`. Cache-key: `(name, sorted_params)` — timeframe implicit via store-instans. `get('ema', period=50)` utan df-arg. En store per fold+TF. Store är strikt bunden till sitt (df, timeframe) — om samma store råkar användas för ett annat segment eller resamplade df kolliderar cachen. Ny instans per fold+TF, alltid. Förhindrar index-mismatch och cache-kollision.

18. **Joblib: workers laddar data från disk.** Skicka inte stora DataFrames till workers via pickling. Workers laddar från parquet (path i args). Precompute features per fold/TF sparas temporärt (parquet/npz), workers läser.

19. **Riktade signaler kräver side-varianter.** Signaler som är inherent riktade (price_action, rsi_divergence) implementeras som två filer: `price_action_long.py` + `price_action_short.py`, `rsi_divergence_bull.py` + `rsi_divergence_bear.py`. Exits som är side-beroende (ema_stop, atr_stop) implementeras likaså som två filer med inverterad logik. EMA cross, PVSRA, volume_filter, time_stop är side-agnostiska och behöver inte splittas.

20. **Fill-modell är side-aware.** Long: entry = open*(1+slip), exit = open*(1-slip). Short: entry = open*(1-slip), exit = open*(1+slip). Slippage gör alltid fill sämre. Enhetstest per side obligatoriskt.

21. **Embargo-bars räknas i strategy.timeframe.** `embargo_bars: 200` i YAML betyder 200 bars i den timeframe strategin kör på (`strategy.timeframe`), INTE 1m-bars. `warmup_bars()` returnerar också bars i strategy.timeframe. Enhetstest: samma embargo_bars på 5min vs 1h ger olika absolut tid. Exempel: `embargo_bars: 200` på 5min = ~17h, på 1h = ~8 dagar.

22. **Canonical spec util.** Implementera `utils/canonical.py` med `canonicalize_spec(spec) -> str` som hanterar float-precision (10 dec), sort_keys, determinism. Används av ALLA: `Strategy.variant_id()`, sweep manifest, repro. Enhetstest: samma spec från olika kodvägar ger identisk hash.

23. **BH-FDR pre-filter är data-sanity, inte performance.** Pre-filter innan p-values: `n_trades >= 30` + `valid variant` (inga NaN, korrekt dtype). Performance-filter (PF, DD, Sharpe) appliceras EFTER signifikanstestning, inte före. Detta bevarar BH-FDR-garantier.

24. **Ingen pyramiding.** Entry-signaler ignoreras när position är öppen. En position åt gången, alltid. Enhetstest: två entries i rad → bara första öppnar position.

25. **Timeframe-mappning.** `'5min'` → `'5T'`, `'15min'` → `'15T'`, `'1h'` → `'1H'`, `'4h'` → `'4H'`, `'1d'` → `'1D'`. All resampling från 1m bas. Definiera i `utils/timeframes.py`.

26. **YAML side-resolver för exits.** Side-beroende exits (`ema_stop`, `atr_stop`) skrivs i YAML som `type: ema_stop`. Composer resolvar automatiskt till `ema_stop_{side}` baserat på `strategy.side`. Exempel: `type: ema_stop` + `side: long` → `ema_stop_long`. Om resolved KEY inte finns i registry → fail-closed error. Side-agnostiska signaler (`time_stop`) används som-är. Resolver-logik implementeras i `combinator/composer.py`.

27. **Funding ignoreras i v2.1 (explicit scope).** Perpetual swap funding rates implementeras INTE i denna version. Detta begränsar realism särskilt för short-positioner med lång hålltid. Resultat ska INTE tolkas som live-ready. Funding läggs till i v2.2/v3.

28. **Fas-checkpoint: pytest + smoke-test är OBLIGATORISK.** Efter varje fas (1–5): kör `pytest` + minimal CLI smoke-test. Om fail: fixa, kör om SAMMA checkpoint tills grönt, commit, först därefter nästa fas. **Ingen ny fas innan grönt.** CC får INTE hoppa över denna regel. Om git inte är tillgängligt: spara diff i `results/checkpoints/phase_X.diff` för spårbarhet. Smoke-tests per fas: Fas 1: `python main.py fetch --help` + `pytest tests/test_data.py tests/test_backtest.py tests/test_walk_forward.py`. Fas 2: `python main.py signals` + `pytest tests/test_signals.py tests/test_lookahead.py`. Fas 3: `python main.py validate config/sweeps/sweep_001_ema_pvsra.yaml` + sweep dry-run. Fas 4: `pytest tests/test_statistics.py` + `python main.py leaderboard --help`. Fas 5: full `pytest` + `python main.py repro`.

29. **BH-FDR-ordning är HÅRD — ingen performance-filtrering före p-values.** Performance-filter (PF, Sharpe, MaxDD, consistency, win rate) får ENBART appliceras EFTER BH-FDR + refined p-values. Före BH-FDR tillåts ENBART sanity-filter. **Sanity-filter definieras exakt som:** (1) `valid variant` — inga NaN i signals/trades, korrekt dtype, (2) `n_trades >= 30`, (3) segment OK (inga data-integritetsproblem), (4) timeframe OK. **Sanity-filter får INTE referera till performance-metrics** — ingen PF, inget Sharpe, ingen DD, ingen consistency, inte ens "PF måste vara definierad" eller "Sharpe måste gå att räkna". Enhetstest: verifiera att antalet varianter som når coarse p-value == antalet sanity-survivors (ingen mellanliggande filtrering). Denna punkt övertrumfar alla andra steg-beskrivningar i dokumentet.

30. **Side-resolver använder SIDE_SPLIT ClassVar.** Side-varianter måste existera som `{type}_{side}` (t.ex. `ema_stop_long`, `ema_stop_short`) och dessa klasser måste sätta `SIDE_SPLIT: ClassVar[bool] = True`. Bas-type `{type}` (t.ex. `ema_stop`) behöver INTE finnas i registry. Resolver aktiveras ENBART när: (1) YAML `type` saknas i registry, OCH (2) `{type}_{side}` finns i registry, OCH (3) den resolvade klassen har `SIDE_SPLIT = True`. Om `{type}_{side}` finns men har `SIDE_SPLIT = False` → INVALID (fail-closed). Om bas-type finns i registry (t.ex. `ema_cross`) → resolver triggar ALDRIG, oavsett om `ema_cross_long` existerar. Enhetstester i `tests/test_composer.py`: (a) `ema_stop` + `side=long` → `ema_stop_long` OK, (b) `ema_cross` + `side=long` → använder `ema_cross` direkt (ingen resolve), (c) bas saknas + suffix finns men `SIDE_SPLIT=False` → INVALID.

31. **FeatureStore-instans per (fold_id, segment_id, timeframe).** Ny FeatureStore skapas för varje unik kombination av fold, segment och timeframe. MTF/HTF-features ligger i en SEPARAT FeatureStore bunden till HTF-df (resamplade data). HTF-store används för `shift(1)`-regeln (Delta 5). Inget delat state mellan stores. Sweep_runner ansvarar för att skapa och tilldela rätt store till rätt kontext.

32. **Ingen silent drop — auditable variant pipeline.** VARJE variant producerar en rad i `variants.jsonl` oavsett utfall. Schema: `{"variant_id": str, "status": "OK"|"INVALID"|"ERROR", "reason_code": str|null, "exception_summary": str|null (max 200 chars), ...metrics if OK}`. Pipeline får ALDRIG droppa en variant tyst p.g.a. exception, NaN, eller valideringsfel — istället skrivs statusraden. `manifest.json` counts (`n_variants`, `n_passed_filter`, `n_significant`) måste matcha variants.jsonl exakt. Enhetstest: räkna rader i jsonl == n_variants i manifest.

33. **Canonical trade return — definieras EN gång.** Implementera `engine/trades.py` med `compute_trade_return(entry_price, exit_price, side, fee_rate, slippage_rate) -> float`. Formler: `long: exit_fill/entry_fill - 1 - 2*fee_rate` där `entry_fill = open*(1+slip)`, `exit_fill = open*(1-slip)`. `short: entry_fill/exit_fill - 1 - 2*fee_rate` där `entry_fill = open*(1-slip)`, `exit_fill = open*(1+slip)`. Slippage härleds från fill-priser, aldrig ad hoc. ALL pipeline-kod (backtest, metrics, permutation test, DSR) ANVÄNDER denna funktion. Inga inline return-beräkningar. Enhetstest: trivial trade (entry=exit) ger negativ return.

34. **No same-bar entry+exit.** Om position öppnas på bar t+1 open kan exit-signal inte trigga förrän tidigast bar t+1 close → exit fill på t+2 open. Minsta möjliga trade varar alltså 1 bar (entry t+1 open, exit t+2 open). Utan denna regel kan stops "teleportera" och skapa omöjliga trades. Enhetstest: entry + omedelbar exit-signal på samma bar → exit sker på nästa bar.

35. **Walk-forward folds skapas INOM varje segment.** Fold-split sker per segment — inte globalt över segment. Segment som är för korta för minst 1 fold (train+test+embargo) droppas med warning. Månadsgränser alignas till segmentets tidsintervall.

36. **Primary timeframe = strategy.timeframe.** Alla signaler returnerar Series på `strategy.timeframe` index. MTF/HTF-features resamplas och alignas till strategy.timeframe (med HTF shift(1) per Delta 5). Ingen signal får returnera data på annan timeframe.

37. **FeatureStore ägs av pipeline, inte av signaler.** Ingen signal får skapa sin egen FeatureStore. Pipeline (sweep_runner / walk_forward) skapar stores och injicerar via `compute(df, cache)`. Detta förhindrar cache-spridning och läckage.

38. **Standardiserade reason_codes för variant status.** Giltig lista: `E_SCHEMA`, `E_SIGNAL_DTYPE`, `E_SIGNAL_NAN`, `E_SIGNAL_INDEX`, `E_LOOKAHEAD`, `E_BACKTEST`, `E_EXCEPTION`, `E_SIDE_UNSUPPORTED`, `E_NO_TRADES`, `E_INSUFFICIENT_TRADES`, `E_SEGMENT_TOO_SHORT`, `E_INVALID_PARAMS`. Andra codes tillåts men måste prefixas `E_`. Definiera i `utils/reason_codes.py` som constants.

39. **Ranking tie-strategi: average rank.** Vid lika värden i rank-percentile-beräkning används `method='average'` (scipy/pandas default). Låst för reproducerbarhet — `min` eller `max` rank ger instabila leaderboards vid repro. Implementera via `pd.Series.rank(method='average')`. Enhetstest: två varianter med identiskt OOS Sharpe får samma rank.

40. **Runtime guardrails för coarse p-values.** Coarse permutation måste vara vektoriserad: sign-flip-matrisen (n_perm × n_trades) beräknas i ett NumPy-anrop, inte i Python-loop per variant. Om `n_sanity_survivors > 50_000`: logga warning + estimated runtime i manifest. CC får INTE implementera permutation test med `for variant in variants: for perm in range(500)` — det måste vara batch-vektoriserat. Rekommenderad approach: beräkna sign-flip matrix en gång per unik trade-count-grupp, återanvänd över varianter med samma n_trades.

---

## Kontext

Du bygger en **forskningsplattform för kvantitativ signalutforskning** i `~/fpr-research/`. Det är INTE en enstaka backtest — det är en miljö för att systematiskt testa tusentals signalkombinationer med strikt statistisk validering.

**AUTONOMT BYGGE:** Arbeta dig igenom ALLA faser (1–5) utan att pausa, fråga om godkännande, eller vänta på input. Fatta egna designbeslut inom ramarna i denna spec. Om du stöter på en tvetydighet — välj det enklare alternativet, dokumentera beslutet i en `DECISIONS.md`, och fortsätt. Stoppa INTE mitt i en fas för att fråga "ska jag fortsätta?". Leverera ett komplett, körbart system med alla tester gröna och sweep_001 exekverad.

### Vad som redan finns (i `~/fpr-backtest/`)

Följande moduler från v1.0 är testade och fungerar. Porta/refaktorera, skriv inte om från scratch:

- `data/fetcher.py` — Bybit API, 1m OHLCV, parquet cache. 280k bars cached.
- `core/indicators.py` — MACD, Stoch RSI, z-scores, EMAs
- `core/phase_detector.py` — Quadrant fas-detektion med hysteresis
- `core/multi_tf.py` — Resample 1m → alla TFs (label='left', closed='left')
- `analysis/surrogate.py` — IAAFT surrogat-generering (1000 st)
- `analysis/statistics.py` — Permutation tests, Cohen's d
- `engine/backtest.py` — Event-driven backtester
- `engine/portfolio.py` — Position sizing, PnL
- `engine/metrics.py` — Sharpe, Sortino, drawdown, profit factor etc

### Mappstruktur på disk

```
~/fpr-backtest/          # EXISTERANDE — rör INTE denna mapp
├── data/
│   ├── fetcher.py
│   └── cache/           # 280k bars BTCUSDT 1m parquet-filer
├── core/
│   ├── indicators.py
│   ├── phase_detector.py
│   └── multi_tf.py
├── analysis/
│   ├── surrogate.py
│   └── statistics.py
└── engine/
    ├── backtest.py
    ├── portfolio.py
    └── metrics.py

~/fpr-research/          # NY — bygg allt här
└── (se arkitektur-sektion nedan)
```

**Instruktion till CC:**

1. Skapa `~/fpr-research/` som ny mapp. ALL ny kod skrivs här.
2. `~/fpr-backtest/` finns på disk och är read-only referens. Ändra INGET där.
3. Kopiera filer som ska portas: `cp ~/fpr-backtest/data/fetcher.py ~/fpr-research/data/fetcher.py` etc. Granska, refaktorera till v2.1-kontrakten, fixa buggar.
4. Kopiera parquet-cachen: `cp -r ~/fpr-backtest/data/cache/ ~/fpr-research/data/cache/` (280k bars, behövs för sweep).
5. Filer från `core/` (indicators, phase_detector, multi_tf) är hjälpfunktioner — porta relevant logik in i `signals/` och `data/feature_store.py`, inte som separata filer.
6. Om en fil från backtest inte behövs i research-arkitekturen — kopiera inte den.

### Vad som INTE finns och ska byggas nytt

1. **Signal-bibliotek** med standardiserat interface (en komponent = en roll)
2. **FeatureStore** — indikator-cache per (feature, params, tf)
3. **Kombinations-motor** (composer + parameter sweep)
4. **Walk-forward engine** med warmup + purge/embargo
5. **Leaderboard + ranking** med normaliserade metrics
6. **Anti-overfitting pipeline** (permutation p-values, BH-FDR, DSR, negativ kontroll)
7. **Regim-analys** (daily-baserad)
8. **CLI** inkl. `validate` och `repro`
9. **Rapportgenerering**
10. **Data integrity gate**

---

## Arkitektur

```
fpr-research/
├── data/
│   ├── fetcher.py              # [PORT] Bybit API + parquet cache
│   ├── validate.py             # [NY] Data integrity gate
│   ├── feature_store.py        # [NY] Indikator-cache
│   └── cache/                  # Parquet-filer
├── utils/
│   ├── canonical.py            # [NY] canonicalize_spec() för variant_id + manifest
│   ├── timeframes.py           # [NY] TF-string → pandas resample code
│   └── reason_codes.py         # [NY] Standardiserade E_* reason codes
├── signals/                    # SIGNAL-BIBLIOTEK (en fil = en roll, en side)
│   ├── base.py                 # [NY] Abstract SignalComponent + registry
│   ├── ema_gating.py           # [NY] filter (side-agnostic)
│   ├── ema_cross.py            # [NY] entry (side-agnostic)
│   ├── ema_stop_long.py        # [NY] exit — close < EMA (long)
│   ├── ema_stop_short.py       # [NY] exit — close > EMA (short)
│   ├── pvsra_entry.py          # [NY] entry (side-agnostic)
│   ├── pvsra_filter.py         # [NY] filter (side-agnostic)
│   ├── price_action_long.py    # [NY] entry — higher lows, BOS (long)
│   ├── price_action_short.py   # [NY] entry — lower highs, BOS short (short)
│   ├── stoch_rsi_entry.py      # [PORT+REFACTOR] entry
│   ├── stoch_rsi_filter.py     # [PORT+REFACTOR] filter
│   ├── macd_phase.py           # [PORT+REFACTOR] context
│   ├── volume_filter.py        # [NY] filter (side-agnostic)
│   ├── rsi_divergence_bull.py  # [NY] entry (long)
│   ├── rsi_divergence_bear.py  # [NY] entry (short)
│   ├── multi_tf_filter.py      # [PORT+REFACTOR] filter
│   ├── multi_tf_context.py     # [PORT+REFACTOR] context
│   ├── atr_stop_long.py        # [NY] exit — trailing below (long)
│   ├── atr_stop_short.py       # [NY] exit — trailing above (short)
│   ├── time_stop.py            # [NY] exit (side-agnostic)
│   └── propagation.py          # [NY] context
├── combinator/                 # KOMBINATIONS-MOTOR
│   ├── composer.py             # [NY] Bygg Strategy, fail-closed
│   ├── param_grid.py           # [NY] Grid + random search
│   └── sweep_runner.py         # [NY] Parallell exekvering (joblib)
├── engine/                     # BACKTEST-MOTOR
│   ├── backtest.py             # [PORT+EXTEND] Event-driven, next-open fill
│   ├── trades.py               # [NY] Canonical trade return beräkning
│   ├── portfolio.py            # [PORT] Position sizing, PnL
│   ├── metrics.py              # [PORT+EXTEND] Fler metrics
│   └── walk_forward.py         # [NY] WF splitter + warmup + purge
├── analysis/                   # ANALYS
│   ├── surrogate.py            # [PORT] IAAFT
│   ├── statistics.py           # [PORT+EXTEND] Permutation p-values
│   ├── multiple_testing.py     # [NY] BH-FDR
│   ├── deflated_sharpe.py      # [NY] DSR
│   ├── negative_control.py     # [NY] Surrogatjämförelse per sweep
│   ├── leaderboard.py          # [NY] Ranking + filtrering
│   ├── regime_analysis.py      # [NY] Daily-baserad regim
│   └── reports.py              # [NY+EXTEND] Markdown + plots
├── config/
│   ├── defaults.yaml           # Globala defaults
│   ├── schema_sweep.json       # [NY] jsonschema för sweep YAML
│   └── sweeps/                 # YAML per experiment
├── results/                    # Output från sweeps
│   ├── sweeps/                 # Per-sweep: manifest.json + variants.jsonl
│   ├── checkpoints/            # Phase diffs för spårbarhet
│   ├── leaderboard/            # Rankade resultat
│   └── reports/                # Genererade rapporter
├── tests/                      # Enhetstester
│   ├── test_signals.py         # Output-typ, NaN, längd, lookahead
│   ├── test_backtest.py        # Fill-modell, fee-beräkning
│   ├── test_walk_forward.py    # Splitter, warmup, purge
│   ├── test_statistics.py      # BH-FDR, DSR korrekthet
│   ├── test_data.py            # Validate.py tester
│   ├── test_lookahead.py       # Append-test för alla signaler
│   └── test_composer.py        # Side-resolver, SIDE_SPLIT tester
├── main.py                     # CLI entry point
├── DECISIONS.md                # CC:s designbeslut under bygget
├── SPEC.md                     # DENNA FIL
└── requirements.txt
```

---

## 0. Data Integrity Gate (`data/validate.py`)

Körs automatiskt innan varje sweep. Om den failar → sweep avbryts.

```python
def validate_ohlcv(df: pd.DataFrame) -> ValidationReport:
    """Validera OHLCV data.

    Checks:
    1. Index är DatetimeIndex, UTC, monotonic increasing
    2. Inga duplicate timestamps
    3. Kolumner: open, high, low, close, volume (alla float64)
    4. high >= max(open, close), low <= min(open, close)
    5. volume >= 0
    6. Rapportera missing minutes count (gaps)
    7. Inga NaN i OHLCV

    Returns ValidationReport med pass/fail + gap-statistik + segment-lista.

    Gap-policy: splittra df i kontinuerliga segment vid gaps > 5 min.
    Returnera List[DataFrame] av segment. WF/backtest körs per segment.
    Om något segment < min_segment_bars (default 5000) → dropp med warning.
    Gaps fylls ALDRIG.
    """
    pass
```

---

## 1. Signal-Bibliotek (`signals/`)

### Interface (`signals/base.py`)

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Any, Literal
import pandas as pd

SignalType = Literal['entry', 'exit', 'filter', 'context']

@dataclass
class SignalComponent(ABC):
    """Bas för alla signalkomponenter.

    Kontrakt:
    - EN komponent = EN roll (entry ELLER filter ELLER exit ELLER context)
    - compute() tar DataFrame + FeatureStore cache
    - compute() returnerar Series med SAMMA index som input
    - Ingen framtidsläcka (verifieras med append-test)
    - Output dtype:
        entry:   int8 i {0, 1}  (1 = signal aktiv, side bestäms av Strategy)
        exit:    bool
        filter:  bool (strict, inga float-truthy)
        context: float64 (multiplier, t.ex. 0.5–1.5. Default 1.0 = ingen justering)
    """
    KEY: ClassVar[str]              # Unik nyckel, t.ex. 'ema_cross'
    SIGNAL_TYPE: ClassVar[SignalType]
    SUPPORTED_SIDES: ClassVar[set] = {'long'}  # {'long'}, {'short'}, eller {'long','short'}
    SIDE_SPLIT: ClassVar[bool] = False       # True för side-splittade signaler (ema_stop_long etc)
    params: Dict[str, Any] = field(default_factory=dict)

    @abstractmethod
    def compute(self, df: pd.DataFrame, cache: 'FeatureStore') -> pd.Series:
        pass

    @abstractmethod
    def param_grid(self) -> Dict[str, List[Any]]:
        """Parameter-ranges för sweep."""
        pass

    def lookback_bars(self) -> int:
        """Max antal historiska bars som behövs.
        Används av walk-forward för warmup-beräkning."""
        return 0

    def with_params(self, **kwargs) -> 'SignalComponent':
        import copy
        new = copy.deepcopy(self)
        new.params.update(kwargs)
        return new

    def required_columns(self) -> List[str]:
        return ['open', 'high', 'low', 'close', 'volume']


# Registry
_REGISTRY: Dict[str, type] = {}

def register_signal(cls):
    key = getattr(cls, 'KEY', None)
    if not key:
        raise ValueError(f"{cls.__name__} saknar KEY")
    if key in _REGISTRY:
        raise ValueError(f"Duplicate KEY: {key}")
    _REGISTRY[key] = cls
    return cls

def get_signal(name: str) -> type:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown signal: {name}. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]

def list_signals() -> List[str]:
    return sorted(_REGISTRY.keys())
```

### Signaler att implementera (prioritetsordning)

**Prioritet 1 — Pris-baserade (bygg först):**

| Signal             | KEY                  | Typ    | Parametrar                                                                                             |
| ------------------ | -------------------- | ------ | ------------------------------------------------------------------------------------------------------ |
| EMA Gating         | `ema_gating`         | filter | period: [20,50,100,200], mode: ['above','fanned']                                                      |
| EMA Cross          | `ema_cross`          | entry  | fast: [5,8,13,21], slow: [21,34,50,100]                                                                |
| PVSRA Entry        | `pvsra_entry`        | entry  | vol_mult: [1.5,2.0,3.0], lookback: [5,10,20]                                                           |
| PVSRA Filter       | `pvsra_filter`       | filter | vol_mult: [1.5,2.0,3.0], window: [3,5,10]                                                              |
| Price Action Long  | `price_action_long`  | entry  | swing_lb: [5,10,20,30], right_bars: [3,5], mode: ['higher_low','bos']. SUPPORTED_SIDES={'long'}        |
| Price Action Short | `price_action_short` | entry  | swing_lb: [5,10,20,30], right_bars: [3,5], mode: ['lower_high','bos_short']. SUPPORTED_SIDES={'short'} |
| EMA Stop Long      | `ema_stop_long`      | exit   | period: [50,100]. close < EMA → exit. SUPPORTED_SIDES={'long'}                                         |
| EMA Stop Short     | `ema_stop_short`     | exit   | period: [50,100]. close > EMA → exit. SUPPORTED_SIDES={'short'}                                        |
| ATR Stop Long      | `atr_stop_long`      | exit   | atr_mult: [1.5,2.0,3.0], atr_period: [14,20]. Trailing below. SUPPORTED_SIDES={'long'}                 |
| ATR Stop Short     | `atr_stop_short`     | exit   | atr_mult: [1.5,2.0,3.0], atr_period: [14,20]. Trailing above. SUPPORTED_SIDES={'short'}                |
| Time Stop          | `time_stop`          | exit   | max_bars: [50,100,200]                                                                                 |

**Prioritet 2 — Indikator-baserade:**

| Signal           | KEY                | Typ    | Parametrar                                            |
| ---------------- | ------------------ | ------ | ----------------------------------------------------- |
| Stoch RSI Entry  | `stoch_rsi_entry`  | entry  | rsi_len: [7,14,21], k_smooth: [3], ob: [80], os: [20] |
| Stoch RSI Filter | `stoch_rsi_filter` | filter | rsi_len: [14], ob: [70,80], os: [20,30]               |
| Volume Filter    | `volume_filter`    | filter | rel_vol: [1.2,1.5,2.0], lookback: [10,20]             |

**Prioritet 3 — Kontext:**

| Signal       | KEY                   | Typ     | Parametrar                                                                      |
| ------------ | --------------------- | ------- | ------------------------------------------------------------------------------- |
| MACD Phase   | `macd_phase`          | context | z_flat: [0.1,0.3,0.5,0.8], confirm: [2,3,5,8]                                   |
| RSI Div Bull | `rsi_divergence_bull` | entry   | lookback: [10,20,50], right_bars: [3,5]. Bullish div. SUPPORTED_SIDES={'long'}  |
| RSI Div Bear | `rsi_divergence_bear` | entry   | lookback: [10,20,50], right_bars: [3,5]. Bearish div. SUPPORTED_SIDES={'short'} |
| MTF Filter   | `mtf_filter`          | filter  | tfs: [...], min_alignment: [2,3]                                                |
| MTF Context  | `mtf_context`         | context | tfs: [...]                                                                      |
| Propagation  | `propagation`         | context | ltf/htf pairs                                                                   |

**Bygg prioritet 1 fullt, testa (inkl. lookahead-test), sedan prioritet 2, etc.**

---

## 2. FeatureStore (`data/feature_store.py`)

```python
class FeatureStore:
    """Memoiserad indikator-cache.

    Bunden till (df, timeframe). En store per (fold_id, segment_id, timeframe).
    MTF/HTF-features ligger i en SEPARAT store bunden till HTF-df.
    Signaler anropar cache.get('ema', period=50) UTAN df-argument.

    VIKTIGT:
    - Skapa ny FeatureStore(df, timeframe) per (fold_id, segment_id, timeframe).
    - MTF: HTF-features i separat FeatureStore bunden till resamplade HTF-df.
    - Cache-key: (name, sorted_params). Timeframe implicit via store.
    - Fold-aware: inga globala percentiler/statistics beräknas.
    - Bara deterministiska funktioner (EMA, MACD, ATR etc) cachas.
    """

    def __init__(self, df: pd.DataFrame, timeframe: str):
        self._df = df
        self._timeframe = timeframe
        self._cache: Dict[tuple, pd.Series] = {}

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @property
    def timeframe(self) -> str:
        return self._timeframe

    def get(self, name: str, **params) -> pd.Series:
        """Cache-key: (name, sorted_params). df bunden i store."""
        key = (name, tuple(sorted(params.items())))
        if key not in self._cache:
            self._cache[key] = self._compute(name, **params)
        return self._cache[key]

    def precompute_for_fold(self, sweep_config):
        """Precomputa alla features för denna fold+TF.
        Extrahera union av alla perioder/params från sweep YAML.
        Sparas till disk (npz/parquet) för joblib workers."""
        pass
```

---

## 3. Strategi-Komposition (`combinator/composer.py`)

```python
@dataclass
class Strategy:
    name: str
    entry: SignalComponent            # Exakt en, SIGNAL_TYPE == 'entry'
    filters: List[SignalComponent]     # 0–N, alla SIGNAL_TYPE == 'filter'
    exits: List[SignalComponent]       # 1+, alla SIGNAL_TYPE == 'exit'
    context: List[SignalComponent]     # 0–N, alla SIGNAL_TYPE == 'context'
    side: Literal['long', 'short']     # Per strategy, aldrig båda samtidigt
    timeframe: str                     # '5min', '15min', '1h' etc
    taker_fee_rate: float = 0.0006    # 0.06% per sida
    slippage_rate: float = 0.0001     # 0.01%

    def __post_init__(self):
        """Fail-closed validering."""
        assert self.side in ('long', 'short'), f"Invalid side: {self.side}"
        assert self.entry.SIGNAL_TYPE == 'entry'
        for f in self.filters:
            assert f.SIGNAL_TYPE == 'filter'
        for e in self.exits:
            assert e.SIGNAL_TYPE == 'exit'
        assert len(self.exits) >= 1, "Minst en exit krävs"
        # Side-validering: alla komponenter måste stödja vald side
        all_components = [self.entry] + self.filters + self.exits + self.context
        for c in all_components:
            if self.side not in c.SUPPORTED_SIDES:
                raise ValueError(
                    f"{c.KEY} stödjer inte side='{self.side}' "
                    f"(stödjer: {c.SUPPORTED_SIDES})"
                )

    def generate_signals(self, df: pd.DataFrame,
                          cache: FeatureStore) -> pd.DataFrame:
        """Kör alla komponenter.

        Fail-closed: om en komponent returnerar fel dtype,
        NaN, eller fel index → variant markeras INVALID.

        entry_signal:  entry.compute() == 1 AND all(f.compute() == True)
        exit_signal:   any(e.compute() == True)
        context_mult:  mean(context multipliers), clipped [0.5, 1.5]
        Context-komponenter returnerar float64 multiplier (t.ex. 0.5–1.5).
        """
        pass

    def warmup_bars(self) -> int:
        """Max lookback across alla komponenter."""
        all_components = [self.entry] + self.filters + self.exits + self.context
        return max(c.lookback_bars() for c in all_components)

    def variant_id(self) -> str:
        """sha256 av canonical spec via utils/canonical.py."""
        from utils.canonical import canonicalize_spec
        spec = {
            'entry': {'key': self.entry.KEY, 'params': self.entry.params},
            'filters': [{'key': f.KEY, 'params': f.params} for f in self.filters],
            'exits': [{'key': e.KEY, 'params': e.params} for e in self.exits],
            'context': [{'key': c.KEY, 'params': c.params} for c in self.context],
            'side': self.side,
            'timeframe': self.timeframe,
            'fee_rate': self.taker_fee_rate,
            'slippage_rate': self.slippage_rate,
        }
        import hashlib
        return hashlib.sha256(canonicalize_spec(spec).encode()).hexdigest()[:12]


# Context → sizing
def context_to_size(context_mults: List[float],
                    base_size: float = 1.0,
                    min_mult: float = 0.5,
                    max_mult: float = 1.5) -> float:
    """size = base_size * clip(mean(context_mults), min, max)
    Om inga context-komponenter: size = base_size (mult = 1.0)."""
    if not context_mults:
        return base_size
    mult = sum(context_mults) / len(context_mults)
    return base_size * max(min_mult, min(max_mult, mult))
```

Entry triggas NUR när:

1. `entry.compute() == 1` (signal aktiv)
2. ALLA filters returnerar `True` (strict bool)
3. Ingen aktiv position (en position åt gången) — **INGEN PYRAMIDING**
4. Strategy `side` bestämmer riktning (long = köp, short = sälj)

Entry-signaler som triggar när position redan är öppen ignoreras tyst. Enhetstest obligatoriskt.

Exit triggas när NÅGON exit returnerar `True`. Position stängs även vid segmentslut (gap-policy).

**Fill-modell (side-aware):**

- Signaler evalueras på bar `close`
- Long entry: `open * (1 + slippage_rate)` — slippage gör köp dyrare
- Long exit: `open * (1 - slippage_rate)` — slippage gör sälj billigare
- Short entry: `open * (1 - slippage_rate)` — slippage gör sälj billigare
- Short exit: `open * (1 + slippage_rate)` — slippage gör köp dyrare
- Fees: `taker_fee_rate` på entry OCH exit (båda sidor)
- Princip: slippage gör ALLTID fill sämre. Enhetstest per side.

---

## 4. Walk-Forward (`engine/walk_forward.py`)

```python
@dataclass
class WalkForwardConfig:
    train_months: int = 3
    test_months: int = 1
    step_months: int = 1
    min_trades_per_fold: int = 5
    embargo_bars: int = 0           # Buffert mellan train/test

def walk_forward_split(df, config, warmup_bars: int = 0) -> List[Tuple[DataFrame, DataFrame]]:
    """Generera train/test splits med embargo.

    |--- train (3 mo) ---|--embargo--|--- test (1 mo) ---|
                                     |--- train (3 mo) ---|--embargo--|--- test (1 mo) ---|

    EMBARGO-REGEL (hård):
    effective_embargo = max(config.embargo_bars, warmup_bars)
    YAML kan inte sätta lägre än warmup. Enhetstest obligatoriskt.

    FOLD-SPLIT PER SEGMENT:
    Folds skapas INOM varje segment — inte globalt över segment.
    Segment som är för korta för minst 1 fold droppas med warning.

    MONTH BOUNDARY RULE:
    Splitta på kalenderdatum (UTC) med inklusiva endpoints.
    Train: första dag 00:00 UTC → sista dag 23:59 UTC.
    Test: första dag 00:00 UTC → sista dag 23:59 UTC.
    Tester ska bekräfta:
    - Inga timestamps i både train och test
    - Embargo verkligen exkluderar rätt antal bars
    - Inga off-by-one vid månadsgräns
    """
    pass

def walk_forward_run(strategy, df, config, cache) -> WalkForwardResult:
    """Kör strategi genom alla walk-forward fönster.

    För varje fönster:
    1. warmup_bars = strategy.warmup_bars()
    2. Kör compute() på tail(train, warmup_bars) + test
    3. Logga trades BARA på test-slice
    4. Spara metrics för test-perioden

    Returns:
    - is_metrics: Aggregerade metrics över train (diagnostik)
    - oos_metrics: Aggregerade metrics över test (rapporteras)
    - per_fold_metrics: Metrics per fönster
    - oos_equity_curve: Konkatenerad equity curve
    - oos_trade_log: Alla OOS trades
    - median_oos_sharpe: Median Sharpe över folds (robust)
    """
    pass
```

**KRITISKT:**

- Warmup = `strategy.warmup_bars()` (max lookback across alla komponenter)
- Embargo = valfri buffert mellan train/test
- Trades räknas BARA på test-data
- Indicator-beräkning via `cache: FeatureStore`

---

## 5. Parameter Sweep (`combinator/sweep_runner.py`)

```python
def run_sweep(sweep_config_path: str, data: pd.DataFrame) -> SweepResult:
    """Kör en full parameter sweep.

    1. Validera data (data integrity gate)
    2. Validera YAML (schema check)
    3. Precomputa features (FeatureStore)
    4. Generera alla varianter (grid/random)
    5. Kör walk-forward på varje variant (joblib)
    6. Data-sanity filter ENBART: n_trades >= 30, valid variant (inga NaN, korrekt dtype)
       OBS: INGEN performance-filtrering här (PF/DD/Sharpe). Det bevarar BH-FDR-garantier.
    7. Coarse p-value (500 perm) för ALLA varianter som passerar sanity-filter
    8. BH-FDR correction på coarse p-values
    9. Refined p-value (10k perm) bara på BH-survivors
    10. DSR på signifikanta varianter
    11. Performance-filter + ranking (Sharpe, PF, DD etc) — EFTER signifikans
    12. Negativ kontroll (distributional): topp-10 på N surrogater, rapportera percentile
    13. Spara manifest.json (inkl seeds) + variants.jsonl + leaderboard.md
    """
    pass
```

### Sweep YAML-format

```yaml
name: "sweep_001_ema_pvsra"
symbol: "BTCUSDT" # Obligatoriskt. Bestämmer vilken parquet-data som laddas.
description: "EMA-cross entries med PVSRA-filter"
theory: "EMA-kors fångar momentum-shift i pris-space. PVSRA filtrerar bort kors utan institutionell volym."
sides: [long] # Kör separat sweep per side. [long, short] = 2x varianter.

entry:
  type: ema_cross
  params:
    fast: [5, 8, 13, 21]
    slow: [21, 34, 50, 100]

filters:
  - type: ema_gating
    params:
      period: [50, 100, 200]
      mode: ["above"]
  - type: pvsra_filter
    params:
      vol_mult: [1.5, 2.0]
      window: [3, 5, 10]

exits:
  - type: ema_stop
    params:
      period: [50]

timeframes: ["5min", "15min", "1h"]

walk_forward:
  train_months: 3
  test_months: 1
  embargo_bars: 200

fees:
  taker_fee_pct: 0.06 # Konverteras till 0.0006 vid parsing
  slippage_pct: 0.01 # Konverteras till 0.0001

statistics:
  coarse_permutation_n: 500 # Steg 1: coarse p-value foer ALLA varianter
  refined_permutation_n: 10000 # Steg 2: refined p-value foer BH-survivors
  surrogate_count: 100 # IAAFT surrogater foer negativ kontroll
  surrogate_pctl_threshold: 90 # Flagga om real < denna percentil
```

**Obligatoriska fält:** `name`, `symbol`, `theory`, `entry`, `exits`, `timeframes`, `fees`.

**Schema-validering:** Implementera `config/schema_sweep.json` (jsonschema). `python main.py validate` kör schema-check + enhetstest för validering. CC får inte improvisera schema.

### manifest.json (sparas per sweep)

```json
{
  "sweep_name": "sweep_001_ema_pvsra",
  "git_commit": "abc1234",
  "data_range": { "start": "2025-08-01", "end": "2025-12-31" },
  "symbol": "BTCUSDT",
  "exchange": "bybit",
  "n_variants": 1440,
  "n_passed_filter": 87,
  "n_significant": 12,
  "runtime_seconds": 3420,
  "config_hash": "sha256...",
  "data_fingerprint": "sha256 av parquet metadata + filelist",
  "environment": {
    "python_version": "3.11.5",
    "pandas_version": "2.1.4",
    "numpy_version": "1.26.2",
    "platform": "linux-x86_64"
  },
  "seeds": {
    "rng_seed_global": 42,
    "seed_permutation": 123,
    "seed_surrogates": 456,
    "seed_sampler": 789
  },
  "p_value_stages": {
    "coarse_n_perm": 500,
    "refined_n_perm": 10000,
    "n_coarse_tested": 1440,
    "n_bh_survivors": 87
  }
}
```

---

## 6. Anti-Overfitting Pipeline (`analysis/`)

### 6.1 p-value-beräkning (`statistics.py`)

```python
def permutation_pvalue(trade_returns: np.ndarray,
                        n_permutations: int = 10000,
                        seed: int = None) -> float:
    """Sign-flip permutation test.

    VIKTIGT: trade_returns MASTE vara NETTO (efter fees+slippage).
    Beräknas via engine/trades.py compute_trade_return() — ALDRIG inline.

    H0: trade returns har noll mean (ingen edge)
    Teststatistik: mean(trade_returns)
    Permutation: flippa tecken slumpmässigt
    p = mean(null_means >= observed_mean)  # EXAKT denna formel

    One-sided (enbart positiv edge). Inga absoluttal.
    Seed loggas i manifest för reproducerbarhet.

    IMPLEMENTATION: Vektoriserad sign-flip matrix (n_perm × n_trades) i NumPy.
    Ingen Python-loop per variant.
    """
    pass
```

### 6.2 Multiple Testing (`multiple_testing.py`)

```python
def benjamini_hochberg(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """BH-FDR correction."""
    pass
```

### 6.3 Deflated Sharpe (`deflated_sharpe.py`)

```python
def deflated_sharpe(observed_sharpe, n_variants_tested, n_obs,
                     sharpe_std, skew, kurt) -> float:
    """Bailey & Lopez de Prado 2014.
    Skew/kurt beräknas på OOS trade returns."""
    pass
```

### 6.4 Negativ kontroll (`negative_control.py`)

```python
def negative_control(top_variants: List[Strategy],
                      real_data: pd.DataFrame,
                      n_surrogates: int = 100,
                      seed: int = 42) -> NegativeControlReport:
    """Kör topp-varianter på IAAFT-surrogater (distributional).

    För varje variant:
    1. Kör på n_surrogates IAAFT-surrogater
    2. Bygg distribution av surrogat-Sharpe
    3. Beräkna surrogate_pctl = percentile rank där real Sharpe hamnar
    4. Flagga om real Sharpe < 90:e percentilen av surrogat-distribution

    Definition: pctl = 100 * mean(sharpe_surrogate <= sharpe_real)

    Returnerar per variant:
    - surrogate_sharpe_distribution (array)
    - surrogate_pctl (float, 0-100)
    - surrogate_mean, surrogate_std (diagnostik)
    - surrogate_z = (sharpe_real - surrogate_mean) / surrogate_std
    - flagged (bool: pctl < threshold)

    Seed loggas i manifest för reproducerbarhet.
    """
    pass
```

### 6.5 Overfitting-diagnostik per variant

```json
{
  "variant_id": "sha256[:12]",
  "status": "OK",
  "reason_code": null,
  "exception_summary": null,
  "is_sharpe": 1.89,
  "oos_sharpe": 1.45,
  "median_oos_sharpe_per_fold": 1.32,
  "sharpe_decay": 0.23,
  "n_params": 3,
  "n_trades_oos": 47,
  "consistency_ratio": 0.75,
  "p_coarse": 0.012,
  "p_refined": 0.003,
  "bh_fdr_significant": true,
  "dsr_pvalue": 0.02,
  "surrogate_pctl": 97.3,
  "surrogate_z": 2.14,
  "surrogate_mean": 0.31,
  "surrogate_std": 0.53,
  "surrogate_flagged": false
}
```

---

## 7. Leaderboard (`analysis/leaderboard.py`)

### Minimum-filter (alla måste passera)

- OOS Sharpe > 1.0
- IS Sharpe > 0.5 (undviker märkliga decay-siffror)
- Trades > 30 (OOS totalt)
- Min 5 trades per fold (annars exkluderas folden)
- Max DD < 20%
- Sharpe decay < 0.6
- BH-FDR adjusted p < 0.05
- Consistency > 60%
- Surrogate pctl > 90 (real Sharpe måste vara extremare än 90% av surrogat-distributionen)

### Ranking

**Normalisera alla metrics till rank-percentile innan viktning** (annars dominerar skalan). **Tie-strategi: `method='average'`** (Delta 39).

| Metric                     | Vikt |
| -------------------------- | ---- |
| Median OOS Sharpe per fold | 30%  |
| Profit Factor              | 20%  |
| Max Drawdown (inverterat)  | 15%  |
| Consistency Ratio          | 15%  |
| Win Rate                   | 10%  |
| Trade Count                | 10%  |

**Stability penalty:** om 1 fold står för > 50% av total vinst → straffa rank med 20%.

---

## 8. Regim-analys (`analysis/regime_analysis.py`)

```python
# Alltid baserad på DAILY data (SMA200 på daily close)
# Mappas till intraday via date-join

def classify_regime(daily_df: pd.DataFrame,
                     slope_window: int = 20) -> pd.Series:
    """Klassificera varje dag som bull/bear/range + vol-regime.

    sma200 = daily close SMA(200)
    slope = (sma200 - sma200.shift(slope_window)) / sma200.shift(slope_window)
    atr = daily ATR(14)
    atr_pct = atr.rolling(252).rank(pct=True)  # 1-year percentile

    Regimes (kan överlappa vol med trend):
    - bull:     close > sma200 AND slope > 0.01
    - bear:     close < sma200 AND slope < -0.01
    - range:    abs(slope) <= 0.01
    - high_vol: atr_pct > 0.75
    - low_vol:  atr_pct < 0.25
    """
    pass
```

Varje variant rapporteras per regim. Flagga varianter som BARA funkar i en regim (> 80% av PnL från en regim).

---

## 9. CLI (`main.py`)

```bash
# Data
python main.py fetch                                      # Hämta/uppdatera
python main.py fetch --start 2025-08-01 --end 2026-02-15

# Validering
python main.py validate config/sweeps/sweep_001.yaml       # Schema-check YAML
#   → visar: antal varianter, feature-union, estimated runtime

# Sweep
python main.py sweep config/sweeps/sweep_001.yaml          # Kör sweep
python main.py sweep config/sweeps/sweep_001.yaml --dry-run # Visa variant-count
python main.py sweep config/sweeps/sweep_001.yaml --jobs 4  # Parallellitet

# Resultat
python main.py leaderboard                                # Visa topp
python main.py leaderboard --min-sharpe 1.5 --min-trades 50
python main.py report VARIANT_ID                           # Detaljrapport
python main.py regimes VARIANT_ID                          # Regim-breakdown

# Reproducerbarhet
python main.py repro VARIANT_ID                            # Återskapar från manifest
#   → kör om variant, assert metrics matchar inom tolerans

# Holdout
python main.py holdout --top 10                            # Topp-10 på holdout

# Verktyg
python main.py signals                                    # Lista signaler
python main.py signals ema_cross --grid                    # Parameter grid
```

---

## 10. Holdout-protokoll

```
IS data:      2025-08-01 → 2025-12-31 (inklusiv)
Holdout data: 2026-01-01 → 2026-02-15 (inklusiv)

1. Kör sweeps på IS data med walk-forward
2. Ranka på OOS metrics (alla walk-forward folds inom IS)
3. Kör negativ kontroll på topp-10
4. Välj finalkandidater (max 10)
5. Kör på holdout (EN gång, ingen iteration)
6. Rapportera holdout-resultat
7. Beslut: paper trading eller tillbaka till forskning
```

---

## 11. Kritiska Regler

1. **VARJE variant walk-forward-valideras.** Inga undantag.
2. **Permutation p-values + BH-FDR på alla sweeps.**
3. **Holdout rörs INTE förrän finalkandidater är valda.**
4. **Fee + slippage från dag 1.** `taker_fee_rate=0.0006`, `slippage_rate=0.0001`.
5. **Fill på nästa bars open.** Ingen same-bar fill.
6. **Regim-analys på varje kandidat.** Daily-baserad.
7. **Theory-fält obligatoriskt i sweep YAML.**
8. **Ingen data leakage.** Resample `label='left', closed='left'`. HTF: shift(1). Pivots: confirmed after `right_bars`.
9. **Enhetstester + lookahead-test.** `pytest tests/` måste passera.
10. **Negativ kontroll.** Topp-varianter ska inte funka på surrogat.
11. **Reproducerbart.** `python main.py repro VARIANT_ID` matchar sparade metrics.
12. **Data integrity gate före varje sweep.**
13. **Ingen pyramiding.** Entry ignoreras när position är öppen.
14. **Fill-modell är side-aware.** Slippage gör alltid fill sämre, både long och short.
15. **Canonical spec util för variant_id.** `utils/canonical.py` används överallt.
16. **YAML side-resolver.** Side-beroende exits resolvas från `type: ema_stop` → `ema_stop_{side}` i composer. Se Delta punkt 26.
17. **Embargo-bars i strategy.timeframe.** Inte 1m-bars. Se Delta punkt 21.
18. **Funding ignoreras.** Resultat är INTE live-ready. Se Delta punkt 27.
19. **Fas-checkpoints är obligatoriska.** `pytest` + CLI smoke-test efter varje fas. Fixa → kör om → commit → nästa fas. Se Delta punkt 28.
20. **BH-FDR-ordning är hård.** Ingen performance-filtrering före p-values. Se Delta punkt 29.
21. **Side-resolver via SIDE_SPLIT.** Bara signaler med `SIDE_SPLIT=True` resolvas. Se Delta punkt 30.
22. **FeatureStore per (fold, segment, timeframe).** Separat HTF-store för MTF. Se Delta punkt 31.
23. **Ingen silent drop.** Varje variant får status OK/INVALID/ERROR i variants.jsonl med standardiserade reason_codes. Se Delta punkt 32+38.
24. **Canonical trade return.** Definieras i `engine/trades.py`, används överallt. Se Delta punkt 33.
25. **No same-bar entry+exit.** Minsta trade = 1 bar. Se Delta punkt 34.
26. **Folds inom segment.** Fold-split per segment, inte globalt. Se Delta punkt 35.
27. **Primary timeframe = strategy.timeframe.** Alla signaler returnerar på denna index. Se Delta punkt 36.
28. **FeatureStore ägs av pipeline.** Signaler skapar aldrig stores. Se Delta punkt 37.
29. **Ranking: average rank vid ties.** Deterministiskt, reproducerbart. Se Delta punkt 39.
30. **Permutation test vektoriserad.** Ingen Python-loop per variant. Se Delta punkt 40.

---

## 12. Byggordning

```
Fas 1: Infrastruktur
  1. Implementera utils/canonical.py + utils/timeframes.py + utils/reason_codes.py + tester
  2. Kopiera och verifiera data/fetcher.py
  3. Implementera data/validate.py + test_data.py
  4. Implementera data/feature_store.py (bunden till df+timeframe)
  5. Implementera signals/base.py (interface + registry + SIDE_SPLIT)
  6. Implementera engine/trades.py (canonical trade return) + test
  7. Implementera engine/walk_forward.py + test (embargo >= warmup, month boundary, folds per segment)
  8. Porta engine/backtest.py (+ next-open fill, side-aware slip, no-pyramiding, no same-bar entry+exit) + test
  9. Porta engine/metrics.py + extend
  10. Skapa config/schema_sweep.json (jsonschema)
  >>> CHECKPOINT FAS 1: pytest tests/test_data.py tests/test_backtest.py tests/test_walk_forward.py
  >>> SMOKE: python main.py fetch --help && python main.py validate --help
  >>> ALLA GRÖNA? → Fortsätt. FAIL? → Fixa, kör om SAMMA checkpoint, commit, först därefter fas 2.

Fas 2: Signaler
  11. Implementera ema_cross.py + ema_gating.py (side-agnostic)
  12. Implementera ema_stop_long.py + ema_stop_short.py (SIDE_SPLIT=True)
  13. Implementera pvsra_entry.py + pvsra_filter.py (side-agnostic)
  14. Implementera price_action_long.py + price_action_short.py (med right_bars confirmation, SIDE_SPLIT=True)
  15. Implementera atr_stop_long.py + atr_stop_short.py (SIDE_SPLIT=True) + time_stop.py
  16. Porta stoch_rsi_entry.py + stoch_rsi_filter.py
  17. Porta macd_phase.py (context)
  18. Implementera volume_filter.py
  19. test_signals.py + test_lookahead.py för alla signaler
  >>> CHECKPOINT FAS 2: pytest tests/test_signals.py tests/test_lookahead.py
  >>> SMOKE: python main.py signals (ska lista alla registrerade signaler)
  >>> ALLA GRÖNA? → Fortsätt. FAIL? → Fixa, kör om SAMMA checkpoint, commit, först därefter fas 3.

Fas 3: Kombinator
  20. Implementera combinator/composer.py (fail-closed, side-validering, YAML side-resolver med SIDE_SPLIT)
  21. Implementera combinator/param_grid.py
  22. Implementera combinator/sweep_runner.py (data-sanity pre-filter, tvåstegs p-values, no silent drop)
  23. Skapa config/sweeps/sweep_001_ema_pvsra.yaml
  24. Integration test: kör sweep_001 med 10 varianter
  >>> CHECKPOINT FAS 3: python main.py validate config/sweeps/sweep_001_ema_pvsra.yaml
  >>> SMOKE: python main.py sweep config/sweeps/sweep_001_ema_pvsra.yaml --dry-run
  >>> ALLA GRÖNA? → Fortsätt. FAIL? → Fixa, kör om SAMMA checkpoint, commit, först därefter fas 4.

Fas 4: Analys
  25. Implementera analysis/statistics.py (vektoriserad permutation p-values, one-sided)
  26. Implementera analysis/multiple_testing.py
  27. Implementera analysis/deflated_sharpe.py
  28. Implementera analysis/negative_control.py (distributional, z-score)
  29. Implementera analysis/leaderboard.py (rank method='average')
  30. Implementera analysis/regime_analysis.py
  31. Implementera analysis/reports.py
  32. Implementera main.py (CLI: alla kommandon)
  >>> CHECKPOINT FAS 4: pytest tests/test_statistics.py
  >>> SMOKE: python main.py leaderboard --help && python main.py report --help
  >>> ALLA GRÖNA? → Fortsätt. FAIL? → Fixa, kör om SAMMA checkpoint, commit, först därefter fas 5.

Fas 5: Validering
  33. Kör sweep_001 på full IS data
  34. Granska leaderboard + negativ kontroll
  35. python main.py repro <topp-variant> → bekräfta reproducerbarhet
  36. Dokumentera findings
  >>> FINAL CHECKPOINT: pytest (alla tester) + python main.py repro <variant>
  >>> Om repro matchar inom ±1% → DONE. Annars → debug.
```

---

## 13. Dependencies

```
pandas>=2.0
numpy>=1.24
scipy>=1.11
matplotlib>=3.7
requests>=2.31
pyarrow>=14.0
pyyaml>=6.0
joblib>=1.3
tqdm>=4.65
pytest>=7.0
jsonschema>=4.0
```

---

## 14. Definition of Done (DoD)

En fas är DONE när alla kriterier är uppfyllda:

### Fas 1 — Infrastruktur

- [ ] `pytest tests/test_data.py` grönt
- [ ] `pytest tests/test_backtest.py` grönt — fill-modell verifierad (next-open + fees + slippage)
- [ ] `pytest tests/test_walk_forward.py` grönt — splits korrekt, warmup funkar, embargo funkar, inga trades i train-slice
- [ ] FeatureStore cachar korrekt (ny instans per fold), ger identiskt resultat vid upprepad anrop
- [ ] `python main.py fetch` laddar data, `validate` passerar, returnerar segment-lista
- [ ] Reality-check test: trivial trade (köp+sälj till samma pris) ger negativ return (fees dras)
- [ ] Embargo >= warmup hårdkodad (test: YAML med embargo=0 + signal med lookback=200 ger effective_embargo=200)
- [ ] Walk-forward month boundary: inga timestamps i både train och test (explicit test)
- [ ] Side-aware fill: long entry dyrare, long exit billigare, short entry billigare, short exit dyrare (explicit per-side test)
- [ ] No-pyramiding: två entries i rad öppnar bara en position (backtest-test)
- [ ] No same-bar entry+exit: entry på t+1 open → exit tidigast t+2 open (backtest-test)
- [ ] Canonical util: samma spec från Strategy.variant_id() och sweep manifest ger identisk hash
- [ ] Canonical trade return: `engine/trades.py` används av backtest (inga inline beräkningar)
- [ ] YAML schema: ogiltiga YAML-filer rejectas av jsonschema
- [ ] FeatureStore: separat instans per (fold, segment, timeframe) — test att två stores med olika df inte delar cache
- [ ] FeatureStore: HTF-store separat från LTF-store (MTF-test)

### Fas 2 — Signaler

- [ ] `pytest tests/test_signals.py` grönt — varje signal: rätt output dtype, rätt längd, inga NaN utanför warmup
- [ ] `pytest tests/test_lookahead.py` grönt — append-test för alla signaler (df[:N] vs df[:N+M] identiska för index ≤ N)
- [ ] `python main.py signals` listar alla registrerade signaler
- [ ] `python main.py signals ema_cross --grid` visar parametergrid
- [ ] Riktade signaler (price_action, rsi_divergence, ema_stop, atr_stop) har separata long/short-filer med SUPPORTED_SIDES korrekt satt
- [ ] Composer rejectar price_action_long med side='short' (fail-closed)
- [ ] Side-splittade signaler har `SIDE_SPLIT = True` (ema_stop_long/short, atr_stop_long/short, price_action_long/short)
- [ ] Side-resolver: `ema_stop` + `side=long` → `ema_stop_long` (test). `ema_cross` + `side=long` → INTE `ema_cross_long` (test)

### Fas 3 — Kombinator

- [ ] `python main.py validate config/sweeps/sweep_001.yaml` — YAML validerar, visar variant-count
- [ ] `python main.py sweep sweep_001.yaml --dry-run` — korrekt antal varianter
- [ ] `python main.py sweep sweep_001.yaml --jobs 4` med 10 varianter producerar:
  - `results/sweeps/sweep_001/manifest.json`
  - `results/sweeps/sweep_001/variants.jsonl`
  - `results/sweeps/sweep_001/leaderboard.md`
- [ ] Composer rejectar felkonfigurerade strategier (fel signal_type → error, inte tyst)
- [ ] Variants.jsonl innehåller ALLA varianter (OK + INVALID + ERROR). Rader i jsonl == n_variants i manifest
- [ ] BH-FDR order: antal varianter som når coarse p-value == antal sanity-survivors (ingen performance-filtrering däremellan)

### Fas 4 — Analys

- [ ] `pytest tests/test_statistics.py` grönt — BH-FDR, DSR, permutation test korrekthet
- [ ] `python main.py leaderboard` visar rankade varianter med normaliserade scores
- [ ] `python main.py report VARIANT_ID` genererar fullständig rapport med: equity curve, trade log, metrics, regime breakdown, overfitting diagnostik
- [ ] `python main.py regimes VARIANT_ID` visar per-regim prestanda
- [ ] `python main.py repro VARIANT_ID` — kör om variant och bekräftar att metrics matchar sparade värden inom ±1% tolerans

### Fas 5 — Validering

- [ ] sweep_001 körd på full IS data (aug–dec 2025)
- [ ] Leaderboard producerat med BH-FDR-korrigerade p-values
- [ ] Negativ kontroll körd på topp-10
- [ ] Minst en `repro`-test passerat
- [ ] Findings dokumenterade i `results/reports/sweep_001_findings.md`

### Globala DoD-krav (alla faser)

- [ ] `pytest` — alla tester gröna, inga warnings
- [ ] Inga hardcoded paths eller magic numbers utanför config/
- [ ] Alla fees/slippage i decimal rate-format
- [ ] Fill-modell: next-open, aldrig same-bar
- [ ] Inga framtidsläckor (lookahead-test grönt)
- [ ] Alla stokastiska operationer använder reproducerbara seeds, loggade i manifest
- [ ] Joblib workers laddar data från disk (parquet path), inte via pickling
- [ ] Trade returns är NETTO (efter fees+slippage) överallt i pipeline — via `engine/trades.py`
- [ ] `utils/canonical.py` används överallt för variant_id + manifest (aldrig inline json.dumps)
- [ ] YAML valideras mot `config/schema_sweep.json` (jsonschema)
- [ ] Riktade signaler har separata long/short-filer med SUPPORTED_SIDES
- [ ] Fill-modell är side-aware (slippage gör alltid fill sämre)
- [ ] Ingen silent drop: varje variant har status OK/INVALID/ERROR i variants.jsonl
- [ ] BH-FDR: inga performance-filter före coarse p-values (bara sanity-filter)
- [ ] Side-resolver: SIDE_SPLIT=True krävs för auto-resolve. Signaler utan SIDE_SPLIT resolvas aldrig
- [ ] FeatureStore: ny instans per (fold, segment, timeframe). HTF i separat store
- [ ] Ranking: tie-strategi = `method='average'` överallt (test: identiska metrics → samma rank)
- [ ] Permutation test: vektoriserad (sign-flip matrix i NumPy, ingen Python-loop per variant)
- [ ] Checkpoint-spårbarhet: commit efter varje fas, eller diff i `results/checkpoints/`
- [ ] No same-bar entry+exit: minsta trade = 1 bar

---

## 15. Vad detta INTE är

- **Inte en Pine Script-generator.** Ingen TradingView-integration.
- **Inte en live trading bot.** Ingen orderläggning.
- **Inte blind optimering.** Varje sweep kräver `theory`.
- **Inte en engångsrapport.** Iterativ forskningsplattform.
- **Shorts är möjliga men opt-in per signal.** Signaler börjar `SUPPORTED_SIDES={'long'}`. Short-stöd adderas efter explicit implementering + test.
- **Inte funding-justerad.** Perpetual swap funding rates ignoreras i v2.1. Resultat är inte live-ready, särskilt short-strategier med lång hålltid. Se Delta punkt 27.

---

## 16. REMINDER: Checkpoint-regel (HÅRD)

Efter VARJE fas: kör `pytest` + den fasens CLI smoke-test (se byggordning sektion 12). **Om NÅGOT test failar → fixa → kör om SAMMA checkpoint → commit → först därefter nästa fas.** Ingen ny fas innan grönt. Gå tillbaka och verifiera ALLA DoD-kriterier i sektion 14. Detta är den viktigaste regeln i hela prompten — den förhindrar att buggar ackumuleras och gör debugging omöjlig i senare faser. Inga undantag.
