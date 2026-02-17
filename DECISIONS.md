# Design Decisions Log

Decisions made during the autonomous build of FPR Research Platform v2.1.

## Phase 1: Infrastructure

### D1: FeatureStore Index Handling

**Decision:** FeatureStore returns Series with the same index as the bound DataFrame, using the DataFrame's natural integer index. Timestamp mapping is handled by the caller.

**Rationale:** Simpler implementation, clearer separation of concerns. The caller (signal or backtest) knows the context and can map to timestamps if needed.

### D2: Embargo Enforcement Location

**Decision:** Embargo is enforced in `walk_forward_split()` by trimming the train DataFrame, not in the backtest.

**Rationale:** Cleaner separation. The walk-forward splitter produces train/test splits that are already properly separated. The backtest doesn't need to know about embargo.

### D3: Trade Return Calculation Centralization

**Decision:** All trade return calculations go through `engine/trades.py`. The backtest calls `compute_trade_return()` which internally calls `compute_fill_price()` for both entry and exit.

**Rationale:** Delta 33 requires a single source of truth. This ensures consistency across backtest, metrics, permutation tests, and DSR calculations.

### D4: Fill Price Computation

**Decision:** Fill prices are computed using raw open price + slippage. Slippage is applied symmetrically but in opposite directions for entry vs exit, and for long vs short.

**Formula:**

- Long entry: `open * (1 + slip)` (worse: pay more)
- Long exit: `open * (1 - slip)` (worse: receive less)
- Short entry: `open * (1 - slip)` (worse: receive less)
- Short exit: `open * (1 + slip)` (worse: pay more)

### D5: Walk-Forward Month Alignment

**Decision:** Folds are aligned to calendar months using pandas `to_period('M')`. Train/test boundaries are determined by month membership.

**Rationale:** Simple and deterministic. Months provide natural boundaries for financial data.

### D6: Segment Handling in Validation

**Decision:** Large gaps (> 5 min) split data into segments. Small segments (< 5000 bars by default) are dropped with a warning. Validation returns both the report and the list of valid segments.

**Rationale:** Gaps represent data quality issues. Trading across gaps is unrealistic. Small segments don't provide enough data for meaningful walk-forward validation.

### D7: No Same-Bar Entry+Exit Implementation

**Decision:** Exit signals are checked only on bars after the entry fill bar. This means if entry signals on bar 5 (fills bar 6), exit can signal on bar 6 but fills on bar 7.

**Rationale:** Delta 34 requires minimum 1-bar trade duration. This implementation is simple and correct.

### D8: Warmup Handling in Backtest

**Decision:** Backtest accepts a `warmup_bars` parameter and skips that many bars at the start. Signals during warmup period are ignored.

**Rationale:** Warmup is needed for indicator calculation. The caller (walk-forward or strategy) knows the required warmup.

### D9: OOS Warmup Uses Full Dataset

**Decision:** Walk-forward OOS (out-of-sample) test periods use warmup bars from before the test period boundary, i.e., from the training data or earlier in the full dataset. The warmup bars are included in the DataFrame passed to backtest but signals during warmup are ignored.

**Rationale:** This is an accepted design choice for the following reasons:

1. **Not information leakage:** Warmup bars are only used for technical indicator computation (EMA convergence, etc.), not for strategy training or signal optimization. The indicator values are purely mechanical calculations.

2. **Preserves test period integrity:** Without this approach, we would lose the first N bars of every test fold to warmup, reducing effective test data and potentially missing important signals at fold boundaries.

3. **Realistic simulation:** In live trading, you always have historical data available for indicator warmup. Artificially constraining warmup to only test-period data would be unrealistic.

4. **Industry standard:** This approach is consistent with how most backtesting frameworks handle indicator warmup in walk-forward validation.

**Note:** The embargo period (gap between train and test) is separate from warmup and is still enforced to prevent lookahead bias in the training phase.

---

_This file will be updated as more decisions are made during Phases 2-5._
