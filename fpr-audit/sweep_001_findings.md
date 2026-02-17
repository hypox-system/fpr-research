# Sweep 001: EMA + PVSRA Findings

**Date:** 2026-02-17
**Sweep Name:** sweep_001_ema_pvsra
**Symbol:** BTCUSDT
**Timeframe:** 1H

## Summary

| Metric           | Value       |
| ---------------- | ----------- |
| Total Variants   | 288         |
| Passed Sanity    | 270 (93.8%) |
| BH-FDR Survivors | 0           |
| Runtime          | 531.4s      |

## Strategy Configuration

- **Entry:** EMA Cross (fast: 5/8/13/21, slow: 21/34/50/100)
- **Filters:** EMA Gating + PVSRA Volume Filter
- **Exit:** EMA Stop (period: 50/100)
- **Side:** Long only
- **Fees:** 0.06% taker + 0.01% slippage

## Key Findings

### 1. All Variants Negative Sharpe

All 270 passing variants showed negative OOS Sharpe ratios, ranging from -6.7 to -24.0. This indicates the simple EMA crossover strategy consistently loses money on BTC 1H data after accounting for fees and slippage.

**Top 5 by OOS Sharpe (least negative):**
| Rank | Variant | OOS Sharpe | Trades | Win Rate |
|------|---------|------------|--------|----------|
| 1 | af60329be6c1 | -6.71 | 371 | 11.1% |
| 2 | da527d5ca96d | -7.28 | 361 | 14.4% |
| 3 | b680b947850e | -7.33 | 520 | 15.2% |
| 4 | 8445c8cc15d6 | -7.44 | 485 | 13.8% |
| 5 | a696afaae012 | -7.54 | 373 | 13.9% |

### 2. Zero BH-FDR Survivors

With 270 p-values tested at α=0.05 after BH correction, zero variants showed statistically significant positive edge. This is expected given:

- All mean returns are negative (p-value = 1.0 by construction)
- The permutation test correctly identifies no positive edge exists

### 3. Sanity Filter Performance

18 variants (6.2%) failed sanity checks:

- Likely due to parameter combinations producing too few trades
- Fast EMA values >= Slow EMA (no valid crossovers)

### 4. IS/OOS Consistency

All variants showed `sharpe_decay = 0` because both IS and OOS Sharpes were negative. The consistency_ratio = 0.0 for all variants indicates no fold had positive returns.

## Conclusions

1. **Strategy Viability:** The EMA crossover + PVSRA filter combination is not viable for BTC 1H trading with current fee structure.

2. **Fee Sensitivity:** With ~10-15% win rate and profit factors around 0.4-0.5, the strategy cannot overcome 0.07% round-trip costs.

3. **Next Steps:**
   - Test on lower fee venues (spot vs perpetual)
   - Add trend filter to avoid chop
   - Consider mean-reversion instead of momentum
   - Test wider stop distances to improve win rate

## Verification

```
# Variant count matches manifest
wc -l variants.jsonl → 288
manifest.n_variants → 288 ✓

# Repro verification (top variant)
python main.py repro af60329be6c1
Found: OOS Sharpe = -6.71, Trades = 371 ✓
```

## Data Fingerprint

- **Config Hash:** 662aaf40e7f79cd310355c27b659e8fd66a7fa8baccb82388ec510de70f9421e
- **Data Fingerprint:** a6f49dba7611ea15
- **Date Range:** 2025-08-01 to 2025-12-31 (IS)
