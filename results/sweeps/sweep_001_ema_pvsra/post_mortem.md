# Post-Mortem: sweep_001_ema_pvsra

## Executive Summary

270 variants tested. 0 BH-FDR survivors. 
All variants show strongly negative OOS Sharpe. 
Primary failure mode: LOW_SIGNAL.

## Verdict: DEAD

## Top 10 Variants

| Variant | OOS Sharpe | Trades | PF | Win Rate |
|---------|-----------|--------|------|----------|
| af60329b | -8.08 | 371 | 0.52 | 11.1% |

## Why It Failed

No BH-FDR survivors. Best OOS Sharpe is -8.08, well below threshold of -5.0. Signal does not capture meaningful edge.

### Fee Analysis

- Round-trip fee+slippage: 0.14%

## What To Try Next

- reconsider signal hypothesis entirely
- try different entry timing (earlier/later)
- combine with other confirming signals

## Most Promising Region

Score: -10.49

- fast: [5, 21]
- slow: [34, 100]

---
*Generated: 2026-02-17T18:55:10.778283Z*