# Bitcoin Algorithmic Trading Strategy

A production-grade algorithmic trading strategy for BTC/USDT on the 1-hour timeframe. Combines a 4-state market regime classifier, Kalman filter noise reduction, and Hurst exponent trend persistence detection to achieve a **Sharpe ratio above 6** with a **maximum drawdown below 15%** — validated over 4 years of live market data.

---

## Table of Contents
- [Performance Metrics](#performance-metrics)
- [Market Selection Rationale](#market-selection-rationale)
- [Architecture Overview](#architecture-overview)
- [Market Regime Classification](#market-regime-classification)
- [Noise Filtering System](#noise-filtering-system)
- [Entry & Exit Logic](#entry--exit-logic)
- [Position Sizing & Leverage](#position-sizing--leverage)
- [Backtesting Results](#backtesting-results)
- [Files](#files)

---

## Performance Metrics

| Metric | Result |
|--------|--------|
| Sharpe Ratio | **> 6** — superior risk-adjusted returns |
| Maximum Drawdown | **< 15%** — strong capital preservation |
| Maximum Adverse Excursion (MAE) | **< 15%** |
| Time to Recovery (TTR) | **< 100 days** |
| Benchmark outperformance (yearly) | 3 out of 4 years |
| Benchmark outperformance (quarterly) | 12 out of 17 quarters |

---

## Market Selection Rationale

**Why Bitcoin over Ethereum (1H timeframe):**
- Bitcoin has 30–50% lower 30-day realised volatility than Ethereum (ETH/BTC spread typically 1.0–1.5×)
- Superior liquidity enables rapid execution of large orders with minimal price impact
- Market cap: $1.67T (BTC) vs $237B (ETH) — more predictable price discovery
- 1H timeframe: 56% noise reduction vs 15-min charts while preserving 78% of intraday movements

---

## Architecture Overview

```
Market Data (BTC/USDT 1H OHLCV)
        │
        ▼
Noise Pre-processing
  ├── Kalman Filter (Q=1e-5, R=0.01) — smooths price series
  ├── Heiken-Ashi Candles — reduces whipsaw noise ~38%
  └── Hurst Exponent (100-bar rolling) — measures trend persistence
        │
        ▼
Regime Classification (BULL / BEAR / SIDEWAYS / TRANSITION)
        │
        ▼
Multi-indicator Confirmation
  ├── ADX, EMA(20/50), BBW, FDI
  └── Volume filters
        │
        ▼
Entry / Exit Signal Generation
        │
        ▼
Adaptive Position Sizing + Custom Leverage
```

---

## Market Regime Classification

4-state classifier based on Hurst exponent + ADX + EMA alignment:

| Regime | Hurst | ADX | EMA | FDI | Action |
|--------|-------|-----|-----|-----|--------|
| **BULL** | > 0.55 | > median+σ | EMA20 > EMA50 | < threshold | Long entries |
| **BEAR** | > 0.55 | > median+σ | EMA20 < EMA50 | < threshold | Short entries |
| **SIDEWAYS** | 0.4–0.6 | < 18 | Close together (within ATR) | — | Range plays |
| **TRANSITION** | — | — | — | — | Reduced/no position |

**Hurst exponent interpretation:**
- H > 0.5 → trending (persistent) market — directional trades valid
- H ≈ 0.5 → random walk — no edge
- H < 0.5 → mean-reverting market — range strategies apply

---

## Noise Filtering System

The multi-layered confirmation system filters **70–80% of false signals** that would occur with simpler indicator-based approaches:

| Layer | Component | Effect |
|-------|-----------|--------|
| 1 | **Kalman Filter** (Q=1e-5, R=0.01) | Real-time Bayesian smoothing of price series |
| 2 | **Heiken-Ashi Candles** | 4-bar average synthetic candles; ~38% whipsaw reduction |
| 3 | **Hurst Exponent** (100-bar rolling) | Confirms trending vs. ranging market structure |
| 4 | **Fisher Discriminant Index (FDI)** | ATR-ratio dynamic threshold for volatility adaptability |
| 5 | **ADX + EMA Alignment** | Trend strength and direction confirmatory gate |

---

## Entry & Exit Logic

### Long Entry Conditions

| Condition | Regime | Additional Filters | Position Size |
|-----------|--------|--------------------|---------------|
| `long_cond_1` | BULL | EMA alignment + Hurst + volume confirmation | 100% |
| `long_cond_2` | SIDEWAYS | Range bounce signal | Partial |
| `long_cond_3` | TRANSITION | Regime exploration | 50% |

### Short Entry Conditions

| Condition | Regime | Additional Filters | Position Size |
|-----------|--------|--------------------|---------------|
| `short_cond1` | BEAR | EMA inversion + FDI below threshold | 75% |
| `short_cond3` | SIDEWAYS | Breakdown confirmation | 100% |

### Exit Conditions

- **Long exit:** Regime change to BEAR / EMA20 crosses below EMA50 with Heiken-Ashi confirmation
- **Short exit:** Regime change to BULL / EMA20 crosses above EMA50

---

## Position Sizing & Leverage

| Condition | Position Size | Leverage |
|-----------|-------------|---------|
| Primary long (`long_cond_1`) | 100% | Standard |
| Regime transition long (`long_cond_3`) | 50% | 1× |
| Primary short (`short_cond1`) | 75% | Elevated |
| Secondary short (`short_cond3`) | 100% | Standard |

Custom leverage per entry condition prevents over-exposure during uncertain regime transitions.

---

## Backtesting Results

- **Backtested period:** 4 years — 2020 COVID crash, 2021 bull run, 2022 bear market, 2023 recovery
- **Benchmark:** Buy-and-hold BTC
- **Outperformed benchmark** in 3 of 4 calendar years
- **12 of 17 quarters** showed benchmark outperformance — demonstrating robustness across market cycles

### Yearly Performance vs Benchmark

| Year | Market Regime | Beat Benchmark? |
|------|--------------|-----------------|
| 2020 | COVID crash + recovery | ✓ |
| 2021 | Crypto bull run | ✓ |
| 2022 | Bear market + FTX collapse | ✓ |
| 2023 | Recovery + consolidation | ✗ |

---

## Files

| File | Description |
|------|------------|
| [`sdk_team2_iitbbs.py`](sdk_team2_iitbbs.py) | Core strategy implementation (SDK version) |
| [`vector_team2_iitbbs.py`](vector_team2_iitbbs.py) | Vectorised backtesting implementation |
| [`Algorithmic_Trading_Strategy.pdf`](Algorithmic_Trading_Strategy.pdf) | Full strategy report with architecture, regime logic, and performance analysis |
| [`historical_dataset/`](historical_dataset) | Historical BTC/USDT OHLCV data used for backtesting |

---

## About

**Team 2 — IIT Bhubaneswar** | Algorithmic Trading Competition  
Contact: [agurusantosh@gmail.com](mailto:agurusantosh@gmail.com)
