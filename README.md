# AlgoTradeX V5

Institutional-style quantitative trading research and orchestration platform built around a risk-first decision stack.

AlgoTradeX V5 upgrades the earlier dashboard into a market-aware multi-model system that combines:

- technical indicator aggregation
- market regime intelligence
- cross-sectional alpha sleeves
- statistical and deep-learning forecasting
- strict pre-trade risk gating
- backtesting and performance diagnostics

The current repository is the Python research, analytics, and orchestration layer. The architecture is intentionally prepared for a future low-latency C++ execution and risk engine.

## Core Capabilities

- Streamlit command center for interactive analysis
- 26-indicator technical engine
- Prophet, TimesFM, naive baseline, and hybrid sequence forecasting
- market intelligence engine for regime, breadth, sector strength, and correlation stress
- alpha sleeves for momentum, volatility regime, and liquidity/volume spikes
- ensemble agreement logic that degrades conviction when models diverge
- strict risk engine with trade-level loss caps and emergency freeze logic
- vectorized strategy backtesting with equity, drawdown, Sharpe, and trade logs
- clean execution intent payload designed for future Python-to-C++ handoff

## System Architecture

```text
AlgoTradeX/
├── app.py
├── models/
│   ├── forecaster.py
│   ├── market_intelligence.py
│   └── alpha_models.py
├── strategies/
│   ├── algotradex_strategies.py
│   └── signal_engine.py
├── risk/
│   └── risk_engine.py
├── utils/
│   ├── data_utils.py
│   └── backtesting_engine.py
├── tests/
│   └── test_engine.py
├── ARCHITECTURE.md
└── requirements.txt
```

## Decision Pipeline

```text
1. Acquire OHLCV data
2. Compute technical indicators
3. Analyze market-wide context
4. Run alpha sleeves
5. Run forecast ensemble
6. Combine signals through the V5 ensemble engine
7. Apply strict risk filter
8. Emit approved action + execution payload
```

## V5 Model Stack

### 1. Market Intelligence Engine

Evaluates broad market conditions rather than relying only on the active ticker.

- regime classification: `TRENDING`, `MEAN_REVERTING`, `HIGH_VOL`, `CRASH`
- market bias scoring
- volatility clustering
- sector relative strength
- cross-asset correlation stress
- risk multiplier for downstream sizing

### 2. Alpha Models

- Cross-Sectional Momentum
  - ranks assets on trailing returns
  - identifies top-decile leadership
- Volatility Regime
  - classifies offensive vs defensive posture
- Liquidity / Volume Spike
  - identifies abnormal participation and impulse confirmation

### 3. Forecasting Stack

- Meta Prophet
- Google TimesFM
- Naive linear baseline
- Hybrid sequence model
  - feature-driven fallback when deep stack cannot train robustly
  - BiLSTM path when sufficient data and dependencies are available

### 4. Ensemble Logic

The final decision engine blends:

- technical score
- sentiment placeholder sleeve
- forecast score
- deep-learning score
- market intelligence bias

Agreement is explicitly modeled. Strong agreement boosts conviction. Weak agreement reduces confidence and can collapse the final action to hold.

## Risk-First Design

Capital preservation is a hard design constraint, not an optional overlay.

- maximum loss per trade: `0.5%` of capital
- position sizing:

```text
position_size = (capital * 0.005) / stop_loss_distance
```

- hard global stop triggers include:
  - crash regime
  - extreme volatility
  - drawdown breach
  - drawdown spike
  - consecutive losing trades
  - model disagreement spike

Emergency behavior:

- flatten all positions
- cancel all pending orders
- freeze new trading activity

## Dashboard

The Streamlit application exposes:

- `Command Deck`
  - approved action
  - ensemble score
  - agreement score
  - risk status
  - execution intent
- `Market Intelligence`
  - regime
  - breadth
  - sector heatmap
  - correlation heatmap
  - alpha sleeve diagnostics
- `Chart`
  - price structure
  - EMA overlays
  - volume
- `Strategy Lab`
  - backtests
  - drawdowns
  - rolling Sharpe
  - monthly returns
- `Forecast`
  - model outputs
  - confidence band
  - deep model status
- `Indicators`
  - 26-indicator panel
- `Raw Data`
  - exportable research artifacts

The app also supports `System`, `Dark`, and `White` theme modes.

## Installation

### Environment

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

## Validation

Run the test suite:

```bash
python -m pytest -q
```

Current local verification used during this version:

```bash
python -m compileall app.py models strategies risk
python -m pytest -q
```

## Research and Production Boundary

This repository currently implements the Python side of the stack:

- research
- feature engineering
- model orchestration
- risk analytics
- dashboarding
- execution intent generation

The execution boundary is intentionally separated so that a future C++20 engine can own:

- market data ingestion
- order routing
- pre-wire risk checks
- low-latency OMS
- exchange connectivity
- sub-millisecond execution paths

## Notes on Scope

This repository does not claim that a low-latency C++ execution engine is already implemented here. The codebase currently prepares the signal and risk payloads that such an engine would consume.

## Disclaimer

This project is for research, simulation, and system design purposes. It is not investment advice and should not be treated as a production brokerage or execution platform without independent validation, risk review, and exchange-grade operational controls.
