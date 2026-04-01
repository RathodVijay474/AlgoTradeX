# AlgoTradeX · Architecture Document

## 1. High-Level System Design

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         USER (Browser / Streamlit)                      │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │  HTTP / WebSocket
┌─────────────────────────────────▼───────────────────────────────────────┐
│  app.py  — Streamlit UI Layer                                           │
│  ┌──────────────────┐  ┌───────────────────┐  ┌────────────────────┐   │
│  │  Command Deck    │  │  Strategy Lab     │  │  Forecast Tab      │   │
│  │  (ensemble sig.) │  │  (backtest UI)    │  │  (prophet/timesfm) │   │
│  └──────────┬───────┘  └────────┬──────────┘  └────────┬───────────┘   │
│             └──────────────────┬┘                       │               │
└─────────────────────────────────┼──────────────────────┬┘───────────────┘
                                  │                       │
        ┌─────────────────────────▼──────┐   ┌───────────▼──────────────┐
        │  strategies/signal_engine.py   │   │  models/forecaster.py    │
        │  ┌──────────────┐              │   │  ┌──────────┐            │
        │  │  Technical   │ 40%  weight  │   │  │ Prophet  │ 45%        │
        │  │  (26 inds.)  │              │   │  ├──────────┤            │
        │  ├──────────────┤              │   │  │ TimesFM  │ 45%        │
        │  │  Sentiment   │ 30%  weight  │   │  ├──────────┤            │
        │  │  (VADER)     │              │   │  │  Naive   │ 10%        │
        │  ├──────────────┤              │   │  └──────────┘            │
        │  │  Forecasting │ 30%  weight  │   └──────────────────────────┘
        │  └──────────────┘              │
        └────────────────────────────────┘
                        │
        ┌───────────────▼────────────────┐   ┌──────────────────────────┐
        │  strategies/                   │   │  utils/                  │
        │  algotradex_strategies.py      │   │  data_utils.py           │
        │  (11 oscillators, 15 MAs,      │   │  backtesting_engine.py   │
        │   custom rule builder)         │   │  (vectorised, no dep)    │
        └───────────────┬────────────────┘   └─────────────┬────────────┘
                        │                                   │
        ┌───────────────▼───────────────────────────────────▼────────────┐
        │  EXTERNAL DATA                                                  │
        │  Yahoo Finance (yfinance)  ·  User CSV upload                  │
        └─────────────────────────────────────────────────────────────────┘
```

---

## 2. Signal Aggregation — In Depth

### 2.1 Technical Component (weight = 0.40)

All 26 indicators run on the OHLCV frame via `run_all_indicators()`.
Each emits one of {Buy, Sell, Neutral}.

```
S_tech = (N_buy - N_sell) / N_total
```

This is a **balanced net signal** normalised to [-1, +1].
A unanimous 26/26 Buy gives S_tech = +1.0.
Equal split gives S_tech = 0.0.

### 2.2 Sentiment Component (weight = 0.30)

Uses VADER (Valence Aware Dictionary for sEntiment Reasoning):

```
S_sent = mean(VADER_compound(headline_i)) for i in headlines
VADER compound ∈ [-1, +1]  (≥+0.05 → positive, ≤-0.05 → negative)
```

In the current build, headlines must be injected via `news_headlines`
parameter. Default = 0.0 (neutral) when no headlines are supplied.

### 2.3 Forecasting Component (weight = 0.30)

```
expected_return = (F - P) / |P|
S_fore = clip(expected_return × 50, -1, +1)
```

The ×50 amplifier maps a 2% predicted return to a ±1 directional score.
Choosing 50 is a calibration hyperparameter — raise to increase forecast
sensitivity, lower to dampen.

### 2.4 Ensemble Aggregation

```python
Ensemble = 0.40 * S_tech + 0.30 * S_sent + 0.30 * S_fore
Ensemble = clip(Ensemble, -1, +1)

# Decision boundaries
if   Ensemble >= +0.15:  signal = Buy
elif Ensemble <= -0.15:  signal = Sell
else:                    signal = Hold

# Confidence (monotone in |Ensemble|)
Confidence = clip(0.50 + 0.45 * |Ensemble|, 0.50, 0.95)
```

The ±0.15 dead-band prevents noise from generating spurious signals
when components disagree. A score of 0.15 on the positive side
corresponds to e.g. 55% Buy technical consensus + neutral everything else.

---

## 3. Forecasting Methodology

### 3.1 Meta Prophet

Prophet models the time series as:

```
y(t) = g(t) + s(t) + h(t) + ε(t)

g(t) — piecewise linear trend with automatic changepoint detection
s(t) — Fourier series seasonality (yearly K=10, weekly K=3)
h(t) — holiday effects (not enabled in current build)
ε(t) — i.i.d. Normal noise
```

Configuration used:
```python
Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode="multiplicative",  # better for financial series
    changepoint_prior_scale=0.08,       # moderate flexibility
    interval_width=0.80,
)
```

### 3.2 Google TimesFM

TimesFM is a **decoder-only transformer** pre-trained by Google Brain on
100 billion real-world time-series observations. It performs
**zero-shot forecasting** — no fine-tuning on the target series.

```
Input  : context window of 128 daily closing prices
Model  : 20-layer decoder, 1280 model dimensions, 200M parameters
Output : 10-step point forecast + quantiles
Used   : forecast[0]  (next-day prediction)
```

TimesFM is accessed via the `timesfm` PyPI package which downloads
the checkpoint from HuggingFace Hub on first use.

### 3.3 Naive Baseline

```
OLS fit: y = α + β·t  over last 30 bars
Forecast: ŷ = α + β·(T+horizon)
```

Always available. Acts as a regularising floor in the ensemble.

### 3.4 Confidence Bounds

```
σ_20 = std(r_t, r_{t-1}, ..., r_{t-19})   where r_t = ΔP/P
lower = F̂ × (1 - max(σ_20, 0.005))
upper = F̂ × (1 + max(σ_20, 0.005))
```

The `max(σ_20, 0.005)` floor prevents degenerate zero-width bands
in ultra-low-volatility periods.

---

## 4. Backtesting Engine

### 4.1 Design

The backtesting engine is **pure pandas/numpy** — no Backtrader/VectorBT
dependency for portability and auditability.

### 4.2 Execution Model

```
For each bar t:
  if signal[t] == 1 and position[t-1] == 0:  # Buy
      exec_price = close[t] × (1 + slippage)
      shares     = (cash - txn_cost × cash) / exec_price
      cash       = 0

  if signal[t] == 0 and position[t-1] == 1:  # Sell
      exec_price = close[t] × (1 - slippage)
      proceeds   = shares × exec_price
      cash       = proceeds × (1 - txn_cost)
      shares     = 0

  portfolio_value[t] = cash + shares × close[t]
```

### 4.3 Performance Metrics

| Metric | Formula |
|--------|---------|
| Total Return | `(V_T / V_0) - 1` |
| Annualised Return | `(1 + R_total)^(252/N) - 1` |
| Annualised Volatility | `σ(r_t) × √252` |
| Sharpe Ratio | `μ(r_t) / σ(r_t) × √252` |
| Sortino Ratio | `μ(r_t) / σ(r_t^-) × √252` |
| Calmar Ratio | `R_annual / \|MDD\|` |
| Max Drawdown | `min((V_t - peak_t) / peak_t)` |

---

## 5. Indicator Library

### Oscillators (11)
| Indicator | Buy Condition | Sell Condition |
|-----------|--------------|----------------|
| RSI(14) | < 30 | > 70 |
| Stochastic %K | < 20 | > 80 |
| CCI(20) | < -100 | > +100 |
| ADX(14) | DI+ > DI- & ADX > 20 | DI- > DI+ & ADX > 20 |
| Awesome Oscillator | AO > 0 & rising | AO < 0 & falling |
| Momentum(10) | > 0 | < 0 |
| MACD(12,26) | histogram > 0 | histogram < 0 |
| Stoch RSI Fast | < 20 | > 80 |
| Williams %R(14) | < -80 | > -20 |
| Bull Bear Power | zero-cross up | zero-cross down |
| Ultimate Oscillator(7,14,28) | < 30 | > 70 |

### Moving Averages (15)
All MA indicators: **Buy if price > MA**, **Sell if price < MA**.
EMA: 10/20/30/50/100/200 · SMA: 10/20/30/50/100/200 ·
Ichimoku Base Line · VWMA(20) · Hull MA(9)

---

## 6. Data Flow (Session State)

```
Streamlit session_state["result"] = {
    "ticker":   str,
    "ohlcv":    pd.DataFrame,    # standardised OHLCV
    "signal":   EnsembleSignal,  # from signal_engine
    "forecast": ForecastResult,  # from forecaster
    "backtest": BacktestResult,  # from backtesting_engine
    "ind_df":   pd.DataFrame,    # 26-row indicator table
    "inputs":   dict,            # sidebar config snapshot
}
```

The heavy computation runs **once per button click** via `run_analysis()`.
All tabs render from the cached `session_state["result"]` — zero redundant
computation on tab switch.

---

## 7. Performance Considerations

| Concern | Mitigation |
|---------|-----------|
| Heavy indicator computation | Run once in `run_analysis()`, cached in session state |
| Prophet fitting latency | Runs in background via `st.spinner()` |
| TimesFM model download | ~800MB first run, then cached by HuggingFace Hub |
| yfinance rate limits | Standard free tier; add `@st.cache_data(ttl=3600)` for production |
| Large OHLCV frames | Slice to selected date range before passing to engines |
