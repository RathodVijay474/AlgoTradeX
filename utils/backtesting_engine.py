"""
backtesting_engine.py
=====================
Vectorised backtesting engine for AlgoTradeX.

Design principles:
  - Pure pandas/numpy — no heavy frameworks for rapid iteration
  - Transaction costs + slippage applied on every position change
  - Correct position sizing (full capital deployment on buy signal)
  - All heavy computation lives HERE, never in the UI loop

Supported strategies:
  dual_ma_crossover   : Short EMA > Long EMA → Long
  macd_trend          : MACD line > Signal line → Long
  rsi_mean_reversion  : RSI < 30 → Buy, RSI > 70 → Sell
  buy_and_hold        : Always fully invested
  custom              : Driven by pre-computed signal column
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    strategy_name: str
    history: pd.DataFrame          # timestamped equity curve + signals
    metrics: dict[str, float]
    trade_log: pd.DataFrame        # individual trade records

    @property
    def equity(self) -> pd.Series:
        return self.history["strategy_equity"]

    @property
    def benchmark(self) -> pd.Series:
        return self.history["benchmark_equity"]


# ---------------------------------------------------------------------------
# Core vectorised engine
# ---------------------------------------------------------------------------

def _apply_costs(price: float, direction: int, cost: float, slip: float) -> float:
    """Return effective execution price after slippage, in the direction of the trade."""
    slippage_adj = price * (1 + direction * slip)
    return slippage_adj


def run_backtest(
    ohlcv: pd.DataFrame,
    signal_series: pd.Series,           # +1 = Long, 0 = Flat (no shorting)
    strategy_name: str = "Strategy",
    initial_capital: float = 100_000.0,
    transaction_cost: float = 0.001,    # fraction per trade
    slippage_pct: float = 0.0005,       # one-way
    start_date: date | None = None,
    end_date: date | None = None,
) -> BacktestResult:
    """
    Vectorised single-strategy backtest.

    Parameters
    ----------
    ohlcv             : Standardised OHLCV frame
    signal_series     : +1 / 0 series aligned to ohlcv.index
    strategy_name     : Label for display
    initial_capital   : Starting portfolio value
    transaction_cost  : Fraction of trade value charged per execution
    slippage_pct      : One-way slippage fraction
    start_date/end_date : Optional date slice

    Returns
    -------
    BacktestResult with full history and performance metrics
    """
    df = ohlcv.copy()
    if start_date:
        df = df[df.index >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df.index <= pd.Timestamp(end_date)]

    if df.empty:
        raise ValueError("No data in the specified date range.")

    # Align signal
    signal = signal_series.reindex(df.index).fillna(0).clip(0, 1)
    df["signal"] = signal

    # Detect position changes
    df["position"]      = df["signal"]
    df["position_prev"] = df["position"].shift(1).fillna(0)
    df["trade"]         = df["position"] - df["position_prev"]   # +1 buy, -1 sell

    # Track portfolio value
    capital   = initial_capital
    shares    = 0.0
    equity    = []
    trade_log = []

    for idx, row in df.iterrows():
        price = float(row["close"])
        pos   = float(row["position"])
        trade = float(row["trade"])

        if trade > 0:                  # Buy
            exec_price = _apply_costs(price, +1, transaction_cost, slippage_pct)
            cost_fee   = capital * transaction_cost
            shares     = (capital - cost_fee) / exec_price
            capital    = 0.0
            trade_log.append({
                "date":      idx,
                "direction": "Buy",
                "price":     round(exec_price, 4),
                "shares":    round(shares, 4),
                "cost":      round(cost_fee, 4),
            })
        elif trade < 0 and shares > 0: # Sell
            exec_price = _apply_costs(price, -1, transaction_cost, slippage_pct)
            proceeds   = shares * exec_price
            fee        = proceeds * transaction_cost
            capital    = proceeds - fee
            trade_log.append({
                "date":      idx,
                "direction": "Sell",
                "price":     round(exec_price, 4),
                "shares":    round(shares, 4),
                "cost":      round(fee, 4),
                "pnl":       round(capital - initial_capital, 4),
            })
            shares = 0.0

        portfolio_value = capital + shares * price
        equity.append(portfolio_value)

    df["strategy_equity"] = equity
    df["benchmark_equity"] = initial_capital * (df["close"] / df["close"].iloc[0])

    # Drawdown
    peak        = df["strategy_equity"].cummax()
    df["drawdown"] = (df["strategy_equity"] - peak) / peak.replace(0, np.nan)

    # Metrics
    from utils.data_utils import compute_full_metrics
    metrics = compute_full_metrics(df["strategy_equity"], df["benchmark_equity"])

    trade_df = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()

    return BacktestResult(
        strategy_name=strategy_name,
        history=df,
        metrics=metrics,
        trade_log=trade_df,
    )


# ---------------------------------------------------------------------------
# Pre-built strategy signal generators
# ---------------------------------------------------------------------------

def _ema(s: pd.Series, p: int) -> pd.Series:
    return s.ewm(span=p, adjust=False).mean()


def _rma(s: pd.Series, p: int) -> pd.Series:
    return s.ewm(alpha=1 / p, adjust=False).mean()


def signal_dual_ma(
    close: pd.Series,
    short_period: int = 20,
    long_period: int  = 50,
) -> pd.Series:
    """Long when short EMA > long EMA, else flat."""
    short_ma = _ema(close, short_period)
    long_ma  = _ema(close, long_period)
    return (short_ma > long_ma).astype(int)


def signal_macd(
    close: pd.Series,
    fast: int   = 12,
    slow: int   = 26,
    signal_p: int = 9,
) -> pd.Series:
    """Long when MACD line > signal line, else flat."""
    macd_line   = _ema(close, fast) - _ema(close, slow)
    signal_line = _ema(macd_line, signal_p)
    return (macd_line > signal_line).astype(int)


def signal_rsi_reversion(
    close: pd.Series,
    period: int       = 14,
    oversold: float   = 30.0,
    overbought: float = 70.0,
) -> pd.Series:
    """Buy when RSI < oversold, sell when RSI > overbought. Hold otherwise."""
    delta    = close.diff()
    gain     = _rma(delta.clip(lower=0), period)
    loss     = _rma((-delta).clip(lower=0), period)
    rs       = gain / loss.replace(0, np.nan)
    rsi      = 100 - (100 / (1 + rs))

    pos = pd.Series(np.nan, index=close.index)
    pos[rsi < oversold]  = 1.0
    pos[rsi > overbought] = 0.0
    return pos.ffill().fillna(0).clip(0, 1)


def signal_buy_and_hold(close: pd.Series) -> pd.Series:
    """Always fully invested."""
    return pd.Series(1.0, index=close.index)


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY = {
    "dual_ma_crossover": {
        "label":   "Dual EMA Crossover",
        "params":  {"short_period": 20, "long_period": 50},
        "fn":      signal_dual_ma,
    },
    "macd_trend": {
        "label":   "MACD Trend",
        "params":  {"fast": 12, "slow": 26, "signal_p": 9},
        "fn":      signal_macd,
    },
    "rsi_mean_reversion": {
        "label":   "RSI Mean Reversion",
        "params":  {"period": 14, "oversold": 30.0, "overbought": 70.0},
        "fn":      signal_rsi_reversion,
    },
    "buy_and_hold": {
        "label":   "Buy & Hold",
        "params":  {},
        "fn":      signal_buy_and_hold,
    },
}


def run_named_strategy(
    ohlcv: pd.DataFrame,
    strategy_key: str,
    params: dict[str, Any] | None = None,
    initial_capital: float = 100_000.0,
    transaction_cost: float = 0.001,
    slippage_pct: float = 0.0005,
    start_date: date | None = None,
    end_date: date | None = None,
) -> BacktestResult:
    """
    Run any registered strategy by key.
    """
    if strategy_key not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy '{strategy_key}'. "
            f"Available: {list(STRATEGY_REGISTRY)}"
        )
    entry  = STRATEGY_REGISTRY[strategy_key]
    fn     = entry["fn"]
    merged = {**entry["params"], **(params or {})}

    close  = ohlcv["close"]
    signal = fn(close, **merged)

    return run_backtest(
        ohlcv=ohlcv,
        signal_series=signal,
        strategy_name=entry["label"],
        initial_capital=initial_capital,
        transaction_cost=transaction_cost,
        slippage_pct=slippage_pct,
        start_date=start_date,
        end_date=end_date,
    )


def list_strategies() -> list[dict]:
    return [
        {"key": k, "label": v["label"], "default_params": v["params"]}
        for k, v in STRATEGY_REGISTRY.items()
    ]
