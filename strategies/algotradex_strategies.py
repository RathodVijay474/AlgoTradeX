"""
algotradex_strategies.py
========================
Complete Technical Strategy Engine for AlgoTradeX
Implements all Oscillators + Moving Averages from the indicator panel,
plus a Custom Strategy Builder framework.

Indicators implemented
----------------------
OSCILLATORS (11)
  RSI (14), Stochastic %K (14,3,3), CCI (20), ADX (14),
  Awesome Oscillator, Momentum (10), MACD Level (12,26),
  Stochastic RSI Fast (3,3,14,14), Williams %R (14),
  Bull Bear Power, Ultimate Oscillator (7,14,28)

MOVING AVERAGES (14)
  EMA(10/20/30/50/100/200), SMA(10/20/30/50/100/200),
  Ichimoku Base Line (9,26,52,26), VWMA(20), Hull MA (9)

CUSTOM STRATEGY BUILDER
  Rule-based engine where users combine any indicator signals
  with AND/OR logic and threshold overrides.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Signal constants
# ---------------------------------------------------------------------------
BUY     = "Buy"
SELL    = "Sell"
NEUTRAL = "Neutral"

SignalType = Literal["Buy", "Sell", "Neutral"]


# ===========================================================================
# Low-level helpers
# ===========================================================================

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def _rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder smoothing (RMA / SMMA)."""
    return series.ewm(alpha=1 / period, adjust=False).mean()


def _tr(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [high - low,
         (high - prev_close).abs(),
         (low  - prev_close).abs()],
        axis=1,
    ).max(axis=1)


def _signal(value: float, buy_cond: bool, sell_cond: bool) -> SignalType:
    if buy_cond:
        return BUY
    if sell_cond:
        return SELL
    return NEUTRAL


# ===========================================================================
# Result container
# ===========================================================================

@dataclass
class IndicatorResult:
    name:   str
    value:  float
    signal: SignalType
    series: pd.Series | None = None          # full history
    extra:  dict[str, pd.Series] = field(default_factory=dict)  # sub-lines

    def to_dict(self) -> dict:
        return {"name": self.name, "value": round(self.value, 4), "signal": self.signal}


# ===========================================================================
# ── OSCILLATORS ─────────────────────────────────────────────────────────────
# ===========================================================================

def calc_rsi(close: pd.Series, period: int = 14) -> IndicatorResult:
    """Relative Strength Index."""
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = _rma(gain, period)
    avg_loss = _rma(loss, period)
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    val = float(rsi.iloc[-1])
    return IndicatorResult(
        name=f"Relative Strength Index ({period})",
        value=val,
        signal=_signal(val, val < 30, val > 70),
        series=rsi,
    )


def calc_stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series,
    k_period: int = 14, smooth_k: int = 3, d_period: int = 3,
) -> IndicatorResult:
    """Stochastic Oscillator %K."""
    lowest_low   = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    raw_k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    k = raw_k.rolling(smooth_k).mean()
    d = k.rolling(d_period).mean()
    val = float(k.iloc[-1])
    return IndicatorResult(
        name=f"Stochastic %K ({k_period}, {smooth_k}, {d_period})",
        value=val,
        signal=_signal(val, val < 20, val > 80),
        series=k,
        extra={"d": d},
    )


def calc_cci(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20
) -> IndicatorResult:
    """Commodity Channel Index."""
    tp   = (high + low + close) / 3
    ma   = tp.rolling(period).mean()
    md   = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    cci  = (tp - ma) / (0.015 * md)
    val  = float(cci.iloc[-1])
    return IndicatorResult(
        name=f"Commodity Channel Index ({period})",
        value=val,
        signal=_signal(val, val < -100, val > 100),
        series=cci,
    )


def calc_adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> IndicatorResult:
    """Average Directional Index."""
    tr  = _tr(high, low, close)
    up  = high.diff()
    dn  = -low.diff()
    dm_plus  = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=close.index)
    dm_minus = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=close.index)
    atr = _rma(tr,       period)
    di_plus  = 100 * _rma(dm_plus,  period) / atr.replace(0, np.nan)
    di_minus = 100 * _rma(dm_minus, period) / atr.replace(0, np.nan)
    dx  = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    adx = _rma(dx, period)
    val = float(adx.iloc[-1])
    # ADX > 25 = trending; signal based on DI crossover
    last_plus  = float(di_plus.iloc[-1])
    last_minus = float(di_minus.iloc[-1])
    return IndicatorResult(
        name=f"Average Directional Index ({period})",
        value=val,
        signal=_signal(val, last_plus > last_minus and val > 20, last_minus > last_plus and val > 20),
        series=adx,
        extra={"di_plus": di_plus, "di_minus": di_minus},
    )


def calc_awesome_oscillator(
    high: pd.Series, low: pd.Series
) -> IndicatorResult:
    """Awesome Oscillator (Bill Williams)."""
    mid = (high + low) / 2
    ao  = _sma(mid, 5) - _sma(mid, 34)
    val = float(ao.iloc[-1])
    prev = float(ao.iloc[-2]) if len(ao) > 1 else 0.0
    return IndicatorResult(
        name="Awesome Oscillator",
        value=val,
        signal=_signal(val, val > 0 and val > prev, val < 0 and val < prev),
        series=ao,
    )


def calc_momentum(close: pd.Series, period: int = 10) -> IndicatorResult:
    """Price Momentum."""
    mom = close - close.shift(period)
    val = float(mom.iloc[-1])
    return IndicatorResult(
        name=f"Momentum ({period})",
        value=val,
        signal=_signal(val, val > 0, val < 0),
        series=mom,
    )


def calc_macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal_period: int = 9
) -> IndicatorResult:
    """MACD Level."""
    macd_line   = _ema(close, fast) - _ema(close, slow)
    signal_line = _ema(macd_line, signal_period)
    histogram   = macd_line - signal_line
    val  = float(macd_line.iloc[-1])
    hist = float(histogram.iloc[-1])
    return IndicatorResult(
        name=f"MACD Level ({fast}, {slow})",
        value=val,
        signal=_signal(val, hist > 0 and macd_line.iloc[-1] > signal_line.iloc[-1],
                            hist < 0 and macd_line.iloc[-1] < signal_line.iloc[-1]),
        series=macd_line,
        extra={"signal": signal_line, "histogram": histogram},
    )


def calc_stochastic_rsi(
    close: pd.Series,
    rsi_period: int = 14,
    stoch_period: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3,
) -> IndicatorResult:
    """Stochastic RSI Fast."""
    delta = close.diff()
    gain  = _rma(delta.clip(lower=0), rsi_period)
    loss  = _rma((-delta).clip(lower=0), rsi_period)
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - (100 / (1 + rs))
    lowest  = rsi.rolling(stoch_period).min()
    highest = rsi.rolling(stoch_period).max()
    stoch_rsi = (rsi - lowest) / (highest - lowest).replace(0, np.nan) * 100
    k = stoch_rsi.rolling(smooth_k).mean()
    d = k.rolling(smooth_d).mean()
    val = float(k.iloc[-1])
    return IndicatorResult(
        name=f"Stochastic RSI Fast ({smooth_k}, {smooth_d}, {rsi_period}, {stoch_period})",
        value=val,
        signal=_signal(val, val < 20, val > 80),
        series=k,
        extra={"d": d},
    )


def calc_williams_r(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> IndicatorResult:
    """Williams Percent Range."""
    highest = high.rolling(period).max()
    lowest  = low.rolling(period).min()
    wr = -100 * (highest - close) / (highest - lowest).replace(0, np.nan)
    val = float(wr.iloc[-1])
    return IndicatorResult(
        name=f"Williams Percent Range ({period})",
        value=val,
        signal=_signal(val, val < -80, val > -20),
        series=wr,
    )


def calc_bull_bear_power(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 13
) -> IndicatorResult:
    """Bull Bear Power (Elder Ray)."""
    ema13     = _ema(close, period)
    bull_power = high  - ema13
    bear_power = low   - ema13
    bbp = bull_power + bear_power
    val = float(bbp.iloc[-1])
    return IndicatorResult(
        name="Bull Bear Power",
        value=val,
        signal=_signal(val,
            val > 0 and float(bbp.iloc[-2]) < 0 if len(bbp) > 1 else val > 0,
            val < 0 and float(bbp.iloc[-2]) > 0 if len(bbp) > 1 else val < 0,
        ),
        series=bbp,
        extra={"bull": bull_power, "bear": bear_power},
    )


def calc_ultimate_oscillator(
    high: pd.Series, low: pd.Series, close: pd.Series,
    p1: int = 7, p2: int = 14, p3: int = 28,
) -> IndicatorResult:
    """Ultimate Oscillator."""
    prev_close = close.shift(1)
    bp = close - pd.concat([low, prev_close], axis=1).min(axis=1)
    tr = pd.concat([high, prev_close], axis=1).max(axis=1) - \
         pd.concat([low,  prev_close], axis=1).min(axis=1)

    def _avg(p: int) -> pd.Series:
        return bp.rolling(p).sum() / tr.rolling(p).sum().replace(0, np.nan)

    uo  = 100 * (4 * _avg(p1) + 2 * _avg(p2) + _avg(p3)) / 7
    val = float(uo.iloc[-1])
    return IndicatorResult(
        name=f"Ultimate Oscillator ({p1}, {p2}, {p3})",
        value=val,
        signal=_signal(val, val < 30, val > 70),
        series=uo,
    )


# ===========================================================================
# ── MOVING AVERAGES ──────────────────────────────────────────────────────────
# ===========================================================================

def calc_ema(close: pd.Series, period: int) -> IndicatorResult:
    ema = _ema(close, period)
    val = float(ema.iloc[-1])
    price = float(close.iloc[-1])
    return IndicatorResult(
        name=f"Exponential Moving Average ({period})",
        value=val,
        signal=_signal(val, price > val, price < val),
        series=ema,
    )


def calc_sma(close: pd.Series, period: int) -> IndicatorResult:
    sma = _sma(close, period)
    val = float(sma.iloc[-1])
    price = float(close.iloc[-1])
    return IndicatorResult(
        name=f"Simple Moving Average ({period})",
        value=val,
        signal=_signal(val, price > val, price < val),
        series=sma,
    )


def calc_ichimoku_base(
    high: pd.Series, low: pd.Series,
    conversion_period: int = 9,
    base_period:       int = 26,
    span_b_period:     int = 52,
    displacement:      int = 26,
) -> IndicatorResult:
    """Ichimoku Base Line (Kijun-sen)."""
    base  = (high.rolling(base_period).max()  + low.rolling(base_period).min())  / 2
    conv  = (high.rolling(conversion_period).max() + low.rolling(conversion_period).min()) / 2
    span_a = (conv + base) / 2
    span_b = (high.rolling(span_b_period).max() + low.rolling(span_b_period).min()) / 2
    val    = float(base.iloc[-1])
    price  = float(high.iloc[-1])  # use last high as proxy for current price
    return IndicatorResult(
        name=f"Ichimoku Base Line ({conversion_period}, {base_period}, {span_b_period}, {displacement})",
        value=val,
        signal=_signal(val, price > float(span_a.iloc[-1]), price < float(span_a.iloc[-1])),
        series=base,
        extra={"conversion": conv, "span_a": span_a, "span_b": span_b},
    )


def calc_vwma(
    close: pd.Series, volume: pd.Series, period: int = 20
) -> IndicatorResult:
    """Volume Weighted Moving Average."""
    vwma = (close * volume).rolling(period).sum() / volume.rolling(period).sum()
    val  = float(vwma.iloc[-1])
    price = float(close.iloc[-1])
    return IndicatorResult(
        name=f"Volume Weighted Moving Average ({period})",
        value=val,
        signal=_signal(val, price > val, price < val),
        series=vwma,
    )


def calc_hull_ma(close: pd.Series, period: int = 9) -> IndicatorResult:
    """Hull Moving Average."""
    half = max(int(period / 2), 1)
    sqrt = max(int(np.sqrt(period)), 1)
    wma_half = close.rolling(half).apply(
        lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.arange(1, len(x)+1).sum(), raw=True
    )
    wma_full = close.rolling(period).apply(
        lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.arange(1, len(x)+1).sum(), raw=True
    )
    raw_hma = 2 * wma_half - wma_full
    hma = raw_hma.rolling(sqrt).apply(
        lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.arange(1, len(x)+1).sum(), raw=True
    )
    val   = float(hma.iloc[-1])
    price = float(close.iloc[-1])
    return IndicatorResult(
        name=f"Hull Moving Average ({period})",
        value=val,
        signal=_signal(val, price > val, price < val),
        series=hma,
    )


# ===========================================================================
# ── MASTER RUNNER ────────────────────────────────────────────────────────────
# ===========================================================================

def run_all_indicators(
    ohlcv: pd.DataFrame,
) -> dict[str, list[IndicatorResult]]:
    """
    Compute every indicator from the panel image and return
    a dict with keys 'oscillators' and 'moving_averages'.

    Expected columns (case-insensitive): open, high, low, close, volume
    """
    df    = ohlcv.copy()
    df.columns = [c.lower() for c in df.columns]
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df.get("volume", pd.Series(np.ones(len(df)), index=df.index))

    oscillators = [
        calc_rsi(close),
        calc_stochastic(high, low, close),
        calc_cci(high, low, close),
        calc_adx(high, low, close),
        calc_awesome_oscillator(high, low),
        calc_momentum(close),
        calc_macd(close),
        calc_stochastic_rsi(close),
        calc_williams_r(high, low, close),
        calc_bull_bear_power(high, low, close),
        calc_ultimate_oscillator(high, low, close),
    ]

    moving_averages = [
        calc_ema(close, 10),
        calc_sma(close, 10),
        calc_ema(close, 20),
        calc_sma(close, 20),
        calc_ema(close, 30),
        calc_sma(close, 30),
        calc_ema(close, 50),
        calc_sma(close, 50),
        calc_ema(close, 100),
        calc_sma(close, 100),
        calc_ema(close, 200),
        calc_sma(close, 200),
        calc_ichimoku_base(high, low),
        calc_vwma(close, volume),
        calc_hull_ma(close),
    ]

    return {"oscillators": oscillators, "moving_averages": moving_averages}


def build_summary_table(results: dict[str, list[IndicatorResult]]) -> pd.DataFrame:
    """Convert all indicator results into a tidy DataFrame."""
    rows = []
    for group, indicators in results.items():
        for ind in indicators:
            rows.append({
                "group":  group.replace("_", " ").title(),
                "name":   ind.name,
                "value":  round(ind.value, 4),
                "signal": ind.signal,
            })
    return pd.DataFrame(rows)


def build_signal_summary(results: dict[str, list[IndicatorResult]]) -> dict[str, Any]:
    """Aggregate signal counts and produce an overall recommendation."""
    all_signals = [
        ind.signal
        for indicators in results.values()
        for ind in indicators
        if not np.isnan(ind.value)
    ]
    buy_count     = all_signals.count(BUY)
    sell_count    = all_signals.count(SELL)
    neutral_count = all_signals.count(NEUTRAL)
    total         = len(all_signals)

    buy_pct  = buy_count  / total if total else 0.0
    sell_pct = sell_count / total if total else 0.0

    if buy_pct >= 0.60:
        overall, strength = "Strong Buy",  "strong"
    elif buy_pct >= 0.45:
        overall, strength = "Buy",         "moderate"
    elif sell_pct >= 0.60:
        overall, strength = "Strong Sell", "strong"
    elif sell_pct >= 0.45:
        overall, strength = "Sell",        "moderate"
    else:
        overall, strength = "Neutral",     "neutral"

    return {
        "overall":       overall,
        "strength":      strength,
        "buy_count":     buy_count,
        "sell_count":    sell_count,
        "neutral_count": neutral_count,
        "total":         total,
        "buy_pct":       buy_pct,
        "sell_pct":      sell_pct,
        "neutral_pct":   neutral_count / total if total else 0.0,
    }


# ===========================================================================
# ── CUSTOM STRATEGY BUILDER ──────────────────────────────────────────────────
# ===========================================================================

@dataclass
class StrategyRule:
    """
    A single rule in a custom strategy.

    indicator_fn : callable that takes (ohlcv_df) → IndicatorResult
    condition    : "above_signal"  → rule fires when signal == BUY
                   "below_signal"  → rule fires when signal == SELL
                   "value_above"   → rule fires when value >  threshold
                   "value_below"   → rule fires when value <  threshold
                   "crossover"     → rule fires when value crossed threshold from below
                   "crossunder"    → rule fires when value crossed threshold from above
    threshold    : used for value_above / value_below / crossover / crossunder
    weight       : how much this rule contributes to the final score [0.0 – 1.0]
    label        : human-readable description
    """
    indicator_fn: Callable[[pd.DataFrame], IndicatorResult]
    condition:    Literal[
        "above_signal", "below_signal",
        "value_above",  "value_below",
        "crossover",    "crossunder",
    ]
    threshold: float | None = None
    weight:    float        = 1.0
    label:     str          = ""

    def evaluate(self, ohlcv: pd.DataFrame) -> tuple[bool, float, IndicatorResult]:
        """
        Returns (rule_fired, contribution_score, indicator_result).
        contribution_score is in [0.0, weight].
        """
        result = self.indicator_fn(ohlcv)
        val    = result.value
        series = result.series

        fired = False
        if self.condition == "above_signal":
            fired = result.signal == BUY
        elif self.condition == "below_signal":
            fired = result.signal == SELL
        elif self.condition == "value_above" and self.threshold is not None:
            fired = val > self.threshold
        elif self.condition == "value_below" and self.threshold is not None:
            fired = val < self.threshold
        elif self.condition == "crossover" and series is not None and self.threshold is not None:
            prev = float(series.iloc[-2]) if len(series) > 1 else val
            fired = prev < self.threshold <= val
        elif self.condition == "crossunder" and series is not None and self.threshold is not None:
            prev = float(series.iloc[-2]) if len(series) > 1 else val
            fired = prev > self.threshold >= val

        contribution = self.weight if fired else 0.0
        return fired, contribution, result


@dataclass
class CustomStrategy:
    """
    A user-defined strategy built from a list of StrategyRules.

    buy_logic  : "all"  → all buy rules must fire (AND)
                 "any"  → at least one buy rule fires (OR)
                 "score"→ score-based threshold
    sell_logic : same options for sell rules
    buy_score_threshold  : required fraction of total buy weight (for "score" mode)
    sell_score_threshold : required fraction of total sell weight (for "score" mode)
    """
    name:                 str
    description:          str
    buy_rules:            list[StrategyRule]  = field(default_factory=list)
    sell_rules:           list[StrategyRule]  = field(default_factory=list)
    buy_logic:            Literal["all","any","score"] = "any"
    sell_logic:           Literal["all","any","score"] = "any"
    buy_score_threshold:  float = 0.60
    sell_score_threshold: float = 0.60

    def evaluate(self, ohlcv: pd.DataFrame) -> dict[str, Any]:
        """
        Evaluate the strategy against latest OHLCV data.
        Returns a rich result dict suitable for the dashboard.
        """
        buy_evaluations  = [r.evaluate(ohlcv) for r in self.buy_rules]
        sell_evaluations = [r.evaluate(ohlcv) for r in self.sell_rules]

        buy_fired      = [e[0] for e in buy_evaluations]
        buy_scores     = [e[1] for e in buy_evaluations]
        buy_results    = [e[2] for e in buy_evaluations]
        sell_fired     = [e[0] for e in sell_evaluations]
        sell_scores    = [e[1] for e in sell_evaluations]
        sell_results   = [e[2] for e in sell_evaluations]

        total_buy_weight  = sum(r.weight for r in self.buy_rules)  or 1.0
        total_sell_weight = sum(r.weight for r in self.sell_rules) or 1.0
        buy_score_pct  = sum(buy_scores)  / total_buy_weight
        sell_score_pct = sum(sell_scores) / total_sell_weight

        # Determine entry signal
        if self.buy_logic == "all":
            buy_signal = all(buy_fired) if buy_fired else False
        elif self.buy_logic == "any":
            buy_signal = any(buy_fired) if buy_fired else False
        else:  # score
            buy_signal = buy_score_pct >= self.buy_score_threshold

        if self.sell_logic == "all":
            sell_signal = all(sell_fired) if sell_fired else False
        elif self.sell_logic == "any":
            sell_signal = any(sell_fired) if sell_fired else False
        else:  # score
            sell_signal = sell_score_pct >= self.sell_score_threshold

        # Priority: sell overrides if both fire
        if buy_signal and sell_signal:
            final_signal = SELL
        elif buy_signal:
            final_signal = BUY
        elif sell_signal:
            final_signal = SELL
        else:
            final_signal = NEUTRAL

        confidence = max(buy_score_pct, sell_score_pct)

        return {
            "strategy_name":    self.name,
            "signal":           final_signal,
            "confidence":       round(float(confidence), 4),
            "buy_score_pct":    round(float(buy_score_pct),  4),
            "sell_score_pct":   round(float(sell_score_pct), 4),
            "buy_rules":        [
                {"label": r.label or r.indicator_fn.__name__,
                 "fired": f, "score": s, "indicator": ir.to_dict()}
                for r, f, s, ir in zip(self.buy_rules,  buy_fired,  buy_scores,  buy_results)
            ],
            "sell_rules": [
                {"label": r.label or r.indicator_fn.__name__,
                 "fired": f, "score": s, "indicator": ir.to_dict()}
                for r, f, s, ir in zip(self.sell_rules, sell_fired, sell_scores, sell_results)
            ],
        }


# ===========================================================================
# ── PRE-BUILT EXAMPLE CUSTOM STRATEGIES ─────────────────────────────────────
# ===========================================================================

def build_rsi_macd_strategy() -> CustomStrategy:
    """Classic RSI + MACD momentum strategy."""
    return CustomStrategy(
        name="RSI + MACD Momentum",
        description=(
            "Buy when RSI exits oversold and MACD histogram is positive. "
            "Sell when RSI exits overbought or MACD crosses below signal."
        ),
        buy_rules=[
            StrategyRule(
                indicator_fn=lambda df: calc_rsi(df["close"]),
                condition="value_below", threshold=35,
                weight=1.5, label="RSI oversold (< 35)",
            ),
            StrategyRule(
                indicator_fn=lambda df: calc_macd(df["close"]),
                condition="above_signal",
                weight=1.0, label="MACD bullish crossover",
            ),
        ],
        sell_rules=[
            StrategyRule(
                indicator_fn=lambda df: calc_rsi(df["close"]),
                condition="value_above", threshold=65,
                weight=1.5, label="RSI overbought (> 65)",
            ),
            StrategyRule(
                indicator_fn=lambda df: calc_macd(df["close"]),
                condition="below_signal",
                weight=1.0, label="MACD bearish crossover",
            ),
        ],
        buy_logic="any", sell_logic="any",
    )


def build_triple_ma_strategy() -> CustomStrategy:
    """Triple EMA trend-following strategy."""
    return CustomStrategy(
        name="Triple EMA Trend",
        description=(
            "Buy when EMA10 > EMA20 > EMA50 (aligned uptrend). "
            "Sell when EMA10 < EMA20 < EMA50 (aligned downtrend)."
        ),
        buy_rules=[
            StrategyRule(
                indicator_fn=lambda df: calc_ema(df["close"], 10),
                condition="above_signal", weight=1.0, label="EMA10 buy signal",
            ),
            StrategyRule(
                indicator_fn=lambda df: calc_ema(df["close"], 20),
                condition="above_signal", weight=1.0, label="EMA20 buy signal",
            ),
            StrategyRule(
                indicator_fn=lambda df: calc_ema(df["close"], 50),
                condition="above_signal", weight=1.0, label="EMA50 buy signal",
            ),
        ],
        sell_rules=[
            StrategyRule(
                indicator_fn=lambda df: calc_ema(df["close"], 10),
                condition="below_signal", weight=1.0, label="EMA10 sell signal",
            ),
            StrategyRule(
                indicator_fn=lambda df: calc_ema(df["close"], 20),
                condition="below_signal", weight=1.0, label="EMA20 sell signal",
            ),
        ],
        buy_logic="all", sell_logic="all",
    )


def build_bollinger_rsi_strategy() -> CustomStrategy:
    """RSI + Stochastic mean-reversion strategy."""
    return CustomStrategy(
        name="Stoch RSI Reversion",
        description=(
            "Buy when both RSI and Stochastic are in oversold territory. "
            "Sell when both indicate overbought conditions."
        ),
        buy_rules=[
            StrategyRule(
                indicator_fn=lambda df: calc_rsi(df["close"]),
                condition="value_below", threshold=30,
                weight=1.0, label="RSI < 30",
            ),
            StrategyRule(
                indicator_fn=lambda df: calc_stochastic(df["high"], df["low"], df["close"]),
                condition="value_below", threshold=20,
                weight=1.0, label="Stoch %K < 20",
            ),
        ],
        sell_rules=[
            StrategyRule(
                indicator_fn=lambda df: calc_rsi(df["close"]),
                condition="value_above", threshold=70,
                weight=1.0, label="RSI > 70",
            ),
            StrategyRule(
                indicator_fn=lambda df: calc_stochastic(df["high"], df["low"], df["close"]),
                condition="value_above", threshold=80,
                weight=1.0, label="Stoch %K > 80",
            ),
        ],
        buy_logic="all", sell_logic="any",
    )


def build_adx_momentum_strategy() -> CustomStrategy:
    """ADX trend strength + Momentum strategy."""
    return CustomStrategy(
        name="ADX + Momentum Breakout",
        description=(
            "Buy when ADX confirms strong uptrend (ADX > 25) and Momentum is positive. "
            "Sell when Awesome Oscillator turns negative."
        ),
        buy_rules=[
            StrategyRule(
                indicator_fn=lambda df: calc_adx(df["high"], df["low"], df["close"]),
                condition="above_signal", weight=1.5, label="ADX bullish DI+ cross",
            ),
            StrategyRule(
                indicator_fn=lambda df: calc_momentum(df["close"]),
                condition="value_above", threshold=0,
                weight=1.0, label="Momentum > 0",
            ),
            StrategyRule(
                indicator_fn=lambda df: calc_awesome_oscillator(df["high"], df["low"]),
                condition="above_signal", weight=0.8, label="AO bullish",
            ),
        ],
        sell_rules=[
            StrategyRule(
                indicator_fn=lambda df: calc_adx(df["high"], df["low"], df["close"]),
                condition="below_signal", weight=1.5, label="ADX bearish DI- cross",
            ),
            StrategyRule(
                indicator_fn=lambda df: calc_momentum(df["close"]),
                condition="value_below", threshold=0,
                weight=1.0, label="Momentum < 0",
            ),
        ],
        buy_logic="score", sell_logic="any",
        buy_score_threshold=0.55,
    )


PRESET_STRATEGIES: list[CustomStrategy] = [
    build_rsi_macd_strategy(),
    build_triple_ma_strategy(),
    build_bollinger_rsi_strategy(),
    build_adx_momentum_strategy(),
]
