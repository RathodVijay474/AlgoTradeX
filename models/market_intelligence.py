"""
market_intelligence.py
======================
Market-wide context engine for AlgoTradeX V5.

Responsibilities
----------------
- Inspect broad market and sector ETFs, not just the active ticker.
- Detect the dominant regime and breadth conditions.
- Estimate volatility clusters and cross-asset correlation stress.
- Produce a risk multiplier that downstream execution can trust.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRENDING = "TRENDING"
MEAN_REVERTING = "MEAN_REVERTING"
HIGH_VOL = "HIGH_VOL"
CRASH = "CRASH"

DEFAULT_MARKET_UNIVERSE: dict[str, str] = {
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100",
    "DIA": "Dow Jones",
    "IWM": "Russell 2000",
    "TLT": "Treasuries",
    "GLD": "Gold",
    "XLF": "Financials",
    "XLK": "Technology",
    "XLE": "Energy",
    "XLV": "Healthcare",
    "XLY": "Consumer Discretionary",
    "XLI": "Industrials",
    "XLP": "Consumer Staples",
}

SECTOR_TICKERS = {
    symbol: label
    for symbol, label in DEFAULT_MARKET_UNIVERSE.items()
    if symbol.startswith("XL")
}


@dataclass
class MarketIntelligenceResult:
    regime: str
    market_bias: int
    confidence: float
    risk_multiplier: float
    breadth_ratio: float
    realized_vol: float
    volatility_cluster: str
    correlation_risk: float
    sector_strength: pd.DataFrame
    cross_asset_correlation: pd.DataFrame
    breadth_metrics: dict[str, float]
    asset_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    notes: list[str] = field(default_factory=list)

    @property
    def market_bias_label(self) -> str:
        return {1: "Bullish", 0: "Neutral", -1: "Bearish"}.get(self.market_bias, "Neutral")

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "regime": self.regime,
            "market_bias": self.market_bias,
            "market_bias_label": self.market_bias_label,
            "confidence": round(self.confidence, 4),
            "risk_multiplier": round(self.risk_multiplier, 4),
            "breadth_ratio": round(self.breadth_ratio, 4),
            "realized_vol": round(self.realized_vol, 4),
            "volatility_cluster": self.volatility_cluster,
            "correlation_risk": round(self.correlation_risk, 4),
            "notes": list(self.notes),
        }


def _standardize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    rename_map = {"adj_close": "adj close", "adjclose": "adj close", "vol": "volume"}
    df = df.rename(columns=rename_map)

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index()

    for col in ("open", "high", "low", "close", "volume"):
        if col not in df.columns:
            if col == "volume":
                df[col] = 0.0
            else:
                raise ValueError(f"Missing column '{col}' in market frame.")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=["open", "high", "low", "close"])


def fetch_market_universe(
    start: date,
    end: date,
    tickers: list[str] | None = None,
    buffer_days: int = 180,
) -> dict[str, pd.DataFrame]:
    """
    Fetch a broad market reference universe used by the regime engine.
    Falls back silently to an empty dict if yfinance is unavailable.
    """
    symbols = tickers or list(DEFAULT_MARKET_UNIVERSE)
    effective_start = start - timedelta(days=buffer_days)

    try:
        import yfinance as yf

        raw = yf.download(
            tickers=symbols,
            start=str(effective_start),
            end=str(end + timedelta(days=1)),
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=False,
        )
    except Exception as exc:
        logger.warning("Market universe download failed: %s", exc)
        return {}

    frames: dict[str, pd.DataFrame] = {}
    if raw.empty:
        return frames

    if isinstance(raw.columns, pd.MultiIndex):
        for symbol in symbols:
            if symbol not in raw.columns.get_level_values(0):
                continue
            try:
                frame = _standardize_ohlcv(raw[symbol])
            except Exception:
                continue
            if not frame.empty:
                frames[symbol] = frame
    else:
        try:
            frame = _standardize_ohlcv(raw)
        except Exception:
            frame = pd.DataFrame()
        if not frame.empty and symbols:
            frames[symbols[0]] = frame

    return frames


def _fallback_market_frames(reference_ohlcv: pd.DataFrame | None) -> dict[str, pd.DataFrame]:
    if reference_ohlcv is None or reference_ohlcv.empty:
        return {}

    base = _standardize_ohlcv(reference_ohlcv)
    synthetic_frames: dict[str, pd.DataFrame] = {}
    phase = np.linspace(0.0, np.pi * 2.0, len(base))

    for idx, symbol in enumerate(DEFAULT_MARKET_UNIVERSE):
        frame = base.copy()
        drift = 1.0 + (idx - len(DEFAULT_MARKET_UNIVERSE) / 2) * 0.002
        wave = 1.0 + 0.015 * np.sin(phase + idx / 3)
        for col in ("open", "high", "low", "close"):
            frame[col] = frame[col].to_numpy(dtype=float) * drift * wave
        frame["volume"] = frame["volume"].to_numpy(dtype=float) * (1.0 + 0.10 * np.cos(phase + idx / 5))
        synthetic_frames[symbol] = frame

    logger.warning("Using synthetic market-intelligence universe fallback.")
    return synthetic_frames


def _atr_percent(df: pd.DataFrame, window: int = 14) -> float:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(window).mean().iloc[-1]
    last_close = max(float(close.iloc[-1]), 1e-8)
    return float(atr / last_close) if pd.notna(atr) else 0.0


def _pairwise_mean_abs_corr(corr_matrix: pd.DataFrame) -> float:
    if corr_matrix.empty:
        return 0.0
    mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    values = corr_matrix.where(mask).stack().abs()
    return float(values.mean()) if not values.empty else 0.0


def _compute_sector_strength(
    frames: dict[str, pd.DataFrame],
    benchmark_returns: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for symbol, sector_name in SECTOR_TICKERS.items():
        frame = frames.get(symbol)
        if frame is None or frame.empty or len(frame) < 65:
            continue

        close = frame["close"]
        volume = frame["volume"]
        ret_20 = float(close.iloc[-1] / close.iloc[-21] - 1.0) if len(close) > 20 else 0.0
        rel_strength = ret_20 - benchmark_returns
        vol_thrust = float(volume.tail(10).mean() / max(volume.tail(60).mean(), 1.0) - 1.0)
        score = float(np.clip(0.55 * rel_strength + 0.30 * ret_20 + 0.15 * vol_thrust, -1.0, 1.0))
        rows.append(
            {
                "sector": sector_name,
                "ticker": symbol,
                "return_20d": ret_20,
                "relative_strength": rel_strength,
                "volume_thrust": vol_thrust,
                "score": score,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["sector", "ticker", "return_20d", "relative_strength", "volume_thrust", "score"])

    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def analyze_market_intelligence(
    start: date | None = None,
    end: date | None = None,
    market_frames: dict[str, pd.DataFrame] | None = None,
    reference_ohlcv: pd.DataFrame | None = None,
    benchmark_ticker: str = "SPY",
) -> MarketIntelligenceResult:
    """
    Analyze market-wide context and return a regime/risk summary.
    """
    if market_frames is None:
        if start is not None and end is not None:
            market_frames = fetch_market_universe(start=start, end=end)
        else:
            market_frames = {}

    if not market_frames:
        market_frames = _fallback_market_frames(reference_ohlcv)

    if not market_frames:
        empty_frame = pd.DataFrame(columns=["sector", "ticker", "return_20d", "relative_strength", "volume_thrust", "score"])
        return MarketIntelligenceResult(
            regime=MEAN_REVERTING,
            market_bias=0,
            confidence=0.35,
            risk_multiplier=0.75,
            breadth_ratio=0.50,
            realized_vol=0.0,
            volatility_cluster="UNKNOWN",
            correlation_risk=0.0,
            sector_strength=empty_frame,
            cross_asset_correlation=pd.DataFrame(),
            breadth_metrics={"above_50dma": 0.5, "positive_20d": 0.5},
            notes=["Market universe unavailable; fallback neutral regime applied."],
        )

    standardized = {symbol: _standardize_ohlcv(frame) for symbol, frame in market_frames.items() if frame is not None and not frame.empty}
    close_panel = pd.concat({symbol: frame["close"] for symbol, frame in standardized.items()}, axis=1).sort_index().ffill().dropna(how="all")
    volume_panel = pd.concat({symbol: frame["volume"] for symbol, frame in standardized.items()}, axis=1).sort_index().ffill().fillna(0.0)

    if benchmark_ticker not in close_panel.columns:
        benchmark_ticker = close_panel.columns[0]

    close_panel = close_panel.dropna(axis=1, how="all")
    if close_panel.shape[0] < 65:
        raise ValueError("Need at least 65 rows across the market universe for regime analysis.")

    aligned_close = close_panel.dropna()
    aligned_volume = volume_panel.reindex(aligned_close.index).fillna(0.0)
    returns = aligned_close.pct_change().dropna()
    benchmark_close = aligned_close[benchmark_ticker]
    benchmark_returns = returns[benchmark_ticker]

    trailing_20 = float(benchmark_close.iloc[-1] / benchmark_close.iloc[-21] - 1.0) if len(benchmark_close) > 20 else 0.0
    trailing_60 = float(benchmark_close.iloc[-1] / benchmark_close.iloc[-61] - 1.0) if len(benchmark_close) > 60 else trailing_20
    realized_vol = float(benchmark_returns.tail(20).std() * np.sqrt(252)) if len(benchmark_returns) >= 20 else 0.0
    drawdown = float(benchmark_close.iloc[-1] / benchmark_close.cummax().iloc[-1] - 1.0)

    above_50dma = (aligned_close.iloc[-1] > aligned_close.tail(50).mean()).mean()
    positive_20d = (aligned_close.iloc[-1] > aligned_close.shift(20).iloc[-1]).mean()
    breadth_ratio = float(np.clip(0.55 * above_50dma + 0.45 * positive_20d, 0.0, 1.0))

    atr_values = [_atr_percent(frame.tail(90)) for frame in standardized.values() if len(frame) >= 30]
    atr_proxy = float(np.mean(atr_values)) if atr_values else realized_vol / np.sqrt(252)
    vol_history = benchmark_returns.rolling(20).std().dropna() * np.sqrt(252)
    vol_z = 0.0
    if len(vol_history) >= 20 and vol_history.std() > 1e-8:
        vol_z = float((vol_history.iloc[-1] - vol_history.mean()) / vol_history.std())

    if vol_z >= 1.25 or atr_proxy >= 0.035:
        volatility_cluster = "ELEVATED"
    elif vol_z <= -0.50 and atr_proxy <= 0.02:
        volatility_cluster = "CALM"
    else:
        volatility_cluster = "NORMAL"

    corr_matrix = returns.tail(20).corr().fillna(0.0)
    correlation_risk = _pairwise_mean_abs_corr(corr_matrix)

    crash_condition = drawdown <= -0.10 or (trailing_20 <= -0.06 and breadth_ratio <= 0.35 and realized_vol >= 0.28)
    high_vol_condition = realized_vol >= 0.30 or vol_z >= 1.50 or (correlation_risk >= 0.70 and breadth_ratio < 0.45)
    trending_condition = abs(trailing_60) >= 0.05 and breadth_ratio >= 0.55 and realized_vol < 0.28

    notes: list[str] = []
    if crash_condition:
        regime = CRASH
        market_bias = -1
        risk_multiplier = 0.25
        notes.append("Crash regime detected: capital preservation mode engaged.")
        logger.warning("Market regime shift detected: CRASH")
    elif high_vol_condition:
        regime = HIGH_VOL
        market_bias = -1 if trailing_20 < 0 else 0
        risk_multiplier = 0.55
        notes.append("High-volatility cluster detected: reduce gross exposure.")
        logger.warning("Market regime shift detected: HIGH_VOL")
    elif trending_condition:
        regime = TRENDING
        market_bias = 1 if trailing_60 >= 0 else -1
        risk_multiplier = 1.05 if market_bias > 0 and volatility_cluster == "CALM" else 0.90
        notes.append("Trending regime detected: directional trades favoured.")
        logger.info("Market regime shift detected: TRENDING")
    else:
        regime = MEAN_REVERTING
        market_bias = 0 if abs(trailing_20) < 0.02 else (1 if trailing_20 > 0 else -1)
        risk_multiplier = 0.80
        notes.append("Mean-reverting tape detected: fade conviction and size conservatively.")
        logger.info("Market regime shift detected: MEAN_REVERTING")

    confidence = float(
        np.clip(
            0.35
            + 0.25 * min(abs(trailing_60) / 0.08, 1.0)
            + 0.20 * min(abs(breadth_ratio - 0.5) / 0.5, 1.0)
            + 0.20 * min(abs(vol_z) / 2.0, 1.0)
            + 0.10 * min(abs(correlation_risk - 0.5) / 0.5, 1.0),
            0.35,
            0.98,
        )
    )

    sector_strength = _compute_sector_strength(standardized, benchmark_returns=trailing_20)
    if not sector_strength.empty:
        leaders = ", ".join(sector_strength.head(3)["sector"].tolist())
        laggards = ", ".join(sector_strength.tail(2)["sector"].tolist())
        notes.append(f"Sector leadership: {leaders}. Relative laggards: {laggards}.")

    return MarketIntelligenceResult(
        regime=regime,
        market_bias=market_bias,
        confidence=confidence,
        risk_multiplier=float(np.clip(risk_multiplier, 0.20, 1.15)),
        breadth_ratio=breadth_ratio,
        realized_vol=realized_vol,
        volatility_cluster=volatility_cluster,
        correlation_risk=correlation_risk,
        sector_strength=sector_strength,
        cross_asset_correlation=corr_matrix.round(3),
        breadth_metrics={
            "above_50dma": float(above_50dma),
            "positive_20d": float(positive_20d),
            "atr_proxy": float(atr_proxy),
            "drawdown": float(drawdown),
        },
        asset_returns=(aligned_close.iloc[-1] / aligned_close.iloc[-21] - 1.0).dropna() if len(aligned_close) > 20 else pd.Series(dtype=float),
        notes=notes,
    )
