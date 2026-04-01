"""
alpha_models.py
===============
Auxiliary alpha sleeves for AlgoTradeX V5.

The signals here are used as a market-aware agreement layer rather than
replacing the existing technical indicator stack.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from models.market_intelligence import MarketIntelligenceResult

logger = logging.getLogger(__name__)


@dataclass
class AlphaModelSignal:
    name: str
    raw_score: float
    confidence: float
    direction: int
    detail: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "raw_score": round(self.raw_score, 4),
            "confidence": round(self.confidence, 4),
            "direction": self.direction,
            "detail": self.detail,
        }


@dataclass
class AlphaModelBundle:
    target_ticker: str
    models: list[AlphaModelSignal]
    composite_score: float
    confidence: float
    agreement_score: float
    ranking_table: pd.DataFrame
    selected_assets: list[str]

    @property
    def direction(self) -> int:
        if self.composite_score > 0.10:
            return 1
        if self.composite_score < -0.10:
            return -1
        return 0

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "target_ticker": self.target_ticker,
            "composite_score": round(self.composite_score, 4),
            "confidence": round(self.confidence, 4),
            "agreement_score": round(self.agreement_score, 4),
            "selected_assets": list(self.selected_assets),
        }


def _close_panel(market_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.concat({symbol: frame["close"] for symbol, frame in market_frames.items()}, axis=1).sort_index().ffill().dropna(how="all")


def cross_sectional_momentum_model(
    target_ticker: str,
    market_frames: dict[str, pd.DataFrame],
    lookback: int = 20,
) -> tuple[AlphaModelSignal, pd.DataFrame, list[str]]:
    panel = _close_panel(market_frames).dropna(axis=1, how="all")
    panel = panel.dropna()

    if panel.empty or target_ticker not in panel.columns or len(panel) <= lookback:
        signal = AlphaModelSignal(
            name="Cross-Sectional Momentum",
            raw_score=0.0,
            confidence=0.35,
            direction=0,
            detail="Insufficient market cross-section for momentum ranking.",
        )
        return signal, pd.DataFrame(columns=["ticker", "return", "rank_pct"]), []

    returns = (panel.iloc[-1] / panel.iloc[-(lookback + 1)] - 1.0).dropna().sort_values(ascending=False)
    rank_pct = returns.rank(pct=True, ascending=True)
    target_rank = float(rank_pct.get(target_ticker, 0.5))
    raw_score = float(np.clip((target_rank - 0.5) * 2.0, -1.0, 1.0))
    direction = 1 if target_rank >= 0.90 else -1 if target_rank <= 0.10 else 0
    confidence = float(np.clip(0.45 + 0.50 * abs(raw_score), 0.35, 0.95))

    ranking_table = (
        pd.DataFrame({"ticker": returns.index, "return": returns.values})
        .assign(rank_pct=lambda df: df["return"].rank(pct=True, ascending=True))
        .sort_values("return", ascending=False)
        .reset_index(drop=True)
    )
    selected_assets = ranking_table[ranking_table["rank_pct"] >= 0.90]["ticker"].tolist()

    signal = AlphaModelSignal(
        name="Cross-Sectional Momentum",
        raw_score=raw_score,
        confidence=confidence,
        direction=direction,
        detail=f"{target_ticker} momentum rank {target_rank:.0%}; top-decile basket size {len(selected_assets)}.",
        metadata={
            "rank_pct": target_rank,
            "selected_assets": selected_assets,
        },
    )
    return signal, ranking_table, selected_assets


def volatility_regime_model(
    ohlcv: pd.DataFrame,
    market_context: MarketIntelligenceResult | None = None,
) -> AlphaModelSignal:
    close = ohlcv["close"]
    returns = close.pct_change().dropna()
    if len(returns) < 30:
        return AlphaModelSignal(
            name="Volatility Regime",
            raw_score=0.0,
            confidence=0.35,
            direction=0,
            detail="Insufficient history for volatility-regime classification.",
        )

    current_vol = float(returns.tail(20).std() * np.sqrt(252))
    long_vol = float(returns.tail(60).std() * np.sqrt(252)) if len(returns) >= 60 else current_vol
    vol_ratio = current_vol / max(long_vol, 1e-8)
    regime_adjustment = 0.0
    if market_context is not None and market_context.regime in {"HIGH_VOL", "CRASH"}:
        regime_adjustment = -0.20
    elif market_context is not None and market_context.regime == "TRENDING":
        regime_adjustment = 0.10

    raw_score = float(np.clip((1.0 - vol_ratio) * 1.35 + regime_adjustment, -1.0, 1.0))
    direction = 1 if raw_score >= 0.15 else -1 if raw_score <= -0.15 else 0
    confidence = float(np.clip(0.40 + 0.45 * abs(raw_score), 0.35, 0.95))

    posture = "aggressive" if direction > 0 else "defensive" if direction < 0 else "balanced"
    return AlphaModelSignal(
        name="Volatility Regime",
        raw_score=raw_score,
        confidence=confidence,
        direction=direction,
        detail=f"Current vol {current_vol:.2%}, long-run vol {long_vol:.2%}; posture {posture}.",
        metadata={
            "current_vol": current_vol,
            "long_vol": long_vol,
            "vol_ratio": vol_ratio,
            "posture": posture,
        },
    )


def liquidity_volume_spike_model(ohlcv: pd.DataFrame) -> AlphaModelSignal:
    volume = ohlcv["volume"].astype(float)
    close = ohlcv["close"].astype(float)
    if len(volume) < 25:
        return AlphaModelSignal(
            name="Liquidity / Volume Spike",
            raw_score=0.0,
            confidence=0.35,
            direction=0,
            detail="Insufficient volume history for spike detection.",
        )

    volume_mean = volume.rolling(20).mean().iloc[-1]
    volume_std = volume.rolling(20).std().iloc[-1]
    volume_z = float((volume.iloc[-1] - volume_mean) / max(volume_std, 1e-8))
    price_impulse = float(close.pct_change().tail(3).sum())
    impulse_sign = np.sign(price_impulse)
    spike_strength = np.tanh(volume_z / 3.0)
    raw_score = float(np.clip(spike_strength * impulse_sign * max(min(abs(price_impulse) * 18.0, 1.0), 0.15 if abs(volume_z) >= 1.0 else 0.0), -1.0, 1.0))
    direction = 1 if raw_score >= 0.12 else -1 if raw_score <= -0.12 else 0
    confidence = float(np.clip(0.35 + 0.25 * min(abs(volume_z) / 2.5, 1.0) + 0.35 * min(abs(price_impulse) / 0.04, 1.0), 0.35, 0.95))

    return AlphaModelSignal(
        name="Liquidity / Volume Spike",
        raw_score=raw_score,
        confidence=confidence,
        direction=direction,
        detail=f"Volume z-score {volume_z:.2f}; 3-bar price impulse {price_impulse:+.2%}.",
        metadata={
            "volume_z": volume_z,
            "price_impulse": price_impulse,
        },
    )


def run_alpha_models(
    target_ticker: str,
    target_ohlcv: pd.DataFrame,
    market_frames: dict[str, pd.DataFrame] | None = None,
    market_context: MarketIntelligenceResult | None = None,
) -> AlphaModelBundle:
    """
    Execute all V5 alpha sleeves and return a composite agreement bundle.
    """
    market_frames = dict(market_frames or {})
    market_frames.setdefault(target_ticker, target_ohlcv)

    momentum_signal, ranking_table, selected_assets = cross_sectional_momentum_model(
        target_ticker=target_ticker,
        market_frames=market_frames,
    )
    vol_signal = volatility_regime_model(target_ohlcv, market_context=market_context)
    liquidity_signal = liquidity_volume_spike_model(target_ohlcv)

    models = [momentum_signal, vol_signal, liquidity_signal]
    weights = np.array([0.45, 0.25, 0.30], dtype=float)
    raw_scores = np.array([model.raw_score for model in models], dtype=float)
    confidences = np.array([model.confidence for model in models], dtype=float)

    composite_score = float(np.clip(np.dot(raw_scores, weights), -1.0, 1.0))
    confidence = float(np.clip(np.dot(confidences, weights), 0.35, 0.95))
    directions = np.array([0 if abs(score) < 0.10 else np.sign(score) for score in raw_scores], dtype=float)
    non_zero = directions[directions != 0]
    agreement_score = float(abs(non_zero.mean())) if len(non_zero) else 0.0

    if agreement_score < 0.35:
        logger.warning(
            "Alpha-model disagreement spike on %s: score=%.2f",
            target_ticker,
            agreement_score,
        )

    return AlphaModelBundle(
        target_ticker=target_ticker,
        models=models,
        composite_score=composite_score,
        confidence=confidence,
        agreement_score=agreement_score,
        ranking_table=ranking_table,
        selected_assets=selected_assets,
    )


def build_alpha_model_df(bundle: AlphaModelBundle) -> pd.DataFrame:
    rows = [model.to_row() for model in bundle.models]
    rows.append(
        {
            "model": "Alpha Composite",
            "raw_score": round(bundle.composite_score, 4),
            "confidence": round(bundle.confidence, 4),
            "direction": bundle.direction,
            "detail": f"Agreement {bundle.agreement_score:.0%}; leaders: {', '.join(bundle.selected_assets[:3]) or 'n/a'}",
        }
    )
    return pd.DataFrame(rows)
