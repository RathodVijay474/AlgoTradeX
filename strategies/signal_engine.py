"""
signal_engine.py
================
AlgoTradeX V5 ensemble signal engine.

V4 weighted technical/sentiment/forecast logic is upgraded with:
- market-intelligence bias
- deep-learning forecast sleeve
- alpha-model agreement gating
- backward-compatible API for older callers/tests
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

BUY = "Buy"
SELL = "Sell"
NEUTRAL = "Neutral"

W_TECH = 0.30
W_SENT = 0.10
W_FORE = 0.20
W_DEEP = 0.25
W_MARKET = 0.15


def _signal_from_score(score: float, threshold: float = 0.05) -> str:
    if score >= threshold:
        return BUY
    if score <= -threshold:
        return SELL
    return NEUTRAL


@dataclass
class ComponentScore:
    name: str
    raw_score: float
    signal: str
    weight: float
    detail: str
    sub_signals: dict[str, Any] = field(default_factory=dict)

    def weighted_score(self) -> float:
        return self.raw_score * self.weight


@dataclass
class EnsembleSignal:
    ticker: str
    timestamp: str
    final_signal: str
    confidence: float
    ensemble_score: float
    components: list[ComponentScore]
    buy_count: int = 0
    sell_count: int = 0
    neutral_count: int = 0
    total_indicators: int = 0
    agreement_score: float = 0.0
    agreement_state: str = "MIXED"
    market_regime: str = "UNKNOWN"
    market_bias: int = 0
    market_bias_label: str = "Neutral"
    risk_multiplier: float = 1.0
    alpha_score: float = 0.0
    alpha_confidence: float = 0.0
    pre_agreement_score: float = 0.0
    pre_agreement_signal: str = NEUTRAL
    execution_notes: list[str] = field(default_factory=list)

    @property
    def action(self) -> str:
        return self.final_signal

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp,
            "final_signal": self.final_signal,
            "confidence": round(self.confidence, 4),
            "ensemble_score": round(self.ensemble_score, 4),
            "agreement_score": round(self.agreement_score, 4),
            "agreement_state": self.agreement_state,
            "market_regime": self.market_regime,
            "market_bias": self.market_bias,
            "risk_multiplier": round(self.risk_multiplier, 4),
            "buy_count": self.buy_count,
            "sell_count": self.sell_count,
            "neutral_count": self.neutral_count,
            "total": self.total_indicators,
        }


def _compute_technical_score(ohlcv: pd.DataFrame) -> ComponentScore:
    try:
        from strategies.algotradex_strategies import build_signal_summary, run_all_indicators

        results = run_all_indicators(ohlcv)
        summary = build_signal_summary(results)
        buy_n = summary["buy_count"]
        sell_n = summary["sell_count"]
        neu_n = summary["neutral_count"]
        total = summary["total"] or 1

        raw_score = (buy_n - sell_n) / total
        sig = BUY if summary["overall"] in ("Strong Buy", "Buy") else SELL if summary["overall"] in ("Strong Sell", "Sell") else NEUTRAL
        return ComponentScore(
            name="Technical Indicators",
            raw_score=float(raw_score),
            signal=sig,
            weight=W_TECH,
            detail=f"{buy_n}B / {sell_n}S / {neu_n}N across {total} indicators",
            sub_signals={
                "summary": summary,
                "buy_count": buy_n,
                "sell_count": sell_n,
                "neutral_count": neu_n,
                "total": total,
            },
        )
    except Exception as exc:
        return ComponentScore(
            name="Technical Indicators",
            raw_score=0.0,
            signal=NEUTRAL,
            weight=W_TECH,
            detail=f"Error: {exc}",
        )


def _compute_sentiment_score(ticker: str, news_headlines: list[str] | None = None) -> ComponentScore:
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        if not news_headlines:
            return ComponentScore(
                name="Market Sentiment",
                raw_score=0.0,
                signal=NEUTRAL,
                weight=W_SENT,
                detail=f"No headlines provided for {ticker}; sentiment neutral.",
            )

        analyzer = SentimentIntensityAnalyzer()
        scores = [analyzer.polarity_scores(text)["compound"] for text in news_headlines]
        avg_compound = float(np.mean(scores))
        return ComponentScore(
            name="Market Sentiment",
            raw_score=avg_compound,
            signal=_signal_from_score(avg_compound, threshold=0.05),
            weight=W_SENT,
            detail=f"Average VADER compound {avg_compound:.3f} across {len(news_headlines)} headlines.",
            sub_signals={"scores": scores, "headlines": news_headlines},
        )
    except Exception:
        return ComponentScore(
            name="Market Sentiment",
            raw_score=0.0,
            signal=NEUTRAL,
            weight=W_SENT,
            detail="VADER sentiment unavailable; sentiment neutral.",
        )


def _compute_forecast_score(
    close: pd.Series,
    forecast_price: float | None = None,
    forecast_result: Any | None = None,
) -> ComponentScore:
    last_price = float(close.iloc[-1])
    if forecast_result is not None:
        forecast_price = getattr(forecast_result, "ensemble_pred", forecast_price)
    if forecast_price is None:
        forecast_price = last_price

    expected_return = (forecast_price - last_price) / max(abs(last_price), 1e-8)
    raw_score = float(np.clip(expected_return * 45.0, -1.0, 1.0))
    return ComponentScore(
        name="Forecasting Model",
        raw_score=raw_score,
        signal=_signal_from_score(raw_score),
        weight=W_FORE,
        detail=f"Last={last_price:.2f} Forecast={forecast_price:.2f} Move={expected_return:+.2%}.",
        sub_signals={
            "last_price": last_price,
            "forecast_price": forecast_price,
            "expected_return": expected_return,
        },
    )


def _compute_deep_learning_score(
    close: pd.Series,
    forecast_result: Any | None = None,
) -> ComponentScore:
    last_price = float(close.iloc[-1])
    deep_pred = getattr(forecast_result, "deep_learning_pred", None) if forecast_result is not None else None
    deep_conf = float(getattr(forecast_result, "deep_learning_confidence", 0.35)) if forecast_result is not None else 0.35
    model_name = getattr(forecast_result, "deep_learning_model", "Unavailable") if forecast_result is not None else "Unavailable"

    if deep_pred is None:
        return ComponentScore(
            name="Deep Learning Model",
            raw_score=0.0,
            signal=NEUTRAL,
            weight=W_DEEP,
            detail="Deep model unavailable for this run.",
            sub_signals={"model_name": model_name, "confidence": deep_conf},
        )

    expected_return = (deep_pred - last_price) / max(abs(last_price), 1e-8)
    raw_score = float(np.clip(expected_return * 60.0 * max(deep_conf, 0.50), -1.0, 1.0))
    return ComponentScore(
        name="Deep Learning Model",
        raw_score=raw_score,
        signal=_signal_from_score(raw_score),
        weight=W_DEEP,
        detail=f"{model_name}: pred {deep_pred:.2f}, move {expected_return:+.2%}, confidence {deep_conf:.0%}.",
        sub_signals={
            "model_name": model_name,
            "confidence": deep_conf,
            "forecast_price": deep_pred,
            "expected_return": expected_return,
        },
    )


def _compute_market_intelligence_score(market_context: Any | None = None) -> ComponentScore:
    if market_context is None:
        return ComponentScore(
            name="Market Intelligence",
            raw_score=0.0,
            signal=NEUTRAL,
            weight=W_MARKET,
            detail="Market regime unavailable; neutral macro bias.",
        )

    if getattr(market_context, "regime", "") == "CRASH":
        raw_score = -1.0
    else:
        bias = float(getattr(market_context, "market_bias", 0))
        confidence = float(getattr(market_context, "confidence", 0.35))
        risk_multiplier = float(getattr(market_context, "risk_multiplier", 0.75))
        raw_score = float(np.clip(bias * confidence * max(risk_multiplier, 0.25), -1.0, 1.0))
        if getattr(market_context, "regime", "") == "HIGH_VOL":
            raw_score = min(raw_score, 0.0) - 0.10

    return ComponentScore(
        name="Market Intelligence",
        raw_score=float(np.clip(raw_score, -1.0, 1.0)),
        signal=_signal_from_score(raw_score),
        weight=W_MARKET,
        detail=(
            f"Regime={getattr(market_context, 'regime', 'UNKNOWN')} "
            f"Bias={getattr(market_context, 'market_bias_label', 'Neutral')} "
            f"Risk x{getattr(market_context, 'risk_multiplier', 1.0):.2f}"
        ),
        sub_signals={
            "regime": getattr(market_context, "regime", "UNKNOWN"),
            "market_bias": getattr(market_context, "market_bias", 0),
            "confidence": getattr(market_context, "confidence", 0.35),
            "risk_multiplier": getattr(market_context, "risk_multiplier", 1.0),
        },
    )


def _compute_alpha_component(alpha_bundle: Any | None = None) -> ComponentScore | None:
    if alpha_bundle is None:
        return None

    detail = (
        f"Composite {getattr(alpha_bundle, 'composite_score', 0.0):+.3f} "
        f"with agreement {getattr(alpha_bundle, 'agreement_score', 0.0):.0%}."
    )
    return ComponentScore(
        name="Alpha Models (Agreement Sleeve)",
        raw_score=float(getattr(alpha_bundle, "composite_score", 0.0)),
        signal=_signal_from_score(float(getattr(alpha_bundle, "composite_score", 0.0))),
        weight=0.0,
        detail=detail,
        sub_signals={
            "agreement_score": getattr(alpha_bundle, "agreement_score", 0.0),
            "selected_assets": getattr(alpha_bundle, "selected_assets", []),
        },
    )


def _compat_args(
    ticker: str | pd.DataFrame,
    ohlcv: pd.DataFrame | None,
) -> tuple[str, pd.DataFrame]:
    if isinstance(ticker, pd.DataFrame) and ohlcv is None:
        return "UNKNOWN", ticker
    if ohlcv is None:
        raise ValueError("`ohlcv` is required.")
    return str(ticker), ohlcv


def compute_ensemble_signal(
    ticker: str | pd.DataFrame,
    ohlcv: pd.DataFrame | None = None,
    news_headlines: list[str] | None = None,
    forecast_price: float | None = None,
    forecast_result: Any | None = None,
    market_context: Any | None = None,
    alpha_bundle: Any | None = None,
) -> EnsembleSignal:
    """
    Compute the V5 ensemble signal.
    Backward compatibility:
    - `compute_ensemble_signal(df)` still works for older tests/callers.
    """
    ticker, ohlcv = _compat_args(ticker, ohlcv)
    close = ohlcv["close"] if "close" in ohlcv.columns else ohlcv.iloc[:, 3]

    tech_comp = _compute_technical_score(ohlcv)
    sent_comp = _compute_sentiment_score(ticker, news_headlines)
    fore_comp = _compute_forecast_score(close, forecast_price=forecast_price, forecast_result=forecast_result)
    deep_comp = _compute_deep_learning_score(close, forecast_result=forecast_result)
    market_comp = _compute_market_intelligence_score(market_context=market_context)
    alpha_comp = _compute_alpha_component(alpha_bundle=alpha_bundle)

    components = [tech_comp, sent_comp, fore_comp, deep_comp, market_comp]
    if alpha_comp is not None:
        components.append(alpha_comp)

    base_score = sum(component.weighted_score() for component in components)
    base_score = float(np.clip(base_score, -1.0, 1.0))
    pre_signal = BUY if base_score >= 0.15 else SELL if base_score <= -0.15 else NEUTRAL

    directional_scores = [tech_comp.raw_score, sent_comp.raw_score, fore_comp.raw_score, deep_comp.raw_score, market_comp.raw_score]
    if alpha_bundle is not None:
        directional_scores.append(float(getattr(alpha_bundle, "composite_score", 0.0)))

    votes = np.array([0.0 if abs(score) < 0.10 else np.sign(score) for score in directional_scores], dtype=float)
    non_zero_votes = votes[votes != 0]
    agreement_score = float(abs(non_zero_votes.mean())) if len(non_zero_votes) else 0.0

    notes: list[str] = []
    if agreement_score >= 0.75:
        adjusted_score = float(np.clip(base_score * (1.0 + 0.15 * agreement_score), -1.0, 1.0))
        agreement_state = "STRONG"
        notes.append("Model agreement strong; conviction increased.")
    elif agreement_score <= 0.35:
        adjusted_score = float(np.clip(base_score * 0.60, -1.0, 1.0))
        agreement_state = "DISAGREE"
        notes.append("Model disagreement spike; conviction reduced.")
        logger.warning("Model disagreement spike on %s: agreement=%.2f", ticker, agreement_score)
    else:
        adjusted_score = float(np.clip(base_score * (0.85 + 0.20 * agreement_score), -1.0, 1.0))
        agreement_state = "MIXED"

    final_signal = BUY if adjusted_score >= 0.15 else SELL if adjusted_score <= -0.15 else NEUTRAL
    if agreement_score <= 0.25 and abs(adjusted_score) < 0.25:
        final_signal = NEUTRAL
        notes.append("Hold enforced because component consensus is too weak.")

    confidence = float(
        np.clip(
            0.42
            + 0.30 * abs(adjusted_score)
            + 0.28 * agreement_score,
            0.35,
            0.98,
        )
    )

    sub = tech_comp.sub_signals
    return EnsembleSignal(
        ticker=ticker,
        timestamp=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        final_signal=final_signal,
        confidence=confidence,
        ensemble_score=adjusted_score,
        components=components,
        buy_count=int(sub.get("buy_count", 0)),
        sell_count=int(sub.get("sell_count", 0)),
        neutral_count=int(sub.get("neutral_count", 0)),
        total_indicators=int(sub.get("total", 0)),
        agreement_score=agreement_score,
        agreement_state=agreement_state,
        market_regime=getattr(market_context, "regime", "UNKNOWN"),
        market_bias=int(getattr(market_context, "market_bias", 0)),
        market_bias_label=str(getattr(market_context, "market_bias_label", "Neutral")),
        risk_multiplier=float(getattr(market_context, "risk_multiplier", 1.0)),
        alpha_score=float(getattr(alpha_bundle, "composite_score", 0.0)) if alpha_bundle is not None else 0.0,
        alpha_confidence=float(getattr(alpha_bundle, "confidence", 0.0)) if alpha_bundle is not None else 0.0,
        pre_agreement_score=base_score,
        pre_agreement_signal=pre_signal,
        execution_notes=notes,
    )


def build_signal_component_df(signal: EnsembleSignal) -> pd.DataFrame:
    rows = []
    for component in signal.components:
        rows.append(
            {
                "Component": component.name,
                "Weight": f"{component.weight:.0%}",
                "Raw Score": round(component.raw_score, 3),
                "Weighted Score": round(component.weighted_score(), 3),
                "Signal": component.signal,
                "Detail": component.detail,
            }
        )
    rows.append(
        {
            "Component": "ENSEMBLE",
            "Weight": "100%",
            "Raw Score": round(signal.ensemble_score, 3),
            "Weighted Score": round(signal.ensemble_score, 3),
            "Signal": signal.final_signal,
            "Detail": f"Confidence {signal.confidence:.0%} | Agreement {signal.agreement_score:.0%} | Regime {signal.market_regime}",
        }
    )
    return pd.DataFrame(rows)
