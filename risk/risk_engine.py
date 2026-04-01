"""
risk_engine.py
==============
Strict risk-first gate for AlgoTradeX V5.

This module decides whether a signal is tradable and emits a payload that can
be handed to a future C++ execution/risk process.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from models.market_intelligence import CRASH, HIGH_VOL, MarketIntelligenceResult

logger = logging.getLogger(__name__)

MAX_LOSS_PER_TRADE_FRACTION = 0.005
GLOBAL_STOP_DRAWDOWN = -0.10
RECENT_DRAWDOWN_SPIKE = -0.04
LOSS_STREAK_LIMIT = 3
DISAGREEMENT_STOP = 0.20


@dataclass
class ExecutionIntent:
    schema_version: str
    symbol: str
    action: str
    side: str
    max_quantity: float
    max_notional: float
    reference_price: float
    stop_loss: float | None
    take_profit: float | None
    risk_status: str
    freeze_trading: bool
    cancel_all_pending: bool
    flatten_all: bool
    reason_codes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "symbol": self.symbol,
            "action": self.action,
            "side": self.side,
            "max_quantity": round(self.max_quantity, 6),
            "max_notional": round(self.max_notional, 4),
            "reference_price": round(self.reference_price, 4),
            "stop_loss": round(self.stop_loss, 4) if self.stop_loss is not None else None,
            "take_profit": round(self.take_profit, 4) if self.take_profit is not None else None,
            "risk_status": self.risk_status,
            "freeze_trading": self.freeze_trading,
            "cancel_all_pending": self.cancel_all_pending,
            "flatten_all": self.flatten_all,
            "reason_codes": list(self.reason_codes),
        }


@dataclass
class RiskDecision:
    approved_signal: str
    risk_status: str
    max_loss_per_trade: float
    stop_loss_distance: float
    stop_loss_price: float | None
    take_profit_price: float | None
    position_size: float
    target_notional: float
    risk_multiplier: float
    freeze_trading: bool
    close_all_positions: bool
    cancel_all_pending_orders: bool
    reasons: list[str] = field(default_factory=list)
    execution_intent: ExecutionIntent | None = None

    @property
    def should_trade(self) -> bool:
        return not self.freeze_trading and self.approved_signal in {"Buy", "Sell"} and self.position_size > 0

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "approved_signal": self.approved_signal,
            "risk_status": self.risk_status,
            "max_loss_per_trade": round(self.max_loss_per_trade, 4),
            "stop_loss_distance": round(self.stop_loss_distance, 4),
            "position_size": round(self.position_size, 4),
            "target_notional": round(self.target_notional, 4),
            "risk_multiplier": round(self.risk_multiplier, 4),
            "freeze_trading": self.freeze_trading,
            "close_all_positions": self.close_all_positions,
            "cancel_all_pending_orders": self.cancel_all_pending_orders,
            "reasons": list(self.reasons),
            "execution_intent": self.execution_intent.to_dict() if self.execution_intent else {},
        }


def _atr(ohlcv: pd.DataFrame, window: int = 14) -> float:
    high = ohlcv["high"].astype(float)
    low = ohlcv["low"].astype(float)
    close = ohlcv["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    value = tr.rolling(window).mean().iloc[-1]
    return float(value) if pd.notna(value) else float((high - low).tail(window).mean())


def _recent_loss_streak(backtest: Any | None) -> int:
    if backtest is None or getattr(backtest, "trade_log", None) is None:
        return 0

    trade_log = backtest.trade_log
    if trade_log is None or trade_log.empty or "pnl" not in trade_log.columns:
        return 0

    streak = 0
    for pnl in reversed(trade_log["pnl"].dropna().tolist()):
        if pnl < 0:
            streak += 1
        else:
            break
    return streak


def _drawdown_signals(backtest: Any | None) -> tuple[float, float]:
    if backtest is None or getattr(backtest, "history", None) is None:
        return 0.0, 0.0

    history = backtest.history
    if history is None or history.empty or "drawdown" not in history.columns:
        return 0.0, 0.0

    latest_drawdown = float(history["drawdown"].iloc[-1])
    recent_spike = float(history["drawdown"].diff().tail(5).min()) if len(history) >= 5 else 0.0
    return latest_drawdown, recent_spike


def evaluate_risk(
    ticker: str,
    ohlcv: pd.DataFrame,
    signal: Any,
    market_context: MarketIntelligenceResult,
    capital: float,
    alpha_bundle: Any | None = None,
    backtest: Any | None = None,
) -> RiskDecision:
    """
    Apply hard capital-preservation rules before any signal is tradeable.
    """
    last_price = float(ohlcv["close"].iloc[-1])
    atr = _atr(ohlcv)
    stop_loss_distance = max(atr * 1.50, last_price * 0.01, 1e-6)
    max_loss_per_trade = capital * MAX_LOSS_PER_TRADE_FRACTION

    agreement_score = float(getattr(signal, "agreement_score", 0.0))
    alpha_agreement = float(getattr(alpha_bundle, "agreement_score", 0.0)) if alpha_bundle is not None else 0.0
    loss_streak = _recent_loss_streak(backtest)
    latest_drawdown, recent_drawdown_spike = _drawdown_signals(backtest)

    reasons: list[str] = []
    freeze_trading = False
    close_all_positions = False
    cancel_all_pending_orders = False
    risk_status = "NORMAL"

    if market_context.regime == CRASH:
        reasons.append("market_regime_crash")
    if latest_drawdown <= GLOBAL_STOP_DRAWDOWN:
        reasons.append("portfolio_drawdown_limit")
    if recent_drawdown_spike <= RECENT_DRAWDOWN_SPIKE:
        reasons.append("drawdown_spike")
    if loss_streak >= LOSS_STREAK_LIMIT:
        reasons.append("continuous_losses")
    if market_context.regime == HIGH_VOL and market_context.confidence >= 0.80:
        reasons.append("extreme_volatility")
    if max(agreement_score, alpha_agreement) <= DISAGREEMENT_STOP:
        reasons.append("model_disagreement_spike")

    if reasons:
        freeze_trading = True
        close_all_positions = True
        cancel_all_pending_orders = True
        risk_status = "EMERGENCY"
        logger.warning("Risk emergency for %s: %s", ticker, ", ".join(reasons))
    elif market_context.regime == HIGH_VOL or agreement_score < 0.45 or alpha_agreement < 0.35:
        risk_status = "WARNING"
        if market_context.regime == HIGH_VOL:
            logger.warning("Risk warning for %s due to high-volatility regime.", ticker)

    raw_position = max_loss_per_trade / stop_loss_distance
    sizing_multiplier = float(
        np.clip(
            market_context.risk_multiplier
            * max(agreement_score, 0.20)
            * max(getattr(signal, "confidence", 0.50), 0.35),
            0.0,
            1.25,
        )
    )
    position_size = raw_position * sizing_multiplier
    target_notional = position_size * last_price

    approved_signal = getattr(signal, "final_signal", "Neutral")
    if freeze_trading or approved_signal not in {"Buy", "Sell"}:
        approved_signal = "Neutral"
        position_size = 0.0
        target_notional = 0.0
    elif risk_status == "WARNING" and abs(getattr(signal, "ensemble_score", 0.0)) < 0.25:
        approved_signal = "Neutral"
        position_size = 0.0
        target_notional = 0.0
        reasons.append("warning_filter_low_conviction")

    if approved_signal == "Buy":
        stop_loss_price = last_price - stop_loss_distance
        take_profit_price = last_price + 2.0 * stop_loss_distance
        action = "OPEN_LONG"
    elif approved_signal == "Sell":
        stop_loss_price = last_price + stop_loss_distance
        take_profit_price = last_price - 2.0 * stop_loss_distance
        action = "OPEN_SHORT"
    else:
        stop_loss_price = None
        take_profit_price = None
        action = "FLATTEN_AND_FREEZE" if freeze_trading else "HOLD"

    execution_intent = ExecutionIntent(
        schema_version="atx.v5.execution",
        symbol=ticker,
        action=action,
        side=approved_signal.upper(),
        max_quantity=float(position_size),
        max_notional=float(target_notional),
        reference_price=last_price,
        stop_loss=stop_loss_price,
        take_profit=take_profit_price,
        risk_status=risk_status,
        freeze_trading=freeze_trading,
        cancel_all_pending=cancel_all_pending_orders,
        flatten_all=close_all_positions,
        reason_codes=list(dict.fromkeys(reasons)),
    )

    return RiskDecision(
        approved_signal=approved_signal,
        risk_status=risk_status,
        max_loss_per_trade=max_loss_per_trade,
        stop_loss_distance=stop_loss_distance,
        stop_loss_price=stop_loss_price,
        take_profit_price=take_profit_price,
        position_size=float(position_size),
        target_notional=float(target_notional),
        risk_multiplier=float(np.clip(sizing_multiplier, 0.0, 1.25)),
        freeze_trading=freeze_trading,
        close_all_positions=close_all_positions,
        cancel_all_pending_orders=cancel_all_pending_orders,
        reasons=list(dict.fromkeys(reasons)),
        execution_intent=execution_intent,
    )
