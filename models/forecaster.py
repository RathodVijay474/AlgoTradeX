"""
forecaster.py
=============
Forecasting engine for AlgoTradeX V5.

Models
------
1. Meta Prophet          : additive trend/seasonality model
2. Google TimesFM        : optional zero-shot foundation model
3. Hybrid BiLSTM sleeve  : price + indicator + sentiment sequence model
4. Naive baseline        : linear extrapolation fallback
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


@dataclass
class ForecastResult:
    last_close: float
    prophet_pred: float | None
    timesfm_pred: float | None
    deep_learning_pred: float | None
    deep_learning_confidence: float
    deep_learning_model: str
    naive_pred: float
    ensemble_pred: float
    lower_bound: float
    upper_bound: float
    horizon_days: int
    methods_used: list[str]
    model_agreement: float
    feature_snapshot: dict[str, float] = field(default_factory=dict)

    @property
    def expected_return_pct(self) -> float:
        return (self.ensemble_pred - self.last_close) / max(abs(self.last_close), 1e-8)

    def summary(self) -> dict[str, Any]:
        return {
            "Last Close": round(self.last_close, 2),
            "Prophet": round(self.prophet_pred, 2) if self.prophet_pred is not None else "N/A",
            "TimesFM": round(self.timesfm_pred, 2) if self.timesfm_pred is not None else "N/A",
            "Deep Learning": round(self.deep_learning_pred, 2) if self.deep_learning_pred is not None else "N/A",
            "Deep Confidence": f"{self.deep_learning_confidence:.0%}",
            "Deep Model": self.deep_learning_model,
            "Naive (linear)": round(self.naive_pred, 2),
            "Ensemble Forecast": round(self.ensemble_pred, 2),
            "Lower Bound": round(self.lower_bound, 2),
            "Upper Bound": round(self.upper_bound, 2),
            "Expected Move": f"{self.expected_return_pct:.2%}",
            "Model Agreement": f"{self.model_agreement:.0%}",
            "Methods Used": ", ".join(self.methods_used),
        }


def _naive_forecast(close: pd.Series, horizon: int = 1) -> float:
    tail = close.tail(30).values
    x = np.arange(len(tail), dtype=float)
    coef = np.polyfit(x, tail, 1)
    return float(coef[0] * (len(tail) - 1 + horizon) + coef[1])


def _run_prophet(close: pd.Series, horizon: int = 1) -> float | None:
    try:
        from prophet import Prophet  # type: ignore

        df = pd.DataFrame({"ds": pd.to_datetime(close.index), "y": close.values}).dropna()
        if len(df) < 30:
            return None

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode="multiplicative",
            interval_width=0.80,
            changepoint_prior_scale=0.08,
        )
        model.fit(df, algorithm="LBFGS")
        future = model.make_future_dataframe(periods=horizon, freq="B")
        forecast = model.predict(future)
        pred_value = float(forecast["yhat"].iloc[-1])
        last = float(close.iloc[-1])
        return float(np.clip(pred_value, last * 0.85, last * 1.15))
    except Exception:
        return None


def _run_timesfm(close: pd.Series, horizon: int = 1, context_len: int = 128) -> float | None:
    """
    Best-effort TimesFM inference.
    The library surface changed across releases, so this function tries a small
    set of likely interfaces and degrades to None on failure.
    """
    try:
        import timesfm  # type: ignore
    except Exception:
        return None

    try:
        if not hasattr(_run_timesfm, "_model"):
            if hasattr(timesfm, "TimesFM_2p5_200M_torch"):
                model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                    "google/timesfm-2.5-200m-pytorch"
                )
                if hasattr(timesfm, "ForecastConfig"):
                    model.compile(
                        timesfm.ForecastConfig(
                            max_context=max(context_len, 256),
                            max_horizon=max(horizon, 4),
                            normalize_inputs=True,
                        )
                    )
                _run_timesfm._model = model
                _run_timesfm._api = "v2p5"
            else:
                hparams = timesfm.TimesFmHparams(
                    context_len=max(context_len, 128),
                    horizon_len=max(horizon, 4),
                    input_patch_len=32,
                    output_patch_len=128,
                    num_layers=20,
                    model_dims=1280,
                    backend="cpu",
                    per_core_batch_size=16,
                )
                checkpoint = timesfm.TimesFmCheckpoint(
                    huggingface_repo_id="google/timesfm-2.0-500m-pytorch"
                )
                _run_timesfm._model = timesfm.TimesFm(hparams=hparams, checkpoint=checkpoint)
                _run_timesfm._api = "legacy"

        context = close.tail(context_len).values.astype(np.float32)
        if _run_timesfm._api == "v2p5":
            point_forecast, _ = _run_timesfm._model.forecast(inputs=[context], horizon=horizon)
            pred_value = float(point_forecast[0, 0])
        else:
            point_forecast, _ = _run_timesfm._model.forecast([context], freq=[0])
            pred_value = float(point_forecast[0][0])

        last = float(close.iloc[-1])
        return float(np.clip(pred_value, last * 0.80, last * 1.20))
    except Exception as exc:
        logger.warning("TimesFM inference failed: %s", exc)
        return None


def _engineer_features(
    close: pd.Series,
    feature_frame: pd.DataFrame | None = None,
    sentiment_score: float = 0.0,
) -> pd.DataFrame:
    features = pd.DataFrame(index=close.index)
    features["close"] = close.astype(float)
    features["return_1"] = close.pct_change()
    features["momentum_5"] = close.pct_change(5)
    features["momentum_20"] = close.pct_change(20)
    features["ma_gap_5"] = close / close.rolling(5).mean() - 1.0
    features["ma_gap_20"] = close / close.rolling(20).mean() - 1.0
    features["vol_20"] = close.pct_change().rolling(20).std().fillna(0.0)
    features["sentiment"] = float(sentiment_score)

    if feature_frame is not None and not feature_frame.empty:
        numeric = feature_frame.select_dtypes(include=[np.number]).copy()
        numeric = numeric.reindex(features.index).ffill().bfill()
        for column in numeric.columns[:8]:
            series = numeric[column].astype(float)
            std = float(series.std())
            if std > 1e-8:
                features[f"feat_{column}"] = (series - float(series.mean())) / std

    return features.dropna()


def _prepare_sequences(
    feature_frame: pd.DataFrame,
    close: pd.Series,
    lookback: int = 20,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    aligned_close = close.reindex(feature_frame.index)
    future_return = aligned_close.shift(-horizon) / aligned_close - 1.0
    matrix = feature_frame.to_numpy(dtype=np.float32)
    targets = future_return.to_numpy(dtype=np.float32)

    sequences: list[np.ndarray] = []
    y_values: list[float] = []
    for idx in range(lookback, len(matrix) - horizon):
        target = targets[idx]
        if np.isnan(target):
            continue
        sequences.append(matrix[idx - lookback:idx])
        y_values.append(float(np.clip(target, -0.20, 0.20)))

    if not sequences:
        return np.empty((0, lookback, feature_frame.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return np.stack(sequences), np.asarray(y_values, dtype=np.float32)


def _fallback_sequence_model(close: pd.Series, features: pd.DataFrame) -> tuple[float, float, str]:
    last_close = float(close.iloc[-1])
    signal_strength = (
        0.45 * float(features["momentum_5"].iloc[-1])
        + 0.25 * float(features["momentum_20"].iloc[-1])
        + 0.20 * float(features["ma_gap_5"].iloc[-1])
        - 0.10 * float(features["vol_20"].iloc[-1])
        + 0.05 * float(features["sentiment"].iloc[-1])
    )
    pred_return = float(np.clip(signal_strength, -0.08, 0.08))
    pred_price = last_close * (1.0 + pred_return)
    confidence = float(
        np.clip(
            0.35
            + 0.30 * min(abs(pred_return) / 0.03, 1.0)
            + 0.20 * min(abs(float(features["momentum_20"].iloc[-1])) / 0.05, 1.0),
            0.35,
            0.75,
        )
    )
    return pred_price, confidence, "Hybrid Sequence Fallback"


def _run_hybrid_sequence_model(
    close: pd.Series,
    feature_frame: pd.DataFrame | None = None,
    sentiment_score: float = 0.0,
    horizon: int = 1,
    lookback: int = 20,
) -> tuple[float | None, float, str, dict[str, float]]:
    features = _engineer_features(close, feature_frame=feature_frame, sentiment_score=sentiment_score)
    if len(features) < max(lookback + 30, 80):
        pred, confidence, model_name = _fallback_sequence_model(close, features)
        snapshot = {col: float(features[col].iloc[-1]) for col in features.columns[:8]}
        return pred, confidence, model_name, snapshot

    X, y = _prepare_sequences(features, close.reindex(features.index), lookback=lookback, horizon=horizon)
    snapshot = {col: float(features[col].iloc[-1]) for col in features.columns[:8]}
    if len(X) < 40:
        pred, confidence, model_name = _fallback_sequence_model(close, features)
        return pred, confidence, model_name, snapshot

    try:
        import torch
        import torch.nn as nn

        torch.manual_seed(7)

        class BiLSTMRegressor(nn.Module):
            def __init__(self, input_size: int, hidden_size: int = 24) -> None:
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    batch_first=True,
                    bidirectional=True,
                    num_layers=1,
                )
                self.head = nn.Sequential(
                    nn.Linear(hidden_size * 2, 24),
                    nn.ReLU(),
                    nn.Linear(24, 1),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                output, _ = self.lstm(x)
                return self.head(output[:, -1, :]).squeeze(-1)

        device = "cpu"
        model = BiLSTMRegressor(X.shape[2]).to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.HuberLoss()

        x_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=device)

        for _ in range(12):
            optimiser.zero_grad()
            pred = model(x_tensor)
            loss = loss_fn(pred, y_tensor)
            loss.backward()
            optimiser.step()

        with torch.no_grad():
            fitted = model(x_tensor).cpu().numpy()
            rmse = float(np.sqrt(np.mean((fitted - y) ** 2)))
            pred_return = float(model(torch.tensor(X[-1:], dtype=torch.float32, device=device)).cpu().item())

        last_close = float(close.iloc[-1])
        pred_return = float(np.clip(pred_return, -0.12, 0.12))
        pred_price = float(np.clip(last_close * (1.0 + pred_return), last_close * 0.80, last_close * 1.20))
        confidence = float(
            np.clip(
                0.40
                + 0.25 * min(abs(pred_return) / 0.04, 1.0)
                + 0.30 * (1.0 / (1.0 + rmse * 80.0)),
                0.35,
                0.92,
            )
        )
        return pred_price, confidence, "BiLSTM Hybrid", snapshot
    except Exception as exc:
        logger.warning("BiLSTM forecaster fallback engaged: %s", exc)
        pred, confidence, model_name = _fallback_sequence_model(close, features)
        return pred, confidence, model_name, snapshot


def _compute_bounds(
    close: pd.Series,
    ensemble_pred: float,
    predictions: list[float],
) -> tuple[float, float, float]:
    returns = close.pct_change().dropna().tail(20)
    vol = float(returns.std()) if len(returns) >= 2 else 0.015
    dispersion = float(np.std(predictions) / max(abs(close.iloc[-1]), 1e-8)) if len(predictions) > 1 else 0.0
    model_agreement = float(np.clip(1.0 - dispersion / 0.03, 0.0, 1.0))
    width = max(vol, 0.005) * (1.0 + (1.0 - model_agreement))
    return float(ensemble_pred * (1 - width)), float(ensemble_pred * (1 + width)), model_agreement


def run_forecast(
    close: pd.Series,
    horizon: int = 1,
    use_prophet: bool = True,
    use_timesfm: bool = True,
    context_len: int = 128,
    feature_frame: pd.DataFrame | None = None,
    sentiment_score: float = 0.0,
    use_deep_learning: bool = True,
) -> ForecastResult:
    """
    Generate a multi-model forecast with an optional deep-learning sleeve.
    """
    close = close.dropna().astype(float)
    last_close = float(close.iloc[-1])
    naive_pred = _naive_forecast(close, horizon)
    methods_used = ["Naive (linear)"]

    prophet_pred: float | None = None
    timesfm_pred: float | None = None
    deep_learning_pred: float | None = None
    deep_learning_confidence = 0.35
    deep_learning_model = "Disabled"
    feature_snapshot: dict[str, float] = {}

    if use_prophet:
        prophet_pred = _run_prophet(close, horizon)
        if prophet_pred is not None:
            methods_used.append("Prophet")

    if use_timesfm:
        timesfm_pred = _run_timesfm(close, horizon=horizon, context_len=context_len)
        if timesfm_pred is not None:
            methods_used.append("TimesFM")

    if use_deep_learning:
        deep_learning_pred, deep_learning_confidence, deep_learning_model, feature_snapshot = _run_hybrid_sequence_model(
            close,
            feature_frame=feature_frame,
            sentiment_score=sentiment_score,
            horizon=horizon,
        )
        if deep_learning_pred is not None:
            methods_used.append(deep_learning_model)

    predictions = [naive_pred]
    weighted_preds = [(naive_pred, 0.20)]

    if prophet_pred is not None:
        predictions.append(prophet_pred)
        weighted_preds.append((prophet_pred, 0.22))
    if timesfm_pred is not None:
        predictions.append(timesfm_pred)
        weighted_preds.append((timesfm_pred, 0.23))
    if deep_learning_pred is not None:
        predictions.append(deep_learning_pred)
        weighted_preds.append((deep_learning_pred, 0.35 * max(deep_learning_confidence, 0.50)))

    total_weight = sum(weight for _, weight in weighted_preds)
    ensemble_pred = float(sum(pred * weight for pred, weight in weighted_preds) / max(total_weight, 1e-8))
    lower_bound, upper_bound, model_agreement = _compute_bounds(close, ensemble_pred, predictions)

    return ForecastResult(
        last_close=last_close,
        prophet_pred=prophet_pred,
        timesfm_pred=timesfm_pred,
        deep_learning_pred=deep_learning_pred,
        deep_learning_confidence=deep_learning_confidence,
        deep_learning_model=deep_learning_model,
        naive_pred=naive_pred,
        ensemble_pred=ensemble_pred,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        horizon_days=horizon,
        methods_used=methods_used,
        model_agreement=model_agreement,
        feature_snapshot=feature_snapshot,
    )
