import numpy as np
import pandas as pd

from models.forecaster import run_forecast
from models.market_intelligence import CRASH, TRENDING, analyze_market_intelligence
from risk.risk_engine import evaluate_risk
from strategies.signal_engine import BUY, NEUTRAL, SELL, compute_ensemble_signal
from utils.data_utils import standardize_ohlcv, validate_ohlcv


def _make_ohlcv(rows: int = 140, slope: float = 0.35, shock: float = 0.0) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, freq="B")
    base = 100 + np.linspace(0, slope * rows, rows)
    wave = 1.8 * np.sin(np.linspace(0, 8, rows))
    close = base + wave
    if shock:
        close[-12:] = close[-12:] + np.linspace(0, shock, 12)
    close = np.maximum(close, 1.0)

    open_ = close * (1 - 0.002)
    high = close * 1.01
    low = close * 0.99
    volume = 1_000_000 + np.linspace(0, 200_000, rows)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )


def _make_market_frames(regime: str = TRENDING) -> dict[str, pd.DataFrame]:
    symbols = ["SPY", "QQQ", "IWM", "XLF", "XLK", "XLE", "XLV"]
    frames: dict[str, pd.DataFrame] = {}
    for idx, symbol in enumerate(symbols):
        if regime == CRASH:
            frames[symbol] = _make_ohlcv(slope=-0.45 - idx * 0.03, shock=-12.0 - idx)
        else:
            frames[symbol] = _make_ohlcv(slope=0.30 + idx * 0.03, shock=4.0 + idx * 0.3)
    return frames


def test_standardize_and_validate_ohlcv():
    raw = pd.DataFrame(
        {
            "Open": np.linspace(100, 120, 80),
            "High": np.linspace(101, 121, 80),
            "Low": np.linspace(99, 119, 80),
            "Close": np.linspace(100.5, 120.5, 80),
            "Volume": np.linspace(1_000, 2_000, 80),
        },
        index=pd.date_range("2024-01-01", periods=80, freq="B"),
    )
    standardized = standardize_ohlcv(raw)
    validate_ohlcv(standardized, min_rows=60)
    assert {"open", "high", "low", "close", "volume"}.issubset(standardized.columns)


def test_forecast_returns_v5_metadata():
    df = _make_ohlcv()
    forecast = run_forecast(
        df["close"],
        use_prophet=False,
        use_timesfm=False,
        feature_frame=df,
        use_deep_learning=True,
    )
    assert forecast.ensemble_pred > 0
    assert "Naive (linear)" in forecast.methods_used
    assert 0.0 <= forecast.model_agreement <= 1.0
    assert forecast.deep_learning_model


def test_market_intelligence_detects_crash_regime():
    market_context = analyze_market_intelligence(
        market_frames=_make_market_frames(regime=CRASH),
        reference_ohlcv=_make_ohlcv(slope=-0.40, shock=-10.0),
    )
    assert market_context.regime == CRASH
    assert market_context.market_bias == -1
    assert market_context.risk_multiplier <= 0.30


def test_signal_engine_backward_compatibility_and_agreement():
    df = _make_ohlcv()
    result = compute_ensemble_signal(df)
    assert result.action in [BUY, SELL, NEUTRAL]
    assert 0.0 <= result.confidence <= 1.0
    assert 0.0 <= result.agreement_score <= 1.0


def test_risk_engine_triggers_global_stop_in_crash():
    ohlcv = _make_ohlcv(slope=-0.25, shock=-9.0)
    market_context = analyze_market_intelligence(
        market_frames=_make_market_frames(regime=CRASH),
        reference_ohlcv=ohlcv,
    )
    signal = compute_ensemble_signal(
        "TEST",
        ohlcv=ohlcv,
        forecast_price=float(ohlcv["close"].iloc[-1] * 1.02),
        market_context=market_context,
    )
    risk = evaluate_risk(
        ticker="TEST",
        ohlcv=ohlcv,
        signal=signal,
        market_context=market_context,
        capital=100_000.0,
    )
    assert risk.risk_status == "EMERGENCY"
    assert risk.approved_signal == NEUTRAL
    assert risk.freeze_trading is True
    assert risk.close_all_positions is True
