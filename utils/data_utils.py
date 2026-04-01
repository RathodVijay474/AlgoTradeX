"""
data_utils.py
=============
Data acquisition and preprocessing utilities for AlgoTradeX.

Handles:
  - Yahoo Finance download via yfinance
  - OHLCV standardisation
  - CSV upload normalisation
  - Basic data validation
"""

from __future__ import annotations

import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}


# ---------------------------------------------------------------------------
# Standardise any OHLCV frame
# ---------------------------------------------------------------------------

def standardize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise column names, ensure DatetimeIndex, and drop bad rows.
    Works with both yfinance MultiIndex output and plain CSVs.
    """
    # Flatten MultiIndex columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # Rename common variants
    renames = {
        "adj_close": "adj close",
        "adjclose":  "adj close",
        "vol":       "volume",
    }
    df = df.rename(columns=renames)

    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        date_col = next(
            (c for c in df.columns if "date" in c or "time" in c),
            None,
        )
        if date_col:
            df = df.set_index(date_col)
        df.index = pd.to_datetime(df.index, utc=False)

    df.index = df.index.tz_localize(None)
    df = df.sort_index()

    # Keep only OHLCV + adj close
    keep = [c for c in df.columns if c in REQUIRED_COLUMNS or c == "adj close"]
    df   = df[keep].copy()

    # Numeric coercion
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]

    # Fill missing volume with 0
    if "volume" not in df.columns:
        df["volume"] = 0
    df["volume"] = df["volume"].fillna(0)

    return df


# ---------------------------------------------------------------------------
# Yahoo Finance downloader
# ---------------------------------------------------------------------------

def fetch_yfinance(
    ticker: str,
    start: date,
    end: date,
    buffer_days: int = 300,
) -> pd.DataFrame:
    """
    Download OHLCV from Yahoo Finance with a lookback buffer for indicators.
    Returns a standardised DataFrame.
    """
    try:
        import yfinance as yf

        effective_start = start - timedelta(days=buffer_days)
        raw = yf.download(
            tickers=ticker.upper().strip(),
            start=str(effective_start),
            end=str(end + timedelta(days=1)),
            interval="1d",
            auto_adjust=False,
            progress=False,
            multi_level_index=False,
        )
        if raw.empty:
            raise ValueError(f"No data returned for '{ticker}'.")
        return standardize_ohlcv(raw)

    except ImportError:
        raise ImportError("yfinance not installed. Run: pip install yfinance")


def slice_date_range(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
    result = df.loc[mask].copy()
    if result.empty:
        raise ValueError("No rows in the selected date range.")
    return result


# ---------------------------------------------------------------------------
# CSV upload handler
# ---------------------------------------------------------------------------

def load_csv_upload(uploaded_file) -> pd.DataFrame:
    """Parse a user-uploaded CSV file into a standardised OHLCV frame."""
    import io
    uploaded_file.seek(0)
    raw = pd.read_csv(io.BytesIO(uploaded_file.read()))
    return standardize_ohlcv(raw)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_ohlcv(df: pd.DataFrame, min_rows: int = 60) -> None:
    """Raise ValueError if the frame is too thin for reliable analysis."""
    if len(df) < min_rows:
        raise ValueError(
            f"Need at least {min_rows} trading days; got {len(df)}. "
            "Widen the date range or choose a different ticker."
        )
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")


# ---------------------------------------------------------------------------
# Return / volatility helpers
# ---------------------------------------------------------------------------

def daily_returns(close: pd.Series) -> pd.Series:
    return close.pct_change().dropna()


def annualised_vol(close: pd.Series, window: int = 20) -> float:
    rets = daily_returns(close).tail(window)
    return float(rets.std() * np.sqrt(252)) if len(rets) >= 2 else 0.0


def sharpe_ratio(equity: pd.Series, risk_free: float = 0.0) -> float:
    rets = equity.pct_change().dropna()
    excess = rets.mean() - risk_free / 252
    vol    = rets.std()
    return float((excess / vol) * np.sqrt(252)) if vol > 1e-8 else 0.0


def sortino_ratio(equity: pd.Series, risk_free: float = 0.0) -> float:
    rets      = equity.pct_change().dropna()
    excess    = rets.mean() - risk_free / 252
    downside  = rets[rets < 0].std()
    return float((excess / downside) * np.sqrt(252)) if downside > 1e-8 else 0.0


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd   = (equity - peak) / peak.replace(0, np.nan)
    return float(dd.min())


def compute_full_metrics(equity: pd.Series, benchmark: pd.Series | None = None) -> dict:
    rets      = equity.pct_change().dropna()
    total_ret = float((equity.iloc[-1] / equity.iloc[0]) - 1)
    n_days    = len(equity)
    ann_ret   = float((1 + total_ret) ** (252 / max(n_days, 1)) - 1)
    ann_vol   = float(rets.std() * np.sqrt(252))
    sharpe    = float((rets.mean() / (rets.std() + 1e-9)) * np.sqrt(252))
    downside  = rets[rets < 0].std()
    sortino   = float((rets.mean() / (downside + 1e-9)) * np.sqrt(252))
    dd        = max_drawdown(equity)
    calmar    = float(ann_ret / abs(dd)) if dd != 0 else 0.0
    n_trades  = int((rets != 0).sum())

    metrics = {
        "Total Return":          total_ret,
        "Annualised Return":     ann_ret,
        "Annualised Volatility": ann_vol,
        "Sharpe Ratio":          sharpe,
        "Sortino Ratio":         sortino,
        "Calmar Ratio":          calmar,
        "Max Drawdown":          dd,
        "Number of Trades":      n_trades,
    }

    if benchmark is not None:
        bm_ret = float((benchmark.iloc[-1] / benchmark.iloc[0]) - 1)
        metrics["Benchmark Return"] = bm_ret
        metrics["Excess Return"]    = total_ret - bm_ret

    return metrics
