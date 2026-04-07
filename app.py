"""
AlgoTradeX V5 Streamlit application.

This entrypoint keeps the existing dashboard workflow but upgrades the runtime
pipeline to:
- market intelligence
- multi-model forecasting with deep learning
- alpha model sleeves
- strict pre-trade risk gating
- theme switching with System / Dark / White modes
"""

from __future__ import annotations

import logging
import os
import re
import sys
import warnings
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import psutil
import streamlit as st
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from dotenv import load_dotenv

from models.alpha_models import AlphaModelBundle, build_alpha_model_df, run_alpha_models
from models.forecaster import ForecastResult, run_forecast
from models.market_intelligence import MarketIntelligenceResult, analyze_market_intelligence, fetch_market_universe
from risk.risk_engine import RiskDecision, evaluate_risk
from strategies.signal_engine import BUY, NEUTRAL, SELL, EnsembleSignal, build_signal_component_df, compute_ensemble_signal
from utils.backtesting_engine import STRATEGY_REGISTRY, list_strategies, run_named_strategy
from utils.data_utils import fetch_yfinance, load_csv_upload, slice_date_range, validate_ohlcv

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

load_dotenv()


def _init_optional_huggingface() -> None:
    token = os.getenv("hugging_face_write")
    if not token:
        return
    try:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)
    except Exception as exc:
        logger.warning("Optional Hugging Face login failed: %s", exc)


_init_optional_huggingface()


DARK_THEME = {
    "BG": "#080d18",
    "SURFACE": "#0f1a2b",
    "SURFACE_ALT": "#09101c",
    "SIDEBAR_BG": "#090f1c",
    "BORDER": "#1a2d45",
    "TEXT": "#dde6f0",
    "MUTED": "#5a7a9a",
    "ACCENT": "#22d3ee",
    "BULL": "#10b981",
    "SELL_C": "#f43f5e",
    "GOLD": "#f59e0b",
    "PURPLE": "#a78bfa",
    "BUTTON_TEXT": "#081018",
    "ALERT_BG": "rgba(34,211,238,0.05)",
    "GRID": "rgba(255,255,255,0.05)",
    "HERO_A": "rgba(9,15,28,0.98)",
    "HERO_B": "rgba(15,26,43,0.96)",
    "GLOW_A": "rgba(34,211,238,0.07)",
    "GLOW_B": "rgba(167,139,250,0.05)",
    "GLOW_C": "rgba(245,158,11,0.04)",
}

LIGHT_THEME = {
    "BG": "#f5f7fb",
    "SURFACE": "#ffffff",
    "SURFACE_ALT": "#eef3fb",
    "SIDEBAR_BG": "#edf2fa",
    "BORDER": "#d6deec",
    "TEXT": "#0f172a",
    "MUTED": "#607189",
    "ACCENT": "#0f766e",
    "BULL": "#16a34a",
    "SELL_C": "#dc2626",
    "GOLD": "#ca8a04",
    "PURPLE": "#7c3aed",
    "BUTTON_TEXT": "#ffffff",
    "ALERT_BG": "rgba(15,118,110,0.06)",
    "GRID": "rgba(15,23,42,0.08)",
    "HERO_A": "rgba(255,255,255,0.98)",
    "HERO_B": "rgba(237,242,250,0.96)",
    "GLOW_A": "rgba(15,118,110,0.08)",
    "GLOW_B": "rgba(124,58,237,0.05)",
    "GLOW_C": "rgba(202,138,4,0.04)",
}

CURRENT_THEME: dict[str, str] = {}
CURRENT_THEME_MODE = "System"
CURRENT_THEME_RESOLVED = "dark"
BG = SURFACE = SURFACE_ALT = SIDEBAR_BG = BORDER = TEXT = MUTED = ACCENT = ""
BULL = SELL_C = GOLD = PURPLE = BUTTON_TEXT = ALERT_BG = GRID = ""
SIGNAL_COLOR: dict[str, str] = {}
PLOTLY_BASE: dict[str, Any] = {}


def detect_system_theme() -> str:
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize",
        ) as key:
            value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
        return "light" if int(value) == 1 else "dark"
    except Exception:
        return "dark"


def _resolve_theme(theme_mode: str) -> str:
    normalized = (theme_mode or "System").strip().lower()
    if normalized in {"white", "light"}:
        return "light"
    if normalized == "dark":
        return "dark"
    return detect_system_theme()


def apply_theme(theme_mode: str) -> None:
    global CURRENT_THEME, CURRENT_THEME_MODE, CURRENT_THEME_RESOLVED
    global BG, SURFACE, SURFACE_ALT, SIDEBAR_BG, BORDER, TEXT, MUTED, ACCENT
    global BULL, SELL_C, GOLD, PURPLE, BUTTON_TEXT, ALERT_BG, GRID
    global SIGNAL_COLOR, PLOTLY_BASE

    CURRENT_THEME_MODE = theme_mode or "System"
    CURRENT_THEME_RESOLVED = _resolve_theme(CURRENT_THEME_MODE)
    CURRENT_THEME = LIGHT_THEME.copy() if CURRENT_THEME_RESOLVED == "light" else DARK_THEME.copy()

    BG = CURRENT_THEME["BG"]
    SURFACE = CURRENT_THEME["SURFACE"]
    SURFACE_ALT = CURRENT_THEME["SURFACE_ALT"]
    SIDEBAR_BG = CURRENT_THEME["SIDEBAR_BG"]
    BORDER = CURRENT_THEME["BORDER"]
    TEXT = CURRENT_THEME["TEXT"]
    MUTED = CURRENT_THEME["MUTED"]
    ACCENT = CURRENT_THEME["ACCENT"]
    BULL = CURRENT_THEME["BULL"]
    SELL_C = CURRENT_THEME["SELL_C"]
    GOLD = CURRENT_THEME["GOLD"]
    PURPLE = CURRENT_THEME["PURPLE"]
    BUTTON_TEXT = CURRENT_THEME["BUTTON_TEXT"]
    ALERT_BG = CURRENT_THEME["ALERT_BG"]
    GRID = CURRENT_THEME["GRID"]

    SIGNAL_COLOR = {BUY: BULL, SELL: SELL_C, NEUTRAL: GOLD}
    PLOTLY_BASE = {
        "template": "plotly_white" if CURRENT_THEME_RESOLVED == "light" else "plotly_dark",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "'IBM Plex Mono', monospace", "color": TEXT, "size": 12},
        "margin": {"l": 24, "r": 24, "t": 48, "b": 24},
        "hovermode": "x unified",
        "hoverlabel": {"bgcolor": SURFACE, "font_color": TEXT, "font_size": 12},
    }


apply_theme("System")


def _build_css() -> str:
    return """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');
html, body, [class*="css"] {{ font-family: 'Space Grotesk', sans-serif; color: {TEXT}; }}
.stApp {{
  background: {BG};
  background-image:
    radial-gradient(ellipse at 8% 4%,  {GLOW_A} 0%, transparent 38%),
    radial-gradient(ellipse at 92% 8%,  {GLOW_B} 0%, transparent 33%),
    radial-gradient(ellipse at 50% 97%, {GLOW_C}  0%, transparent 38%);
}}
[data-testid="stSidebar"] {{ background: {SIDEBAR_BG} !important; border-right: 1px solid {BORDER}; }}
.main .block-container {{ max-width: 1600px; padding-top: 0.85rem; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 0; background: {SURFACE_ALT}; border-radius: 10px; padding: 4px; border: 1px solid {BORDER}; }}
.stTabs [data-baseweb="tab"] {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.76rem; letter-spacing: 0.08em; color: {MUTED}; padding: 0.45rem 1.1rem; border-radius: 8px; }}
.stTabs [aria-selected="true"] {{ background: {BORDER} !important; color: {ACCENT} !important; }}
[data-testid="metric-container"] {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px; padding: 0.85rem 1rem; box-shadow: 0 2px 16px rgba(0,0,0,0.12); }}
[data-testid="metric-container"] label {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.67rem; letter-spacing: 0.12em; text-transform: uppercase; color: {MUTED}; }}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{ font-family: 'IBM Plex Mono', monospace; font-size: 1.25rem; font-weight: 700; color: {TEXT}; }}
.atx-hero {{ background: linear-gradient(135deg, {HERO_A} 0%, {HERO_B} 100%); border: 1px solid {BORDER}; border-radius: 16px; padding: 1.5rem 2rem; margin-bottom: 1.2rem; position: relative; overflow: hidden; }}
.atx-wordmark {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; letter-spacing: 0.24em; text-transform: uppercase; color: {ACCENT}; margin-bottom: 0.3rem; }}
.atx-title {{ font-family: 'Space Grotesk', sans-serif; font-size: 2.15rem; font-weight: 700; color: {TEXT}; margin: 0 0 0.3rem 0; line-height: 1.15; }}
.atx-title span {{ color: {ACCENT}; }}
.atx-sub {{ color: {MUTED}; font-size: 0.92rem; line-height: 1.5; }}
.atx-badge {{ display: inline-block; font-family: 'IBM Plex Mono', monospace; font-size: 0.63rem; letter-spacing: 0.1em; text-transform: uppercase; background: {ALERT_BG}; color: {ACCENT}; border: 1px solid {BORDER}; border-radius: 20px; padding: 0.18rem 0.6rem; margin-right: 0.4rem; margin-top: 0.7rem; }}
.sl {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.67rem; letter-spacing: 0.18em; text-transform: uppercase; color: {MUTED}; margin-bottom: 0.5rem; margin-top: 1.1rem; }}
.sig-pill {{ display: inline-block; font-family: 'IBM Plex Mono', monospace; font-size: 1rem; font-weight: 700; letter-spacing: 0.05em; padding: 0.3rem 1.1rem; border-radius: 24px; margin: 0.2rem 0; }}
.sig-buy {{ background: rgba(16,185,129,0.15); color: {BULL}; border: 1px solid rgba(16,185,129,0.32); }}
.sig-sell {{ background: rgba(244,63,94,0.15); color: {SELL_C}; border: 1px solid rgba(244,63,94,0.32); }}
.sig-hold {{ background: rgba(245,158,11,0.15); color: {GOLD}; border: 1px solid rgba(245,158,11,0.32); }}
.status-pill {{ display: inline-block; font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; border-radius: 999px; padding: 0.28rem 0.8rem; border: 1px solid transparent; }}
.status-normal {{ color: {BULL}; background: rgba(16,185,129,0.13); border-color: rgba(16,185,129,0.28); }}
.status-warning {{ color: {GOLD}; background: rgba(245,158,11,0.13); border-color: rgba(245,158,11,0.28); }}
.status-emergency {{ color: {SELL_C}; background: rgba(244,63,94,0.13); border-color: rgba(244,63,94,0.28); }}
.panel-card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 12px; padding: 1rem 1.1rem; }}
.stButton > button {{ background: linear-gradient(135deg, {ACCENT} 0%, {PURPLE} 100%); color: {BUTTON_TEXT}; border: none; border-radius: 8px; font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem; font-weight: 700; letter-spacing: 0.06em; padding: 0.55rem 1.4rem; }}
.stDownloadButton > button {{ background: {SURFACE}; color: {TEXT}; border: 1px solid {BORDER}; border-radius: 8px; font-family: 'IBM Plex Mono', monospace; font-size: 0.76rem; }}
.stAlert {{ background: {ALERT_BG} !important; border: 1px solid {BORDER} !important; border-radius: 10px !important; }}
details {{ background: {SURFACE} !important; border: 1px solid {BORDER} !important; border-radius: 10px !important; }}
</style>
""".format(**CURRENT_THEME)


def _inject_css() -> None:
    st.markdown(_build_css(), unsafe_allow_html=True)


def _fig(**kwargs: Any) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(**{**PLOTLY_BASE, **kwargs})
    return fig


def _styled_status(label: str) -> str:
    tone = {
        "NORMAL": "status-normal",
        "WARNING": "status-warning",
        "EMERGENCY": "status-emergency",
    }.get(label.upper(), "status-warning")
    return f"<span class='status-pill {tone}'>{label}</span>"


def _sanitize_name(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def build_feature_frame(ohlcv: pd.DataFrame, indicator_results: dict[str, list[Any]]) -> pd.DataFrame:
    features = pd.DataFrame(index=ohlcv.index)
    features["open"] = ohlcv["open"].astype(float)
    features["high"] = ohlcv["high"].astype(float)
    features["low"] = ohlcv["low"].astype(float)
    features["close"] = ohlcv["close"].astype(float)
    features["volume"] = ohlcv["volume"].astype(float)
    features["return_1"] = ohlcv["close"].pct_change()
    features["range_pct"] = (ohlcv["high"] - ohlcv["low"]) / ohlcv["close"].replace(0, np.nan)
    features["volume_z"] = (
        (ohlcv["volume"] - ohlcv["volume"].rolling(20).mean())
        / ohlcv["volume"].rolling(20).std().replace(0, np.nan)
    )

    added_columns = 0
    for indicators in indicator_results.values():
        for indicator in indicators:
            if added_columns >= 18:
                break
            if getattr(indicator, "series", None) is not None:
                features[_sanitize_name(indicator.name)] = pd.to_numeric(
                    indicator.series.reindex(features.index),
                    errors="coerce",
                )
                added_columns += 1
            for key, series in getattr(indicator, "extra", {}).items():
                if added_columns >= 18:
                    break
                features[f"{_sanitize_name(indicator.name)}_{_sanitize_name(key)}"] = pd.to_numeric(
                    series.reindex(features.index),
                    errors="coerce",
                )
                added_columns += 1
        if added_columns >= 18:
            break

    return features.replace([np.inf, -np.inf], np.nan).ffill().bfill()


def fig_candlestick(ohlcv: pd.DataFrame, signal: EnsembleSignal) -> go.Figure:
    df = ohlcv.copy()
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.72, 0.28],
        subplot_titles=(f"{signal.ticker} Price and Consensus", "Volume"),
    )
    for ann in fig.layout.annotations:
        ann.font = {"color": MUTED, "size": 11}

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
            increasing_line_color=BULL,
            decreasing_line_color=SELL_C,
            increasing_fillcolor=BULL,
            decreasing_fillcolor=SELL_C,
        ),
        row=1,
        col=1,
    )
    ema20 = df["close"].ewm(span=20, adjust=False).mean()
    ema50 = df["close"].ewm(span=50, adjust=False).mean()
    fig.add_trace(go.Scatter(x=df.index, y=ema20, name="EMA20", line={"color": ACCENT, "width": 1.5}), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ema50, name="EMA50", line={"color": GOLD, "width": 1.5, "dash": "dash"}), row=1, col=1)

    vol_colors = [BULL if c >= o else SELL_C for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["volume"], name="Volume", marker_color=vol_colors, opacity=0.6), row=2, col=1)

    fig.update_layout(
        **PLOTLY_BASE,
        height=680,
        title={"text": f"<b>{signal.ticker}</b> price structure", "font": {"size": 15, "color": TEXT}},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "bgcolor": "rgba(0,0,0,0)", "font": {"size": 11}},
        xaxis_rangeslider_visible=False,
    )
    fig.update_yaxes(title_text="Price", gridcolor=GRID)
    fig.update_xaxes(gridcolor=GRID)
    return fig


def fig_signal_gauge(signal: EnsembleSignal) -> go.Figure:
    color = SIGNAL_COLOR.get(signal.final_signal, GOLD)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=signal.confidence * 100,
            number={"suffix": "%", "font": {"size": 40, "color": color, "family": "IBM Plex Mono"}},
            title={"text": f"<b>{signal.final_signal}</b> confidence", "font": {"size": 15, "color": TEXT}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": MUTED, "tickfont": {"color": MUTED}},
                "bar": {"color": color, "thickness": 0.22},
                "bgcolor": SURFACE,
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": SURFACE_ALT},
                    {"range": [50, 75], "color": BORDER},
                    {"range": [75, 100], "color": ALERT_BG},
                ],
            },
        )
    )
    fig.update_layout(**PLOTLY_BASE, height=320)
    return fig


def fig_component_bar(signal: EnsembleSignal) -> go.Figure:
    comp_df = build_signal_component_df(signal)
    comp_df = comp_df[comp_df["Component"] != "ENSEMBLE"]
    colors = [BULL if score >= 0 else SELL_C for score in comp_df["Weighted Score"]]
    fig = _fig(height=320, title={"text": "<b>Signal component weights</b>", "font": {"size": 13, "color": TEXT}})
    fig.add_trace(
        go.Bar(
            x=comp_df["Component"],
            y=comp_df["Weighted Score"],
            marker_color=colors,
            text=[f"{score:.3f}" for score in comp_df["Weighted Score"]],
            textposition="outside",
            textfont={"color": TEXT, "size": 12},
            customdata=comp_df["Detail"],
            hovertemplate="%{x}<br>Score: %{y:.3f}<br>%{customdata}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_color=BORDER, line_width=1)
    fig.update_yaxes(title_text="Weighted score", range=[-0.5, 0.5], gridcolor=GRID)
    fig.update_xaxes(gridcolor=GRID)
    return fig


def fig_equity_curve(result: Any, strategy_label: str) -> go.Figure:
    history = result.history
    fig = _fig(height=420, title={"text": "<b>Strategy equity vs buy and hold</b>", "font": {"size": 13, "color": TEXT}})
    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history["benchmark_equity"],
            name="Buy and Hold",
            fill="tozeroy",
            fillcolor=ALERT_BG,
            line={"color": MUTED, "width": 1.8, "dash": "dash"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history["strategy_equity"],
            name=strategy_label,
            line={"color": GOLD, "width": 2.5},
        )
    )
    fig.update_yaxes(title_text="Portfolio value", gridcolor=GRID)
    fig.update_xaxes(title_text="Date", gridcolor=GRID)
    return fig


def fig_drawdown(result: Any) -> go.Figure:
    history = result.history
    fig = _fig(height=280, title={"text": "<b>Drawdown (%)</b>", "font": {"size": 13, "color": TEXT}})
    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history["drawdown"] * 100,
            name="Drawdown",
            fill="tozeroy",
            fillcolor="rgba(244,63,94,0.12)",
            line={"color": SELL_C, "width": 2.0},
        )
    )
    fig.update_yaxes(title_text="Drawdown %", gridcolor=GRID)
    fig.update_xaxes(gridcolor=GRID)
    return fig


def fig_monthly_heatmap(result: Any) -> go.Figure:
    equity = result.history["strategy_equity"]
    monthly = equity.resample("ME").last().pct_change().dropna() * 100
    monthly_df = pd.DataFrame({"year": monthly.index.year, "month": monthly.index.month, "ret": monthly.values})
    pivot = monthly_df.pivot(index="year", columns="month", values="ret")
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    display_columns = [month_labels[idx - 1] for idx in pivot.columns]
    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=display_columns,
            y=[str(year) for year in pivot.index],
            colorscale=[[0.0, SELL_C], [0.5, BORDER], [1.0, BULL]],
            zmid=0,
            text=[[f"{value:.1f}%" if not np.isnan(value) else "" for value in row] for row in pivot.values],
            texttemplate="%{text}",
            textfont={"size": 10, "color": TEXT},
            hovertemplate="Year %{y} | %{x}<br>Return: %{z:.2f}%<extra></extra>",
            colorbar={"title": "Ret %", "tickfont": {"color": TEXT}},
        )
    )
    fig.update_layout(
        **PLOTLY_BASE,
        height=max(240, len(pivot) * 38 + 80),
        title={"text": "<b>Monthly return heatmap</b>", "font": {"size": 13, "color": TEXT}},
    )
    return fig


def fig_forecast_chart(ohlcv: pd.DataFrame, forecast: ForecastResult) -> go.Figure:
    tail = ohlcv.tail(40)
    next_date = tail.index[-1] + pd.Timedelta(days=1)
    last_close = float(tail["close"].iloc[-1])

    fig = _fig(height=380, title={"text": "<b>Forecast ensemble</b>", "font": {"size": 13, "color": TEXT}})
    fig.add_trace(go.Scatter(x=tail.index, y=tail["close"], name="Close", line={"color": TEXT, "width": 2.0}))
    fig.add_trace(
        go.Scatter(
            x=[tail.index[-1], next_date],
            y=[last_close, forecast.ensemble_pred],
            name="Ensemble",
            mode="lines+markers",
            line={"color": GOLD, "width": 2.2, "dash": "dot"},
            marker={"size": 9, "color": GOLD},
        )
    )
    if forecast.prophet_pred is not None:
        fig.add_trace(
            go.Scatter(
                x=[tail.index[-1], next_date],
                y=[last_close, forecast.prophet_pred],
                name="Prophet",
                mode="lines+markers",
                line={"color": PURPLE, "width": 1.8, "dash": "dot"},
                marker={"size": 7, "color": PURPLE},
            )
        )
    if forecast.timesfm_pred is not None:
        fig.add_trace(
            go.Scatter(
                x=[tail.index[-1], next_date],
                y=[last_close, forecast.timesfm_pred],
                name="TimesFM",
                mode="lines+markers",
                line={"color": ACCENT, "width": 1.8, "dash": "dot"},
                marker={"size": 7, "color": ACCENT},
            )
        )
    if forecast.deep_learning_pred is not None:
        fig.add_trace(
            go.Scatter(
                x=[tail.index[-1], next_date],
                y=[last_close, forecast.deep_learning_pred],
                name=forecast.deep_learning_model,
                mode="lines+markers",
                line={"color": BULL, "width": 1.8, "dash": "dash"},
                marker={"size": 7, "color": BULL},
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[next_date, next_date],
            y=[forecast.lower_bound, forecast.upper_bound],
            name="Confidence band",
            mode="markers",
            marker={"size": 16, "symbol": "line-ns-open", "color": GOLD, "line": {"width": 2, "color": GOLD}},
        )
    )
    fig.update_yaxes(title_text="Price", gridcolor=GRID)
    fig.update_xaxes(title_text="Date", gridcolor=GRID)
    return fig


def fig_rolling_sharpe(result: Any) -> go.Figure:
    returns = result.history["strategy_equity"].pct_change().dropna()
    window = 30
    rolling = (returns.rolling(window).mean() / returns.rolling(window).std().clip(lower=1e-9)) * np.sqrt(252)
    pos = rolling.copy()
    neg = rolling.copy()
    pos[pos < 0] = np.nan
    neg[neg >= 0] = np.nan
    fig = _fig(height=280, title={"text": f"<b>Rolling {window}-day Sharpe</b>", "font": {"size": 13, "color": TEXT}})
    fig.add_trace(go.Scatter(x=rolling.index, y=pos, name="Positive", fill="tozeroy", fillcolor="rgba(16,185,129,0.12)", line={"color": BULL, "width": 1.8}))
    fig.add_trace(go.Scatter(x=rolling.index, y=neg, name="Negative", fill="tozeroy", fillcolor="rgba(244,63,94,0.12)", line={"color": SELL_C, "width": 1.8}))
    fig.add_hline(y=0, line_color=BORDER, line_width=1)
    fig.update_yaxes(title_text="Sharpe", gridcolor=GRID)
    fig.update_xaxes(gridcolor=GRID)
    return fig


def fig_indicator_bar(indicator_df: pd.DataFrame) -> go.Figure:
    color_map = {BUY: BULL, SELL: SELL_C, NEUTRAL: MUTED}
    colors = [color_map.get(value, MUTED) for value in indicator_df["signal"]]
    fig = _fig(height=max(400, len(indicator_df) * 22 + 60), title={"text": "<b>Indicator signal panel</b>", "font": {"size": 13, "color": TEXT}})
    fig.add_trace(
        go.Bar(
            x=indicator_df["value"],
            y=indicator_df["name"],
            orientation="h",
            marker_color=colors,
            opacity=0.82,
            hovertemplate="%{y}<br>Value: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="Value", gridcolor=GRID)
    fig.update_yaxes(tickfont={"size": 9}, gridcolor=GRID)
    return fig


def fig_sector_heatmap(market_context: MarketIntelligenceResult) -> go.Figure | None:
    sector_df = market_context.sector_strength.copy()
    if sector_df.empty:
        return None
    fig = go.Figure(
        go.Heatmap(
            z=[sector_df["score"].tolist()],
            x=sector_df["sector"].tolist(),
            y=["Relative strength"],
            zmin=-1,
            zmax=1,
            colorscale=[[0.0, SELL_C], [0.5, BORDER], [1.0, BULL]],
            text=[[f"{score:+.2f}" for score in sector_df["score"]]],
            texttemplate="%{text}",
            textfont={"color": TEXT, "size": 11},
            customdata=np.stack(
                [
                    sector_df["ticker"].to_numpy(),
                    sector_df["return_20d"].to_numpy(),
                    sector_df["relative_strength"].to_numpy(),
                    sector_df["volume_thrust"].to_numpy(),
                ],
                axis=-1,
            ),
            hovertemplate=(
                "%{x} (%{customdata[0]})"
                "<br>20d return: %{customdata[1]:+.2%}"
                "<br>Relative strength: %{customdata[2]:+.2%}"
                "<br>Volume thrust: %{customdata[3]:+.2%}<extra></extra>"
            ),
            colorbar={"title": "Score", "tickfont": {"color": TEXT}},
        )
    )
    fig.update_layout(**PLOTLY_BASE, height=300, title={"text": "<b>Sector heatmap</b>", "font": {"size": 13, "color": TEXT}})
    return fig


def fig_correlation_heatmap(market_context: MarketIntelligenceResult) -> go.Figure | None:
    corr = market_context.cross_asset_correlation.copy()
    if corr.empty:
        return None
    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            zmin=-1,
            zmax=1,
            colorscale=[[0.0, SELL_C], [0.5, BORDER], [1.0, BULL]],
            hovertemplate="%{y} vs %{x}<br>Corr: %{z:.2f}<extra></extra>",
            colorbar={"title": "Corr", "tickfont": {"color": TEXT}},
        )
    )
    fig.update_layout(**PLOTLY_BASE, height=520, title={"text": "<b>Cross-asset correlation</b>", "font": {"size": 13, "color": TEXT}})
    return fig


def fig_alpha_models(alpha_df: pd.DataFrame) -> go.Figure:
    colors = [BULL if value > 0 else SELL_C if value < 0 else GOLD for value in alpha_df["raw_score"]]
    fig = _fig(height=320, title={"text": "<b>Alpha sleeve scores</b>", "font": {"size": 13, "color": TEXT}})
    fig.add_trace(
        go.Bar(
            x=alpha_df["model"],
            y=alpha_df["raw_score"],
            marker_color=colors,
            text=[f"{value:+.2f}" for value in alpha_df["raw_score"]],
            textposition="outside",
            hovertemplate="%{x}<br>Score: %{y:.3f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_color=BORDER, line_width=1)
    fig.update_yaxes(title_text="Raw score", range=[-1.0, 1.0], gridcolor=GRID)
    fig.update_xaxes(gridcolor=GRID)
    return fig


def render_kpi_row(metrics: dict[str, Any], label: str = "") -> None:
    cols = st.columns(8)
    items = [
        ("Total Return", f"{metrics.get('Total Return', 0):.2%}", metrics.get("Total Return", 0) >= 0),
        ("Ann. Return", f"{metrics.get('Annualised Return', 0):.2%}", metrics.get("Annualised Return", 0) >= 0),
        ("Sharpe", f"{metrics.get('Sharpe Ratio', 0):.2f}", metrics.get("Sharpe Ratio", 0) >= 1),
        ("Sortino", f"{metrics.get('Sortino Ratio', 0):.2f}", metrics.get("Sortino Ratio", 0) >= 1),
        ("Calmar", f"{metrics.get('Calmar Ratio', 0):.2f}", metrics.get("Calmar Ratio", 0) >= 0.5),
        ("Max Drawdown", f"{metrics.get('Max Drawdown', 0):.2%}", metrics.get("Max Drawdown", 0) >= -0.1),
        ("Ann. Vol", f"{metrics.get('Annualised Volatility', 0):.2%}", True),
        ("Trades", f"{int(metrics.get('Number of Trades', 0))}", True),
    ]
    for col, (name, value, positive) in zip(cols, items):
        delta_color = "normal" if positive else "inverse"
        col.metric(f"{label + ' | ' if label else ''}{name}", value, delta=value, delta_color=delta_color)


def render_system_health() -> None:
    st.markdown("---")
    st.markdown("<div class='sl'>Engine Guardian</div>", unsafe_allow_html=True)
    cpu_load = psutil.cpu_percent()
    ram_load = psutil.virtual_memory().percent
    col1, col2 = st.columns(2)
    col1.metric("CPU Load", f"{cpu_load:.0f}%")
    col2.metric("RAM Load", f"{ram_load:.0f}%")
    if cpu_load > 80:
        st.warning("High compute load detected.")
    else:
        st.success("Runtime stable.")
    st.caption(f"Theme resolved to {CURRENT_THEME_RESOLVED.title()}. Execution boundary ready for future C++ gateway.")


def render_sidebar() -> dict[str, Any]:
    default_end = date.today()
    default_start = default_end - timedelta(days=365 * 2)
    strategy_list = list_strategies()

    with st.sidebar:
        st.markdown(
            f"<div style='font-family:IBM Plex Mono;font-size:0.7rem;letter-spacing:0.2em;text-transform:uppercase;color:{ACCENT};margin-bottom:0.8rem;'>ATX V5</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<div class='sl'>Theme</div>", unsafe_allow_html=True)
        current_theme_mode = st.session_state.get("theme_mode", "System")
        if current_theme_mode == "Light":
            current_theme_mode = "White"
        theme_mode = st.selectbox(
            "Theme mode",
            ["System", "Dark", "White"],
            index=["System", "Dark", "White"].index(current_theme_mode if current_theme_mode in {"System", "Dark", "White"} else "System"),
            key="theme_mode",
            help="System follows the current Windows app theme.",
        )
        st.caption(f"Resolved theme: {CURRENT_THEME_RESOLVED.title()}")

        st.markdown("<div class='sl'>Data Source</div>", unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload OHLCV CSV", type=["csv"])
        ticker = st.text_input("Ticker Symbol", value="AAPL", help="Yahoo Finance ticker, for example AAPL or RELIANCE.NS")
        date_range = st.date_input(
            "Date Range",
            value=(default_start, default_end),
            min_value=date(2000, 1, 1),
            max_value=default_end,
        )

        st.markdown("<div class='sl'>Strategy Lab</div>", unsafe_allow_html=True)
        strategy_labels = [entry["label"] for entry in strategy_list]
        strategy_keys = {entry["label"]: entry["key"] for entry in strategy_list}
        selected_label = st.selectbox("Benchmark Strategy", strategy_labels)
        selected_key = strategy_keys[selected_label]
        default_params = STRATEGY_REGISTRY[selected_key]["params"]

        strategy_params: dict[str, Any] = {}
        if default_params:
            with st.expander("Strategy Parameters", expanded=True):
                for key, value in default_params.items():
                    label = key.replace("_", " ").title()
                    if isinstance(value, int):
                        strategy_params[key] = st.number_input(label, min_value=1, max_value=500, value=value, step=1)
                    elif isinstance(value, float):
                        strategy_params[key] = st.number_input(label, min_value=0.0, max_value=100.0, value=value, step=1.0)

        st.markdown("<div class='sl'>Portfolio</div>", unsafe_allow_html=True)
        initial_capital = st.number_input("Initial Capital ($)", min_value=1_000.0, max_value=10_000_000.0, value=100_000.0, step=5_000.0)
        transaction_cost = st.number_input("Transaction Cost", min_value=0.0, max_value=0.02, value=0.001, step=0.0001, format="%.4f")
        slippage_pct = st.number_input("Slippage", min_value=0.0, max_value=0.02, value=0.0005, step=0.0001, format="%.4f")

        st.markdown("<div class='sl'>Forecast Stack</div>", unsafe_allow_html=True)
        use_prophet = st.checkbox("Enable Prophet", value=True)
        use_timesfm = st.checkbox("Enable TimesFM", value=True)
        use_deep_learning = st.checkbox("Enable deep learning sleeve", value=True)

        st.markdown("<div class='sl'>Risk Guardrails</div>", unsafe_allow_html=True)
        st.caption("Max loss per trade: 0.5% of capital")
        st.caption("Emergency stop: drawdown spike, loss streak, crash regime, or model disagreement")

        st.markdown("---")
        run_clicked = st.button("Run Analysis", type="primary", use_container_width=True)

        render_system_health()

    if len(date_range) != 2:
        raise ValueError("Select both a start and end date.")

    return {
        "ticker": ticker.strip().upper() or "AAPL",
        "uploaded": uploaded,
        "start_date": date_range[0],
        "end_date": date_range[1],
        "strategy_key": selected_key,
        "strategy_label": selected_label,
        "strategy_params": strategy_params,
        "initial_capital": float(initial_capital),
        "transaction_cost": float(transaction_cost),
        "slippage_pct": float(slippage_pct),
        "use_prophet": use_prophet,
        "use_timesfm": use_timesfm,
        "use_deep_learning": use_deep_learning,
        "theme_mode": theme_mode,
        "run_clicked": run_clicked,
    }


def run_analysis(inputs: dict[str, Any], progress_cb) -> dict[str, Any]:
    from strategies.algotradex_strategies import build_summary_table, run_all_indicators

    progress_cb("Fetching market data...")
    if inputs["uploaded"]:
        ohlcv_full = load_csv_upload(inputs["uploaded"])
    else:
        ohlcv_full = fetch_yfinance(inputs["ticker"], inputs["start_date"], inputs["end_date"])

    ohlcv = slice_date_range(ohlcv_full, inputs["start_date"], inputs["end_date"])
    validate_ohlcv(ohlcv)

    progress_cb("Computing technical indicators...")
    indicator_results = run_all_indicators(ohlcv)
    indicator_df = build_summary_table(indicator_results)
    feature_frame = build_feature_frame(ohlcv, indicator_results)

    progress_cb("Running market intelligence...")
    market_frames = fetch_market_universe(start=inputs["start_date"], end=inputs["end_date"])
    market_context = analyze_market_intelligence(
        start=inputs["start_date"],
        end=inputs["end_date"],
        market_frames=market_frames,
        reference_ohlcv=ohlcv_full,
    )

    progress_cb("Running alpha models...")
    alpha_bundle = run_alpha_models(
        target_ticker=inputs["ticker"],
        target_ohlcv=ohlcv,
        market_frames=market_frames,
        market_context=market_context,
    )
    alpha_df = build_alpha_model_df(alpha_bundle)

    progress_cb("Running forecast ensemble...")
    forecast = run_forecast(
        ohlcv["close"],
        horizon=1,
        use_prophet=inputs["use_prophet"],
        use_timesfm=inputs["use_timesfm"],
        feature_frame=feature_frame,
        sentiment_score=0.0,
        use_deep_learning=inputs["use_deep_learning"],
    )

    progress_cb("Computing ensemble agreement...")
    signal = compute_ensemble_signal(
        ticker=inputs["ticker"],
        ohlcv=ohlcv,
        news_headlines=None,
        forecast_result=forecast,
        market_context=market_context,
        alpha_bundle=alpha_bundle,
    )

    progress_cb(f"Backtesting {inputs['strategy_label']}...")
    backtest = run_named_strategy(
        ohlcv=ohlcv_full,
        strategy_key=inputs["strategy_key"],
        params=inputs["strategy_params"],
        initial_capital=inputs["initial_capital"],
        transaction_cost=inputs["transaction_cost"],
        slippage_pct=inputs["slippage_pct"],
        start_date=inputs["start_date"],
        end_date=inputs["end_date"],
    )

    progress_cb("Applying risk gate...")
    risk_decision = evaluate_risk(
        ticker=inputs["ticker"],
        ohlcv=ohlcv,
        signal=signal,
        market_context=market_context,
        capital=inputs["initial_capital"],
        alpha_bundle=alpha_bundle,
        backtest=backtest,
    )

    return {
        "ticker": inputs["ticker"],
        "ohlcv": ohlcv,
        "signal": signal,
        "forecast": forecast,
        "market_context": market_context,
        "alpha_bundle": alpha_bundle,
        "alpha_df": alpha_df,
        "risk_decision": risk_decision,
        "backtest": backtest,
        "indicator_df": indicator_df,
        "feature_frame": feature_frame,
        "inputs": inputs,
    }


def tab_command_deck(result: dict[str, Any]) -> None:
    signal: EnsembleSignal = result["signal"]
    forecast: ForecastResult = result["forecast"]
    market_context: MarketIntelligenceResult = result["market_context"]
    risk_decision: RiskDecision = result["risk_decision"]
    alpha_bundle: AlphaModelBundle = result["alpha_bundle"]
    ohlcv = result["ohlcv"]

    approved_signal = risk_decision.approved_signal
    pill_class = {"Buy": "sig-buy", "Sell": "sig-sell"}.get(approved_signal, "sig-hold")
    st.markdown(
        f"<div class='sl'>Risk approved action</div><div class='sig-pill {pill_class}'>{approved_signal}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(_styled_status(risk_decision.risk_status), unsafe_allow_html=True)

    last_price = float(ohlcv["close"].iloc[-1])
    prev_price = float(ohlcv["close"].iloc[-2]) if len(ohlcv) > 1 else last_price
    day_change = (last_price - prev_price) / prev_price if prev_price else 0.0
    metrics = st.columns(7)
    metrics[0].metric("Last Close", f"{last_price:.2f}", f"{day_change:+.2%}")
    metrics[1].metric("Ensemble Score", f"{signal.ensemble_score:+.3f}")
    metrics[2].metric("Agreement", f"{signal.agreement_score:.0%}")
    metrics[3].metric("Market Bias", market_context.market_bias_label)
    metrics[4].metric("Risk Status", risk_decision.risk_status)
    metrics[5].metric("Position Size", f"{risk_decision.position_size:,.2f}")
    metrics[6].metric("Forecast", f"{forecast.ensemble_pred:.2f}", f"{forecast.expected_return_pct:+.2%}")

    gauge_col, bar_col = st.columns([1, 1.6])
    with gauge_col:
        st.plotly_chart(fig_signal_gauge(signal), use_container_width=True)
    with bar_col:
        st.plotly_chart(fig_component_bar(signal), use_container_width=True)

    note_col, risk_col = st.columns(2)
    with note_col:
        st.markdown("<div class='sl'>Model agreement</div>", unsafe_allow_html=True)
        st.markdown(
            (
                f"<div class='panel-card'>"
                f"<strong>Pre-agreement signal:</strong> {signal.pre_agreement_signal}<br>"
                f"<strong>Agreement state:</strong> {signal.agreement_state}<br>"
                f"<strong>Forecast model agreement:</strong> {forecast.model_agreement:.0%}<br>"
                f"<strong>Alpha agreement:</strong> {alpha_bundle.agreement_score:.0%}<br>"
                f"<strong>Execution notes:</strong> {', '.join(signal.execution_notes) or 'None'}"
                f"</div>"
            ),
            unsafe_allow_html=True,
        )
    with risk_col:
        st.markdown("<div class='sl'>Risk gate</div>", unsafe_allow_html=True)
        risk_text = ", ".join(risk_decision.reasons) if risk_decision.reasons else "No active risk blocks."
        st.markdown(
            (
                f"<div class='panel-card'>"
                f"<strong>Approved signal:</strong> {risk_decision.approved_signal}<br>"
                f"<strong>Stop distance:</strong> {risk_decision.stop_loss_distance:.4f}<br>"
                f"<strong>Target notional:</strong> ${risk_decision.target_notional:,.2f}<br>"
                f"<strong>Freeze trading:</strong> {risk_decision.freeze_trading}<br>"
                f"<strong>Reasons:</strong> {risk_text}"
                f"</div>"
            ),
            unsafe_allow_html=True,
        )

    with st.expander("Signal component detail", expanded=False):
        st.dataframe(build_signal_component_df(signal), use_container_width=True, hide_index=True)
    with st.expander("Execution intent", expanded=False):
        st.json(risk_decision.execution_intent.to_dict() if risk_decision.execution_intent else {})

    if risk_decision.freeze_trading:
        st.error("Trading is frozen by the risk engine. All new execution should be blocked.")
    elif risk_decision.risk_status == "WARNING":
        st.warning("Risk engine downgraded conviction. Position sizing is reduced or neutralized.")
    else:
        st.success("Signal passed the risk gate under current market conditions.")


def tab_market_intelligence(result: dict[str, Any]) -> None:
    market_context: MarketIntelligenceResult = result["market_context"]
    alpha_bundle: AlphaModelBundle = result["alpha_bundle"]
    alpha_df = result["alpha_df"]

    metrics = st.columns(6)
    metrics[0].metric("Regime", market_context.regime)
    metrics[1].metric("Market Bias", market_context.market_bias_label)
    metrics[2].metric("Confidence", f"{market_context.confidence:.0%}")
    metrics[3].metric("Risk Multiplier", f"{market_context.risk_multiplier:.2f}x")
    metrics[4].metric("Breadth", f"{market_context.breadth_ratio:.0%}")
    metrics[5].metric("Correlation Risk", f"{market_context.correlation_risk:.0%}")

    sector_fig = fig_sector_heatmap(market_context)
    corr_fig = fig_correlation_heatmap(market_context)
    alpha_fig = fig_alpha_models(alpha_df)

    col1, col2 = st.columns([1.05, 1.35])
    with col1:
        if sector_fig is not None:
            st.plotly_chart(sector_fig, use_container_width=True)
        else:
            st.info("Sector strength is unavailable for this run.")
        st.plotly_chart(alpha_fig, use_container_width=True)
    with col2:
        if corr_fig is not None:
            st.plotly_chart(corr_fig, use_container_width=True)
        else:
            st.info("Cross-asset correlation is unavailable for this run.")

    st.markdown("<div class='sl'>Alpha sleeve detail</div>", unsafe_allow_html=True)
    st.dataframe(alpha_df, use_container_width=True, hide_index=True)

    with st.expander("Sector ranking table", expanded=False):
        st.dataframe(market_context.sector_strength, use_container_width=True, hide_index=True)
    with st.expander("Market regime summary", expanded=False):
        st.json(market_context.to_summary_dict())
        st.caption("Notes: " + (" | ".join(market_context.notes) if market_context.notes else "None"))
        st.caption(f"Selected momentum basket: {', '.join(alpha_bundle.selected_assets) or 'None'}")


def tab_chart(result: dict[str, Any]) -> None:
    st.plotly_chart(fig_candlestick(result["ohlcv"], result["signal"]), use_container_width=True)


def tab_strategy_lab(result: dict[str, Any]) -> None:
    backtest = result["backtest"]
    inputs = result["inputs"]
    risk_decision: RiskDecision = result["risk_decision"]

    st.markdown(
        (
            "<div class='panel-card'>"
            f"<div class='sl' style='margin-top:0;'>Strategy laboratory</div>"
            f"<strong>{backtest.strategy_name}</strong><br>"
            f"Research sandbox aligned with V5 signals. Current live risk status: {risk_decision.risk_status}."
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    st.markdown("<div class='sl'>Performance dossier</div>", unsafe_allow_html=True)
    render_kpi_row(backtest.metrics, backtest.strategy_name)
    st.plotly_chart(fig_equity_curve(backtest, backtest.strategy_name), use_container_width=True)

    col_drawdown, col_sharpe = st.columns(2)
    with col_drawdown:
        st.plotly_chart(fig_drawdown(backtest), use_container_width=True)
    with col_sharpe:
        st.plotly_chart(fig_rolling_sharpe(backtest), use_container_width=True)

    st.plotly_chart(fig_monthly_heatmap(backtest), use_container_width=True)

    if not backtest.trade_log.empty:
        trade_log = backtest.trade_log
        completed_trades = int((trade_log["direction"] == "Sell").sum()) if "direction" in trade_log.columns else len(trade_log)
        winners = trade_log[trade_log.get("pnl", pd.Series(dtype=float)) > 0] if "pnl" in trade_log.columns else pd.DataFrame()
        cols = st.columns(3)
        cols[0].metric("Completed Trades", completed_trades)
        cols[1].metric("Win Rate", f"{(len(winners) / max(completed_trades, 1)):.0%}")
        cols[2].metric("Total Costs", f"{trade_log.get('cost', pd.Series(dtype=float)).sum():,.2f}")
        with st.expander("Trade log", expanded=False):
            st.dataframe(trade_log, use_container_width=True, hide_index=True)

    with st.expander("Experiment configuration", expanded=False):
        st.json(
            {
                "strategy": backtest.strategy_name,
                "initial_capital": inputs["initial_capital"],
                "transaction_cost": inputs["transaction_cost"],
                "slippage_pct": inputs["slippage_pct"],
                **inputs["strategy_params"],
            }
        )


def tab_forecast(result: dict[str, Any]) -> None:
    forecast: ForecastResult = result["forecast"]
    ohlcv = result["ohlcv"]

    st.warning("Forecasts are probabilistic research outputs, not guaranteed execution instructions.")

    metrics = st.columns(5)
    metrics[0].metric("Last Close", f"{forecast.last_close:.2f}")
    metrics[1].metric("Ensemble Forecast", f"{forecast.ensemble_pred:.2f}", f"{forecast.expected_return_pct:+.2%}")
    metrics[2].metric("Deep Learning", f"{forecast.deep_learning_pred:.2f}" if forecast.deep_learning_pred is not None else "N/A", f"{forecast.deep_learning_confidence:.0%}")
    metrics[3].metric("Model Agreement", f"{forecast.model_agreement:.0%}")
    metrics[4].metric("Forecast Band", f"{forecast.lower_bound:.2f} - {forecast.upper_bound:.2f}")

    breakdown = st.columns(4)
    breakdown[0].metric("Prophet", f"{forecast.prophet_pred:.2f}" if forecast.prophet_pred is not None else "N/A")
    breakdown[1].metric("TimesFM", f"{forecast.timesfm_pred:.2f}" if forecast.timesfm_pred is not None else "N/A")
    breakdown[2].metric(forecast.deep_learning_model, f"{forecast.deep_learning_pred:.2f}" if forecast.deep_learning_pred is not None else "N/A")
    breakdown[3].metric("Naive", f"{forecast.naive_pred:.2f}")

    st.plotly_chart(fig_forecast_chart(ohlcv, forecast), use_container_width=True)

    with st.expander("Forecast methodology", expanded=False):
        st.markdown(
            f"""
Prophet, TimesFM, a hybrid sequence model, and a naive linear baseline contribute to the V5 forecast stack.

- Methods used: {", ".join(forecast.methods_used)}
- Deep sleeve: {forecast.deep_learning_model}
- Deep confidence: {forecast.deep_learning_confidence:.0%}
- Forecast model agreement: {forecast.model_agreement:.0%}
"""
        )
        if forecast.feature_snapshot:
            st.json(forecast.feature_snapshot)


def tab_indicator_panel(result: dict[str, Any]) -> None:
    indicator_df = result["indicator_df"]
    signal: EnsembleSignal = result["signal"]

    st.markdown("<div class='sl'>Indicator panel</div>", unsafe_allow_html=True)
    counts = st.columns(3)
    counts[0].metric("Buy", signal.buy_count)
    counts[1].metric("Sell", signal.sell_count)
    counts[2].metric("Neutral", signal.neutral_count)

    filter_value = st.selectbox("Filter by signal", ["All", "Buy", "Sell", "Neutral"])
    display_df = indicator_df if filter_value == "All" else indicator_df[indicator_df["signal"] == filter_value]

    st.plotly_chart(fig_indicator_bar(display_df), use_container_width=True)
    with st.expander("Full indicator table", expanded=True):
        st.dataframe(display_df, use_container_width=True, hide_index=True)


def tab_raw_data(result: dict[str, Any]) -> None:
    ohlcv = result["ohlcv"]
    backtest = result["backtest"]
    ticker = result["ticker"]

    downloads = st.columns(4)
    downloads[0].download_button(
        "Market Data CSV",
        data=ohlcv.to_csv().encode(),
        mime="text/csv",
        file_name=f"{ticker.lower()}_market_data.csv",
        use_container_width=True,
    )
    downloads[1].download_button(
        "Backtest CSV",
        data=backtest.history.to_csv().encode(),
        mime="text/csv",
        file_name=f"{ticker.lower()}_{result['inputs']['strategy_key']}_backtest.csv",
        use_container_width=True,
    )
    downloads[2].download_button(
        "Indicators CSV",
        data=result["indicator_df"].to_csv(index=False).encode(),
        mime="text/csv",
        file_name=f"{ticker.lower()}_indicators.csv",
        use_container_width=True,
    )
    downloads[3].download_button(
        "Alpha Models CSV",
        data=result["alpha_df"].to_csv(index=False).encode(),
        mime="text/csv",
        file_name=f"{ticker.lower()}_alpha_models.csv",
        use_container_width=True,
    )

    with st.expander("Market data", expanded=False):
        st.dataframe(ohlcv.tail(100), use_container_width=True)
    with st.expander("Backtest history", expanded=False):
        st.dataframe(backtest.history.tail(100), use_container_width=True)
    with st.expander("Trade log", expanded=False):
        if not backtest.trade_log.empty:
            st.dataframe(backtest.trade_log, use_container_width=True, hide_index=True)
        else:
            st.info("No trades executed in this period.")
    with st.expander("Execution and risk payload", expanded=False):
        st.json(result["risk_decision"].to_summary_dict())
    with st.expander("Market intelligence payload", expanded=False):
        st.json(result["market_context"].to_summary_dict())


def main() -> None:
    st.set_page_config(
        page_title="AlgoTradeX V5",
        layout="wide",
        page_icon="ATX",
        initial_sidebar_state="expanded",
    )

    st.session_state.setdefault("theme_mode", "System")
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("error", None)

    apply_theme(st.session_state.get("theme_mode", "System"))
    _inject_css()

    st.markdown(
        """
<div class="atx-hero">
  <div class="atx-wordmark">ATX V5 Quant Intelligence System</div>
  <h1 class="atx-title">AlgoTradex <span></span> By Vijay</h1>
  <p class="atx-sub">
    Multi-model forecasting, market regime analysis, alpha sleeves, and strict risk-first execution control.
    Signals are tradeable only when models agree, market conditions support them, and the risk gate approves them.
  </p>
  <div>
    <span class="atx-badge">Market Intelligence</span>
    <span class="atx-badge">Deep Learning Forecast</span>
    <span class="atx-badge">Alpha Models</span>
    <span class="atx-badge">Risk First</span>
    <span class="atx-badge">Python to C++ Boundary Ready</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    try:
        inputs = render_sidebar()
    except Exception as exc:
        st.error(f"Configuration error: {exc}")
        return

    if inputs["run_clicked"] or st.session_state["result"] is None:
        holder = st.empty()
        try:
            with st.spinner(""):
                result = run_analysis(inputs, progress_cb=lambda message: holder.info(message))
            st.session_state["result"] = result
            st.session_state["error"] = None
        except Exception as exc:
            logger.exception("Analysis failed")
            st.session_state["error"] = str(exc)
        finally:
            holder.empty()

    if st.session_state["error"]:
        st.error(st.session_state["error"])

    if st.session_state["result"] is None:
        st.markdown(
            f"""
<div style="text-align:center;padding:5rem 2rem;color:{MUTED};font-family:'IBM Plex Mono',monospace;">
  <div style="font-size:1.05rem;font-weight:600;color:{TEXT};margin-bottom:0.5rem;">
    Ready to run the V5 pipeline
  </div>
  <div style="font-size:0.84rem;max-width:520px;margin:auto;line-height:1.7;">
    Select a ticker, configure the strategy and forecast stack, then run analysis.
    The dashboard will execute market intelligence, alpha models, forecasting, ensemble logic, and risk gating.
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
        return

    result = st.session_state["result"]
    tabs = st.tabs(
        [
            "Command Deck",
            "Market Intelligence",
            "Chart",
            "Strategy Lab",
            "Forecast",
            "Indicators",
            "Raw Data",
        ]
    )

    with tabs[0]:
        tab_command_deck(result)
    with tabs[1]:
        tab_market_intelligence(result)
    with tabs[2]:
        tab_chart(result)
    with tabs[3]:
        tab_strategy_lab(result)
    with tabs[4]:
        tab_forecast(result)
    with tabs[5]:
        tab_indicator_panel(result)
    with tabs[6]:
        tab_raw_data(result)


if __name__ == "__main__":
    main()
