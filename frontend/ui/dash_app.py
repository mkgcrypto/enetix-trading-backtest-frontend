import datetime as dt
import io
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dash import Dash, Input, Output, State, callback_context, dash_table, dcc, html, no_update
from dash.exceptions import PreventUpdate

from frontend.api_client import (
    BackendApiError,
    cancel_job,
    enqueue_discover,
    enqueue_leaderboard_refresh,
    enqueue_optimize,
    enqueue_patterns_refresh,
    get_discovery_stats,
    get_job,
    get_job_events,
    get_leaderboard_latest,
    get_patterns,
    list_jobs,
    run_backtest,
)

from frontend.features.discovery_helpers import (
    count_filtered_combinations,
    create_margin_discovery_grid,
    estimate_discovery_time,
    get_cpu_count,
)
from frontend.features.optimization import ResultConstraints, analyze_results, create_custom_grid
from frontend.features.patterns import DiscoveredRule, get_rule_summary
from frontend.features.presets import DEFAULT_PRESET, STRATEGY_PRESETS, apply_preset_direction
from frontend.ui.dash_helpers import build_backtest_figure, build_empty_figure, build_entry_diagnostics, build_trades_table

CHECK_ON = "on"
INLINE_OPTION_STYLE = {
    "display": "inline-flex",
    "alignItems": "center",
    "margin": "0 10px 6px 0",
    "textTransform": "none",
    "letterSpacing": "normal",
}


class _DashRequestFilter(logging.Filter):
    def __init__(self, *, blocked_paths: tuple[str, ...]) -> None:
        super().__init__()
        self._blocked_paths = blocked_paths

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(path in message for path in self._blocked_paths)


def _configure_logging() -> None:
    logger = logging.getLogger("werkzeug")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.addFilter(_DashRequestFilter(blocked_paths=("/_dash-update-component",)))
    logger.handlers = [handler]
    logger.propagate = False


def _to_checklist(flag: bool) -> List[str]:
    return [CHECK_ON] if flag else []


def _from_checklist(value: Optional[List[str]]) -> bool:
    return bool(value)


def _parse_date(value: Any) -> Optional[dt.date]:
    if value is None:
        return None
    if isinstance(value, dt.date):
        return value
    try:
        return dt.date.fromisoformat(str(value))
    except ValueError:
        return None


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_ts(value: Any) -> Optional[dt.datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, dt.datetime):
        return value
    try:
        return dt.datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _heartbeat_age_seconds(value: Optional[str]) -> Optional[float]:
    ts = _parse_ts(value)
    if ts is None:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.UTC)
    now = dt.datetime.now(dt.UTC)
    return max(0.0, (now - ts).total_seconds())


def _format_age(seconds: Optional[float]) -> str:
    if seconds is None:
        return ""
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        minutes = int(seconds // 60)
        return f"{minutes}m {int(seconds % 60)}s"
    if seconds < 86400:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    return f"{days}d {hours}h"


def _worker_stale_seconds() -> float:
    return _safe_float(os.getenv("WORKER_STALE_SECONDS"), 900.0)


def _worker_status(heartbeat_at: Optional[str]) -> tuple[str, str]:
    age_seconds = _heartbeat_age_seconds(heartbeat_at)
    if age_seconds is None:
        return "unknown", ""
    status = "active" if age_seconds <= _worker_stale_seconds() else "stale"
    return status, _format_age(age_seconds)


def _resolve_preset_direction(value: Optional[str]) -> Optional[str]:
    if value in {"Short", "Long", "Both"}:
        return value
    return None


def _format_margin_summary(preset: Dict[str, Any]) -> str:
    if preset.get("trade_mode") != "Margin / Futures":
        return ""
    parts = []
    max_leverage = preset.get("max_leverage")
    if max_leverage is not None:
        parts.append(f"max lev {max_leverage:g}x")
    max_util = preset.get("max_margin_utilization")
    if max_util is not None:
        parts.append(f"max util {max_util:g}%")
    maintenance = preset.get("maintenance_margin_pct")
    if maintenance is not None:
        parts.append(f"maint {maintenance:g}%")
    if not parts:
        return ""
    return "Margin: " + ", ".join(parts)


def _build_preset_info(preset: Dict[str, Any], direction_override: Optional[str]) -> str:
    name = preset.get("name", "")
    description = preset.get("description", "")
    header = f"{name}: {description}" if description else name
    info_parts = [header] if header else []

    direction = preset.get("trade_direction")
    if direction:
        label = "Direction override" if direction_override else "Default direction"
        info_parts.append(f"{label}: {direction}")

    margin = _format_margin_summary(preset)
    if margin:
        info_parts.append(margin)

    return " | ".join(info_parts)


def _format_preset_recommendation(preset: Dict[str, Any]) -> Optional[str]:
    if not preset:
        return None
    timeframe = preset.get("recommended_timeframe")
    lookback_days = preset.get("recommended_lookback_days")
    parts = []
    if timeframe:
        parts.append(f"{timeframe} candles")
    if lookback_days:
        preset_key = _preset_key_for_lookback(int(lookback_days))
        label = DATE_PRESET_LABELS.get(preset_key, f"{int(lookback_days)}D")
        parts.append(f"{label} lookback")
    if not parts:
        return None
    return "Recommended data: " + " · ".join(parts)


def _date_range_from_preset(preset: str, anchor_date: dt.date) -> Optional[Tuple[dt.date, dt.date]]:
    if preset == "ytd":
        return dt.date(anchor_date.year, 1, 1), anchor_date
    days = DATE_PRESET_LOOKBACK_DAYS.get(preset)
    if not days:
        return None
    return anchor_date - dt.timedelta(days=days), anchor_date


def _preset_key_for_lookback(days: Optional[int]) -> Optional[str]:
    if not days:
        return None
    for key, value in DATE_PRESET_LOOKBACK_DAYS.items():
        if int(value) == int(days):
            return key
    return None


def _deserialize_df(payload: Optional[str]) -> pd.DataFrame:
    if not payload:
        return pd.DataFrame()
    return pd.read_json(io.StringIO(payload), orient="split")


def _default_ui_values() -> Dict[str, Any]:
    return {
        "w_exchange": "bitstamp",
        "w_symbol": "BTC/USD",
        "w_timeframe": "30m",
        "w_start_date": dt.date(2022, 1, 1),
        "w_end_date": dt.datetime.now(dt.UTC).date(),
        "w_date_preset": "manual",
        "w_cash": 10_000,
        "w_commission": 0.001,
        "w_bb_len": DEFAULT_PRESET.get("bb_len", 20),
        "w_bb_std": DEFAULT_PRESET.get("bb_std", 2.0),
        "w_bb_basis_type": DEFAULT_PRESET.get("bb_basis_type", "sma"),
        "w_kc_ema_len": DEFAULT_PRESET.get("kc_ema_len", 20),
        "w_kc_atr_len": DEFAULT_PRESET.get("kc_atr_len", 14),
        "w_kc_mult": DEFAULT_PRESET.get("kc_mult", 2.0),
        "w_kc_mid_is_ema": _to_checklist(DEFAULT_PRESET.get("kc_mid_type", "ema") == "ema"),
        "w_rsi_len_30m": DEFAULT_PRESET.get("rsi_len_30m", 14),
        "w_rsi_smoothing_type": DEFAULT_PRESET.get("rsi_smoothing_type", "ema"),
        "w_rsi_ma_len": DEFAULT_PRESET.get("rsi_ma_len", 10),
        "w_rsi_ma_type": DEFAULT_PRESET.get("rsi_ma_type", "sma"),
        "w_rsi_min": DEFAULT_PRESET.get("rsi_min", 70),
        "w_rsi_ma_min": DEFAULT_PRESET.get("rsi_ma_min", 70),
        "w_rsi_max": DEFAULT_PRESET.get("rsi_max", 30),
        "w_rsi_ma_max": DEFAULT_PRESET.get("rsi_ma_max", 30),
        "w_use_rsi_relation": _to_checklist(DEFAULT_PRESET.get("use_rsi_relation", True)),
        "w_rsi_relation": DEFAULT_PRESET.get("rsi_relation", ">="),
        "w_entry_band_mode": DEFAULT_PRESET.get("entry_band_mode", "Either"),
        "w_trade_direction": DEFAULT_PRESET.get("trade_direction", "Short"),
        "w_exit_channel": DEFAULT_PRESET.get("exit_channel", "BB"),
        "w_exit_level": DEFAULT_PRESET.get("exit_level", "mid"),
        "w_trade_mode": DEFAULT_PRESET.get("trade_mode", "Margin / Futures"),
        "w_use_stop": _to_checklist(DEFAULT_PRESET.get("use_stop", True)),
        "w_stop_mode": DEFAULT_PRESET.get("stop_mode", "Fixed %"),
        "w_stop_pct": float(DEFAULT_PRESET.get("stop_pct", 2.0) or 2.0),
        "w_stop_atr_mult": float(DEFAULT_PRESET.get("stop_atr_mult", 2.0) or 2.0),
        "w_use_trailing": _to_checklist(DEFAULT_PRESET.get("use_trailing", False)),
        "w_trail_pct": float(DEFAULT_PRESET.get("trail_pct", 1.0) or 1.0),
        "w_max_bars_in_trade": int(DEFAULT_PRESET.get("max_bars_in_trade", 100)),
        "w_daily_loss_limit": float(DEFAULT_PRESET.get("daily_loss_limit", 3.0)),
        "w_risk_per_trade_pct": float(DEFAULT_PRESET.get("risk_per_trade_pct", 1.0)),
        "w_max_leverage": float(DEFAULT_PRESET.get("max_leverage", 5.0) or 5.0),
        "w_maintenance_margin_pct": float(DEFAULT_PRESET.get("maintenance_margin_pct", 0.5) or 0.5),
        "w_enable_max_margin_util": [],
        "w_max_margin_utilization": float(DEFAULT_PRESET.get("max_margin_utilization", 70.0) or 70.0),
        "show_candles": _to_checklist(True),
        "lock_rsi_y": _to_checklist(True),
    }


def _preset_to_ui_values(preset: Dict[str, Any]) -> Dict[str, Any]:
    values = _default_ui_values()
    if not preset:
        return values

    values.update({
        "w_bb_len": preset.get("bb_len", values["w_bb_len"]),
        "w_bb_std": preset.get("bb_std", values["w_bb_std"]),
        "w_bb_basis_type": preset.get("bb_basis_type", values["w_bb_basis_type"]),
        "w_kc_ema_len": preset.get("kc_ema_len", values["w_kc_ema_len"]),
        "w_kc_atr_len": preset.get("kc_atr_len", values["w_kc_atr_len"]),
        "w_kc_mult": preset.get("kc_mult", values["w_kc_mult"]),
        "w_kc_mid_is_ema": _to_checklist(preset.get("kc_mid_type", "ema") == "ema"),
        "w_rsi_len_30m": preset.get("rsi_len_30m", values["w_rsi_len_30m"]),
        "w_rsi_smoothing_type": preset.get("rsi_smoothing_type", values["w_rsi_smoothing_type"]),
        "w_rsi_ma_len": preset.get("rsi_ma_len", values["w_rsi_ma_len"]),
        "w_rsi_ma_type": preset.get("rsi_ma_type", values["w_rsi_ma_type"]),
        "w_rsi_min": preset.get("rsi_min", values["w_rsi_min"]),
        "w_rsi_ma_min": preset.get("rsi_ma_min", values["w_rsi_ma_min"]),
        "w_rsi_max": preset.get("rsi_max", values["w_rsi_max"]),
        "w_rsi_ma_max": preset.get("rsi_ma_max", values["w_rsi_ma_max"]),
        "w_use_rsi_relation": _to_checklist(preset.get("use_rsi_relation", True)),
        "w_rsi_relation": preset.get("rsi_relation", values["w_rsi_relation"]),
        "w_entry_band_mode": preset.get("entry_band_mode", values["w_entry_band_mode"]),
        "w_trade_direction": preset.get("trade_direction", values["w_trade_direction"]),
        "w_exit_channel": preset.get("exit_channel", values["w_exit_channel"]),
        "w_exit_level": preset.get("exit_level", values["w_exit_level"]),
        "w_trade_mode": preset.get("trade_mode", values["w_trade_mode"]),
        "w_use_stop": _to_checklist(preset.get("use_stop", True)),
        "w_stop_mode": preset.get("stop_mode", values["w_stop_mode"]),
        "w_stop_pct": float(preset.get("stop_pct", values["w_stop_pct"]) or values["w_stop_pct"]),
        "w_stop_atr_mult": float(preset.get("stop_atr_mult", values["w_stop_atr_mult"]) or values["w_stop_atr_mult"]),
        "w_use_trailing": _to_checklist(preset.get("use_trailing", False)),
        "w_trail_pct": float(preset.get("trail_pct", values["w_trail_pct"]) or values["w_trail_pct"]),
        "w_max_bars_in_trade": int(preset.get("max_bars_in_trade", values["w_max_bars_in_trade"])),
        "w_daily_loss_limit": float(preset.get("daily_loss_limit", values["w_daily_loss_limit"])),
        "w_risk_per_trade_pct": float(preset.get("risk_per_trade_pct", values["w_risk_per_trade_pct"])),
        "w_max_leverage": float(preset.get("max_leverage", values["w_max_leverage"]) or values["w_max_leverage"]),
        "w_maintenance_margin_pct": float(preset.get("maintenance_margin_pct", values["w_maintenance_margin_pct"]) or values["w_maintenance_margin_pct"]),
        "w_max_margin_utilization": float(preset.get("max_margin_utilization", values["w_max_margin_utilization"]) or values["w_max_margin_utilization"]),
    })

    if preset.get("max_margin_utilization") is not None:
        values["w_enable_max_margin_util"] = _to_checklist(True)

    return values


def _build_params(values: Dict[str, Any]) -> Dict[str, Any]:
    start_date = _parse_date(values.get("w_start_date"))
    end_date = _parse_date(values.get("w_end_date"))
    if start_date is None or end_date is None:
        return {}

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)

    use_stop = _from_checklist(values.get("w_use_stop"))
    trade_mode = values.get("w_trade_mode")

    max_leverage = _safe_float(values.get("w_max_leverage"), 5.0)
    maintenance_margin_pct = _safe_float(values.get("w_maintenance_margin_pct"), 0.5)
    max_margin_utilization = _safe_float(values.get("w_max_margin_utilization"), 70.0)
    enable_max_margin_util = _from_checklist(values.get("w_enable_max_margin_util"))

    if trade_mode != "Margin / Futures":
        max_leverage = None
        maintenance_margin_pct = None
        max_margin_utilization = None
    elif not use_stop or not enable_max_margin_util:
        max_margin_utilization = None

    params = {
        "exchange": values.get("w_exchange"),
        "symbol": values.get("w_symbol"),
        "timeframe": values.get("w_timeframe"),
        "start_ts": start_ts.isoformat(),
        "end_ts": end_ts.isoformat(),
        "bb_len": _safe_int(values.get("w_bb_len"), 20),
        "bb_std": _safe_float(values.get("w_bb_std"), 2.0),
        "bb_basis_type": values.get("w_bb_basis_type"),
        "kc_ema_len": _safe_int(values.get("w_kc_ema_len"), 20),
        "kc_atr_len": _safe_int(values.get("w_kc_atr_len"), 14),
        "kc_mult": _safe_float(values.get("w_kc_mult"), 2.0),
        "kc_mid_type": "ema" if _from_checklist(values.get("w_kc_mid_is_ema")) else "sma",
        "rsi_len_30m": _safe_int(values.get("w_rsi_len_30m"), 14),
        "rsi_ma_len": _safe_int(values.get("w_rsi_ma_len"), 10),
        "rsi_smoothing_type": values.get("w_rsi_smoothing_type"),
        "rsi_ma_type": values.get("w_rsi_ma_type"),
        "rsi_min": _safe_float(values.get("w_rsi_min"), 70.0),
        "rsi_ma_min": _safe_float(values.get("w_rsi_ma_min"), 70.0),
        "rsi_max": _safe_float(values.get("w_rsi_max"), 30.0),
        "rsi_ma_max": _safe_float(values.get("w_rsi_ma_max"), 30.0),
        "use_rsi_relation": _from_checklist(values.get("w_use_rsi_relation")),
        "rsi_relation": values.get("w_rsi_relation"),
        "entry_band_mode": values.get("w_entry_band_mode"),
        "trade_direction": values.get("w_trade_direction"),
        "exit_channel": values.get("w_exit_channel"),
        "exit_level": values.get("w_exit_level"),
        "cash": _safe_float(values.get("w_cash"), 10_000.0),
        "commission": _safe_float(values.get("w_commission"), 0.001),
        "trade_mode": trade_mode,
        "use_stop": use_stop,
        "stop_mode": values.get("w_stop_mode"),
        "stop_pct": _safe_float(values.get("w_stop_pct"), 2.0),
        "stop_atr_mult": _safe_float(values.get("w_stop_atr_mult"), 2.0),
        "use_trailing": _from_checklist(values.get("w_use_trailing")),
        "trail_pct": _safe_float(values.get("w_trail_pct"), 1.0),
        "max_bars_in_trade": _safe_int(values.get("w_max_bars_in_trade"), 100),
        "daily_loss_limit": _safe_float(values.get("w_daily_loss_limit"), 3.0),
        "risk_per_trade_pct": _safe_float(values.get("w_risk_per_trade_pct"), 1.0),
        "max_leverage": max_leverage,
        "maintenance_margin_pct": maintenance_margin_pct,
        "max_margin_utilization": max_margin_utilization,
    }

    return params


DEFAULTS = _default_ui_values()
BACKTEST_CACHE: Dict[tuple, Dict[str, Any]] = {}

EXCHANGE_OPTIONS = ["coinbase", "kraken", "gemini", "bitstamp", "binanceus"]
TIMEFRAME_OPTIONS = [
    "1m", "3m", "5m", "15m", "30m",
    "1h", "2h", "4h", "6h", "12h",
    "1d", "3d", "1w",
]
DATE_PRESET_LOOKBACK_DAYS = {
    "last_30d": 30,
    "last_90d": 90,
    "last_180d": 180,
    "last_1y": 365,
    "last_2y": 730,
}
DATE_PRESET_OPTIONS = [
    {"label": "Custom", "value": "manual"},
    {"label": "30D", "value": "last_30d"},
    {"label": "90D", "value": "last_90d"},
    {"label": "180D", "value": "last_180d"},
    {"label": "1Y", "value": "last_1y"},
    {"label": "2Y", "value": "last_2y"},
    {"label": "YTD", "value": "ytd"},
]
DATE_PRESET_LABELS = {
    "last_30d": "30D",
    "last_90d": "90D",
    "last_180d": "180D",
    "last_1y": "1Y",
    "last_2y": "2Y",
    "ytd": "YTD",
}
BB_BASIS_OPTIONS = ["sma", "ema"]
RSI_SMOOTHING_OPTIONS = ["ema", "sma", "rma"]
RSI_MA_OPTIONS = ["sma", "ema"]
RSI_RELATION_OPTIONS = [">=", ">", "<=", "<"]
ENTRY_BAND_OPTIONS = ["Either", "KC", "BB", "Both", "Squeeze"]
EXIT_CHANNEL_OPTIONS = ["BB", "KC"]
EXIT_LEVEL_OPTIONS = ["mid", "lower"]
TRADE_DIRECTION_OPTIONS = [
    {"label": "Short only", "value": "Short"},
    {"label": "Long only", "value": "Long"},
    {"label": "Long + Short (Blend)", "value": "Both"},
]
TRADE_MODE_OPTIONS = ["Simple (1x spot-style)", "Margin / Futures"]
STOP_MODE_OPTIONS = ["Fixed %", "ATR"]

PRESET_OPTIONS = ["Custom"] + list(STRATEGY_PRESETS.keys())
PRESET_LABELS = ["Custom (Manual Configuration)"] + [
    STRATEGY_PRESETS[key]["name"] for key in STRATEGY_PRESETS.keys()
]
PRESET_DIRECTION_DEFAULT = "Preset"
PRESET_DIRECTION_OPTIONS = [
    {"label": "Use preset direction", "value": "Preset"},
    {"label": "Short only", "value": "Short"},
    {"label": "Long only", "value": "Long"},
    {"label": "Long + Short (Blend)", "value": "Both"},
]
DATA_SCOPE_KEYS = {"w_exchange", "w_symbol", "w_timeframe", "w_start_date", "w_end_date"}

app = Dash(__name__)
app.title = "BB + KC + RSI Backtester"

app.layout = html.Div(
    [
        dcc.Store(id="store-results"),
        dcc.Store(id="store-params", data=_build_params(DEFAULTS)),
        dcc.Store(id="store-last-params"),
        dcc.Store(id="store-dirty", data=False),
        dcc.Store(id="store-selected-trade"),
        dcc.Store(id="store-load-strategy"),
        dcc.Store(id="store-opt-results"),
        dcc.Store(id="store-wf-results"),
        dcc.Store(id="store-leaderboard-selected"),
        dcc.Store(id="store-opt-job"),
        dcc.Store(id="store-disc-job"),
        dcc.Store(id="store-lb-job"),
        dcc.Store(id="store-lb-snapshot"),
        dcc.Store(id="store-pattern-job"),
        dcc.Store(id="store-pattern-rules"),
        dcc.Store(id="store-job-queue"),
        dcc.Store(id="store-job-selected"),
        dcc.Store(id="store-job-detail"),
        dcc.Store(id="store-job-events"),
        dcc.Interval(id="backend-poll", interval=1000, n_intervals=0),
        dcc.Download(id="download-diagnostics"),
        dcc.Download(id="download-optimization"),
        dcc.Download(id="download-walkforward"),
        dcc.Download(id="download-leaderboard"),
        html.Div(id="theme-sink", style={"display": "none"}),
        html.Div(
            [
                html.H2("BB + KC + RSI Long/Short Strategy", className="page-title"),
                html.Div(
                    [
                        html.Span("Settings", className="settings-label"),
                        dcc.Checklist(
                            id="theme-toggle",
                            options=[{"label": "Dark mode", "value": CHECK_ON}],
                            value=[],
                            persistence=True,
                            persistence_type="local",
                            className="theme-toggle",
                        ),
                    ],
                    className="header-controls",
                ),
            ],
            className="header-bar",
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Button("Run Backtest", id="run-backtest", n_clicks=0, className="btn-primary"),
                        dcc.Loading(
                            id="loading-backtest",
                            type="default",
                            children=html.Div(id="run-status", style={"marginTop": "8px"}),
                        ),
                        html.Hr(),
                        html.Label("Load Preset"),
                        dcc.Dropdown(
                            id="preset_selector",
                            options=[
                                {"label": label, "value": value}
                                for label, value in zip(PRESET_LABELS, PRESET_OPTIONS)
                            ],
                            value="Custom",
                            clearable=False,
                        ),
                        html.Label("Preset direction"),
                        dcc.Dropdown(
                            id="preset_direction",
                            options=PRESET_DIRECTION_OPTIONS,
                            value=PRESET_DIRECTION_DEFAULT,
                            clearable=False,
                        ),
                        html.Div(id="preset-info", style={"marginTop": "6px", "fontSize": "12px"}),
                        html.Div(
                            "Presets only change strategy settings. Data & timeframe stay as-is.",
                            className="helper-text",
                        ),
                        html.Div(
                            [
                                html.Div(id="preset-reco-text", className="preset-reco-text"),
                                html.Button(
                                    "Apply Recommended Data",
                                    id="apply-preset-data",
                                    n_clicks=0,
                                    className="btn-secondary",
                                ),
                            ],
                            id="preset-reco",
                            className="preset-reco",
                            style={"display": "none"},
                        ),
                        html.Hr(),
                        html.Details(
                            [
                                html.Summary("Data & Timeframe"),
                                html.Label("Exchange"),
                                dcc.Dropdown(
                                    id="w_exchange",
                                    options=[{"label": e, "value": e} for e in EXCHANGE_OPTIONS],
                                    value=DEFAULTS["w_exchange"],
                                    clearable=False,
                                ),
                                html.Label("Symbol"),
                                dcc.Input(
                                    id="w_symbol",
                                    type="text",
                                    value=DEFAULTS["w_symbol"],
                                ),
                                html.Label("Timeframe"),
                                dcc.RadioItems(
                                    id="w_timeframe",
                                    options=[{"label": t, "value": t} for t in TIMEFRAME_OPTIONS],
                                    value=DEFAULTS["w_timeframe"],
                                    labelStyle=INLINE_OPTION_STYLE,
                                ),
                                html.Label("Lookback window"),
                                dcc.RadioItems(
                                    id="w_date_preset",
                                    options=DATE_PRESET_OPTIONS,
                                    value=DEFAULTS["w_date_preset"],
                                    labelStyle=INLINE_OPTION_STYLE,
                                ),
                                html.Label("Custom range (UTC)"),
                                dcc.DatePickerRange(
                                    id="w_date_range",
                                    start_date=DEFAULTS["w_start_date"],
                                    end_date=DEFAULTS["w_end_date"],
                                ),
                            ],
                            open=True,
                            className="section-card",
                        ),
                        html.Details(
                            [
                                html.Summary("Indicators (BB, KC, RSI)"),
                                html.Label("BB length"),
                                dcc.Input(id="w_bb_len", type="number", value=DEFAULTS["w_bb_len"], min=5, max=200),
                                html.Label("BB std dev"),
                                dcc.Input(id="w_bb_std", type="number", value=DEFAULTS["w_bb_std"], min=1.0, max=4.0, step=0.1),
                                html.Label("BB basis type"),
                                dcc.Dropdown(
                                    id="w_bb_basis_type",
                                    options=[{"label": o, "value": o} for o in BB_BASIS_OPTIONS],
                                    value=DEFAULTS["w_bb_basis_type"],
                                    clearable=False,
                                ),
                                html.Label("KC EMA/SMA length (mid)"),
                                dcc.Input(id="w_kc_ema_len", type="number", value=DEFAULTS["w_kc_ema_len"], min=5, max=200),
                                html.Label("KC ATR length"),
                                dcc.Input(id="w_kc_atr_len", type="number", value=DEFAULTS["w_kc_atr_len"], min=5, max=200),
                                html.Label("KC ATR multiplier"),
                                dcc.Input(id="w_kc_mult", type="number", value=DEFAULTS["w_kc_mult"], min=0.5, max=5.0, step=0.1),
                                dcc.Checklist(
                                    id="w_kc_mid_is_ema",
                                    options=[{"label": "KC mid uses EMA", "value": CHECK_ON}],
                                    value=DEFAULTS["w_kc_mid_is_ema"],
                                ),
                                html.Label("RSI length"),
                                dcc.Input(id="w_rsi_len_30m", type="number", value=DEFAULTS["w_rsi_len_30m"], min=5, max=100),
                                html.Label("RSI smoothing"),
                                dcc.Dropdown(
                                    id="w_rsi_smoothing_type",
                                    options=[{"label": o, "value": o} for o in RSI_SMOOTHING_OPTIONS],
                                    value=DEFAULTS["w_rsi_smoothing_type"],
                                    clearable=False,
                                ),
                                html.Label("RSI MA length"),
                                dcc.Input(id="w_rsi_ma_len", type="number", value=DEFAULTS["w_rsi_ma_len"], min=2, max=100),
                                html.Label("RSI MA type"),
                                dcc.Dropdown(
                                    id="w_rsi_ma_type",
                                    options=[{"label": o, "value": o} for o in RSI_MA_OPTIONS],
                                    value=DEFAULTS["w_rsi_ma_type"],
                                    clearable=False,
                                ),
                            ],
                            className="section-card",
                        ),
                        html.Details(
                            [
                                html.Summary("Entry / Exit"),
                                html.Label("Trade direction"),
                                dcc.Dropdown(
                                    id="w_trade_direction",
                                    options=TRADE_DIRECTION_OPTIONS,
                                    value=DEFAULTS["w_trade_direction"],
                                    clearable=False,
                                ),
                                html.Label("RSI minimum (short entry)"),
                                dcc.Input(id="w_rsi_min", type="number", value=DEFAULTS["w_rsi_min"], min=0, max=100),
                                html.Label("RSI MA minimum (short entry)"),
                                dcc.Input(id="w_rsi_ma_min", type="number", value=DEFAULTS["w_rsi_ma_min"], min=0, max=100),
                                html.Label("RSI maximum (long entry)"),
                                dcc.Input(id="w_rsi_max", type="number", value=DEFAULTS["w_rsi_max"], min=0, max=100),
                                html.Label("RSI MA maximum (long entry)"),
                                dcc.Input(id="w_rsi_ma_max", type="number", value=DEFAULTS["w_rsi_ma_max"], min=0, max=100),
                                dcc.Checklist(
                                    id="w_use_rsi_relation",
                                    options=[{"label": "Use RSI vs RSI MA", "value": CHECK_ON}],
                                    value=DEFAULTS["w_use_rsi_relation"],
                                ),
                                html.Label("RSI relation"),
                                dcc.Dropdown(
                                    id="w_rsi_relation",
                                    options=[{"label": o, "value": o} for o in RSI_RELATION_OPTIONS],
                                    value=DEFAULTS["w_rsi_relation"],
                                    clearable=False,
                                ),
                                html.Label("Entry band mode"),
                                dcc.Dropdown(
                                    id="w_entry_band_mode",
                                    options=[{"label": o, "value": o} for o in ENTRY_BAND_OPTIONS],
                                    value=DEFAULTS["w_entry_band_mode"],
                                    clearable=False,
                                ),
                                html.Label("Exit channel"),
                                dcc.Dropdown(
                                    id="w_exit_channel",
                                    options=[{"label": o, "value": o} for o in EXIT_CHANNEL_OPTIONS],
                                    value=DEFAULTS["w_exit_channel"],
                                    clearable=False,
                                ),
                                html.Label("Exit level (short: mid/lower, long: mid/upper)"),
                                dcc.Dropdown(
                                    id="w_exit_level",
                                    options=[{"label": o, "value": o} for o in EXIT_LEVEL_OPTIONS],
                                    value=DEFAULTS["w_exit_level"],
                                    clearable=False,
                                ),
                            ],
                            className="section-card",
                        ),
                        html.Details(
                            [
                                html.Summary("Capital & Risk"),
                                html.Label("Starting cash"),
                                dcc.Input(id="w_cash", type="number", value=DEFAULTS["w_cash"], min=100, max=1_000_000_000, step=100),
                                html.Label("Commission (fraction)"),
                                dcc.Input(id="w_commission", type="number", value=DEFAULTS["w_commission"], min=0.0, max=0.01, step=0.0001),
                                html.Label("Trade mode"),
                                dcc.Dropdown(
                                    id="w_trade_mode",
                                    options=[{"label": o, "value": o} for o in TRADE_MODE_OPTIONS],
                                    value=DEFAULTS["w_trade_mode"],
                                    clearable=False,
                                ),
                                dcc.Checklist(
                                    id="w_use_stop",
                                    options=[{"label": "Enable stop loss", "value": CHECK_ON}],
                                    value=DEFAULTS["w_use_stop"],
                                ),
                                html.Label("Stop mode"),
                                dcc.Dropdown(
                                    id="w_stop_mode",
                                    options=[{"label": o, "value": o} for o in STOP_MODE_OPTIONS],
                                    value=DEFAULTS["w_stop_mode"],
                                    clearable=False,
                                ),
                                html.Label("Stop %"),
                                dcc.Input(id="w_stop_pct", type="number", value=DEFAULTS["w_stop_pct"], min=0.1, max=20.0, step=0.1),
                                html.Label("Stop ATR mult"),
                                dcc.Input(id="w_stop_atr_mult", type="number", value=DEFAULTS["w_stop_atr_mult"], min=0.1, max=10.0, step=0.1),
                                dcc.Checklist(
                                    id="w_use_trailing",
                                    options=[{"label": "Enable trailing stop", "value": CHECK_ON}],
                                    value=DEFAULTS["w_use_trailing"],
                                ),
                                html.Label("Trail %"),
                                dcc.Input(id="w_trail_pct", type="number", value=DEFAULTS["w_trail_pct"], min=0.1, max=10.0, step=0.1),
                                html.Label("Max bars in trade"),
                                dcc.Input(id="w_max_bars_in_trade", type="number", value=DEFAULTS["w_max_bars_in_trade"], min=1, max=1000, step=1),
                                html.Label("Daily loss limit %"),
                                dcc.Input(id="w_daily_loss_limit", type="number", value=DEFAULTS["w_daily_loss_limit"], min=0.0, max=50.0, step=0.5),
                                html.Label("Risk per trade % of equity"),
                                dcc.Input(id="w_risk_per_trade_pct", type="number", value=DEFAULTS["w_risk_per_trade_pct"], min=0.1, max=100.0, step=0.5),
                            ],
                            className="section-card",
                        ),
                        html.Details(
                            [
                                html.Summary("Margin Settings"),
                                html.Label("Max leverage"),
                                dcc.Input(id="w_max_leverage", type="number", value=DEFAULTS["w_max_leverage"], min=1.0, max=125.0, step=0.5),
                                html.Label("Maintenance margin %"),
                                dcc.Input(id="w_maintenance_margin_pct", type="number", value=DEFAULTS["w_maintenance_margin_pct"], min=0.1, max=50.0, step=0.1),
                                dcc.Checklist(
                                    id="w_enable_max_margin_util",
                                    options=[{"label": "Limit margin utilization?", "value": CHECK_ON}],
                                    value=DEFAULTS["w_enable_max_margin_util"],
                                ),
                                html.Label("Max margin utilization %"),
                                dcc.Input(id="w_max_margin_utilization", type="number", value=DEFAULTS["w_max_margin_utilization"], min=10.0, max=100.0, step=5.0),
                            ],
                            className="section-card",
                        ),
                    ],
                    style={
                        "flex": "0 0 320px",
                        "padding": "12px",
                        "border": "1px solid #ddd",
                        "borderRadius": "8px",
                        "maxHeight": "90vh",
                        "overflowY": "auto",
                    },
                    className="sidebar",
                ),
                html.Div(
                    [
                        html.Div(id="dirty-warning", style={"marginBottom": "8px", "color": "#b45309"}),
                        dcc.Tabs(
                            id="main-tabs",
                            value="dashboard",
                            children=[
                                dcc.Tab(
                                    label="Dashboard",
                                    value="dashboard",
                                    children=[
                                        html.Div(id="dashboard-message", style={"margin": "8px 0"}),
                                        html.Div(id="dashboard-metrics", className="metric-grid"),
                                        html.Div(
                                            [
                                                dcc.Checklist(
                                                    id="show_candles",
                                                    options=[{"label": "Show Candlesticks", "value": CHECK_ON}],
                                                    value=DEFAULTS["show_candles"],
                                                ),
                                                dcc.Checklist(
                                                    id="lock_rsi_y",
                                                    options=[{"label": "Lock RSI Y-axis (0-100)", "value": CHECK_ON}],
                                                    value=DEFAULTS["lock_rsi_y"],
                                                ),
                                            ],
                                            style={"display": "flex", "gap": "16px", "margin": "8px 0"},
                                        ),
                                        dcc.Loading(
                                            dcc.Graph(id="backtest-figure"),
                                            type="default",
                                        ),
                                    ],
                                ),
                                dcc.Tab(
                                    label="Trades & Diagnostics",
                                    value="trades",
                                    children=[
                                        html.Div(id="trades-message", style={"margin": "8px 0"}),
                                        dcc.Loading(
                                            type="default",
                                            children=[
                                                dash_table.DataTable(
                                                    id="trades-table",
                                                    data=[],
                                                    columns=[],
                                                    row_selectable="single",
                                                    page_size=15,
                                                    sort_action="native",
                                                    filter_action="native",
                                                    style_table={"overflowX": "auto"},
                                                ),
                                            ],
                                        ),
                                        html.Hr(),
                                        html.Div(id="diagnostics-message", style={"margin": "8px 0"}),
                                        dcc.Loading(
                                            type="default",
                                            children=[
                                                dash_table.DataTable(
                                                    id="diagnostics-table",
                                                    data=[],
                                                    columns=[],
                                                    page_size=15,
                                                    sort_action="native",
                                                    filter_action="native",
                                                    style_table={"overflowX": "auto", "maxHeight": "400px", "overflowY": "auto"},
                                                ),
                                            ],
                                        ),
                                        html.Button("Download Diagnostics", id="download-diagnostics-btn", n_clicks=0, className="btn-secondary", style={"marginTop": "8px"}),
                                    ],
                                ),
                                dcc.Tab(
                                    label="Analysis & Discovery",
                                    value="analysis",
                                    children=[
                                        dcc.Tabs(
                                            id="analysis-tabs",
                                            value="optimization",
                                            children=[
                                                dcc.Tab(
                                                    label="Parameter Optimization",
                                                    value="optimization",
                                                    children=[
                                                        html.P(
                                                            "Run a grid search to find optimal parameter combinations."
                                                        ),
                                                        html.Div(
                                                            [
                                                                html.Div(
                                                                    [
                                                                        html.H4("Parameter Ranges"),
                                                                        html.Label("RSI Minimum Range"),
                                                                        dcc.RangeSlider(
                                                                            id="opt_rsi_range",
                                                                            min=60,
                                                                            max=85,
                                                                            value=[65, 75],
                                                                            step=1,
                                                                            tooltip={"placement": "bottom"},
                                                                        ),
                                                                        html.Label("Stop Loss Range (%, or ATR mult if stop mode = ATR)"),
                                                                        dcc.RangeSlider(
                                                                            id="opt_stop_range",
                                                                            min=0.5,
                                                                            max=5.0,
                                                                            value=[1.5, 3.0],
                                                                            step=0.5,
                                                                            tooltip={"placement": "bottom"},
                                                                        ),
                                                                        html.Label("Band Multiplier Range"),
                                                                        dcc.RangeSlider(
                                                                            id="opt_band_range",
                                                                            min=1.5,
                                                                            max=3.0,
                                                                            value=[1.8, 2.2],
                                                                            step=0.1,
                                                                            tooltip={"placement": "bottom"},
                                                                        ),
                                                                    ],
                                                                    style={"flex": "1", "minWidth": "280px"},
                                                                ),
                                                                html.Div(
                                                                    [
                                                                        html.H4("Optimization Settings"),
                                                                        html.Label("Validation mode"),
                                                                        dcc.RadioItems(
                                                                            id="opt_validation_mode",
                                                                            options=[
                                                                                {"label": "In-sample (single window)", "value": "single"},
                                                                                {"label": "Walk-forward (train/test)", "value": "walk"},
                                                                            ],
                                                                            value="single",
                                                                        ),
                                                                        html.Label("Optimize for"),
                                                                        dcc.Dropdown(
                                                                            id="opt_metric",
                                                                            options=[
                                                                                {"label": "profit_factor", "value": "profit_factor"},
                                                                                {"label": "sharpe_ratio", "value": "sharpe_ratio"},
                                                                                {"label": "sortino_ratio", "value": "sortino_ratio"},
                                                                                {"label": "win_rate", "value": "win_rate"},
                                                                                {"label": "total_equity_return_pct", "value": "total_equity_return_pct"},
                                                                            ],
                                                                            value="profit_factor",
                                                                            clearable=False,
                                                                        ),
                                                                        html.Label("Grid Steps"),
                                                                        dcc.Slider(
                                                                            id="opt_grid_steps",
                                                                            min=2,
                                                                            max=5,
                                                                            value=3,
                                                                            step=1,
                                                                            marks={2: "2", 3: "3", 4: "4", 5: "5"},
                                                                        ),
                                                                        dcc.Checklist(
                                                                            id="opt_include_entry_modes",
                                                                            options=[{"label": "Test all entry band modes", "value": CHECK_ON}],
                                                                            value=_to_checklist(True),
                                                                        ),
                                                                        dcc.Checklist(
                                                                            id="opt_include_exit_levels",
                                                                            options=[{"label": "Test both exit levels", "value": CHECK_ON}],
                                                                            value=[],
                                                                        ),
                                                                        html.Label("Minimum trades required"),
                                                                        dcc.Input(id="opt_min_trades", type="number", value=5, min=1, max=50, step=1),
                                                                        html.H4("Hard Constraints (optional)"),
                                                                        html.Label("Min Profit Factor (0 = off)"),
                                                                        dcc.Input(id="opt_min_pf", type="number", value=0.0, min=0.0, max=10.0, step=0.1),
                                                                        html.Label("Min Win Rate % (0 = off)"),
                                                                        dcc.Input(id="opt_min_wr", type="number", value=0.0, min=0.0, max=100.0, step=1.0),
                                                                        html.Label("Max Drawdown % (0 = off)"),
                                                                        dcc.Input(id="opt_max_dd", type="number", value=0.0, min=0.0, max=100.0, step=1.0),
                                                                        html.Label("Min Total Return % (0 = off)"),
                                                                        dcc.Input(id="opt_min_total_ret", type="number", value=0.0, min=-100.0, max=1000.0, step=1.0),
                                                                        html.Div(
                                                                            [
                                                                                html.H4("Walk-forward settings"),
                                                                                html.Label("Train window (days)"),
                                                                                dcc.Input(id="wf_train_days", type="number", value=180, min=30, max=3650, step=30),
                                                                                html.Label("Test window (days)"),
                                                                                dcc.Input(id="wf_test_days", type="number", value=30, min=7, max=365, step=7),
                                                                                html.Label("Max folds (0 = all)"),
                                                                                dcc.Input(id="wf_max_folds", type="number", value=8, min=0, max=200, step=1),
                                                                            ],
                                                                            style={"marginTop": "12px"},
                                                                        ),
                                                                    ],
                                                                    style={"flex": "1", "minWidth": "280px"},
                                                                ),
                                                            ],
                                                            style={"display": "flex", "gap": "24px", "flexWrap": "wrap"},
                                                        ),
                                                        html.Div(id="opt-estimate", style={"margin": "8px 0"}),
                                                        html.Button("Run Optimization", id="opt-run-btn", n_clicks=0, className="btn-primary"),
                                                        dcc.Loading(
                                                            id="loading-optimization",
                                                            type="default",
                                                            children=html.Div(id="opt-status", style={"margin": "8px 0"}),
                                                        ),
                                                        html.H4("Top Configurations"),
                                                        dash_table.DataTable(
                                                            id="opt-results-table",
                                                            data=[],
                                                            columns=[],
                                                            page_size=20,
                                                            sort_action="native",
                                                            filter_action="native",
                                                            style_table={"overflowX": "auto"},
                                                        ),
                                                        html.Div(id="opt-summary", style={"marginTop": "12px"}),
                                                        html.Button("Download Optimization Results", id="opt-download-btn", n_clicks=0, className="btn-secondary", style={"marginTop": "8px"}),
                                                        html.H4("Walk-forward Results"),
                                                        dash_table.DataTable(
                                                            id="wf-results-table",
                                                            data=[],
                                                            columns=[],
                                                            page_size=20,
                                                            sort_action="native",
                                                            filter_action="native",
                                                            style_table={"overflowX": "auto"},
                                                        ),
                                                        html.Div(id="wf-summary", style={"marginTop": "12px"}),
                                                        html.Button("Download Walk-forward Results", id="wf-download-btn", n_clicks=0, className="btn-secondary", style={"marginTop": "8px"}),
                                                    ],
                                                ),
                                                dcc.Tab(
                                                    label="Strategy Discovery",
                                                    value="discovery",
                                                    children=[
                                                        html.P(
                                                            "Automatically discover winning strategies by testing parameter combinations."
                                                        ),
                                                        html.Div(
                                                            [
                                                                html.Div(
                                                                    [
                                                                        html.H4("Performance"),
                                                                        html.Label("Parallel Workers"),
                                                                        dcc.Slider(
                                                                            id="disc_n_workers",
                                                                            min=1,
                                                                            max=max(1, get_cpu_count()),
                                                                            value=max(1, get_cpu_count() - 1),
                                                                            step=1,
                                                                            tooltip={"placement": "bottom"},
                                                                        ),
                                                                        dcc.Checklist(
                                                                            id="disc_use_parallel",
                                                                            options=[{"label": "Enable Parallel Processing", "value": CHECK_ON}],
                                                                            value=_to_checklist(True),
                                                                        ),
                                                                        dcc.Checklist(
                                                                            id="disc_skip_tested",
                                                                            options=[{"label": "Skip tested combinations", "value": CHECK_ON}],
                                                                            value=_to_checklist(True),
                                                                        ),
                                                                    ],
                                                                    style={"flex": "1", "minWidth": "260px"},
                                                                ),
                                                                html.Div(
                                                                    [
                                                                        html.H4("Strategy Parameters"),
                                                                        html.Label("RSI Min Range"),
                                                                        dcc.RangeSlider(
                                                                            id="disc_rsi_range",
                                                                            min=60,
                                                                            max=82,
                                                                            value=[68, 74],
                                                                            step=1,
                                                                            tooltip={"placement": "bottom"},
                                                                        ),
                                                                        html.Label("RSI MA Min Range"),
                                                                        dcc.RangeSlider(
                                                                            id="disc_rsi_ma_range",
                                                                            min=58,
                                                                            max=80,
                                                                            value=[66, 72],
                                                                            step=1,
                                                                            tooltip={"placement": "bottom"},
                                                                        ),
                                                                        html.Label("Band Mult Range"),
                                                                        dcc.RangeSlider(
                                                                            id="disc_band_range",
                                                                            min=1.5,
                                                                            max=2.8,
                                                                            value=[1.9, 2.1],
                                                                            step=0.1,
                                                                            tooltip={"placement": "bottom"},
                                                                        ),
                                                                    ],
                                                                    style={"flex": "1", "minWidth": "260px"},
                                                                ),
                                                                html.Div(
                                                                    [
                                                                        html.H4("Margin / Risk"),
                                                                        html.Label("Leverage"),
                                                                        dcc.Dropdown(
                                                                            id="disc_leverage_options",
                                                                            options=[{"label": str(v), "value": v} for v in [2.0, 3.0, 5.0, 10.0, 20.0]],
                                                                            value=[2.0, 5.0, 10.0],
                                                                            multi=True,
                                                                        ),
                                                                        html.Label("Risk %"),
                                                                        dcc.Dropdown(
                                                                            id="disc_risk_options",
                                                                            options=[{"label": str(v), "value": v} for v in [0.5, 1.0, 1.5, 2.0, 3.0]],
                                                                            value=[0.5, 1.0, 2.0],
                                                                            multi=True,
                                                                        ),
                                                                        dcc.Checklist(
                                                                            id="disc_include_atr",
                                                                            options=[{"label": "Include ATR Stops", "value": CHECK_ON}],
                                                                            value=_to_checklist(True),
                                                                        ),
                                                                        dcc.Checklist(
                                                                            id="disc_include_trailing",
                                                                            options=[{"label": "Include Trailing", "value": CHECK_ON}],
                                                                            value=_to_checklist(True),
                                                                        ),
                                                                    ],
                                                                    style={"flex": "1", "minWidth": "260px"},
                                                                ),
                                                                html.Div(
                                                                    [
                                                                        html.H4("Win Criteria"),
                                                                        html.Label("Min Return %"),
                                                                        dcc.Input(id="disc_min_return", type="number", value=0.0, min=-100.0, max=100.0),
                                                                        html.Label("Max Drawdown %"),
                                                                        dcc.Input(id="disc_max_dd", type="number", value=20.0, min=1.0, max=50.0),
                                                                        html.Label("Min Trades"),
                                                                        dcc.Input(id="disc_min_trades", type="number", value=10, min=5, max=100),
                                                                        html.Label("Min Profit Factor"),
                                                                        dcc.Input(id="disc_min_pf", type="number", value=1.0, min=0.5, max=3.0, step=0.1),
                                                                    ],
                                                                    style={"flex": "1", "minWidth": "240px"},
                                                                ),
                                                            ],
                                                            style={"display": "flex", "gap": "24px", "flexWrap": "wrap"},
                                                        ),
                                                        html.Div(id="disc-estimate", style={"margin": "8px 0"}),
                                                        html.Button("Run Strategy Discovery", id="disc-run-btn", n_clicks=0, className="btn-primary"),
                                                        dcc.Loading(
                                                            id="loading-discovery",
                                                            type="default",
                                                            children=html.Div(id="disc-status", style={"margin": "8px 0"}),
                                                        ),
                                                        html.Div(id="disc-results", style={"margin": "8px 0"}),
                                                    ],
                                                ),
                                                dcc.Tab(
                                                    label="Leaderboard",
                                                    value="leaderboard",
                                                    children=[
                                                        html.H4("Winning Strategies Leaderboard"),
                                                        html.Div(id="lb-stats", style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
                                                        html.Div(
                                                            [
                                                                html.Label("Sort by"),
                                                                dcc.Dropdown(
                                                                    id="lb_sort_by",
                                                                    options=[
                                                                        {"label": "total_return", "value": "total_return"},
                                                                        {"label": "profit_factor", "value": "profit_factor"},
                                                                        {"label": "sharpe_ratio", "value": "sharpe_ratio"},
                                                                        {"label": "win_rate", "value": "win_rate"},
                                                                        {"label": "max_drawdown", "value": "max_drawdown"},
                                                                    ],
                                                                    value="total_return",
                                                                    clearable=False,
                                                                ),
                                                                html.Label("Show top"),
                                                                dcc.Dropdown(
                                                                    id="lb_top_n",
                                                                    options=[{"label": str(v), "value": v} for v in [10, 25, 50, 100]],
                                                                    value=10,
                                                                    clearable=False,
                                                                ),
                                                        html.Button("Refresh", id="lb-refresh-btn", n_clicks=0, className="btn-secondary"),
                                                            ],
                                                            style={"display": "flex", "gap": "12px", "alignItems": "center"},
                                                        ),
                                                        dcc.Loading(
                                                            id="loading-leaderboard",
                                                            type="default",
                                                            children=html.Div(id="lb-status", style={"margin": "8px 0"}),
                                                        ),
                                                        html.Div(id="lb-updated", style={"margin": "8px 0", "fontSize": "12px"}),
                                                        dcc.Loading(
                                                            id="loading-leaderboard-table",
                                                            type="default",
                                                            children=html.Div([
                                                                dash_table.DataTable(
                                                                    id="lb-table",
                                                                    data=[],
                                                                    columns=[],
                                                                    row_selectable="single",
                                                                    page_size=15,
                                                                    sort_action="native",
                                                                    filter_action="native",
                                                                    style_table={"overflowX": "auto"},
                                                                    hidden_columns=["FullHash", "ParamsJson"],
                                                                ),
                                                                html.Div(id="lb-details", style={"marginTop": "12px"}),
                                                            ]),
                                                        ),
                                                        html.Button("Load This Strategy", id="lb-load-btn", n_clicks=0, className="btn-primary"),
                                                        html.Button("Export Leaderboard CSV", id="lb-download-btn", n_clicks=0, className="btn-secondary", style={"marginLeft": "8px"}),
                                                    ],
                                                ),
                                                dcc.Tab(
                                                    label="Pattern Recognition",
                                                    value="patterns",
                                                    children=[
                                                        html.H4("Discovered Patterns & Rules"),
                                                        html.Button("Analyze Patterns", id="patterns-btn", n_clicks=0, className="btn-primary"),
                                                        dcc.Loading(
                                                            id="loading-patterns",
                                                            type="default",
                                                            children=html.Div(id="patterns-status", style={"margin": "8px 0"}),
                                                        ),
                                                        dcc.Loading(
                                                            id="loading-patterns-summary",
                                                            type="default",
                                                            children=dcc.Markdown(id="patterns-summary"),
                                                        ),
                                                    ],
                                                ),
                                                dcc.Tab(
                                                    label="Job Queue",
                                                    value="jobs",
                                                    children=[
                                                        html.H4("Background Job Queue"),
                                                        html.Div(
                                                            [
                                                                html.Label("Status"),
                                                                dcc.Dropdown(
                                                                    id="jobs-status-filter",
                                                                    options=[
                                                                        {"label": "all", "value": "all"},
                                                                        {"label": "queued", "value": "queued"},
                                                                        {"label": "running", "value": "running"},
                                                                        {"label": "succeeded", "value": "succeeded"},
                                                                        {"label": "failed", "value": "failed"},
                                                                        {"label": "canceled", "value": "canceled"},
                                                                    ],
                                                                    value="all",
                                                                    clearable=False,
                                                                    style={"minWidth": "160px"},
                                                                ),
                                                                html.Label("Type"),
                                                                dcc.Dropdown(
                                                                    id="jobs-type-filter",
                                                                    options=[
                                                                        {"label": "all", "value": "all"},
                                                                        {"label": "optimize", "value": "optimize"},
                                                                        {"label": "discover", "value": "discover"},
                                                                        {"label": "leaderboard_refresh", "value": "leaderboard_refresh"},
                                                                        {"label": "patterns_refresh", "value": "patterns_refresh"},
                                                                        {"label": "prices_ingest", "value": "prices_ingest"},
                                                                    ],
                                                                    value="all",
                                                                    clearable=False,
                                                                    style={"minWidth": "220px"},
                                                                ),
                                                                html.Label("Limit"),
                                                                dcc.Dropdown(
                                                                    id="jobs-limit",
                                                                    options=[{"label": str(v), "value": v} for v in [25, 50, 100, 200]],
                                                                    value=50,
                                                                    clearable=False,
                                                                    style={"minWidth": "120px"},
                                                                ),
                                                                dcc.Checklist(
                                                                    id="jobs-auto-refresh",
                                                                    options=[{"label": "Auto-refresh", "value": CHECK_ON}],
                                                                    value=_to_checklist(True),
                                                                    style={"marginTop": "22px"},
                                                                ),
                                                                html.Button(
                                                                    "Refresh",
                                                                    id="jobs-refresh-btn",
                                                                    n_clicks=0,
                                                                    className="btn-secondary",
                                                                    style={"marginTop": "22px"},
                                                                ),
                                                            ],
                                                            style={"display": "flex", "gap": "12px", "alignItems": "center", "flexWrap": "wrap"},
                                                        ),
                                                        dcc.Loading(
                                                            id="loading-jobs-status",
                                                            type="default",
                                                            children=html.Div(id="jobs-status", style={"margin": "8px 0"}),
                                                        ),
                                                        html.H5("Workers"),
                                                        html.Div(id="workers-summary", style={"margin": "8px 0"}),
                                                        dcc.Loading(
                                                            id="loading-jobs-tables",
                                                            type="default",
                                                            children=html.Div([
                                                                dash_table.DataTable(
                                                                    id="workers-table",
                                                                    data=[],
                                                                    columns=[],
                                                                    page_size=8,
                                                                    sort_action="native",
                                                                    style_table={"overflowX": "auto"},
                                                                ),
                                                                dash_table.DataTable(
                                                                    id="jobs-table",
                                                                    data=[],
                                                                    columns=[],
                                                                    row_selectable="single",
                                                                    page_size=15,
                                                                    sort_action="native",
                                                                    filter_action="native",
                                                                    style_table={"overflowX": "auto"},
                                                                ),
                                                            ]),
                                                        ),
                                                        html.Hr(),
                                                        html.Div(
                                                            [
                                                                html.Button("Refresh Details", id="jobs-detail-refresh-btn", n_clicks=0, className="btn-secondary"),
                                                                html.Button(
                                                                    "Cancel Selected",
                                                                    id="jobs-cancel-btn",
                                                                    n_clicks=0,
                                                                    className="btn-secondary",
                                                                    style={"marginLeft": "8px"},
                                                                ),
                                                            ],
                                                            style={"display": "flex", "alignItems": "center"},
                                                        ),
                                                        dcc.Loading(
                                                            id="loading-job-details",
                                                            type="default",
                                                            children=html.Div([
                                                                html.Div(id="jobs-detail-status", style={"margin": "8px 0"}),
                                                                html.Div(id="jobs-detail-summary", style={"margin": "8px 0", "fontSize": "12px"}),
                                                                dcc.Markdown(id="jobs-detail-json"),
                                                                html.H4("Events"),
                                                                dash_table.DataTable(
                                                                    id="jobs-events-table",
                                                                    data=[],
                                                                    columns=[],
                                                                    page_size=10,
                                                                    sort_action="native",
                                                                    style_table={"overflowX": "auto", "maxHeight": "260px", "overflowY": "auto"},
                                                                ),
                                                            ]),
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                    style={"flex": "1", "padding": "12px"},
                    className="main-panel",
                ),
            ],
            style={"display": "flex", "gap": "16px"},
            className="app-shell",
        ),
    ],
    style={"padding": "16px"},
    className="app-root",
    id="app-root",
)

INPUT_FIELDS = [
    {"id": "w_exchange", "prop": "value", "key": "w_exchange"},
    {"id": "w_symbol", "prop": "value", "key": "w_symbol"},
    {"id": "w_timeframe", "prop": "value", "key": "w_timeframe"},
    {"id": "w_date_range", "prop": "start_date", "key": "w_start_date"},
    {"id": "w_date_range", "prop": "end_date", "key": "w_end_date"},
    {"id": "w_bb_len", "prop": "value", "key": "w_bb_len"},
    {"id": "w_bb_std", "prop": "value", "key": "w_bb_std"},
    {"id": "w_bb_basis_type", "prop": "value", "key": "w_bb_basis_type"},
    {"id": "w_kc_ema_len", "prop": "value", "key": "w_kc_ema_len"},
    {"id": "w_kc_atr_len", "prop": "value", "key": "w_kc_atr_len"},
    {"id": "w_kc_mult", "prop": "value", "key": "w_kc_mult"},
    {"id": "w_kc_mid_is_ema", "prop": "value", "key": "w_kc_mid_is_ema"},
    {"id": "w_rsi_len_30m", "prop": "value", "key": "w_rsi_len_30m"},
    {"id": "w_rsi_smoothing_type", "prop": "value", "key": "w_rsi_smoothing_type"},
    {"id": "w_rsi_ma_len", "prop": "value", "key": "w_rsi_ma_len"},
    {"id": "w_rsi_ma_type", "prop": "value", "key": "w_rsi_ma_type"},
    {"id": "w_trade_direction", "prop": "value", "key": "w_trade_direction"},
    {"id": "w_rsi_min", "prop": "value", "key": "w_rsi_min"},
    {"id": "w_rsi_ma_min", "prop": "value", "key": "w_rsi_ma_min"},
    {"id": "w_rsi_max", "prop": "value", "key": "w_rsi_max"},
    {"id": "w_rsi_ma_max", "prop": "value", "key": "w_rsi_ma_max"},
    {"id": "w_use_rsi_relation", "prop": "value", "key": "w_use_rsi_relation"},
    {"id": "w_rsi_relation", "prop": "value", "key": "w_rsi_relation"},
    {"id": "w_entry_band_mode", "prop": "value", "key": "w_entry_band_mode"},
    {"id": "w_exit_channel", "prop": "value", "key": "w_exit_channel"},
    {"id": "w_exit_level", "prop": "value", "key": "w_exit_level"},
    {"id": "w_cash", "prop": "value", "key": "w_cash"},
    {"id": "w_commission", "prop": "value", "key": "w_commission"},
    {"id": "w_trade_mode", "prop": "value", "key": "w_trade_mode"},
    {"id": "w_use_stop", "prop": "value", "key": "w_use_stop"},
    {"id": "w_stop_mode", "prop": "value", "key": "w_stop_mode"},
    {"id": "w_stop_pct", "prop": "value", "key": "w_stop_pct"},
    {"id": "w_stop_atr_mult", "prop": "value", "key": "w_stop_atr_mult"},
    {"id": "w_use_trailing", "prop": "value", "key": "w_use_trailing"},
    {"id": "w_trail_pct", "prop": "value", "key": "w_trail_pct"},
    {"id": "w_max_bars_in_trade", "prop": "value", "key": "w_max_bars_in_trade"},
    {"id": "w_daily_loss_limit", "prop": "value", "key": "w_daily_loss_limit"},
    {"id": "w_risk_per_trade_pct", "prop": "value", "key": "w_risk_per_trade_pct"},
    {"id": "w_max_leverage", "prop": "value", "key": "w_max_leverage"},
    {"id": "w_maintenance_margin_pct", "prop": "value", "key": "w_maintenance_margin_pct"},
    {"id": "w_enable_max_margin_util", "prop": "value", "key": "w_enable_max_margin_util"},
    {"id": "w_max_margin_utilization", "prop": "value", "key": "w_max_margin_utilization"},
]

# Filtered list excluding data scope fields for sync_inputs_from_preset callback
STRATEGY_INPUT_FIELDS = [field for field in INPUT_FIELDS if field["key"] not in DATA_SCOPE_KEYS]


def _values_from_args(values: List[Any]) -> Dict[str, Any]:
    return {field["key"]: value for field, value in zip(INPUT_FIELDS, values)}


app.clientside_callback(
    """
    function(toggleValue) {
        var isDark = Array.isArray(toggleValue) && toggleValue.indexOf("on") !== -1;
        var selected = isDark ? "dark" : "light";
        document.documentElement.setAttribute("data-theme", selected);
        return "";
    }
    """,
    Output("theme-sink", "children"),
    Input("theme-toggle", "value"),
)


@app.callback(
    [Output(field["id"], field["prop"]) for field in STRATEGY_INPUT_FIELDS]
    + [Output("preset-info", "children"), Output("preset_selector", "value")],
    [Input("preset_selector", "value"), Input("store-load-strategy", "data"), Input("preset_direction", "value")],
    prevent_initial_call=True,
)
def sync_inputs_from_preset(preset_value, loaded_strategy, preset_direction):
    """Sync strategy parameters (excluding data scope fields) from preset selection."""
    if not callback_context.triggered:
        raise PreventUpdate

    trigger_id = callback_context.triggered[0]["prop_id"].split(".")[0]
    if trigger_id == "store-load-strategy" and loaded_strategy:
        resolved = apply_preset_direction(loaded_strategy)
        ui_values = _preset_to_ui_values(resolved)
        output_values = [ui_values.get(field["key"]) for field in STRATEGY_INPUT_FIELDS]
        info = "Loaded strategy parameters."
        return output_values + [info, "Custom"]

    if preset_value == "Custom":
        info = "Custom (Manual Configuration)"
        return [no_update] * len(STRATEGY_INPUT_FIELDS) + [info, no_update]

    preset = STRATEGY_PRESETS.get(preset_value)
    if not preset:
        return [no_update] * len(STRATEGY_INPUT_FIELDS) + [no_update, no_update]

    direction_override = _resolve_preset_direction(preset_direction)
    resolved = apply_preset_direction(preset, direction_override)
    ui_values = _preset_to_ui_values(resolved)
    output_values = [ui_values.get(field["key"]) for field in STRATEGY_INPUT_FIELDS]
    info = _build_preset_info(resolved, direction_override)
    return output_values + [info, preset_value]


@app.callback(
    [Output("preset-reco", "style"), Output("preset-reco-text", "children"), Output("apply-preset-data", "disabled")],
    Input("preset_selector", "value"),
)
def update_preset_recommendation(preset_value):
    if not preset_value or preset_value == "Custom":
        return {"display": "none"}, "", True

    preset = STRATEGY_PRESETS.get(preset_value)
    if not preset:
        return {"display": "none"}, "", True

    text = _format_preset_recommendation(preset)
    if not text:
        return {"display": "none"}, "", True

    return {"display": "flex"}, text, False


@app.callback(
    [
        Output("w_timeframe", "value"),
        Output("w_date_preset", "value"),
        Output("w_date_range", "start_date"),
        Output("w_date_range", "end_date"),
    ],
    [
        Input("apply-preset-data", "n_clicks"),
        Input("w_date_preset", "value"),
        Input("w_date_range", "end_date"),
    ],
    [
        State("preset_selector", "value"),
        State("w_timeframe", "value"),
        State("w_date_preset", "value"),
    ],
    prevent_initial_call=True,
)
def manage_data_scope_fields(apply_clicks, date_preset_input, end_date_input, preset_value, current_timeframe, current_date_preset):
    """Consolidated callback to manage w_timeframe, w_date_preset, and w_date_range fields."""
    if not callback_context.triggered:
        raise PreventUpdate

    trigger_id = callback_context.triggered[0]["prop_id"].split(".")[0]

    # Handle "Apply Recommended Data Scope" button click
    if trigger_id == "apply-preset-data":
        if not apply_clicks or not preset_value or preset_value == "Custom":
            raise PreventUpdate

        preset = STRATEGY_PRESETS.get(preset_value)
        if not preset:
            raise PreventUpdate

        timeframe = preset.get("recommended_timeframe")
        lookback_days = preset.get("recommended_lookback_days")
        if not timeframe and not lookback_days:
            raise PreventUpdate

        date_preset = _preset_key_for_lookback(int(lookback_days)) if lookback_days else None
        end_date = dt.datetime.now(dt.UTC).date()

        # Calculate start_date based on date_preset
        start_date = no_update
        if date_preset:
            anchor_date = end_date
            resolved = _date_range_from_preset(date_preset, anchor_date)
            if resolved:
                start_date, _ = resolved

        return (
            timeframe or no_update,
            date_preset or no_update,
            start_date,
            end_date if date_preset else no_update,
        )

    # Handle date preset changes (recalculate start_date)
    if trigger_id == "w_date_preset":
        if not date_preset_input or date_preset_input == "manual":
            raise PreventUpdate

        anchor_date = _parse_date(end_date_input) or dt.datetime.now(dt.UTC).date()
        resolved = _date_range_from_preset(date_preset_input, anchor_date)
        if not resolved:
            raise PreventUpdate

        start_date, _ = resolved
        return no_update, no_update, start_date, no_update

    # Handle end_date changes (recalculate start_date if preset is active)
    if trigger_id == "w_date_range":
        if not current_date_preset or current_date_preset == "manual":
            raise PreventUpdate

        anchor_date = _parse_date(end_date_input) or dt.datetime.now(dt.UTC).date()
        resolved = _date_range_from_preset(current_date_preset, anchor_date)
        if not resolved:
            raise PreventUpdate

        start_date, _ = resolved
        return no_update, no_update, start_date, no_update

    raise PreventUpdate


@app.callback(
    [Output("store-results", "data"), Output("store-last-params", "data"), Output("store-dirty", "data", allow_duplicate=True), Output("store-selected-trade", "data"), Output("run-status", "children")],
    Input("run-backtest", "n_clicks"),
    State("store-params", "data"),
    prevent_initial_call=True,
)
def run_backtest_callback(n_clicks, params):
    if not n_clicks:
        raise PreventUpdate

    if not params:
        print("[BACKTEST] ERROR: Missing parameters")
        return no_update, no_update, no_update, no_update, "Missing parameters."

    try:
        params = dict(params)
        cache_key = tuple(sorted((k, str(v)) for k, v in params.items()))
        if cache_key in BACKTEST_CACHE:
            cached = BACKTEST_CACHE[cache_key]
            print("[BACKTEST] Loaded from cache")
            return cached, params, False, None, "✓ Backtest loaded from cache."
        print(f"[BACKTEST] Requesting backend backtest with {len(params)} params")
        results = run_backtest(params)
        if not isinstance(results, dict) or results.get("error"):
            err = results.get("error") if isinstance(results, dict) else "Unknown error"
            return {"error": err}, no_update, True, None, f"✗ Backtest failed: {err}"
        BACKTEST_CACHE[cache_key] = results
        trades = _deserialize_df(results.get("trades"))
        num_trades = len(trades) if trades is not None else 0
        print(f"[BACKTEST] Success! Returning results")
        return results, params, False, None, f"✓ Backtest complete! {num_trades} trades executed."
    except BackendApiError as exc:
        print(f"[BACKTEST] Backend error: {exc}")
        return {"error": str(exc)}, no_update, True, None, f"✗ Backend error: {exc}"
    except Exception as exc:
        print(f"[BACKTEST] ERROR: {exc}")
        import traceback
        traceback.print_exc()
        return {"error": str(exc)}, no_update, True, None, f"✗ Backtest failed: {exc}"


@app.callback(
    [Output("store-params", "data"), Output("store-dirty", "data"), Output("dirty-warning", "children")],
    [Input(field["id"], field["prop"]) for field in INPUT_FIELDS],
    State("store-last-params", "data"),
)
def update_params_state(*args):
    input_values = list(args[: len(INPUT_FIELDS)])
    last_params = args[len(INPUT_FIELDS)] if len(args) > len(INPUT_FIELDS) else None
    values = _values_from_args(input_values)
    params = _build_params(values)
    if not params:
        return {}, False, ""

    dirty = bool(last_params) and last_params != params
    warning = "Parameters changed. Results shown are from the previous run." if dirty else ""
    return params, dirty, warning


@app.callback(
    [Output("dashboard-message", "children"), Output("dashboard-metrics", "children"), Output("backtest-figure", "figure")],
    [
        Input("store-results", "data"),
        Input("show_candles", "value"),
        Input("lock_rsi_y", "value"),
        Input("store-selected-trade", "data"),
        Input("theme-toggle", "value"),
    ],
)
def update_dashboard(results, show_candles, lock_rsi_y, selected_trade, theme_toggle):
    print(f"[DASHBOARD] update_dashboard called, results: {type(results)}, has_data: {bool(results)}")
    theme_value = "dark" if _from_checklist(theme_toggle) else "light"
    if not results:
        return "Run a backtest to see results.", [], build_empty_figure(theme_value)
    if isinstance(results, dict) and results.get("error"):
        print(f"[DASHBOARD] Error in results: {results.get('error')}")
        return results.get("error"), [], build_empty_figure(theme_value)

    stats = results.get("stats", {})
    ds = _deserialize_df(results.get("ds"))
    trades = _deserialize_df(results.get("trades"))
    equity_curve = results.get("equity_curve", [])
    params = results.get("params", {})

    trades_table = build_trades_table(trades)

    metrics = [
        ("Total Return", f"{float(stats.get('total_equity_return_pct', 0.0)):.2f}%"),
        ("Win Rate", f"{float(stats.get('win_rate', 0.0)):.1f}%"),
        ("Profit Factor", f"{float(stats.get('profit_factor', 0.0)):.2f}"),
        ("Max Drawdown", f"{float(stats.get('max_drawdown_pct', 0.0)):.2f}%"),
        ("Sharpe Ratio", f"{float(stats.get('sharpe_ratio', 0.0)):.2f}"),
        ("Trades", str(len(trades_table))),
    ]

    metric_cards = [
        html.Div(
            [html.Div(label, style={"fontSize": "12px"}), html.Div(value, style={"fontWeight": "bold"})],
            className="metric-card",
        )
        for label, value in metrics
    ]

    fig = build_backtest_figure(
        ds,
        trades_table,
        equity_curve,
        params,
        selected_trade=selected_trade,
        show_candles=_from_checklist(show_candles),
        lock_rsi_y=_from_checklist(lock_rsi_y),
        theme=theme_value,
    )

    return "", metric_cards, fig


@app.callback(
    [
        Output("trades-message", "children"),
        Output("trades-table", "data"),
        Output("trades-table", "columns"),
        Output("diagnostics-message", "children"),
        Output("diagnostics-table", "data"),
        Output("diagnostics-table", "columns"),
    ],
    Input("store-results", "data"),
)
def update_trades(results):
    if not results:
        return "Run a backtest to see trades.", [], [], "", [], []
    if isinstance(results, dict) and results.get("error"):
        return results.get("error"), [], [], "", [], []

    trades = _deserialize_df(results.get("trades"))
    ds = _deserialize_df(results.get("ds"))

    trades_table = build_trades_table(trades)
    if trades_table.empty:
        trades_message = "No trades for the selected settings/window."
        trades_data = []
        trades_columns = []
    else:
        trades_message = ""
        tz = "America/Los_Angeles"
        trades_view = trades_table.copy()
        trades_view["Entry (Local)"] = pd.to_datetime(trades_view["EntryTime"], utc=True).dt.tz_convert(tz).dt.strftime("%m/%d/%Y %H:%M")
        trades_view["Exit (Local)"] = pd.to_datetime(trades_view["ExitTime"], utc=True).dt.tz_convert(tz).dt.strftime("%m/%d/%Y %H:%M")
        preferred_cols = [
            "Entry (Local)",
            "Exit (Local)",
            "EntryBar",
            "ExitBar",
            "Side",
            "Price@Entry",
            "Price@Exit",
            "Price Move %",
            "Duration",
            "PnL (per unit)",
        ]
        cols_present = [c for c in preferred_cols if c in trades_view.columns]
        trades_data = trades_view[cols_present].to_dict("records")
        trades_columns = [{"name": c, "id": c} for c in cols_present]

    diag = build_entry_diagnostics(trades, ds)
    if diag.empty:
        diag_message = "No diagnostics available."
        diag_data = []
        diag_columns = []
    else:
        diag_message = ""
        if "EntryTime" in diag.columns:
            diag["EntryTime"] = pd.to_datetime(diag["EntryTime"], utc=True, errors="coerce").dt.strftime("%m/%d/%Y %H:%M")
        diag_data = diag.to_dict("records")
        diag_columns = [{"name": c, "id": c} for c in diag.columns]

    return trades_message, trades_data, trades_columns, diag_message, diag_data, diag_columns


@app.callback(
    Output("store-selected-trade", "data", allow_duplicate=True),
    Input("trades-table", "selected_rows"),
    State("trades-table", "data"),
    prevent_initial_call=True,
)
def select_trade(selected_rows, rows):
    if not selected_rows or not rows:
        return None
    idx = selected_rows[0]
    if idx is None or idx >= len(rows):
        return None
    return rows[idx]


@app.callback(
    Output("download-diagnostics", "data"),
    Input("download-diagnostics-btn", "n_clicks"),
    State("diagnostics-table", "data"),
    prevent_initial_call=True,
)
def download_diagnostics(n_clicks, data):
    if not n_clicks or not data:
        raise PreventUpdate
    df = pd.DataFrame(data)
    return dcc.send_data_frame(df.to_csv, "entry_diagnostics.csv", index=False)


@app.callback(
    Output("opt-estimate", "children"),
    [
        Input("opt_rsi_range", "value"),
        Input("opt_stop_range", "value"),
        Input("opt_band_range", "value"),
        Input("opt_grid_steps", "value"),
        Input("opt_include_entry_modes", "value"),
        Input("opt_include_exit_levels", "value"),
        Input("opt_validation_mode", "value"),
        Input("store-results", "data"),
    ],
    State("wf_train_days", "value"),
    State("wf_test_days", "value"),
    State("wf_max_folds", "value"),
)
def update_opt_estimate(rsi_range, stop_range, band_range, steps, include_entry_modes, include_exit_levels, validation_mode, results, wf_train_days, wf_test_days, wf_max_folds):
    if not rsi_range or not stop_range or not band_range:
        return ""

    base_params = dict(results.get("params", {})) if isinstance(results, dict) else {}
    stop_mode = base_params.get("stop_mode")
    param_grid = create_custom_grid(
        rsi_range=tuple(rsi_range),
        stop_range=tuple(stop_range),
        band_mult_range=tuple(band_range),
        steps=int(steps or 3),
        include_entry_modes=_from_checklist(include_entry_modes),
        include_exit_levels=_from_checklist(include_exit_levels),
        stop_mode=stop_mode,
    )

    total_combos = 1
    for v in param_grid.values():
        total_combos *= len(v)

    if validation_mode == "walk":
        if not results or results.get("error"):
            return f"Grid size: {total_combos:,} per fold. Run a backtest to estimate folds."
        df = _deserialize_df(results.get("df"))
        try:
            df_days = max(0, (df.index.max() - df.index.min()).days)
            est_folds = max(0, (df_days - int(wf_train_days)) // max(1, int(wf_test_days)))
        except Exception:
            est_folds = 0
        if wf_max_folds and int(wf_max_folds) > 0:
            est_folds = min(est_folds, int(wf_max_folds))
        return f"Grid size: {total_combos:,} per fold. Estimated folds: {est_folds}."

    return f"{total_combos:,} combinations will be tested."


@app.callback(
    [Output("store-opt-job", "data"), Output("store-opt-results", "data"), Output("store-wf-results", "data"), Output("opt-status", "children")],
    Input("opt-run-btn", "n_clicks"),
    State("store-results", "data"),
    State("opt_rsi_range", "value"),
    State("opt_stop_range", "value"),
    State("opt_band_range", "value"),
    State("opt_grid_steps", "value"),
    State("opt_include_entry_modes", "value"),
    State("opt_include_exit_levels", "value"),
    State("opt_metric", "value"),
    State("opt_min_trades", "value"),
    State("opt_min_pf", "value"),
    State("opt_min_wr", "value"),
    State("opt_max_dd", "value"),
    State("opt_min_total_ret", "value"),
    State("opt_validation_mode", "value"),
    State("wf_train_days", "value"),
    State("wf_test_days", "value"),
    State("wf_max_folds", "value"),
    prevent_initial_call=True,
)
def run_optimization(n_clicks, results, rsi_range, stop_range, band_range, steps, include_entry_modes, include_exit_levels, metric, min_trades, min_pf, min_wr, max_dd, min_total_ret, validation_mode, wf_train_days, wf_test_days, wf_max_folds):
    if not n_clicks:
        raise PreventUpdate
    if not results or results.get("error"):
        return no_update, no_update, no_update, "Run a backtest before optimization."

    base_params = dict(results.get("params", {}))
    exchange = base_params.get("exchange")
    symbol = base_params.get("symbol")
    timeframe = base_params.get("timeframe")
    start_ts = base_params.get("start_ts")
    end_ts = base_params.get("end_ts")
    if not all([exchange, symbol, timeframe, start_ts, end_ts]):
        return no_update, no_update, no_update, "Missing backtest parameters (exchange/symbol/timeframe/dates)."

    param_grid = create_custom_grid(
        rsi_range=tuple(rsi_range),
        stop_range=tuple(stop_range),
        band_mult_range=tuple(band_range),
        steps=int(steps or 3),
        include_entry_modes=_from_checklist(include_entry_modes),
        include_exit_levels=_from_checklist(include_exit_levels),
        stop_mode=base_params.get("stop_mode"),
    )

    constraints = ResultConstraints(
        min_trades=int(min_trades or 1),
        min_profit_factor=(float(min_pf) if min_pf and float(min_pf) > 0 else None),
        min_win_rate=(float(min_wr) if min_wr and float(min_wr) > 0 else None),
        max_drawdown=(float(max_dd) if max_dd and float(max_dd) > 0 else None),
        min_total_return=(float(min_total_ret) if min_total_ret not in (None, 0, 0.0) else None),
    )

    try:
        constraints_payload: Dict[str, Any] = {"min_trades": int(constraints.min_trades)}
        if constraints.min_profit_factor is not None:
            constraints_payload["min_profit_factor"] = float(constraints.min_profit_factor)
        if constraints.min_win_rate is not None:
            constraints_payload["min_win_rate"] = float(constraints.min_win_rate)
        if constraints.max_drawdown is not None:
            constraints_payload["max_drawdown"] = float(constraints.max_drawdown)
        if constraints.min_total_return is not None:
            constraints_payload["min_total_return"] = float(constraints.min_total_return)

        payload: Dict[str, Any] = {
            "exchange": exchange,
            "symbol": symbol,
            "timeframe": timeframe,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "base_params": base_params,
            "param_grid": param_grid,
            "metric": metric,
            "min_trades": int(min_trades or 1),
            "constraints": constraints_payload,
            "validation_mode": validation_mode,
            "top_n": 20,
        }

        if validation_mode == "walk":
            payload["walk_forward"] = {
                "train_days": int(wf_train_days),
                "test_days": int(wf_test_days),
                "step_days": None,
                "max_folds": (None if int(wf_max_folds) == 0 else int(wf_max_folds)),
            }
            payload["top_n_train"] = 20

        job = enqueue_optimize(payload)
        job_id = int(job.get("id"))
        return job_id, None, None, f"Optimization queued (job #{job_id})."
    except Exception as exc:
        return None, None, None, f"Optimization enqueue failed: {exc}"


@app.callback(
    [
        Output("store-opt-job", "data", allow_duplicate=True),
        Output("store-opt-results", "data", allow_duplicate=True),
        Output("store-wf-results", "data", allow_duplicate=True),
        Output("opt-status", "children", allow_duplicate=True),
    ],
    Input("backend-poll", "n_intervals"),
    State("store-opt-job", "data"),
    prevent_initial_call=True,
)
def poll_optimization_job(_tick, job_id):
    if not job_id:
        raise PreventUpdate

    try:
        job = get_job(int(job_id))
    except BackendApiError as exc:
        return no_update, no_update, no_update, f"Backend error polling job #{job_id}: {exc}"

    status = job.get("status")
    if status in {"queued", "running"}:
        cur = int(job.get("progress_current") or 0)
        total = int(job.get("progress_total") or 0)
        msg = job.get("progress_message") or ""
        suffix = f" ({cur}/{total})" if total else ""
        msg_part = f" — {msg}" if msg else ""
        return no_update, no_update, no_update, f"Optimization {status}{suffix}{msg_part}"

    if status == "succeeded":
        result = job.get("result") or {}
        kind = result.get("kind") or "grid_search"
        results_obj = result.get("results") or {}
        records = results_obj.get("records") or []
        df = pd.DataFrame(records)

        if kind == "walk_forward":
            wf_payload = {
                "results": df.to_json(orient="split"),
                "summary": result.get("summary") or {},
            }
            return None, None, wf_payload, "Walk-forward complete."

        opt_payload = df.to_json(orient="split")
        return None, opt_payload, None, f"Optimization complete! Found {len(df)} valid configurations."

    if status == "failed":
        return None, None, None, f"Optimization failed: {job.get('error') or 'unknown error'}"

    if status == "canceled":
        return None, None, None, "Optimization canceled."

    return None, None, None, f"Optimization ended with status={status}"


@app.callback(
    [Output("opt-results-table", "data"), Output("opt-results-table", "columns"), Output("opt-summary", "children"), Output("wf-results-table", "data"), Output("wf-results-table", "columns"), Output("wf-summary", "children")],
    [Input("store-opt-results", "data"), Input("store-wf-results", "data")],
)
def render_opt_results(opt_payload, wf_payload):
    opt_data = []
    opt_columns = []
    opt_summary = ""
    wf_data = []
    wf_columns = []
    wf_summary = ""

    if opt_payload:
        df = _deserialize_df(opt_payload)
        opt_data = df.to_dict("records")
        opt_columns = [{"name": c, "id": c} for c in df.columns]
        analysis = analyze_results(df)
        best = analysis.get("best_params", {})
        if best:
            best_lines = [html.Li(f"{k}: {v}") for k, v in best.items()]
            opt_summary = html.Div([
                html.H5("Best Configuration"),
                html.Ul(best_lines),
                html.Div(f"Top Profit Factor: {analysis.get('top_profit_factor', 0):.2f}"),
                html.Div(f"Avg PF (Top 10): {analysis.get('avg_profit_factor_top_10', 0):.2f}"),
            ])

    if wf_payload:
        df = _deserialize_df(wf_payload.get("results"))
        wf_data = df.to_dict("records")
        wf_columns = [{"name": c, "id": c} for c in df.columns]
        summary = wf_payload.get("summary", {})
        wf_summary = html.Div([
            html.Div(f"Avg OOS Return: {summary.get('oos_avg_return', 0):.2f}%"),
            html.Div(f"Median OOS Return: {summary.get('oos_median_return', 0):.2f}%"),
            html.Div(f"Avg OOS Win Rate: {summary.get('oos_avg_win_rate', 0):.1f}%"),
            html.Div(f"Avg OOS Profit Factor: {summary.get('oos_avg_profit_factor', 0):.2f}"),
            html.Div(f"Avg OOS Max DD: {summary.get('oos_avg_max_drawdown', 0):.2f}%"),
        ])

    return opt_data, opt_columns, opt_summary, wf_data, wf_columns, wf_summary


@app.callback(
    Output("download-optimization", "data"),
    Input("opt-download-btn", "n_clicks"),
    State("store-opt-results", "data"),
    prevent_initial_call=True,
)
def download_opt_results(n_clicks, payload):
    if not n_clicks or not payload:
        raise PreventUpdate
    df = _deserialize_df(payload)
    return dcc.send_data_frame(df.to_csv, "optimization_results.csv", index=False)


@app.callback(
    Output("download-walkforward", "data"),
    Input("wf-download-btn", "n_clicks"),
    State("store-wf-results", "data"),
    prevent_initial_call=True,
)
def download_wf_results(n_clicks, payload):
    if not n_clicks or not payload:
        raise PreventUpdate
    df = _deserialize_df(payload.get("results"))
    return dcc.send_data_frame(df.to_csv, "walk_forward_results.csv", index=False)


@app.callback(
    Output("disc-estimate", "children"),
    [
        Input("disc_rsi_range", "value"),
        Input("disc_rsi_ma_range", "value"),
        Input("disc_band_range", "value"),
        Input("disc_leverage_options", "value"),
        Input("disc_risk_options", "value"),
        Input("disc_include_atr", "value"),
        Input("disc_include_trailing", "value"),
        Input("disc_skip_tested", "value"),
        Input("disc_use_parallel", "value"),
        Input("disc_n_workers", "value"),
    ],
)
def update_discovery_estimate(rsi_range, rsi_ma_range, band_range, leverage_options, risk_options, include_atr, include_trailing, skip_tested, use_parallel, n_workers):
    disc_grid = create_margin_discovery_grid(
        rsi_range=(int(rsi_range[0]), int(rsi_range[1])),
        rsi_ma_range=(int(rsi_ma_range[0]), int(rsi_ma_range[1])),
        band_mult_range=tuple(band_range),
        leverage_options=leverage_options or [5.0],
        risk_pct_options=risk_options or [1.0],
        include_atr_stops=_from_checklist(include_atr),
        include_trailing=_from_checklist(include_trailing),
        rsi_step=2,
        mult_step=0.1,
    )

    tested_count = 0
    try:
        tested_count = int(get_discovery_stats().get("total_runs", 0))
    except Exception:
        tested_count = 0
    effective_workers = int(n_workers or 1) if _from_checklist(use_parallel) else 1
    filtered_total = count_filtered_combinations(disc_grid)
    estimate = estimate_discovery_time(
        disc_grid,
        tested_count if _from_checklist(skip_tested) else 0,
        n_workers=effective_workers,
        total_override=filtered_total,
    )

    return f"{estimate['remaining']:,} new to test | Estimated time: {estimate['human_readable']}"


@app.callback(
    [Output("store-disc-job", "data"), Output("disc-status", "children"), Output("disc-results", "children")],
    Input("disc-run-btn", "n_clicks"),
    State("store-results", "data"),
    State("disc_rsi_range", "value"),
    State("disc_rsi_ma_range", "value"),
    State("disc_band_range", "value"),
    State("disc_leverage_options", "value"),
    State("disc_risk_options", "value"),
    State("disc_include_atr", "value"),
    State("disc_include_trailing", "value"),
    State("disc_min_return", "value"),
    State("disc_max_dd", "value"),
    State("disc_min_trades", "value"),
    State("disc_min_pf", "value"),
    State("disc_skip_tested", "value"),
    State("disc_use_parallel", "value"),
    State("disc_n_workers", "value"),
    prevent_initial_call=True,
)
def run_discovery_callback(n_clicks, results, rsi_range, rsi_ma_range, band_range, leverage_options, risk_options, include_atr, include_trailing, min_return, max_dd, min_trades, min_pf, skip_tested, use_parallel, n_workers):
    if not n_clicks:
        raise PreventUpdate
    if not results or results.get("error"):
        return no_update, "Run a backtest before discovery.", ""

    base_params = dict(results.get("params", {}))
    exchange = base_params.get("exchange")
    symbol = base_params.get("symbol")
    timeframe = base_params.get("timeframe")
    start_ts = base_params.get("start_ts")
    end_ts = base_params.get("end_ts")
    if not all([exchange, symbol, timeframe, start_ts, end_ts]):
        return no_update, "Missing backtest parameters (exchange/symbol/timeframe/dates).", ""

    disc_grid = create_margin_discovery_grid(
        rsi_range=(int(rsi_range[0]), int(rsi_range[1])),
        rsi_ma_range=(int(rsi_ma_range[0]), int(rsi_ma_range[1])),
        band_mult_range=tuple(band_range),
        leverage_options=leverage_options or [5.0],
        risk_pct_options=risk_options or [1.0],
        include_atr_stops=_from_checklist(include_atr),
        include_trailing=_from_checklist(include_trailing),
        rsi_step=2,
        mult_step=0.1,
    )

    try:
        payload: Dict[str, Any] = {
            "exchange": exchange,
            "symbol": symbol,
            "timeframe": timeframe,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "base_params": base_params,
            "param_grid": disc_grid,
            "win_criteria": {
                "min_total_return": float(min_return),
                "max_drawdown": float(max_dd),
                "min_trades": int(min_trades),
                "min_profit_factor": float(min_pf),
            },
            "skip_tested": _from_checklist(skip_tested),
            "batch_size": 50,
            "use_parallel": _from_checklist(use_parallel),
            "n_workers": int(n_workers or 1),
        }
        job = enqueue_discover(payload)
        job_id = int(job.get("id"))
        return job_id, f"Discovery queued (job #{job_id}).", ""
    except Exception as exc:
        return None, f"Discovery enqueue failed: {exc}", ""


@app.callback(
    [
        Output("store-disc-job", "data", allow_duplicate=True),
        Output("disc-status", "children", allow_duplicate=True),
        Output("disc-results", "children", allow_duplicate=True),
    ],
    Input("backend-poll", "n_intervals"),
    State("store-disc-job", "data"),
    prevent_initial_call=True,
)
def poll_discovery_job(_tick, job_id):
    if not job_id:
        raise PreventUpdate

    try:
        job = get_job(int(job_id))
    except BackendApiError as exc:
        return no_update, f"Backend error polling job #{job_id}: {exc}", no_update

    status = job.get("status")
    if status in {"queued", "running"}:
        cur = int(job.get("progress_current") or 0)
        total = int(job.get("progress_total") or 0)
        msg = job.get("progress_message") or ""
        suffix = f" ({cur}/{total})" if total else ""
        msg_part = f" — {msg}" if msg else ""
        return no_update, f"Discovery {status}{suffix}{msg_part}", no_update

    if status == "succeeded":
        result = job.get("result") or {}
        summary = result.get("summary") or {}
        winners_found = int(summary.get("winners_found") or 0)
        return None, "Discovery complete.", f"Found {winners_found} winners."

    if status == "failed":
        return None, f"Discovery failed: {job.get('error') or 'unknown error'}", ""

    if status == "canceled":
        return None, "Discovery canceled.", ""

    return None, f"Discovery ended with status={status}", ""


@app.callback(
    Output("store-lb-snapshot", "data"),
    Input("analysis-tabs", "value"),
)
def load_leaderboard_snapshot(tab_value):
    if tab_value != "leaderboard":
        return no_update

    try:
        snapshot = get_leaderboard_latest()
        return snapshot
    except BackendApiError as exc:
        if exc.status == 404:
            return None
        return no_update


@app.callback(
    [Output("store-lb-job", "data"), Output("lb-status", "children")],
    Input("lb-refresh-btn", "n_clicks"),
    State("lb_sort_by", "value"),
    State("lb_top_n", "value"),
    prevent_initial_call=True,
)
def enqueue_leaderboard_job(n_clicks, sort_by, top_n):
    if not n_clicks:
        raise PreventUpdate

    try:
        job = enqueue_leaderboard_refresh(
            {
                "sort_by": sort_by,
                "top_n": int(top_n),
                "min_trades": 10,
            }
        )
        job_id = int(job.get("id"))
        return job_id, f"Leaderboard refresh queued (job #{job_id})."
    except Exception as exc:
        return None, f"Leaderboard refresh enqueue failed: {exc}"


@app.callback(
    [
        Output("store-lb-job", "data", allow_duplicate=True),
        Output("store-lb-snapshot", "data", allow_duplicate=True),
        Output("lb-status", "children", allow_duplicate=True),
    ],
    Input("backend-poll", "n_intervals"),
    State("store-lb-job", "data"),
    prevent_initial_call=True,
)
def poll_leaderboard_job(_tick, job_id):
    if not job_id:
        raise PreventUpdate

    try:
        job = get_job(int(job_id))
    except BackendApiError as exc:
        return no_update, no_update, f"Backend error polling job #{job_id}: {exc}"

    status = job.get("status")
    if status in {"queued", "running"}:
        msg = job.get("progress_message") or ""
        msg_part = f" — {msg}" if msg else ""
        return no_update, no_update, f"Leaderboard {status}{msg_part}"

    if status == "succeeded":
        try:
            snapshot = get_leaderboard_latest()
            snap_id = snapshot.get("snapshot_id")
            created_at = snapshot.get("created_at")
            msg = "Leaderboard refreshed."
            if snap_id and created_at:
                msg = f"Leaderboard refreshed (snapshot #{snap_id} @ {created_at})."
            return None, snapshot, msg
        except Exception:
            return None, job.get("result"), "Leaderboard refreshed."

    if status == "failed":
        return None, no_update, f"Leaderboard refresh failed: {job.get('error') or 'unknown error'}"

    if status == "canceled":
        return None, no_update, "Leaderboard refresh canceled."

    return None, no_update, f"Leaderboard job ended with status={status}"


@app.callback(
    [Output("lb-stats", "children"), Output("lb-table", "data"), Output("lb-table", "columns")],
    [Input("analysis-tabs", "value"), Input("store-lb-snapshot", "data"), Input("lb_sort_by", "value"), Input("lb_top_n", "value")],
)
def update_leaderboard(tab_value, snapshot, sort_by, top_n):
    if tab_value != "leaderboard":
        return no_update, no_update, no_update

    if not snapshot:
        stats = [html.Div("No leaderboard snapshot yet. Click Refresh.")]
        return stats, [], []

    if isinstance(snapshot, dict) and "payload" in snapshot:
        payload = snapshot.get("payload") or {}
    else:
        payload = snapshot or {}

    stats_obj = payload.get("stats") or {}
    strategies = payload.get("strategies") or []

    stats = [
        html.Div(f"Total Winners: {int(stats_obj.get('total_winners') or 0)}"),
        html.Div(f"Best Return: {float(stats_obj.get('best_return') or 0):.2f}%"),
        html.Div(f"Avg Return: {float(stats_obj.get('avg_return') or 0):.2f}%"),
        html.Div(f"Avg Drawdown: {float(stats_obj.get('avg_drawdown') or 0):.2f}%"),
    ]

    def _metric_value(item: dict) -> float:
        metrics = item.get("metrics") or {}
        try:
            return float(metrics.get(sort_by, 0) or 0)
        except Exception:
            return 0.0

    reverse = str(sort_by) != "max_drawdown"
    sorted_strats = sorted(strategies, key=_metric_value, reverse=reverse)
    limited = sorted_strats[: int(top_n or 10)]

    lb_data = []
    for s in limited:
        metrics = s.get("metrics") or {}
        params = s.get("params") or {}
        params_hash = s.get("params_hash") or ""
        lb_data.append(
            {
                "Rank": int(s.get("rank") or 0),
                "Return %": f"{float(metrics.get('total_return') or 0):.2f}",
                "Max DD %": f"{float(metrics.get('max_drawdown') or 0):.2f}",
                "Profit Factor": f"{float(metrics.get('profit_factor') or 0):.2f}",
                "Win Rate %": f"{float(metrics.get('win_rate') or 0):.1f}",
                "Sharpe": f"{float(metrics.get('sharpe_ratio') or 0):.2f}",
                "Trades": int(metrics.get("num_trades") or 0),
                "Hash": str(params_hash)[:8],
                "FullHash": str(params_hash),
                "ParamsJson": json.dumps(params, default=str),
            }
        )

    columns = [{"name": c, "id": c} for c in lb_data[0].keys()] if lb_data else []
    return stats, lb_data, columns


@app.callback(
    Output("lb-updated", "children"),
    Input("store-lb-snapshot", "data"),
)
def render_leaderboard_updated(snapshot):
    if not snapshot or not isinstance(snapshot, dict):
        return ""

    created_at = snapshot.get("created_at")
    snap_id = snapshot.get("snapshot_id") or snapshot.get("id")
    if created_at and snap_id:
        return f"Last updated: {created_at} (snapshot #{snap_id})"
    if created_at:
        return f"Last updated: {created_at}"
    return ""


@app.callback(
    [Output("lb-details", "children"), Output("store-leaderboard-selected", "data")],
    Input("lb-table", "selected_rows"),
    State("lb-table", "data"),
)
def leaderboard_details(selected_rows, rows):
    if not selected_rows or not rows:
        return "", None

    row = rows[selected_rows[0]]
    params_json = row.get("ParamsJson")
    if not params_json:
        return "", None

    try:
        params = json.loads(params_json)
    except Exception:
        params = {}

    params_items = list((params or {}).items())
    details = html.Ul([html.Li(f"{k}: {v}") for k, v in params_items])
    return details, params


@app.callback(
    Output("store-load-strategy", "data"),
    Input("lb-load-btn", "n_clicks"),
    State("store-leaderboard-selected", "data"),
    prevent_initial_call=True,
)
def load_leaderboard_strategy(n_clicks, params_hash):
    if not n_clicks:
        raise PreventUpdate
    if not params_hash or not isinstance(params_hash, dict):
        return no_update

    return params_hash


@app.callback(
    Output("download-leaderboard", "data"),
    Input("lb-download-btn", "n_clicks"),
    State("lb-table", "data"),
    prevent_initial_call=True,
)
def download_leaderboard(n_clicks, rows):
    if not n_clicks or not rows:
        raise PreventUpdate
    df = pd.DataFrame(rows)
    return dcc.send_data_frame(df.to_csv, "winning_strategies.csv", index=False)


@app.callback(
    Output("store-pattern-rules", "data"),
    Input("analysis-tabs", "value"),
)
def load_patterns_on_tab(tab_value):
    if tab_value != "patterns":
        return no_update
    try:
        return get_patterns(min_confidence=0.3)
    except Exception:
        return no_update


@app.callback(
    [Output("store-pattern-job", "data"), Output("patterns-status", "children")],
    Input("patterns-btn", "n_clicks"),
    prevent_initial_call=True,
)
def enqueue_patterns_job(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    try:
        job = enqueue_patterns_refresh({"min_confidence": 0.3, "min_occurrence_pct": 10.0})
        job_id = int(job.get("id"))
        return job_id, f"Pattern analysis queued (job #{job_id})."
    except Exception as exc:
        return None, f"Pattern analysis enqueue failed: {exc}"


@app.callback(
    [
        Output("store-pattern-job", "data", allow_duplicate=True),
        Output("store-pattern-rules", "data", allow_duplicate=True),
        Output("patterns-status", "children", allow_duplicate=True),
    ],
    Input("backend-poll", "n_intervals"),
    State("store-pattern-job", "data"),
    prevent_initial_call=True,
)
def poll_patterns_job(_tick, job_id):
    if not job_id:
        raise PreventUpdate

    try:
        job = get_job(int(job_id))
    except BackendApiError as exc:
        return no_update, no_update, f"Backend error polling job #{job_id}: {exc}"

    status = job.get("status")
    if status in {"queued", "running"}:
        msg = job.get("progress_message") or ""
        msg_part = f" — {msg}" if msg else ""
        return no_update, no_update, f"Pattern analysis {status}{msg_part}"

    if status == "succeeded":
        try:
            rules = get_patterns(min_confidence=0.3)
        except Exception:
            rules = []
        return None, rules, "Pattern analysis complete."

    if status == "failed":
        return None, no_update, f"Pattern analysis failed: {job.get('error') or 'unknown error'}"

    if status == "canceled":
        return None, no_update, "Pattern analysis canceled."

    return None, no_update, f"Pattern analysis ended with status={status}"


@app.callback(
    Output("patterns-summary", "children"),
    [Input("analysis-tabs", "value"), Input("store-pattern-rules", "data")],
)
def render_patterns(tab_value, rules_payload):
    if tab_value != "patterns":
        return no_update

    if not rules_payload:
        return "No patterns discovered yet."

    rules: List[DiscoveredRule] = []
    for r in rules_payload:
        if not isinstance(r, dict):
            continue
        try:
            rules.append(
                DiscoveredRule(
                    rule_id=r.get("rule_id"),
                    parameter=r.get("parameter", ""),
                    condition=r.get("condition", ""),
                    occurrence_pct=float(r.get("occurrence_pct") or 0),
                    avg_return_with=float(r.get("avg_return_with") or 0),
                    avg_return_without=float(r.get("avg_return_without") or 0),
                    confidence=float(r.get("confidence") or 0),
                    description=r.get("description", ""),
                    discovered_at=r.get("discovered_at") or dt.datetime.now(dt.UTC).isoformat(),
                )
            )
        except Exception:
            continue

    summary = get_rule_summary(rules)
    return f"```text\n{summary}\n```"


@app.callback(
    [Output("store-job-queue", "data"), Output("jobs-status", "children")],
    [
        Input("analysis-tabs", "value"),
        Input("jobs-refresh-btn", "n_clicks"),
        Input("backend-poll", "n_intervals"),
        Input("jobs-status-filter", "value"),
        Input("jobs-type-filter", "value"),
        Input("jobs-limit", "value"),
        Input("jobs-auto-refresh", "value"),
    ],
)
def refresh_job_queue(tab_value, _refresh_clicks, _tick, status_filter, type_filter, limit, auto_refresh):
    if tab_value != "jobs":
        return no_update, no_update

    trigger_id = (callback_context.triggered_id or "").split(".")[0]
    if trigger_id == "backend-poll" and not _from_checklist(auto_refresh):
        return no_update, no_update

    status = None if status_filter in (None, "", "all") else str(status_filter)
    job_type = None if type_filter in (None, "", "all") else str(type_filter)

    try:
        jobs = list_jobs(status=status, job_type=job_type, limit=int(limit or 50), offset=0)
    except BackendApiError as exc:
        return no_update, f"Backend error loading jobs: {exc}"

    counts: Dict[str, int] = {}
    for j in jobs:
        st = str(j.get("status") or "unknown")
        counts[st] = counts.get(st, 0) + 1

    parts = [f"{k}: {v}" for k, v in sorted(counts.items())]
    summary = " | ".join(parts) if parts else "no jobs"
    return jobs, f"Loaded {len(jobs)} jobs — {summary}"


@app.callback(
    [Output("workers-summary", "children"), Output("workers-table", "data"), Output("workers-table", "columns")],
    [
        Input("store-job-queue", "data"),
        Input("jobs-status-filter", "value"),
        Input("jobs-type-filter", "value"),
    ],
)
def render_workers_table(jobs, status_filter, type_filter):
    if not jobs:
        return "No workers in current view.", [], []

    worker_map: Dict[str, Dict[str, Any]] = {}
    for j in jobs:
        worker_id = j.get("worker_id") or ""
        if not worker_id:
            continue
        info = worker_map.setdefault(
            worker_id,
            {
                "worker_id": worker_id,
                "running_jobs": 0,
                "last_job_id": None,
                "last_heartbeat_raw": "",
                "last_heartbeat": None,
            },
        )
        if str(j.get("status") or "") == "running":
            info["running_jobs"] += 1

        heartbeat_at = j.get("heartbeat_at")
        parsed = _parse_ts(heartbeat_at)
        if parsed and (info["last_heartbeat"] is None or parsed > info["last_heartbeat"]):
            info["last_heartbeat"] = parsed
            info["last_heartbeat_raw"] = heartbeat_at
            info["last_job_id"] = j.get("id")

    if not worker_map:
        return "No workers in current view.", [], []

    rows = []
    active = 0
    stale = 0
    unknown = 0
    for worker_id in sorted(worker_map.keys()):
        info = worker_map[worker_id]
        status, age = _worker_status(info.get("last_heartbeat_raw"))
        if status == "active":
            active += 1
        elif status == "stale":
            stale += 1
        else:
            unknown += 1
        rows.append(
            {
                "Worker": worker_id,
                "Status": status,
                "Running Jobs": info.get("running_jobs", 0),
                "Last Job": info.get("last_job_id") or "",
                "Last Heartbeat": info.get("last_heartbeat_raw") or "",
                "Heartbeat Age": age,
            }
        )

    filter_bits = []
    if status_filter not in (None, "", "all"):
        filter_bits.append(f"status={status_filter}")
    if type_filter not in (None, "", "all"):
        filter_bits.append(f"type={type_filter}")
    filter_note = f" (filter: {', '.join(filter_bits)})" if filter_bits else ""
    summary = f"Workers: {len(rows)} | active={active} stale={stale} unknown={unknown}{filter_note}"

    columns = [{"name": c, "id": c} for c in rows[0].keys()] if rows else []
    return summary, rows, columns


@app.callback(
    [Output("jobs-table", "data"), Output("jobs-table", "columns")],
    Input("store-job-queue", "data"),
)
def render_jobs_table(jobs):
    if not jobs:
        return [], []

    rows = []
    for j in jobs:
        cur = int(j.get("progress_current") or 0)
        total = int(j.get("progress_total") or 0)
        progress = f"{cur}/{total}" if total else (str(cur) if cur else "")
        attempts = int(j.get("attempts") or 0)
        worker_status, heartbeat_age = _worker_status(j.get("heartbeat_at"))
        rows.append(
            {
                "ID": int(j.get("id") or 0),
                "Type": j.get("job_type") or "",
                "Status": j.get("status") or "",
                "Priority": int(j.get("priority") or 0),
                "Created": j.get("created_at") or "",
                "Started": j.get("started_at") or "",
                "Finished": j.get("finished_at") or "",
                "Worker": j.get("worker_id") or "",
                "Worker Status": worker_status,
                "Heartbeat Age": heartbeat_age,
                "Attempts": attempts,
                "Progress": progress,
                "Message": j.get("progress_message") or "",
            }
        )

    columns = [{"name": c, "id": c} for c in rows[0].keys()] if rows else []
    return rows, columns


@app.callback(
    Output("store-job-selected", "data"),
    Input("jobs-table", "selected_rows"),
    State("jobs-table", "data"),
)
def select_job_from_queue(selected_rows, rows):
    if not selected_rows or not rows:
        return None
    idx = selected_rows[0]
    if idx is None or idx >= len(rows):
        return None
    return rows[idx].get("ID")


@app.callback(
    [Output("store-job-detail", "data"), Output("store-job-events", "data"), Output("jobs-detail-status", "children")],
    [
        Input("analysis-tabs", "value"),
        Input("store-job-selected", "data"),
        Input("jobs-detail-refresh-btn", "n_clicks"),
        Input("jobs-cancel-btn", "n_clicks"),
        Input("backend-poll", "n_intervals"),
        Input("jobs-auto-refresh", "value"),
    ],
    prevent_initial_call=True,
)
def load_job_detail(tab_value, job_id, _refresh_clicks, cancel_clicks, _tick, auto_refresh):
    if tab_value != "jobs":
        return no_update, no_update, no_update
    if not job_id:
        return None, None, ""

    trigger_id = (callback_context.triggered_id or "").split(".")[0]
    if trigger_id == "backend-poll" and not _from_checklist(auto_refresh):
        return no_update, no_update, no_update

    try:
        if trigger_id == "jobs-cancel-btn" and cancel_clicks:
            cancel_job(int(job_id))

        job = get_job(int(job_id))
        events = get_job_events(int(job_id), limit=200)
        return job, events, ""
    except BackendApiError as exc:
        return no_update, no_update, f"Backend error loading job #{job_id}: {exc}"


@app.callback(
    [
        Output("jobs-detail-summary", "children"),
        Output("jobs-detail-json", "children"),
        Output("jobs-events-table", "data"),
        Output("jobs-events-table", "columns"),
    ],
    [Input("analysis-tabs", "value"), Input("store-job-detail", "data"), Input("store-job-events", "data")],
)
def render_job_detail(tab_value, job, events):
    if tab_value != "jobs":
        return no_update, no_update, no_update, no_update

    if not job:
        return "", "", [], []

    job_id = job.get("id")
    status = job.get("status")
    job_type = job.get("job_type")
    worker_id = job.get("worker_id") or ""
    heartbeat_at = job.get("heartbeat_at") or ""
    attempts = int(job.get("attempts") or 0)
    created_at = job.get("created_at") or ""
    started_at = job.get("started_at") or ""
    finished_at = job.get("finished_at") or ""
    progress_current = int(job.get("progress_current") or 0)
    progress_total = int(job.get("progress_total") or 0)
    progress_message = job.get("progress_message") or ""
    worker_status, heartbeat_age = _worker_status(heartbeat_at)

    summary = (
        f"Job #{job_id} — {job_type} — {status} | "
        f"worker={worker_id or 'n/a'} status={worker_status} attempts={attempts} "
        f"heartbeat={heartbeat_at or 'n/a'} age={heartbeat_age or 'n/a'} | "
        f"created={created_at} started={started_at} finished={finished_at} | "
        f"progress={progress_current}/{progress_total} {progress_message}"
    )

    payload = job.get("payload") or {}
    result = job.get("result")
    error = job.get("error")
    detail = {"payload": payload, "result": result, "error": error}
    json_block = f"```json\n{json.dumps(detail, indent=2, default=str)}\n```"

    ev_rows = []
    if isinstance(events, list):
        for e in events:
            if not isinstance(e, dict):
                continue
            ev_rows.append(
                {
                    "ts": e.get("ts") or "",
                    "level": e.get("level") or "",
                    "message": e.get("message") or "",
                }
            )
    ev_cols = [{"name": c, "id": c} for c in ev_rows[0].keys()] if ev_rows else []

    return summary, json_block, ev_rows, ev_cols


def main() -> None:
    _configure_logging()
    port = int(os.environ.get("PORT", 8050))
    app.run_server(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
