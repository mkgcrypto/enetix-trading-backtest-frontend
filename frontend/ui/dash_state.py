import datetime as dt
from typing import Any, Dict, List

import pandas as pd

from frontend.features.presets import DEFAULT_PRESET
from frontend.ui.dash_constants import DATA_SCOPE_KEYS
from frontend.ui.dash_utils import _from_checklist, _parse_date, _safe_float, _safe_int, _to_checklist


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

STRATEGY_INPUT_FIELDS = [field for field in INPUT_FIELDS if field["key"] not in DATA_SCOPE_KEYS]


def _values_from_args(values: List[Any]) -> Dict[str, Any]:
    return {field["key"]: value for field, value in zip(INPUT_FIELDS, values)}
