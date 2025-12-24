import datetime as dt
import io
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from frontend.ui.dash_constants import CHECK_ON, DATE_PRESET_LABELS, DATE_PRESET_LOOKBACK_DAYS


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
