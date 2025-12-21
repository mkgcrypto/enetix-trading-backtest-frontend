from __future__ import annotations

from typing import Any, Dict, List, Tuple
import multiprocessing as mp


def count_combinations(param_grid: Dict[str, List[Any]]) -> int:
    """Calculate total number of parameter combinations."""
    total = 1
    for values in param_grid.values():
        total *= len(values)
    return total


def count_filtered_combinations(param_grid: Dict[str, List[Any]]) -> int:
    """
    Calculate total combinations after removing duplicates from dependent params.

    This mirrors the backend's normalization rules:
    - use_trailing=False makes trail_pct irrelevant
    - use_stop=False makes stop_pct/stop_atr_mult/stop_mode irrelevant
    - use_rsi_relation=False makes rsi_relation irrelevant
    - stop_mode selects between stop_pct and stop_atr_mult
    """
    handled = set()
    total = 1

    def _len(values: List[Any]) -> int:
        return max(1, len(values))

    def _count_stop_modes(
        modes: List[Any],
        stop_pct: List[Any],
        stop_atr: List[Any],
    ) -> int:
        count = 0
        for mode in modes:
            if mode == "Fixed %":
                count += _len(stop_pct)
            elif mode == "ATR":
                count += _len(stop_atr)
            else:
                count += _len(stop_pct) * _len(stop_atr)
        return count

    if "use_stop" in param_grid:
        stop_count = 0
        if True in param_grid["use_stop"]:
            stop_count += _count_stop_modes(
                param_grid.get("stop_mode", ["Fixed %"]),
                param_grid.get("stop_pct", []),
                param_grid.get("stop_atr_mult", []),
            )
        if False in param_grid["use_stop"]:
            stop_count += 1
        total *= max(1, stop_count)
        handled.update({"use_stop", "stop_mode", "stop_pct", "stop_atr_mult"})
    elif "stop_mode" in param_grid:
        total *= _count_stop_modes(
            param_grid["stop_mode"],
            param_grid.get("stop_pct", []),
            param_grid.get("stop_atr_mult", []),
        )
        handled.update({"stop_mode", "stop_pct", "stop_atr_mult"})
    else:
        if "stop_pct" in param_grid:
            total *= _len(param_grid["stop_pct"])
            handled.add("stop_pct")
        if "stop_atr_mult" in param_grid:
            total *= _len(param_grid["stop_atr_mult"])
            handled.add("stop_atr_mult")

    if "use_trailing" in param_grid:
        trailing_count = 0
        if True in param_grid["use_trailing"]:
            trailing_count += _len(param_grid.get("trail_pct", []))
        if False in param_grid["use_trailing"]:
            trailing_count += 1
        total *= max(1, trailing_count)
        handled.update({"use_trailing", "trail_pct"})
    elif "trail_pct" in param_grid:
        total *= _len(param_grid["trail_pct"])
        handled.add("trail_pct")

    if "use_rsi_relation" in param_grid:
        relation_count = 0
        if True in param_grid["use_rsi_relation"]:
            relation_count += _len(param_grid.get("rsi_relation", []))
        if False in param_grid["use_rsi_relation"]:
            relation_count += 1
        total *= max(1, relation_count)
        handled.update({"use_rsi_relation", "rsi_relation"})
    elif "rsi_relation" in param_grid:
        total *= _len(param_grid["rsi_relation"])
        handled.add("rsi_relation")

    for key, values in param_grid.items():
        if key in handled:
            continue
        total *= len(values)

    return total


def estimate_discovery_time(
    param_grid: Dict[str, List[Any]],
    tested_count: int = 0,
    seconds_per_test: float = 0.1,
    n_workers: int = 1,
    total_override: int | None = None,
) -> Dict[str, Any]:
    """Estimate time required for a discovery run."""
    total = int(total_override) if total_override is not None else count_combinations(param_grid)
    remaining = max(0, total - tested_count)

    effective_workers = max(1, n_workers)
    seconds = (remaining * seconds_per_test) / effective_workers
    minutes = seconds / 60
    hours = minutes / 60

    return {
        "total_combinations": total,
        "already_tested": tested_count,
        "remaining": remaining,
        "n_workers": effective_workers,
        "estimated_seconds": round(seconds, 1),
        "estimated_minutes": round(minutes, 1),
        "estimated_hours": round(hours, 2),
        "human_readable": (
            f"{hours:.1f} hours" if hours >= 1
            else f"{minutes:.0f} minutes" if minutes >= 1
            else f"{seconds:.0f} seconds"
        ),
    }


def get_cpu_count() -> int:
    """Get the number of available CPU cores."""
    return mp.cpu_count()


def create_margin_discovery_grid(
    rsi_range: Tuple[int, int] = (68, 74),
    rsi_ma_range: Tuple[int, int] = (66, 72),
    band_mult_range: Tuple[float, float] = (1.9, 2.1),
    leverage_options: List[float] | None = None,
    risk_pct_options: List[float] | None = None,
    include_atr_stops: bool = True,
    include_trailing: bool = True,
    rsi_step: int = 2,
    mult_step: float = 0.1,
) -> Dict[str, List[Any]]:
    """Create a discovery grid that includes margin/futures parameters."""
    def arange(start, end, step):
        values = []
        current = start
        while current <= end + step / 2:
            values.append(round(current, 2) if isinstance(step, float) else int(current))
            current += step
        return values

    if leverage_options is None:
        leverage_options = [2.0, 5.0, 10.0]
    if risk_pct_options is None:
        risk_pct_options = [0.5, 1.0, 2.0]

    grid = {
        "rsi_min": arange(rsi_range[0], rsi_range[1], rsi_step),
        "rsi_ma_min": arange(rsi_ma_range[0], rsi_ma_range[1], rsi_step),
        "bb_std": arange(band_mult_range[0], band_mult_range[1], mult_step),
        "kc_mult": arange(band_mult_range[0], band_mult_range[1], mult_step),
        "entry_band_mode": ["Either", "KC", "Both"],
        "exit_level": ["mid", "lower"],
        "trade_mode": ["Margin / Futures"],
        "max_leverage": leverage_options,
        "risk_per_trade_pct": risk_pct_options,
        "maintenance_margin_pct": [0.5],
        "use_stop": [True],
        "stop_pct": [1.5, 2.0, 2.5],
    }

    if include_atr_stops:
        grid["stop_mode"] = ["Fixed %", "ATR"]
        grid["stop_atr_mult"] = [1.5, 2.0, 2.5]
    else:
        grid["stop_mode"] = ["Fixed %"]

    if include_trailing:
        grid["use_trailing"] = [True, False]
        grid["trail_pct"] = [1.0, 1.5, 2.0]
    else:
        grid["use_trailing"] = [False]

    return grid
