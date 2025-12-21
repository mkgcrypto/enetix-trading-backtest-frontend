from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class ResultConstraints:
    """Hard constraints applied during optimization."""
    min_trades: int = 5
    min_win_rate: Optional[float] = None
    min_profit_factor: Optional[float] = None
    max_drawdown: Optional[float] = None
    min_total_return: Optional[float] = None


def create_custom_grid(
    rsi_range: tuple = (65, 75),
    stop_range: tuple = (1.5, 3.0),
    band_mult_range: tuple = (1.8, 2.2),
    steps: int = 3,
    include_entry_modes: bool = True,
    include_exit_levels: bool = False,
    stop_mode: Optional[str] = None,
) -> Dict[str, List[Any]]:
    """Create a custom parameter grid based on ranges.

    When stop_mode is "ATR", stop_range is interpreted as an ATR multiplier range.
    """
    def linspace(start, end, n):
        return [round(start + i * (end - start) / (n - 1), 1) for i in range(n)]

    grid = {
        "rsi_min": [int(x) for x in linspace(rsi_range[0], rsi_range[1], steps)],
        "rsi_ma_min": [int(x) for x in linspace(rsi_range[0], rsi_range[1], steps)],
        "bb_std": linspace(band_mult_range[0], band_mult_range[1], steps),
        "kc_mult": linspace(band_mult_range[0], band_mult_range[1], steps),
    }

    stop_values = linspace(stop_range[0], stop_range[1], steps)
    if (stop_mode or "").strip().upper() == "ATR":
        grid["stop_atr_mult"] = stop_values
    else:
        grid["stop_pct"] = stop_values

    if include_entry_modes:
        grid["entry_band_mode"] = ["Either", "KC", "BB", "Both"]

    if include_exit_levels:
        grid["exit_level"] = ["mid", "lower"]

    return grid


def analyze_results(results_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze optimization results to find patterns."""
    if results_df.empty:
        return {"error": "No results to analyze"}

    metric_cols = {
        "profit_factor",
        "win_rate",
        "total_return",
        "max_drawdown",
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "num_trades",
        "avg_return",
    }
    param_cols = [c for c in results_df.columns if c not in metric_cols]

    best_row = results_df.iloc[0]
    best_params = {col: best_row[col] for col in param_cols}

    top_10 = results_df.head(10)
    param_modes = {}
    for col in param_cols:
        if col in top_10.columns:
            mode_val = top_10[col].mode()
            if len(mode_val) > 0:
                param_modes[col] = mode_val.iloc[0]

    recommended_ranges = {}
    for col in param_cols:
        if col in top_10.columns:
            vals = top_10[col]
            if vals.dtype in [np.float64, np.int64, float, int]:
                recommended_ranges[col] = {
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                    "mean": float(vals.mean()),
                }

    return {
        "best_params": best_params,
        "common_in_top_10": param_modes,
        "recommended_ranges": recommended_ranges,
        "top_profit_factor": float(results_df["profit_factor"].iloc[0]),
    }
