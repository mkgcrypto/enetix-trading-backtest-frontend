from __future__ import annotations

import numpy as np


def calculate_drawdown(equity_curve: np.ndarray) -> tuple:
    """
    Calculate drawdown series and maximum drawdown from equity curve.

    Args:
        equity_curve: Array of equity values over time.

    Returns:
        Tuple of (drawdown_pct_array, max_drawdown_pct).
    """
    if equity_curve is None or len(equity_curve) == 0:
        return np.array([]), 0.0

    running_max = np.maximum.accumulate(equity_curve)
    drawdown_pct = (running_max - equity_curve) / running_max * 100.0
    max_drawdown_pct = float(np.nanmax(drawdown_pct))

    return drawdown_pct, max_drawdown_pct
