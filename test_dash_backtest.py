#!/usr/bin/env python
"""
Test script to verify the Dash backtest functionality works correctly.
"""
import datetime as dt
from typing import Optional
from unittest.mock import patch

IMPORT_ERROR: Optional[Exception] = None

try:
    import pandas as pd

    from frontend.ui.dash_app import (
        _default_ui_values,
        _build_params,
        run_backtest_callback,
        update_dashboard,
        update_trades,
    )
except Exception as exc:  # pragma: no cover
    pd = None  # type: ignore[assignment]
    IMPORT_ERROR = exc


def _build_dummy_results(params):
    index = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    df = pd.DataFrame(
        {
            "Open": [100, 102, 101, 103, 104],
            "High": [101, 103, 102, 104, 105],
            "Low": [99, 101, 100, 102, 103],
            "Close": [100, 102, 101, 103, 104],
            "Volume": [10, 12, 9, 11, 10],
        },
        index=index,
    )
    ds = df.copy()
    ds["bb_mid"] = ds["Close"].rolling(2, min_periods=1).mean()
    ds["bb_up"] = ds["bb_mid"] + 1
    ds["bb_low"] = ds["bb_mid"] - 1
    ds["kc_mid"] = ds["bb_mid"]
    ds["kc_up"] = ds["bb_mid"] + 1
    ds["kc_low"] = ds["bb_mid"] - 1
    ds["rsi30"] = 55
    ds["rsi30_ma"] = 50

    trades = pd.DataFrame(
        {
            "EntryBar": [1],
            "ExitBar": [3],
            "EntryTimeUTC": [index[1].isoformat()],
            "ExitTimeUTC": [index[3].isoformat()],
            "EntryPrice": [102.0],
            "ExitPrice": [103.0],
            "Side": ["Short"],
            "ExitReason": ["signal_exit"],
            "ReturnPct": [-0.01],
        }
    )

    return {
        "df": df.to_json(orient="split", date_format="iso"),
        "ds": ds.to_json(orient="split", date_format="iso"),
        "trades": trades.to_json(orient="split", date_format="iso"),
        "stats": {
            "total_equity_return_pct": 1.0,
            "win_rate": 50.0,
            "profit_factor": 1.2,
            "max_drawdown_pct": 2.0,
            "sharpe_ratio": 0.5,
        },
        "equity_curve": [10000, 10010, 10005, 10020, 10030],
        "params": params,
    }


def test_params_initialization():
    """Test that default params are built correctly."""
    if IMPORT_ERROR:
        raise RuntimeError(f"Missing optional dependencies: {IMPORT_ERROR}")
    defaults = _default_ui_values()
    params = _build_params(defaults)

    assert params is not None, "Params should not be None"
    assert len(params) > 0, "Params should have keys"
    assert "exchange" in params, "Params should have exchange"
    assert "symbol" in params, "Params should have symbol"
    assert "timeframe" in params, "Params should have timeframe"

    return params


def test_backtest_callback():
    """Test the backtest callback with a mocked API response."""
    if IMPORT_ERROR:
        raise RuntimeError(f"Missing optional dependencies: {IMPORT_ERROR}")

    defaults = _default_ui_values()
    defaults["w_start_date"] = dt.date.today() - dt.timedelta(days=30)
    defaults["w_end_date"] = dt.date.today()

    params = _build_params(defaults)
    dummy_results = _build_dummy_results(params)

    with patch("frontend.ui.dash_app.run_backtest", return_value=dummy_results):
        results, last_params, dirty, selected_trade, status = run_backtest_callback(1, params)

    assert results is not None, "Results should not be None"
    assert not isinstance(results, dict) or not results.get("error"), "Backtest should succeed"
    assert status
    assert last_params == params
    assert dirty is False
    assert selected_trade is None

    return results


def test_dashboard_update(results):
    """Test that dashboard updates correctly with results."""
    if IMPORT_ERROR:
        raise RuntimeError(f"Missing optional dependencies: {IMPORT_ERROR}")

    message, metrics, figure = update_dashboard(results, ["on"], ["on"], None)

    assert message == "" or isinstance(message, str), "Message should be empty or string"
    assert isinstance(metrics, list), "Metrics should be a list"
    assert figure is not None, "Figure should not be None"


def test_trades_update(results):
    """Test that trades table updates correctly."""
    if IMPORT_ERROR:
        raise RuntimeError(f"Missing optional dependencies: {IMPORT_ERROR}")

    (trades_msg, trades_data, trades_cols,
     diag_msg, diag_data, diag_cols) = update_trades(results)

    assert isinstance(trades_data, list)
    assert isinstance(trades_cols, list)
    assert isinstance(diag_data, list)
    assert isinstance(diag_cols, list)
    assert isinstance(trades_msg, str)
    assert isinstance(diag_msg, str)


if __name__ == "__main__":
    if IMPORT_ERROR:
        print(f"\n- SKIPPED (missing optional deps): {IMPORT_ERROR}")
    else:
        params = test_params_initialization()
        results = test_backtest_callback()
        test_dashboard_update(results)
        test_trades_update(results)
        print("\n✓ Dash backtest smoke tests passed")
