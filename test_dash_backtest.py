#!/usr/bin/env python
"""
Test script to verify the Dash backtest functionality works correctly.
"""
import datetime as dt
from unittest.mock import patch

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("dash")

from frontend.ui.dash_app import (
    _default_ui_values,
    _build_params,
    run_backtest_callback,
    update_dashboard,
    update_trades,
)


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


@pytest.fixture()
def params():
    defaults = _default_ui_values()
    defaults["w_start_date"] = dt.date.today() - dt.timedelta(days=30)
    defaults["w_end_date"] = dt.date.today()
    return _build_params(defaults)


@pytest.fixture()
def results(params):
    return _build_dummy_results(params)


def test_params_initialization():
    """Test that default params are built correctly."""
    defaults = _default_ui_values()
    params = _build_params(defaults)

    assert params is not None, "Params should not be None"
    assert len(params) > 0, "Params should have keys"
    assert "exchange" in params, "Params should have exchange"
    assert "symbol" in params, "Params should have symbol"
    assert "timeframe" in params, "Params should have timeframe"


def test_backtest_callback(params, results):
    """Test the backtest callback with a mocked API response."""
    with patch("frontend.ui.dash_app.run_backtest", return_value=results):
        run_results, last_params, dirty, selected_trade, status = run_backtest_callback(1, params)

    assert run_results is not None, "Results should not be None"
    assert not isinstance(run_results, dict) or not run_results.get("error"), "Backtest should succeed"
    assert status
    assert last_params == params
    assert dirty is False
    assert selected_trade is None


def test_dashboard_update(results):
    """Test that dashboard updates correctly with results."""
    message, metrics, figure = update_dashboard(results, ["on"], ["on"], None, ["on"])

    assert message == "" or isinstance(message, str), "Message should be empty or string"
    assert isinstance(metrics, list), "Metrics should be a list"
    assert figure is not None, "Figure should not be None"


def test_trades_update(results):
    """Test that trades table updates correctly."""
    (trades_msg, trades_data, trades_cols,
     diag_msg, diag_data, diag_cols) = update_trades(results)

    assert isinstance(trades_data, list)
    assert isinstance(trades_cols, list)
    assert isinstance(diag_data, list)
    assert isinstance(diag_cols, list)
    assert isinstance(trades_msg, str)
    assert isinstance(diag_msg, str)
