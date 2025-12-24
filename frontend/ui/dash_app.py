import datetime as dt
import json
import logging
import os
import sys
from typing import Any, Dict, List

import pandas as pd
from dash import Dash, Input, Output, State, callback_context, dcc, html, no_update
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
)
from frontend.features.optimization import ResultConstraints, analyze_results, create_custom_grid
from frontend.features.patterns import DiscoveredRule, get_rule_summary
from frontend.features.presets import STRATEGY_PRESETS, apply_preset_direction
from frontend.ui.dash_helpers import build_backtest_figure, build_empty_figure, build_entry_diagnostics, build_trades_table
from frontend.ui.dash_layout import build_layout
from frontend.ui.dash_state import (
    DEFAULTS,
    INPUT_FIELDS,
    STRATEGY_INPUT_FIELDS,
    _build_params,
    _default_ui_values,
    _preset_to_ui_values,
    _values_from_args,
)
from frontend.ui.dash_utils import (
    _build_preset_info,
    _date_range_from_preset,
    _deserialize_df,
    _format_preset_recommendation,
    _from_checklist,
    _parse_date,
    _parse_ts,
    _preset_key_for_lookback,
    _resolve_preset_direction,
    _worker_status,
)


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

BACKTEST_CACHE: Dict[tuple, Dict[str, Any]] = {}

app = Dash(__name__)
app.title = "BB + KC + RSI Backtester"

app.layout = build_layout(DEFAULTS, _build_params(DEFAULTS))

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
