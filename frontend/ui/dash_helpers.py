import math
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from frontend.features.metrics import calculate_drawdown


def _resolve_figure_theme(theme: str) -> dict:
    theme_value = (theme or "light").lower()
    dark_mode = theme_value == "dark"
    return {
        "dark_mode": dark_mode,
        "axis_color": "#e2e8f0" if dark_mode else "#1f2933",
        "grid_color": "rgba(148, 163, 184, 0.2)" if dark_mode else "rgba(15, 23, 42, 0.08)",
        "axis_line_color": "rgba(148, 163, 184, 0.4)" if dark_mode else "rgba(15, 23, 42, 0.2)",
        "plot_bg": "#0f1720" if dark_mode else "#ffffff",
        "bb_line_color": "#94a3b8" if dark_mode else "gray",
        "kc_line_color": "#60a5fa" if dark_mode else "blue",
        "annotation_color": "#e2e8f0" if dark_mode else "#1f2933",
    }


def _apply_figure_theme(fig: go.Figure, theme: dict) -> None:
    axis_color = theme["axis_color"]
    grid_color = theme["grid_color"]
    axis_line_color = theme["axis_line_color"]
    plot_bg = theme["plot_bg"]

    fig.update_layout(
        paper_bgcolor=plot_bg,
        plot_bgcolor=plot_bg,
        font=dict(color=axis_color),
    )
    fig.update_annotations(font_color=axis_color)
    fig.update_xaxes(
        showline=True,
        linecolor=axis_line_color,
        gridcolor=grid_color,
        zerolinecolor=grid_color,
        tickfont=dict(color=axis_color),
        titlefont=dict(color=axis_color),
    )
    fig.update_yaxes(
        showline=True,
        linecolor=axis_line_color,
        gridcolor=grid_color,
        zerolinecolor=grid_color,
        tickfont=dict(color=axis_color),
        titlefont=dict(color=axis_color),
    )


def build_empty_figure(theme: str = "light") -> go.Figure:
    fig = go.Figure()
    theme_values = _resolve_figure_theme(theme)
    _apply_figure_theme(fig, theme_values)
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=700)
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    return fig


def build_trades_table(trades_obj) -> pd.DataFrame:
    if trades_obj is None:
        return pd.DataFrame()

    t = pd.DataFrame(trades_obj).copy()
    if t.empty:
        return pd.DataFrame()

    required = [
        "EntryBar", "ExitBar", "EntryTimeUTC", "ExitTimeUTC",
        "EntryPrice", "ExitPrice", "Side", "ExitReason", "ReturnPct",
    ]

    missing = [c for c in required if c not in t.columns]
    if missing:
        raise ValueError(f"Engine trades missing required columns: {missing}")

    out = pd.DataFrame({
        "EntryTime": pd.to_datetime(t["EntryTimeUTC"], utc=True),
        "ExitTime": pd.to_datetime(t["ExitTimeUTC"], utc=True),
        "EntryBar": t["EntryBar"].astype("Int64"),
        "ExitBar": t["ExitBar"].astype("Int64"),
        "Side": t["Side"],
        "Price@Entry": pd.to_numeric(t["EntryPrice"]),
        "Price@Exit": pd.to_numeric(t["ExitPrice"]),
        "ExitReason": t["ExitReason"],
    })

    optional = [
        "Size", "NotionalEntry", "RealizedPnL",
        "EquityAfter", "EquityBefore", "R_multiple", "SkippedDailyLoss",
        "EffectiveLeverage", "MarginUtilAtEntry", "LiqPrice",
    ]
    for col in optional:
        if col in t.columns:
            out[col] = t[col]

    ret_pct = pd.to_numeric(t["ReturnPct"], errors="coerce")
    out["Price Move %"] = (ret_pct * 100).round(3)
    out["Duration"] = out["ExitTime"] - out["EntryTime"]
    side = out["Side"].fillna("Short")
    pnl_per_unit = out["Price@Exit"] - out["Price@Entry"]
    out["PnL (per unit)"] = pnl_per_unit.where(side == "Long", -pnl_per_unit)

    return out


def build_entry_diagnostics(trades_table: pd.DataFrame, ds: pd.DataFrame) -> pd.DataFrame:
    if trades_table is None or trades_table.empty:
        return pd.DataFrame()

    time_col = "EntryTimeUTC" if "EntryTimeUTC" in trades_table.columns else (
        "EntryTime" if "EntryTime" in trades_table.columns else None
    )
    bar_col = "EntryBar" if "EntryBar" in trades_table.columns else None
    if bar_col is None:
        return pd.DataFrame()

    has_exit_reason = "ExitReason" in trades_table.columns
    margin_mode = False
    if "MarginUtilAtEntry" in trades_table.columns:
        if trades_table["MarginUtilAtEntry"].notna().any():
            margin_mode = True

    rows = []
    for _, r in trades_table.iterrows():
        ei = r.get(bar_col)
        try:
            ei = int(ei) if pd.notna(ei) else None
        except Exception:
            ei = None

        side = r.get("Side", "Short")

        def _safe(col, idx):
            try:
                return float(ds[col].iloc[idx])
            except Exception:
                return float("nan")

        equity_before = float(r.get("EquityBefore")) if "EquityBefore" in trades_table.columns and pd.notna(r.get("EquityBefore")) else float("nan")
        equity_after = float(r.get("EquityAfter")) if "EquityAfter" in trades_table.columns and pd.notna(r.get("EquityAfter")) else float("nan")
        size = float(r.get("Size")) if "Size" in trades_table.columns and pd.notna(r.get("Size")) else float("nan")
        notional = float(r.get("NotionalEntry")) if "NotionalEntry" in trades_table.columns and pd.notna(r.get("NotionalEntry")) else float("nan")

        if equity_before and not math.isnan(equity_before) and equity_before > 0:
            effective_lev = notional / equity_before
        else:
            effective_lev = float("nan")

        if margin_mode and equity_before and not math.isnan(equity_before) and equity_before > 0:
            margin_util = r.get("MarginUtilAtEntry", float("nan"))
        else:
            margin_util = float("nan")

        if ei is None or ei < 0 or ei >= len(ds):
            row = {
                "EntryTime": r.get(time_col),
                "EntryBar": r.get(bar_col),
                "Side": side,
                "Price@Entry (ds)": float("nan"),
                "RSI@Entry": float("nan"),
                "RSI_MA@Entry": float("nan"),
                "TouchKC?": None,
                "TouchBB?": None,
                "BandOK?": None,
                "Size": size,
                "NotionalEntry": notional,
                "EquityBefore": equity_before,
                "EquityAfter": equity_after,
                "RealizedPnL": r.get("RealizedPnL"),
                "EffectiveLeverage": effective_lev,
            }

            if margin_mode:
                r_mult = r.get("R_multiple")
                liq = r.get("LiqPrice")

                row["R_multiple"] = r_mult
                row["MarginUtilAtEntry"] = margin_util
                row["HighMarginUtil(>80%)"] = (
                    isinstance(margin_util, (int, float))
                    and not math.isnan(margin_util)
                    and margin_util > 0.8
                )
                row["LiqPrice"] = liq

            if has_exit_reason:
                row["ExitReason"] = r["ExitReason"]
            rows.append(row)
            continue

        px = _safe("Close", ei)
        high_px = _safe("High", ei)
        low_px = _safe("Low", ei)
        rsi_v = _safe("rsi30", ei) if "rsi30" in ds.columns else float("nan")
        rma_v = _safe("rsi30_ma", ei) if "rsi30_ma" in ds.columns else float("nan")
        bb_u = _safe("bb_up", ei) if "bb_up" in ds.columns else float("nan")
        bb_l = _safe("bb_low", ei) if "bb_low" in ds.columns else float("nan")
        kc_u = _safe("kc_up", ei) if "kc_up" in ds.columns else float("nan")
        kc_l = _safe("kc_low", ei) if "kc_low" in ds.columns else float("nan")

        if side == "Long":
            touch_kc = (low_px <= kc_l) if pd.notna(kc_l) else False
            touch_bb = (low_px <= bb_l) if pd.notna(bb_l) else False
        else:
            touch_kc = (high_px >= kc_u) if pd.notna(kc_u) else False
            touch_bb = (high_px >= bb_u) if pd.notna(bb_u) else False

        row = {
            "EntryTime": r.get(time_col),
            "EntryBar": ei,
            "Side": side,
            "Price@Entry (ds)": px,
            "RSI@Entry": rsi_v,
            "RSI_MA@Entry": rma_v,
            "TouchKC?": bool(touch_kc),
            "TouchBB?": bool(touch_bb),
            "Size": size,
            "NotionalEntry": notional,
            "EquityBefore": equity_before,
            "EquityAfter": equity_after,
            "RealizedPnL": r.get("RealizedPnL"),
            "EffectiveLeverage": effective_lev,
        }

        if margin_mode:
            r_mult = r.get("R_multiple")
            liq = r.get("LiqPrice")

            row["R_multiple"] = r_mult
            row["MarginUtilAtEntry"] = margin_util
            row["HighMarginUtil(>80%)"] = (
                isinstance(margin_util, (int, float))
                and not math.isnan(margin_util)
                and margin_util > 0.8
            )
            row["LiqPrice"] = liq

        if has_exit_reason:
            row["ExitReason"] = r["ExitReason"]
        rows.append(row)

    return pd.DataFrame(rows)


def build_backtest_figure(
    ds: pd.DataFrame,
    trades_table: pd.DataFrame,
    equity_curve,
    params: dict,
    *,
    selected_trade: Optional[dict] = None,
    show_candles: bool = True,
    lock_rsi_y: bool = True,
    theme: str = "light",
):
    if ds is None or ds.empty:
        return build_empty_figure(theme)

    theme_values = _resolve_figure_theme(theme)
    dark_mode = theme_values["dark_mode"]
    axis_color = theme_values["axis_color"]
    bb_line_color = theme_values["bb_line_color"]
    kc_line_color = theme_values["kc_line_color"]
    annotation_color = theme_values["annotation_color"]

    ds_full = ds
    ds_plot = ds_full.copy()
    max_points = 5000
    if len(ds_plot) > max_points:
        step = max(1, len(ds_plot) // max_points)
        ds_plot = ds_plot.iloc[::step]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.50, 0.25, 0.25],
        subplot_titles=("Price + BB + KC", "RSI (30m) + MA", "Equity & Drawdown"),
    )

    if show_candles:
        fig.add_trace(
            go.Candlestick(
                x=ds_plot.index,
                open=ds_plot["Open"],
                high=ds_plot["High"],
                low=ds_plot["Low"],
                close=ds_plot["Close"],
                name="Price",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(x=ds_plot.index, y=ds_plot["Close"], mode="lines", name="Close"),
            row=1,
            col=1,
        )

    bb_basis_type = params.get("bb_basis_type", "sma")
    kc_mid_type = params.get("kc_mid_type", "ema")
    rsi_ma_type = params.get("rsi_ma_type", "sma")

    fig.add_trace(
        go.Scatter(
            x=ds_plot.index,
            y=ds_plot["bb_mid"],
            mode="lines",
            name=f"BB mid ({bb_basis_type.upper()})",
            line=dict(width=1, dash="dot"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=ds_plot.index,
            y=ds_plot["bb_up"],
            mode="lines",
            name="BB upper",
            line=dict(color=bb_line_color, width=1),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=ds_plot.index,
            y=ds_plot["bb_low"],
            mode="lines",
            name="BB lower",
            line=dict(color=bb_line_color, width=1),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=ds_plot.index,
            y=ds_plot["kc_mid"],
            mode="lines",
            name=f"KC mid ({kc_mid_type.upper()})",
            line=dict(width=1),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=ds_plot.index,
            y=ds_plot["kc_up"],
            mode="lines",
            name="KC upper",
            line=dict(color=kc_line_color, width=1),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=ds_plot.index,
            y=ds_plot["kc_low"],
            mode="lines",
            name="KC lower",
            line=dict(color=kc_line_color, width=1),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=ds_plot.index,
            y=ds_plot["rsi30"],
            mode="lines",
            name="RSI(30m)",
            line=dict(color="purple"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=ds_plot.index,
            y=ds_plot["rsi30_ma"],
            mode="lines",
            name=f"RSI MA ({rsi_ma_type.upper()})",
            line=dict(color="orange"),
        ),
        row=2,
        col=1,
    )

    rsi_min = params.get("rsi_min", 70)
    rsi_ma_min = params.get("rsi_ma_min", 70)
    rsi_max = params.get("rsi_max")
    rsi_ma_max = params.get("rsi_ma_max")
    fig.add_hline(
        y=rsi_min,
        line_dash="dot",
        line_color=bb_line_color,
        annotation_text="RSI Min (Short)",
        annotation_font_color=annotation_color,
        row=2,
        col=1,
    )
    fig.add_hline(
        y=rsi_ma_min,
        line_dash="dot",
        line_color=bb_line_color,
        annotation_text="RSI MA Min (Short)",
        annotation_font_color=annotation_color,
        row=2,
        col=1,
    )
    if rsi_max is not None:
        try:
            fig.add_hline(
                y=float(rsi_max),
                line_dash="dot",
                line_color="green",
                annotation_text="RSI Max (Long)",
                annotation_font_color=annotation_color,
                row=2,
                col=1,
            )
        except (TypeError, ValueError):
            pass
    if rsi_ma_max is not None:
        try:
            fig.add_hline(
                y=float(rsi_ma_max),
                line_dash="dot",
                line_color="darkgreen",
                annotation_text="RSI MA Max (Long)",
                annotation_font_color=annotation_color,
                row=2,
                col=1,
            )
        except (TypeError, ValueError):
            pass

    if equity_curve is not None and len(equity_curve) > 0:
        if len(ds_plot) < len(equity_curve):
            step = max(1, len(equity_curve) // len(ds_plot))
            eq_sampled = equity_curve[::step][: len(ds_plot)]
        else:
            eq_sampled = equity_curve[: len(ds_plot)]

        eq_sampled = eq_sampled[: len(ds_plot.index)]

        drawdown_pct, _ = calculate_drawdown(eq_sampled)

        fig.add_trace(
            go.Scatter(
                x=ds_plot.index[: len(eq_sampled)],
                y=eq_sampled,
                mode="lines",
                name="Equity",
                line=dict(color="royalblue", width=2),
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=ds_plot.index[: len(drawdown_pct)],
                y=-drawdown_pct,
                mode="lines",
                fill="tozeroy",
                name="Drawdown %",
                line=dict(color="crimson", width=1),
                fillcolor="rgba(220, 20, 60, 0.3)",
            ),
            row=3,
            col=1,
        )

    if selected_trade is None:
        if trades_table is not None and not trades_table.empty:
            if "Side" in trades_table.columns:
                short_trades = trades_table[trades_table["Side"] == "Short"]
                long_trades = trades_table[trades_table["Side"] == "Long"]

                if not short_trades.empty:
                    fig.add_trace(
                        go.Scattergl(
                            x=short_trades["EntryTime"],
                            y=short_trades["Price@Entry"],
                            mode="markers",
                            marker=dict(size=8, symbol="triangle-down", color="crimson"),
                            name="Short Entry",
                            showlegend=True,
                        ),
                        row=1,
                        col=1,
                    )
                    fig.add_trace(
                        go.Scattergl(
                            x=short_trades["ExitTime"],
                            y=short_trades["Price@Exit"],
                            mode="markers",
                            marker=dict(size=8, symbol="circle-open", color="crimson"),
                            name="Short Exit",
                            showlegend=True,
                        ),
                        row=1,
                        col=1,
                    )

                if not long_trades.empty:
                    fig.add_trace(
                        go.Scattergl(
                            x=long_trades["EntryTime"],
                            y=long_trades["Price@Entry"],
                            mode="markers",
                            marker=dict(size=8, symbol="triangle-up", color="seagreen"),
                            name="Long Entry",
                            showlegend=True,
                        ),
                        row=1,
                        col=1,
                    )
                    fig.add_trace(
                        go.Scattergl(
                            x=long_trades["ExitTime"],
                            y=long_trades["Price@Exit"],
                            mode="markers",
                            marker=dict(size=8, symbol="circle-open", color="seagreen"),
                            name="Long Exit",
                            showlegend=True,
                        ),
                        row=1,
                        col=1,
                    )
            else:
                fig.add_trace(
                    go.Scattergl(
                        x=trades_table["EntryTime"],
                        y=trades_table["Price@Entry"],
                        mode="markers",
                        marker=dict(size=8, symbol="triangle-down", color="red"),
                        name="Entry",
                        showlegend=True,
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Scattergl(
                        x=trades_table["ExitTime"],
                        y=trades_table["Price@Exit"],
                        mode="markers",
                        marker=dict(size=8, symbol="circle-dot", color="green"),
                        name="Exit",
                        showlegend=True,
                    ),
                    row=1,
                    col=1,
                )
    else:
        try:
            if "EntryBar" in selected_trade and pd.notna(selected_trade.get("EntryBar")):
                ei = int(selected_trade["EntryBar"])
                xi_raw = selected_trade.get("ExitBar", ei)
                xi = int(xi_raw) if (xi_raw is not None and str(xi_raw).isdigit()) else ei
            else:
                e_ts = pd.to_datetime(selected_trade.get("EntryTime"))
                x_ts = pd.to_datetime(selected_trade.get("ExitTime") or e_ts)
                ei = int(ds_full.index.get_indexer([e_ts], method="nearest")[0])
                xi = int(ds_full.index.get_indexer([x_ts], method="nearest")[0])

            pad = 30
            i0 = max(0, min(ei, xi) - pad)
            i1 = min(len(ds_full) - 1, max(ei, xi) + pad)
            x0, x1 = ds_full.index[i0], ds_full.index[i1]

            fig.update_xaxes(range=[x0, x1], row=1, col=1)
            fig.update_xaxes(range=[x0, x1], row=2, col=1)
            fig.update_xaxes(range=[x0, x1], row=3, col=1)

            window_high = float(ds_full["High"].iloc[i0 : i1 + 1].max())
            window_low = float(ds_full["Low"].iloc[i0 : i1 + 1].min())
            span = max(1e-9, window_high - window_low)
            pad_y = 0.07 * span
            fig.update_yaxes(range=[window_low - pad_y, window_high + pad_y], row=1, col=1)

            e_ts = pd.to_datetime(selected_trade.get("EntryTime")) if "EntryTime" in selected_trade else ds_full.index[ei]
            x_ts = pd.to_datetime(selected_trade.get("ExitTime")) if "ExitTime" in selected_trade else (ds_full.index[xi] if xi is not None else None)
            e_px = selected_trade.get("Price@Entry")
            x_px = selected_trade.get("Price@Exit")

            side = selected_trade.get("Side", "Short") if isinstance(selected_trade, dict) else "Short"
            entry_symbol = "triangle-up" if side == "Long" else "triangle-down"
            entry_color = "seagreen" if side == "Long" else "crimson"

            if pd.notna(e_ts) and pd.notna(e_px):
                fig.add_trace(
                    go.Scattergl(
                        x=[e_ts],
                        y=[e_px],
                        mode="markers",
                        marker=dict(size=16, symbol=entry_symbol, color=entry_color, line=dict(width=1)),
                        name="Selected Entry",
                        showlegend=True,
                    ),
                    row=1,
                    col=1,
                )
                fig.add_vline(x=e_ts, line_width=1.5, line_dash="dot", line_color=entry_color, row=1, col=1)

            if pd.notna(x_ts) and pd.notna(x_px):
                fig.add_trace(
                    go.Scattergl(
                        x=[x_ts],
                        y=[x_px],
                        mode="markers",
                        marker=dict(size=16, symbol="x", color="limegreen", line=dict(width=2)),
                        name="Selected Exit",
                        showlegend=True,
                    ),
                    row=1,
                    col=1,
                )
                fig.add_vline(x=x_ts, line_width=1.5, line_dash="dot", line_color="limegreen", row=1, col=1)
        except Exception:
            pass

    legend = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    if dark_mode:
        legend.update({
            "font": {"color": axis_color},
            "bgcolor": "rgba(12, 18, 26, 0.6)",
            "bordercolor": "rgba(42, 52, 64, 0.6)",
            "borderwidth": 1,
        })

    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        legend=legend,
        hovermode="x unified",
        height=700,
    )
    _apply_figure_theme(fig, theme_values)
    if lock_rsi_y:
        fig.update_yaxes(range=[0, 100], fixedrange=True, row=2, col=1)
    fig.update_xaxes(rangeslider=dict(visible=False), row=1, col=1)
    fig.update_xaxes(rangeslider=dict(visible=False), row=2, col=1)
    fig.update_xaxes(rangeslider=dict(visible=False), row=3, col=1)
    fig.update_yaxes(title_text="Equity / DD%", row=3, col=1)

    return fig
