from frontend.features.presets import STRATEGY_PRESETS

CHECK_ON = "on"
INLINE_OPTION_STYLE = {
    "display": "inline-flex",
    "alignItems": "center",
    "margin": "0 10px 6px 0",
    "textTransform": "none",
    "letterSpacing": "normal",
}

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
