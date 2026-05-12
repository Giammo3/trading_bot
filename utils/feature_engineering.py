"""
Unified feature engineering for Forex AI Trading Bot.

Consolidates and standardizes features from:
  - utils/features.py      (original standalone, used by binary_02 / reversal / binary_03)
  - scripts/features/feature_engineering.py  (module version, used by combo)

Usage:
    from utils.feature_engineering import apply_all_features
    df = apply_all_features(df)
"""

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, ADXIndicator
from ta.volatility import BollingerBands


def apply_all_features(df, dropna=True):
    """
    Apply the full set of technical indicators and engineered features
    to a Forex OHLC DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: open, high, low, close.
        May contain a 'timestamp' column or a DatetimeIndex for time-based
        features; if neither is present, session features are skipped.
    dropna : bool, default True
        Whether to drop rows with NaN (introduced by rolling windows and
        the volatility-filtered z-score).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with all feature columns appended.

    Notes
    -----
    Returns a *copy*; the original DataFrame is not modified.
    """
    df = df.copy()

    # ------------------------------------------------------------------
    # Timestamp handling — support column, DatetimeIndex, or absence
    # ------------------------------------------------------------------
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        timestamps = pd.DatetimeIndex(df["timestamp"])
    elif isinstance(df.index, pd.DatetimeIndex):
        timestamps = df.index
    else:
        timestamps = None

    # Ensure numeric types
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ------------------------------------------------------------------
    # 1. CANDLE BODY & WICK
    # ------------------------------------------------------------------
    df["body_size"] = abs(df["close"] - df["open"])
    df["wick_size"] = df["high"] - df["low"]
    df["candle_type"] = (df["close"] > df["open"]).astype(int)
    df["wick_body_ratio"] = df["wick_size"] / (df["body_size"] + 1e-6)
    df["body"] = df["close"] - df["open"]                     # signed body
    df["pip_move"] = df["body"] / 0.0001                       # body in pips

    # ------------------------------------------------------------------
    # 2. MOVING AVERAGES
    # ------------------------------------------------------------------
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma10"] = df["close"].rolling(10).mean()
    df["ma50"] = df["close"].rolling(50).mean()
    df["ma200"] = df["close"].rolling(200).mean()
    df["ma50_slope"] = df["ma50"].diff()                       # trend velocity

    # ------------------------------------------------------------------
    # 3. RETURNS (decimal form: 0.01 = 1 %)
    # ------------------------------------------------------------------
    df["return"] = (df["close"] - df["open"]) / df["open"]     # candle return
    df["return_pct"] = df["close"].pct_change()                # close-to-close

    # ------------------------------------------------------------------
    # 4. VOLATILITY
    # ------------------------------------------------------------------
    df["volatility_10"] = df["return_pct"].rolling(10).std()   # financial vol

    # ------------------------------------------------------------------
    # 5. MOMENTUM INDICATORS
    # ------------------------------------------------------------------
    # RSI
    df["rsi_14"] = RSIIndicator(close=df["close"], window=14).rsi()

    # MACD
    _macd = MACD(close=df["close"])
    df["macd"] = _macd.macd()
    df["macd_signal"] = _macd.macd_signal()
    df["macd_diff"] = df["macd"] - df["macd_signal"]

    # Price momentum + acceleration
    df["momentum_5"] = df["close"] - df["close"].shift(5)
    df["acceleration_5"] = df["momentum_5"] - df["momentum_5"].shift(1)
    # Normalised acceleration (signal / noise)
    df["acceleration_5_norm"] = df["acceleration_5"] / (df["volatility_10"] + 1e-6)

    # ------------------------------------------------------------------
    # 6. TREND STRENGTH (ADX)
    # ------------------------------------------------------------------
    _adx = ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=14)
    df["adx"] = _adx.adx()
    df["plus_di"] = _adx.adx_pos()
    df["minus_di"] = _adx.adx_neg()

    # 3-class direction:  -1 (bearish), 0 (flat), +1 (bullish)
    df["adx_direction"] = 0
    df.loc[(df["adx"] > 20) & (df["plus_di"] > df["minus_di"]), "adx_direction"] = 1
    df.loc[(df["adx"] > 20) & (df["minus_di"] > df["plus_di"]), "adx_direction"] = -1

    # ------------------------------------------------------------------
    # 7. BOLLINGER BANDS
    # ------------------------------------------------------------------
    _bb = BollingerBands(close=df["close"], window=20, window_dev=2)
    df["bb_upper"] = _bb.bollinger_hband()
    df["bb_lower"] = _bb.bollinger_lband()
    df["bb_width"] = df["bb_upper"] - df["bb_lower"]

    # ------------------------------------------------------------------
    # 8. BREAKOUTS
    # ------------------------------------------------------------------
    # Rolling high/low breakouts (original)
    df["_rolling_high"] = df["high"].rolling(20).max()
    df["_rolling_low"] = df["low"].rolling(20).min()
    df["breakout_up"] = (df["close"] > df["_rolling_high"].shift(1)).astype(int)
    df["breakout_down"] = (df["close"] < df["_rolling_low"].shift(1)).astype(int)

    # Bollinger-band breakouts (alternative definition)
    df["breakout_up_bb"] = (df["close"] > df["bb_upper"]).astype(int)
    df["breakout_down_bb"] = (df["close"] < df["bb_lower"]).astype(int)

    # ------------------------------------------------------------------
    # 9. DISTANCE & MEAN REVERSION
    # ------------------------------------------------------------------
    df["distance_from_ma50"] = df["close"] - df["ma50"]

    _std_50 = df["close"].rolling(50).std()
    df["zscore_ma50"] = (df["close"] - df["ma50"]) / (_std_50 + 1e-6)
    df["zscore_ma50_clipped"] = df["zscore_ma50"].clip(-3, 3)

    # Filtered z-score — NaN when volatility is too low to trust
    _vol_low = df["volatility_10"].quantile(0.25)
    df["zscore_ma50_filtered"] = df["zscore_ma50"]
    df.loc[df["volatility_10"] < _vol_low, "zscore_ma50_filtered"] = np.nan

    # ------------------------------------------------------------------
    # 10. MARKET SESSION (requires a timestamp source)
    # ------------------------------------------------------------------
    if timestamps is not None:
        df["hour_of_day"] = timestamps.hour
        df["day_of_week"] = timestamps.dayofweek

        # Realistic EUR/USD sessions in UTC
        def _session_label(hour):
            if 0 <= hour < 5:
                return "Asia"
            elif 5 <= hour < 8:
                return "London_Open"
            elif 8 <= hour < 13:
                return "London_Morning"
            elif 13 <= hour < 17:
                return "LON_NY_Overlap"
            elif 17 <= hour < 22:
                return "NY"
            else:
                return "Off_Hours"

        _session_map = {
            "Asia": 0, "London_Open": 1, "London_Morning": 2,
            "LON_NY_Overlap": 3, "NY": 4, "Off_Hours": 5,
        }
        df["market_session"] = df["hour_of_day"].apply(_session_label)
        df["market_session_code"] = df["market_session"].map(_session_map)
        # True London-NY overlap (max liquidity)
        df["is_lon_ny_overlap"] = (df["hour_of_day"].between(13, 17)).astype(int)

    # ------------------------------------------------------------------
    # 11. LIQUIDITY PROXY (volume-free; relies on session codes above)
    # ------------------------------------------------------------------
    if timestamps is not None:
        _liquidity_weights = {0: 0.3, 1: 0.7, 2: 0.85, 3: 1.0, 4: 0.8, 5: 0.2}
        df["liquidity_proxy"] = (
            (1 / (df["wick_body_ratio"] + 1e-6)) * 0.5
            + df["market_session_code"].map(_liquidity_weights) * 0.3
            + (1 / (df["volatility_10"] + 1e-6)) * 0.2
        )
        df["valid_trade"] = (
            df["liquidity_proxy"] > df["liquidity_proxy"].quantile(0.25)
        )

    # ------------------------------------------------------------------
    # 12. CANDLESTICK PATTERNS
    # ------------------------------------------------------------------
    # Bullish engulfing
    df["bullish_engulfing"] = (
        (df["open"].shift(1) > df["close"].shift(1))       # prev candle red
        & (df["close"] > df["open"])                        # current green
        & (df["close"] > df["open"].shift(1))               # closes above prev open
        & (df["open"] < df["close"].shift(1))               # opens below prev close
    ).astype(int)

    # Doji detection
    _doji_mask = df["body_size"] < (df["high"] - df["low"]) * 0.1
    df["doji_type"] = 0                                    # not a doji
    df.loc[_doji_mask & (df["adx"] < 20), "doji_type"] = 1    # continuation
    df.loc[_doji_mask & (df["adx"] > 25), "doji_type"] = 2    # reversal

    # ------------------------------------------------------------------
    # 13. CLEANUP
    # ------------------------------------------------------------------
    # Drop intermediate helper columns
    _helpers = ["_rolling_high", "_rolling_low", "zscore_ma50", "market_session"]
    df.drop(columns=[c for c in _helpers if c in df.columns], errors="ignore", inplace=True)

    # Rows made NaN by rolling windows + z-score volatility filter
    if dropna:
        df.dropna(inplace=True)

    return df