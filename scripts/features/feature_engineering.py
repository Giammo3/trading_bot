import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, ADXIndicator
from ta.volatility import BollingerBands

def apply_all_features(df):
    # === BODY & WICK ===
    df["body_size"] = abs(df["close"] - df["open"])
    df["wick_size"] = abs(df["high"] - df["low"])
    df["candle_type"] = np.where(df["close"] > df["open"], 1, 0)
    df["wick_body_ratio"] = df["wick_size"] / (df["body_size"] + 1e-6)

    # === MEDIE MOBILI ===
    df["ma5"] = df["close"].rolling(window=5).mean()
    df["ma10"] = df["close"].rolling(window=10).mean()
    df["ma50"] = df["close"].rolling(window=50).mean()
    df["ma200"] = df["close"].rolling(window=200).mean()

    df["ma50_slope"] = df["ma50"].diff()

    # === VOLATILITÀ ===
    df["return_pct"] = df["close"].pct_change()
    df["volatility_10"] = df["return_pct"].rolling(window=10).std()

    # === RSI ===
    df["rsi_14"] = RSIIndicator(close=df["close"], window=14).rsi()

    # === MACD ===
    macd = MACD(close=df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = df["macd"] - df["macd_signal"]

    # === BOLLINGER BANDS ===
    bb = BollingerBands(close=df["close"])
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()

    # === DISTANZE ===
    df["distance_from_ma50"] = df["close"] - df["ma50"]

    # === TIME FEATURES ===
    df["hour_of_day"] = pd.to_datetime(df["timestamp"]).dt.hour
    df["day_of_week"] = pd.to_datetime(df["timestamp"]).dt.dayofweek
    df["is_lon_ny_overlap"] = df["hour_of_day"].between(13, 17).astype(int)

    # === ENGULFING + DOJI (semplificati) ===
    df["bullish_engulfing"] = (
        (df["close"].shift(1) < df["open"].shift(1)) &
        (df["close"] > df["open"]) &
        (df["close"] > df["open"].shift(1)) &
        (df["open"] < df["close"].shift(1))
    ).astype(int)

    df["doji_type"] = ((df["body_size"] < (df["wick_size"] * 0.1))).astype(int)

    # === ADX ===
    adx = ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=14)
    df["adx"] = adx.adx()
    df["adx_direction"] = (adx.adx_pos() > adx.adx_neg()).astype(int)

    # === BREAKOUT FEATURE ===
    df["breakout_up"] = (df["close"] > df["bb_upper"]).astype(int)
    df["breakout_down"] = (df["close"] < df["bb_lower"]).astype(int)

    # === LIQUIDITY PROXY ===
    #df["liquidity_proxy"] = df["volume"].rolling(window=10).mean()

    # === ACCELERATION (norm su return) ===
    df["acceleration_5_norm"] = df["return_pct"].rolling(5).mean()

    # === ZSCORE SU MA50 (con rolling std) ===
    df["zscore_ma50_filtered"] = (
        (df["close"] - df["ma50"]) / (df["ma50"].rolling(20).std() + 1e-6)
    )

    # === Drop righe con NaN ===
    df.dropna(inplace=True)

    return df
