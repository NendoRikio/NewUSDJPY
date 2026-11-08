"""
Technical analysis module for FX trading.
Computes moving averages, Bollinger Bands, RSI, MACD, ATR, and support/resistance.
"""
import numpy as np
import pandas as pd


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes key technical indicators on OHLCV data.
    Required columns: ['Open', 'High', 'Low', 'Close']
    """
    if df is None or len(df) < 20:
        return df

    data = df.copy()

    # Moving Averages
    data['SMA20'] = data['Close'].rolling(window=20).mean()
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    if len(data) >= 200:
        data['SMA200'] = data['Close'].rolling(window=200).mean()
    else:
        data['SMA200'] = data['Close'].rolling(window=len(data)).mean()

    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA20'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()

    # Bollinger Bands (20, 2 std)
    rolling_std = data['Close'].rolling(window=20).std()
    data['BB_Mid'] = data['SMA20']
    data['BB_Upper2'] = data['BB_Mid'] + (rolling_std * 2)
    data['BB_Lower2'] = data['BB_Mid'] - (rolling_std * 2)
    data['BB_Upper1'] = data['BB_Mid'] + (rolling_std * 1)
    data['BB_Lower1'] = data['BB_Mid'] - (rolling_std * 1)

    # RSI (14)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss.replace(0, np.nan))
    data['RSI14'] = 100 - (100 / (1 + rs))
    data['RSI14'] = data['RSI14'].fillna(50)

    # MACD (12, 26, 9)
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']

    # ATR (14) - Critical for Pips Volatility Estimation
    high_low = data['High'] - data['Low']
    high_close = (data['High'] - data['Close'].shift()).abs()
    low_close = (data['Low'] - data['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data['ATR14'] = tr.rolling(window=14).mean()
    # Backfill ATR if NaN at start
    data['ATR14'] = data['ATR14'].bfill()

    return data


def find_support_resistance(df: pd.DataFrame, window: int = 15) -> dict:
    """
    Finds nearest key support and resistance levels.
    """
    if df is None or len(df) < window * 2:
        return {'supports': [], 'resistances': []}

    highs = df['High'].values
    lows = df['Low'].values
    curr_price = df['Close'].iloc[-1]

    resistances = []
    supports = []

    for i in range(window, len(df) - window):
        # Local peak
        if highs[i] == max(highs[i - window : i + window + 1]):
            resistances.append(highs[i])
        # Local trough
        if lows[i] == min(lows[i - window : i + window + 1]):
            supports.append(lows[i])

    # Filter levels near current price
    active_res = sorted([r for r in set(resistances) if r > curr_price])
    active_sup = sorted([s for s in set(supports) if s < curr_price], reverse=True)

    return {
        'resistances': active_res[:3],
        'supports': active_sup[:3],
        'nearest_res': active_res[0] if active_res else None,
        'nearest_sup': active_sup[0] if active_sup else None,
    }
