"""
Technical analysis module for FX trading.
Computes moving averages, Bollinger Bands, RSI, MACD, ATR,
Ichimoku Kinko Hyo, ADX/DMI, Slow Stochastics, and support/resistance levels.
"""
from datetime import datetime
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

    # --- 1. Moving Averages ---
    data['SMA20'] = data['Close'].rolling(window=20).mean()
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    if len(data) >= 200:
        data['SMA200'] = data['Close'].rolling(window=200).mean()
    else:
        data['SMA200'] = data['Close'].rolling(window=len(data)).mean()

    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA20'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()

    # --- 2. Bollinger Bands (20, 2 std & 1 std) ---
    rolling_std = data['Close'].rolling(window=20).std()
    data['BB_Mid'] = data['SMA20']
    data['BB_Upper2'] = data['BB_Mid'] + (rolling_std * 2)
    data['BB_Lower2'] = data['BB_Mid'] - (rolling_std * 2)
    data['BB_Upper1'] = data['BB_Mid'] + (rolling_std * 1)
    data['BB_Lower1'] = data['BB_Mid'] - (rolling_std * 1)
    # Band Width & %B
    data['BB_Width'] = (data['BB_Upper2'] - data['BB_Lower2']) / data['BB_Mid'].replace(0, np.nan)
    data['BB_PctB'] = (data['Close'] - data['BB_Lower2']) / (data['BB_Upper2'] - data['BB_Lower2']).replace(0, np.nan)

    # --- 3. RSI (14) ---
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss.replace(0, np.nan))
    data['RSI14'] = 100 - (100 / (1 + rs))
    data['RSI14'] = data['RSI14'].fillna(50)

    # --- 4. MACD (12, 26, 9) ---
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']

    # --- 5. ATR (14) ---
    high_low = data['High'] - data['Low']
    high_close = (data['High'] - data['Close'].shift()).abs()
    low_close = (data['Low'] - data['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data['ATR14'] = tr.rolling(window=14).mean()
    data['ATR14'] = data['ATR14'].bfill()

    # --- 6. Ichimoku Kinko Hyo (一目均衡表 9, 26, 52) ---
    high9 = data['High'].rolling(window=9).max()
    low9 = data['Low'].rolling(window=9).min()
    data['Ichimoku_Tenkan'] = (high9 + low9) / 2  # 転換線

    high26 = data['High'].rolling(window=26).max()
    low26 = data['Low'].rolling(window=26).min()
    data['Ichimoku_Kijun'] = (high26 + low26) / 2  # 基準線

    # 先行スパンA: (転換線 + 基準線) / 2 を26期間先行
    span_a_raw = (data['Ichimoku_Tenkan'] + data['Ichimoku_Kijun']) / 2
    data['Ichimoku_SpanA'] = span_a_raw.shift(26)

    # 先行スパンB: 過去52期間の最高値・最安値半値を26期間先行
    high52 = data['High'].rolling(window=52).max()
    low52 = data['Low'].rolling(window=52).min()
    span_b_raw = (high52 + low52) / 2
    data['Ichimoku_SpanB'] = span_b_raw.shift(26)

    # 直近の雲の厚みと位置（先行スパン実体）
    # ※シフト前の値（現在の最新価格に対する現在時点の雲実体）
    data['Ichimoku_Cloud_Top'] = np.maximum(data['Ichimoku_SpanA'], data['Ichimoku_SpanB'])
    data['Ichimoku_Cloud_Bottom'] = np.minimum(data['Ichimoku_SpanA'], data['Ichimoku_SpanB'])

    # --- 7. ADX & DMI (14) - トレンド強度と方向性判定 ---
    up_move = data['High'] - data['High'].shift(1)
    down_move = data['Low'].shift(1) - data['Low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr_smooth = data['ATR14'].replace(0, np.nan)
    data['Plus_DI14'] = 100 * (pd.Series(plus_dm, index=data.index).rolling(window=14).mean() / atr_smooth)
    data['Minus_DI14'] = 100 * (pd.Series(minus_dm, index=data.index).rolling(window=14).mean() / atr_smooth)
    data['Plus_DI14'] = data['Plus_DI14'].fillna(20)
    data['Minus_DI14'] = data['Minus_DI14'].fillna(20)

    dx = 100 * (data['Plus_DI14'] - data['Minus_DI14']).abs() / (data['Plus_DI14'] + data['Minus_DI14']).replace(0, np.nan)
    data['ADX14'] = dx.rolling(window=14).mean()
    data['ADX14'] = data['ADX14'].bfill().fillna(20)

    # --- 8. Slow Stochastics (14, 3, 3) ---
    low14 = data['Low'].rolling(window=14).min()
    high14 = data['High'].rolling(window=14).max()
    fast_k = 100 * (data['Close'] - low14) / (high14 - low14).replace(0, np.nan)
    data['Stoch_K'] = fast_k.rolling(window=3).mean().fillna(50)
    data['Stoch_D'] = data['Stoch_K'].rolling(window=3).mean().fillna(50)

    return data


def find_support_resistance(df: pd.DataFrame, window: int = 15) -> dict:
    """
    Finds nearest key support and resistance levels.
    """
    if df is None or len(df) < window * 2:
        return {'supports': [], 'resistances': [], 'nearest_res': None, 'nearest_sup': None}

    highs = df['High'].values
    lows = df['Low'].values
    curr_price = df['Close'].iloc[-1]

    resistances = []
    supports = []

    for i in range(window, len(df) - window):
        if highs[i] == max(highs[i - window : i + window + 1]):
            resistances.append(highs[i])
        if lows[i] == min(lows[i - window : i + window + 1]):
            supports.append(lows[i])

    active_res = sorted([r for r in set(resistances) if r > curr_price])
    active_sup = sorted([s for s in set(supports) if s < curr_price], reverse=True)

    return {
        'resistances': active_res[:3],
        'supports': active_sup[:3],
        'nearest_res': active_res[0] if active_res else None,
        'nearest_sup': active_sup[0] if active_sup else None,
    }


def analyze_price_action(df: pd.DataFrame) -> dict:
    """
    Analyzes recent candlestick price action (pin bars, candles, momentum).
    """
    if df is None or len(df) < 3:
        return {'signal': 'NEUTRAL', 'description': 'データ不足', 'score_bias': 0.0}

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    body = abs(latest['Close'] - latest['Open'])
    total_range = latest['High'] - latest['Low']
    if total_range <= 0:
        total_range = 1e-6

    upper_wick = latest['High'] - max(latest['Close'], latest['Open'])
    lower_wick = min(latest['Close'], latest['Open']) - latest['Low']

    # Pin bar detection (wick is > 60% of candle range)
    is_bullish_pin = (lower_wick / total_range > 0.58) and (body / total_range < 0.35)
    is_bearish_pin = (upper_wick / total_range > 0.58) and (body / total_range < 0.35)

    # Bullish / Bearish Engulfing
    is_bullish_engulfing = (latest['Close'] > latest['Open']) and (prev['Close'] < prev['Open']) and (latest['Close'] > prev['Open']) and (latest['Open'] < prev['Close'])
    is_bearish_engulfing = (latest['Close'] < latest['Open']) and (prev['Close'] > prev['Open']) and (latest['Close'] < prev['Open']) and (latest['Open'] > prev['Close'])

    if is_bullish_pin:
        return {'signal': 'BULLISH', 'pattern': '長い下ヒゲ（下値支持ピンバー）', 'score_bias': 4.0}
    elif is_bearish_pin:
        return {'signal': 'BEARISH', 'pattern': '長い上ヒゲ（上値抵抗ピンバー）', 'score_bias': -4.0}
    elif is_bullish_engulfing:
        return {'signal': 'BULLISH', 'pattern': '強気の包み足（陽線アウトサイドバー）', 'score_bias': 4.0}
    elif is_bearish_engulfing:
        return {'signal': 'BEARISH', 'pattern': '弱気の包み足（陰線アウトサイドバー）', 'score_bias': -4.0}
    else:
        return {'signal': 'NEUTRAL', 'pattern': '通常ローソク足形成中', 'score_bias': 0.0}


def get_advanced_technical_summary(df: pd.DataFrame, curr_price: float) -> dict:
    """
    Produces a comprehensive diagnostic of all technical indicators
    including market regime (trending vs ranging), Ichimoku cloud state,
    ADX momentum, and Stochastics crossover.
    """
    if df is None or len(df) < 30:
        return {}

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    # 1. Market Regime (ADX + Bollinger Band Width)
    adx = latest.get('ADX14', 20.0)
    plus_di = latest.get('Plus_DI14', 20.0)
    minus_di = latest.get('Minus_DI14', 20.0)
    bb_pct_b = latest.get('BB_PctB', 0.5)

    if adx >= 25.0:
        if plus_di > minus_di:
            regime = 'STRONG_UPTREND'
            regime_label = '強力な上昇トレンド相場'
            regime_desc = f'ADX {adx:.1f} (トレンド発生中) | +DI ({plus_di:.1f}) が -DI ({minus_di:.1f}) を圧倒'
            regime_color = '#10b981'
        else:
            regime = 'STRONG_DOWNTREND'
            regime_label = '強力な下落トレンド相場'
            regime_desc = f'ADX {adx:.1f} (トレンド発生中) | -DI ({minus_di:.1f}) が +DI ({plus_di:.1f}) を圧倒'
            regime_color = '#ef4444'
    elif adx >= 20.0:
        regime = 'DEVELOPING_TREND'
        regime_label = 'トレンド形成初動'
        regime_desc = f'ADX {adx:.1f} (方向性模索中) | DI差: {plus_di - minus_di:+.1f}'
        regime_color = '#38bdf8'
    else:
        regime = 'RANGE'
        regime_label = 'レンジ・もみ合い相場'
        regime_desc = f'ADX {adx:.1f} (弱トレンド) | オシレーター逆張りが有効な局面'
        regime_color = '#f59e0b'

    # 2. Ichimoku Analysis
    tenkan = latest.get('Ichimoku_Tenkan', curr_price)
    kijun = latest.get('Ichimoku_Kijun', curr_price)
    span_a = latest.get('Ichimoku_SpanA', curr_price)
    span_b = latest.get('Ichimoku_SpanB', curr_price)

    # Cloud position
    cloud_top = max(span_a, span_b) if not pd.isna(span_a) and not pd.isna(span_b) else None
    cloud_bottom = min(span_a, span_b) if not pd.isna(span_a) and not pd.isna(span_b) else None

    if cloud_top and curr_price > cloud_top:
        ichimoku_pos = 'ABOVE_CLOUD'
        ichimoku_pos_label = '雲の上（強力な買い支持帯の上を推移）'
        ichimoku_color = '#10b981'
    elif cloud_bottom and curr_price < cloud_bottom:
        ichimoku_pos = 'BELOW_CLOUD'
        ichimoku_pos_label = '雲の下（上値の重い抵抗帯の下を推移）'
        ichimoku_color = '#ef4444'
    else:
        ichimoku_pos = 'IN_CLOUD'
        ichimoku_pos_label = '雲の中（もみ合い・方向感乏しい）'
        ichimoku_color = '#f59e0b'

    # Tenkan / Kijun signal
    tk_cross = 'BULLISH' if tenkan > kijun else ('BEARISH' if tenkan < kijun else 'NEUTRAL')
    tk_cross_label = '好転（転換線 > 基準線）' if tk_cross == 'BULLISH' else '逆転（転換線 < 基準線）'

    # 3. Slow Stochastics Analysis
    stoch_k = latest.get('Stoch_K', 50.0)
    stoch_d = latest.get('Stoch_D', 50.0)
    prev_k = prev.get('Stoch_K', 50.0)
    prev_d = prev.get('Stoch_D', 50.0)

    is_gc = (prev_k <= prev_d) and (stoch_k > stoch_d)
    is_dc = (prev_k >= prev_d) and (stoch_k < stoch_d)

    if stoch_k > 80:
        stoch_status = '買われすぎ圏 (>80)'
        stoch_color = '#ef4444'
    elif stoch_k < 20:
        stoch_status = '売られすぎ圏 (<20)'
        stoch_color = '#10b981'
    elif is_gc:
        stoch_status = 'ゴールデンクロス発生 (反発サイン)'
        stoch_color = '#10b981'
    elif is_dc:
        stoch_status = 'デッドクロス発生 (失速サイン)'
        stoch_color = '#ef4444'
    else:
        stoch_status = f'%K: {stoch_k:.1f} / %D: {stoch_d:.1f}'
        stoch_color = '#94a3b8'

    # 4. Bollinger Band Analysis
    is_bandwalk_up = bb_pct_b >= 0.85 and curr_price >= latest.get('BB_Upper1', curr_price)
    is_bandwalk_down = bb_pct_b <= 0.15 and curr_price <= latest.get('BB_Lower1', curr_price)

    bb_state = 'NORMAL'
    if is_bandwalk_up:
        bb_state = 'BANDWALK_UP'
        bb_state_label = 'バンドウォーク上昇（+1σ〜+2σ上を猛進中）'
    elif is_bandwalk_down:
        bb_state = 'BANDWALK_DOWN'
        bb_state_label = 'バンドウォーク下落（-1σ〜-2σ下を急落中）'
    elif latest.get('BB_Width', 0.05) < df['BB_Width'].rolling(20).mean().iloc[-1] * 0.7:
        bb_state = 'SQUEEZE'
        bb_state_label = 'スクイーズ収縮（大相場の前兆・ブレイク直前）'
    else:
        bb_state_label = 'バンド内推移'

    # Price action
    pa = analyze_price_action(df)

    return {
        'regime': regime,
        'regime_label': regime_label,
        'regime_desc': regime_desc,
        'regime_color': regime_color,
        'adx': adx,
        'plus_di': plus_di,
        'minus_di': minus_di,
        'ichimoku_pos': ichimoku_pos,
        'ichimoku_pos_label': ichimoku_pos_label,
        'ichimoku_color': ichimoku_color,
        'tenkan': tenkan,
        'kijun': kijun,
        'tk_cross_label': tk_cross_label,
        'stoch_k': stoch_k,
        'stoch_d': stoch_d,
        'stoch_status': stoch_status,
        'stoch_color': stoch_color,
        'bb_pct_b': bb_pct_b,
        'bb_state': bb_state,
        'bb_state_label': bb_state_label,
        'price_action': pa
    }
