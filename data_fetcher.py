"""
Forex market data fetcher module using yfinance.
Supports USD/JPY and JPY/USD pairs.
"""
from datetime import datetime, timezone
import pandas as pd
import yfinance as yf


# Pair configuration
PAIR_CONFIG = {
    'USD/JPY': {
        'ticker': 'USDJPY=X',
        'pip_value': 0.01,         # 1 pip = 0.01 JPY (1 sen)
        'decimal_places': 3,
        'label': '米ドル / 日本円 (USD/JPY)',
        'description': '1ドルあたりの円価格'
    },
    'JPY/USD': {
        'ticker': 'JPYUSD=X',
        'pip_value': 0.000001,     # 1 pip = 0.000001 USD (10^-6 USD, equivalent to 0.01 JPY in FX)
        'decimal_places': 6,
        'label': '日本円 / 米ドル (JPY/USD)',
        'description': '1円あたりの米ドル価格 (1 pip = 0.000001 USD)'
    }
}


def fetch_pair_data(pair: str = 'USD/JPY', period: str = '1mo', interval: str = '1h') -> tuple[pd.DataFrame, dict]:
    """
    Fetches historical OHLCV data and summary statistics for the selected pair.
    Returns: (DataFrame with OHLCV, summary dict)
    """
    cfg = PAIR_CONFIG.get(pair, PAIR_CONFIG['USD/JPY'])
    ticker_sym = cfg['ticker']

    ticker = yf.Ticker(ticker_sym)
    df = ticker.history(period=period, interval=interval)

    if df.empty:
        # Fallback if specific pair failed
        if pair == 'JPY/USD':
            # Derive from USDJPY=X if JPYUSD=X is sparse
            inv_df = yf.Ticker('USDJPY=X').history(period=period, interval=interval)
            if not inv_df.empty:
                df = pd.DataFrame(index=inv_df.index)
                df['Open'] = 1.0 / inv_df['Open']
                df['High'] = 1.0 / inv_df['Low']
                df['Low'] = 1.0 / inv_df['High']
                df['Close'] = 1.0 / inv_df['Close']
                df['Volume'] = inv_df.get('Volume', 0)

    if df.empty:
        raise ValueError(f"Could not retrieve data for {ticker_sym}")

    # Remove timezone issues for plotting
    if df.index.tz is not None:
        df.index = df.index.tz_convert('Asia/Tokyo')

    latest_close = df['Close'].iloc[-1]
    prev_close = df['Close'].iloc[-2] if len(df) > 1 else latest_close
    open_today = df['Open'].iloc[-24] if len(df) >= 24 else df['Open'].iloc[0]

    change_val = latest_close - prev_close
    change_pips = change_val / cfg['pip_value']
    pct_change = (change_val / prev_close) * 100 if prev_close != 0 else 0

    high_24h = df['High'].iloc[-24:].max() if len(df) >= 24 else df['High'].max()
    low_24h = df['Low'].iloc[-24:].min() if len(df) >= 24 else df['Low'].min()

    summary = {
        'pair': pair,
        'ticker': ticker_sym,
        'current_price': latest_close,
        'prev_close': prev_close,
        'change_val': change_val,
        'change_pips': round(change_pips, 1),
        'pct_change': round(pct_change, 2),
        'high_24h': high_24h,
        'low_24h': low_24h,
        'pip_value': cfg['pip_value'],
        'decimal_places': cfg['decimal_places'],
        'timestamp': df.index[-1].strftime('%Y-%m-%d %H:%M JST'),
        'total_bars': len(df)
    }

    return df, summary
