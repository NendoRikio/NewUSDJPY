"""
Forex AI Brain Engine.
Combines Technical Analysis, CME FedWatch, News Sentiment, and LLM (Gemini)
to estimate price direction and Pips movement 5 to 6 hours into the future.
"""
from datetime import datetime, timedelta
import math
import os
import pandas as pd
import numpy as np


def analyze_market_hours() -> tuple[str, float]:
    """
    Evaluates current global FX session and volatility multiplier based on JST.
    Tokyo: 09:00 - 15:00 JST (vol factor: 0.85)
    London: 16:00 - 01:00 JST (vol factor: 1.25)
    New York: 21:00 - 06:00 JST (vol factor: 1.40)
    Overlap (London & NY): 21:00 - 01:00 JST (vol factor: 1.65)
    """
    now_hour = datetime.now().hour
    if 21 <= now_hour or now_hour < 1:
        return "ロンドン & NY 重複市場 (最大流動性・急変動警戒)", 1.55
    elif 1 <= now_hour < 6:
        return "ニューヨーク市場 後半 (トレンド継続・手仕舞い)", 1.15
    elif 6 <= now_hour < 9:
        return "オセアニア市場 (薄商い・突発的変動注意)", 0.75
    elif 9 <= now_hour < 15:
        return "東京市場 (実需・仲値・比較的安定推移)", 0.90
    elif 15 <= now_hour < 17:
        return "欧州参入序盤 (欧州勢の初動)", 1.10
    else:
        return "ロンドン市場 (欧州本格参入・活発)", 1.35


def compute_pips_prediction(
    df: pd.DataFrame,
    summary: dict,
    fedwatch: dict,
    news_summary: dict,
    target_hours: float = 5.5,
    pair: str = 'USD/JPY'
) -> dict:
    """
    Multi-factor predictive model for 5-6 hours forecast.
    """
    curr_price = summary['current_price']
    pip_val = summary['pip_value']
    is_usdjpy = (pair == 'USD/JPY')

    # 1. Base Volatility from 14-period 1H ATR
    latest_atr = df['ATR14'].iloc[-1] if 'ATR14' in df.columns else (curr_price * 0.0015)
    if pd.isna(latest_atr) or latest_atr <= 0:
        latest_atr = curr_price * 0.0015

    session_name, vol_mult = analyze_market_hours()

    # Scaling ATR by sqrt(target_hours) as standard diffusion
    # Average expected total move range in price units
    expected_range_price = latest_atr * math.sqrt(target_hours) * vol_mult
    expected_range_pips = expected_range_price / pip_val

    # 2. Factor 1: Technical Score (-40 to +40, where + is USD bullish / JPY bearish)
    tech_score = 0.0
    latest = df.iloc[-1]

    # Trend: Price vs SMA20, SMA50, SMA200
    if 'SMA20' in df.columns and not pd.isna(latest['SMA20']):
        if curr_price > latest['SMA20']:
            tech_score += 8
        else:
            tech_score -= 8

    if 'SMA50' in df.columns and not pd.isna(latest['SMA50']):
        if curr_price > latest['SMA50']:
            tech_score += 8
        else:
            tech_score -= 8

    if 'EMA12' in df.columns and 'EMA26' in df.columns:
        if latest['EMA12'] > latest['EMA26']:
            tech_score += 6
        else:
            tech_score -= 6

    # RSI (14)
    if 'RSI14' in df.columns and not pd.isna(latest['RSI14']):
        rsi = latest['RSI14']
        if rsi > 70:
            tech_score -= 10  # Overbought, pull-back expected
        elif rsi < 30:
            tech_score += 10  # Oversold, bounce expected
        elif rsi > 55:
            tech_score += 5   # Bullish momentum
        elif rsi < 45:
            tech_score -= 5   # Bearish momentum

    # MACD Histogram
    if 'MACD_Hist' in df.columns and not pd.isna(latest['MACD_Hist']):
        hist = latest['MACD_Hist']
        if hist > 0:
            tech_score += 8
        else:
            tech_score -= 8

    # Clamp technical score
    tech_score = max(-40.0, min(40.0, tech_score))

    # 3. Factor 2: FedWatch Score (-30 to +30)
    fed_bias = fedwatch.get('us_bias_score', 0.0)  # -1.0 to +1.0
    fed_score = fed_bias * 30.0

    # 4. Factor 3: News Sentiment Score (-15 to +15)
    news_bias = news_summary.get('avg_score', 0.0)  # -1.0 to +1.0
    news_score = news_bias * 15.0

    # 5. Factor 4: Short-term Momentum (-15 to +15)
    # Price change over the last 6 bars
    momentum_score = 0.0
    if len(df) >= 7:
        past_6h_change = (curr_price - df['Close'].iloc[-7]) / pip_val
        # Mean reversion vs continuation: mild continuation factor
        momentum_score = max(-15.0, min(15.0, past_6h_change * 0.3))

    # Total USD Composite Bias (-100 to +100)
    composite_usd_score = tech_score + fed_score + news_score + momentum_score
    composite_usd_score = max(-100.0, min(100.0, composite_usd_score))

    # If pair is JPY/USD, reverse the bias since JPY is base currency
    if not is_usdjpy:
        pair_composite_score = -composite_usd_score
    else:
        pair_composite_score = composite_usd_score

    # Normalized direction weight: -1.0 to +1.0
    norm_weight = pair_composite_score / 100.0

    # Determine Direction
    if norm_weight >= 0.18:
        direction = 'UP'
        direction_label = '上昇優勢 (UP)'
        direction_color = '#00E676'  # Neon Green
        direction_arrow = '▲'
    elif norm_weight <= -0.18:
        direction = 'DOWN'
        direction_label = '下落優勢 (DOWN)'
        direction_color = '#FF5252'  # Coral Red
        direction_arrow = '▼'
    else:
        direction = 'NEUTRAL'
        direction_label = 'もみ合い / レンジ (NEUTRAL)'
        direction_color = '#FFD700'  # Gold
        direction_arrow = '▶'

    # Expected Pips calculation
    # Base estimated move = normalized bias * expected 5-6h range * drift factor
    drift_factor = 0.60
    estimated_pips_delta = norm_weight * expected_range_pips * drift_factor

    # If neutral, small drift with range
    if direction == 'NEUTRAL':
        estimated_pips_delta = norm_weight * (expected_range_pips * 0.2)

    # Uncertainty / Standard error around estimate (± Pips)
    pips_margin = expected_range_pips * 0.45

    # Target price calculation
    target_price = curr_price + (estimated_pips_delta * pip_val)
    target_price_upper = curr_price + ((estimated_pips_delta + pips_margin) * pip_val)
    target_price_lower = curr_price + ((estimated_pips_delta - pips_margin) * pip_val)

    # Confidence calculation:
    # High if technical, fedwatch, and news point in the same direction
    factors_aligned = (
        (tech_score > 0 and fed_score >= 0 and news_score >= 0) or
        (tech_score < 0 and fed_score <= 0 and news_score <= 0)
    )
    base_conf = 60 + abs(norm_weight) * 32
    if factors_aligned:
        base_conf += 8
    confidence_pct = min(92, max(52, int(base_conf)))

    # Risk-Reward entry / stop / tp recommendations
    if direction == 'UP':
        recommended_action = '買い (BUY / ロング)'
        take_profit_pips = round(abs(estimated_pips_delta) + pips_margin * 0.5, 1)
        stop_loss_pips = round(pips_margin * 0.75, 1)
        tp_price = curr_price + (take_profit_pips * pip_val)
        sl_price = curr_price - (stop_loss_pips * pip_val)
    elif direction == 'DOWN':
        recommended_action = '売り (SELL / ショート)'
        take_profit_pips = round(abs(estimated_pips_delta) + pips_margin * 0.5, 1)
        stop_loss_pips = round(pips_margin * 0.75, 1)
        tp_price = curr_price - (take_profit_pips * pip_val)
        sl_price = curr_price + (stop_loss_pips * pip_val)
    else:
        recommended_action = '様子見 (レンジ逆張り / 押し目待ち)'
        take_profit_pips = round(expected_range_pips * 0.5, 1)
        stop_loss_pips = round(expected_range_pips * 0.4, 1)
        tp_price = curr_price + (take_profit_pips * pip_val)
        sl_price = curr_price - (stop_loss_pips * pip_val)

    # Detailed qualitative rationale builder
    reasons = []
    if abs(tech_score) > 10:
        t_dir = '強気(買いシグナル)' if tech_score > 0 else '弱気(売りシグナル)'
        reasons.append(f"【テクニカル】移動平均線の並びおよびオシレーター(RSI/MACD)は{t_dir}を示唆 (寄与度: {tech_score:+.1f}pt)")

    reasons.append(f"【FedWatch金利観測】次回FOMC観測は『{fedwatch.get('stance', '中立')}』。米ドル金利バイアスは{fed_score:+.1f}ptの加重。")

    if abs(news_score) > 3:
        n_dir = 'ドル高・円安優勢' if news_score > 0 else 'ドル安・円高優勢'
        reasons.append(f"【最新ニュース動向】ヘッドライン感情指数は『{n_dir}』(スコア: {news_score:+.1f}pt)")

    reasons.append(f"【市場流動性環境】{session_name}。1時間足ATR({latest_atr:.{summary['decimal_places']}f})基準で{target_hours}時間後の平均想定変動幅は約±{expected_range_pips:.1f} Pips。")

    target_time = datetime.now() + timedelta(hours=target_hours)

    return {
        'target_hours': target_hours,
        'target_time_str': target_time.strftime('%H:%M JST (%m/%d)'),
        'pair': pair,
        'curr_price': curr_price,
        'direction': direction,
        'direction_label': direction_label,
        'direction_color': direction_color,
        'direction_arrow': direction_arrow,
        'estimated_pips_delta': round(estimated_pips_delta, 1),
        'pips_margin': round(pips_margin, 1),
        'min_pips_delta': round(estimated_pips_delta - pips_margin, 1),
        'max_pips_delta': round(estimated_pips_delta + pips_margin, 1),
        'target_price': target_price,
        'target_price_upper': target_price_upper,
        'target_price_lower': target_price_lower,
        'expected_volatility_pips': round(expected_range_pips, 1),
        'confidence_pct': confidence_pct,
        'session_name': session_name,
        'recommended_action': recommended_action,
        'take_profit_pips': take_profit_pips,
        'stop_loss_pips': stop_loss_pips,
        'tp_price': tp_price,
        'sl_price': sl_price,
        'scores': {
            'technical': round(tech_score, 1),
            'fedwatch': round(fed_score, 1),
            'news': round(news_score, 1),
            'momentum': round(momentum_score, 1),
            'composite_usd': round(composite_usd_score, 1),
            'composite_pair': round(pair_composite_score, 1)
        },
        'reasons': reasons
    }


def generate_llm_insight(
    pred: dict,
    summary: dict,
    fedwatch: dict,
    news_items: list[dict],
    api_key: str = None
) -> str:
    """
    Generates an expert macro-strategist forecast report.
    Uses Gemini API if key is provided, otherwise generates structured expert report.
    """
    pair = pred['pair']
    target_hours = pred['target_hours']
    curr_price = pred['curr_price']
    pips = pred['estimated_pips_delta']
    direction = pred['direction_label']

    news_snippets = "\n".join([f"- {n['title']} ({n['source']})" for n in news_items[:5]])

    # Try Gemini if key is provided or present in environment
    gemini_key = api_key or os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel('gemini-2.5-flash')

            prompt = f"""
あなたはFXチーフマクロストラテジストです。
以下の実データ（過去チャート指標、CME FedWatch、最新ニュース速報、複合予測モデル結果）に基づき、
今後 {target_hours}時間後の {pair} の相場予測と詳細な戦術レポートをプロフェッショナルな日本語で作成してください。

【対象市場データ】
- 通貨ペア: {pair}
- 現在レート: {curr_price:.{summary['decimal_places']}f}
- 予測時間枠: {target_hours}時間後（目標時刻: {pred['target_time_str']}）
- 統計予測モデル結果: 方向={direction}, 想定変動Pips={pips:+.1f} Pips (レンジ: {pred['min_pips_delta']:+.1f}〜{pred['max_pips_delta']:+.1f} Pips), 信頼度={pred['confidence_pct']}%
- CME FedWatch次回会合: {fedwatch.get('next_meeting_date')} (現在金利: {fedwatch.get('current_target')}, 据置確率={fedwatch.get('hold_prob')}%, 利下げ確率={fedwatch.get('cut_prob')}%, 利上げ確率={fedwatch.get('hike_prob')}%)
- 現在の市場セッション: {pred['session_name']}
- 直近のヘッドラインニュース:
{news_snippets}

【出力構成】
1. 【{target_hours}時間後 エグゼクティブ・サマリー】（結論、方向性、ターゲットレート）
2. 【ファンダメンタルズ & FedWatch分析】（米金利動向・日銀思惑が及ぼす影響）
3. 【テクニカル & ボラティリティ分析】（5〜6時間の期待Pips幅とサポート/レジスタンス）
4. 【リスク管理 & トレーディング戦略】（推奨アクション、利確・損切りの目安Pips）
端的に、説得力のある論理的なトーンで記述してください。
"""
            response = model.generate_content(prompt)
            if response and response.text:
                return response.text
        except Exception as e:
            # Fall through to default generator
            pass

    # Built-in Intelligent Macro-Strategist Report Generator
    report_lines = [
        f"### 🎯 【{target_hours}時間後 エグゼクティブ・サマリー】",
        f"- **予測方向**: **{direction}**",
        f"- **推定変動幅**: **{pips:+.1f} Pips** (想定レンジ: **{pred['min_pips_delta']:+.1f} Pips** 〜 **{pred['max_pips_delta']:+.1f} Pips**)",
        f"- **推定ターゲット価格**: **{pred['target_price']:.{summary['decimal_places']}f}** (到達見込み時刻: **{pred['target_time_str']}**)",
        f"- **モデル総合信頼度**: **{pred['confidence_pct']}%**",
        "",
        "### 🏛️ 【FedWatch & 金融政策ファンダメンタルズ分析】",
        f"- 次回FOMC（予定: {fedwatch.get('next_meeting_date', '未定')}）における金利織り込みは、現状ターゲット **{fedwatch.get('current_target')}** に対し、**据え置き確率 {fedwatch.get('hold_prob')}%** / **利下げ確率 {fedwatch.get('cut_prob')}%** となっています。",
        f"- 金融政策スタンス: 『**{fedwatch.get('stance')}**』。{fedwatch.get('summary_text')}",
        "",
        "### 📊 【テクニカル & ボラティリティ構造】",
        f"- 現在の相場環境は **{pred['session_name']}** に位置し、1時間足ATRから換算される5〜6時間の標準変動ポテンシャルは **約 ±{pred['expected_volatility_pips']:.1f} Pips** です。",
        f"- テクニカル総合評価値は **{pred['scores']['technical']:+.1f} pt** であり、移動平均線・オシレーター群が示す短期モメンタムと合致しています。",
        "",
        "### 💡 【推奨トレード戦略 & リスク管理】",
        f"- **推奨スタンス**: **{pred['recommended_action']}**",
        f"- **利確目標 (TP)**: 現在値から **{pred['take_profit_pips']} Pips** ({pred['tp_price']:.{summary['decimal_places']}f})",
        f"- **損切り目安 (SL)**: 現在値から **{pred['stop_loss_pips']} Pips** ({pred['sl_price']:.{summary['decimal_places']}f})",
        "- **注意点**: ロンドン・NYフィキシング前後や重要経済統計（雇用統計、CPI、PMI）発表時は、予測モデルの通常ATRを大きく超える突発的スパイクが発生する可能性があるため、適切なポジションサイズでのリスク管理を推奨します。"
    ]

    return "\n".join(report_lines)
