"""
Forex AI Brain Engine.
Combines Advanced Technical Analysis (MA, RSI, MACD, Ichimoku, ADX/DMI, Stochastics, Price Action),
CME FedWatch, News Sentiment, and LLM (Gemini)
to estimate price direction and Pips movement 5 to 6 hours into the future.
"""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import math
import os
import pandas as pd
import numpy as np

from technical import get_advanced_technical_summary, find_support_resistance


def analyze_market_hours(now_dt: datetime = None) -> tuple[str, float]:
    """
    Evaluates current global FX session and volatility multiplier based on JST.
    Tokyo: 09:00 - 15:00 JST (vol factor: 0.85)
    London: 16:00 - 01:00 JST (vol factor: 1.25)
    New York: 21:00 - 06:00 JST (vol factor: 1.40)
    Overlap (London & NY): 21:00 - 01:00 JST (vol factor: 1.65)
    """
    if now_dt is None:
        now_dt = datetime.now(ZoneInfo('Asia/Tokyo'))
    now_hour = now_dt.hour
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
    pair: str = 'USD/JPY',
    base_time: datetime = None
) -> dict:
    """
    Multi-factor predictive model for 5-6 hours forecast with regime-adaptive technical scoring.
    """
    if base_time is None:
        base_time = datetime.now(ZoneInfo('Asia/Tokyo'))
    elif base_time.tzinfo is None:
        base_time = base_time.replace(tzinfo=ZoneInfo('Asia/Tokyo'))

    curr_price = summary['current_price']
    pip_val = summary['pip_value']
    is_usdjpy = (pair == 'USD/JPY')

    # 1. Base Volatility from 14-period 1H ATR
    latest_atr = df['ATR14'].iloc[-1] if 'ATR14' in df.columns else (curr_price * 0.0015)
    if pd.isna(latest_atr) or latest_atr <= 0:
        latest_atr = curr_price * 0.0015

    session_name, vol_mult = analyze_market_hours(base_time)

    # Scaling ATR by sqrt(target_hours) as standard diffusion
    expected_range_price = latest_atr * math.sqrt(target_hours) * vol_mult
    expected_range_pips = expected_range_price / pip_val

    # 2. Advanced Technical Diagnostics (Ichimoku, ADX/DMI, Stochastics, Price Action)
    tech_diag = get_advanced_technical_summary(df, curr_price)
    sr_levels = find_support_resistance(df)
    latest = df.iloc[-1]

    # --- Factor 1: Technical Score (-40 to +40, where + is USD bullish / JPY bearish) ---
    tech_score = 0.0
    tech_details = []

    regime = tech_diag.get('regime', 'DEVELOPING_TREND')
    is_strong_trend = (regime in ['STRONG_UPTREND', 'STRONG_DOWNTREND'])
    is_ranging = (regime == 'RANGE')

    # A. Moving Averages Trend (SMA20, SMA50, EMA12 vs EMA26)
    ma_bullish_count = 0
    if 'SMA20' in df.columns and curr_price > latest['SMA20']:
        ma_bullish_count += 1
    if 'SMA50' in df.columns and curr_price > latest['SMA50']:
        ma_bullish_count += 1
    if 'EMA12' in df.columns and 'EMA26' in df.columns and latest['EMA12'] > latest['EMA26']:
        ma_bullish_count += 1

    if ma_bullish_count == 3:
        tech_score += 10.0
        tech_details.append("移動平均線パーフェクトオーダー(強気整列)")
    elif ma_bullish_count >= 2:
        tech_score += 5.0
    elif ma_bullish_count == 0:
        tech_score -= 10.0
        tech_details.append("移動平均線パーフェクトオーダー(弱気整列)")
    else:
        tech_score -= 5.0

    # B. Ichimoku Kinko Hyo (一目均衡表: 雲・転換線・基準線)
    ichimoku_pos = tech_diag.get('ichimoku_pos')
    if ichimoku_pos == 'ABOVE_CLOUD':
        tech_score += 6.0
        tech_details.append("一目均衡表: 雲の上で推移(強力な支持帯)")
    elif ichimoku_pos == 'BELOW_CLOUD':
        tech_score -= 6.0
        tech_details.append("一目均衡表: 雲の下で推移(上値重い抵抗帯)")
    
    if tech_diag.get('tenkan', 0) > tech_diag.get('kijun', 0):
        tech_score += 4.0
    else:
        tech_score -= 4.0

    # C. ADX & DMI Trend Strength & Direction
    adx_val = tech_diag.get('adx', 20.0)
    plus_di = tech_diag.get('plus_di', 20.0)
    minus_di = tech_diag.get('minus_di', 20.0)

    if is_strong_trend:
        if plus_di > minus_di:
            tech_score += 8.0
            tech_details.append(f"ADX {adx_val:.1f}: 強力な上昇トレンド発生中(+DI優勢)")
        else:
            tech_score -= 8.0
            tech_details.append(f"ADX {adx_val:.1f}: 強力な下降トレンド発生中(-DI優勢)")

    # D. Oscillator Dynamics (RSI & Slow Stochastics) - Adaptive to Market Regime
    rsi_val = latest.get('RSI14', 50.0)
    stoch_k = tech_diag.get('stoch_k', 50.0)
    stoch_d = tech_diag.get('stoch_d', 50.0)

    if is_strong_trend:
        # In a strong trend, overbought/oversold indicates strong continuation momentum (Bandwalk)
        if plus_di > minus_di:
            if rsi_val >= 55.0:
                tech_score += 5.0
            if stoch_k > stoch_d:
                tech_score += 4.0
        else:
            if rsi_val <= 45.0:
                tech_score -= 5.0
            if stoch_k < stoch_d:
                tech_score -= 4.0
    else:
        # In range or developing trend, standard mean-reversion rules apply
        if rsi_val > 70.0:
            tech_score -= 7.0
            tech_details.append("RSI 70超(過熱・短期反落警戒)")
        elif rsi_val < 30.0:
            tech_score += 7.0
            tech_details.append("RSI 30未満(売られすぎ・自律反発期待)")
        elif rsi_val > 52.0:
            tech_score += 3.0
        elif rsi_val < 48.0:
            tech_score -= 3.0

        # Slow Stochastics crossover
        if stoch_k > 80.0:
            tech_score -= 5.0
        elif stoch_k < 20.0:
            tech_score += 5.0
        elif (stoch_k > stoch_d) and (stoch_k < 75.0):
            tech_score += 4.0
        elif (stoch_k < stoch_d) and (stoch_k > 25.0):
            tech_score -= 4.0

    # E. MACD Momentum
    macd_hist = latest.get('MACD_Hist', 0.0)
    if not pd.isna(macd_hist):
        if macd_hist > 0:
            tech_score += 5.0
        else:
            tech_score -= 5.0

    # F. Bollinger Band Bandwalk / Squeeze
    bb_state = tech_diag.get('bb_state')
    if bb_state == 'BANDWALK_UP':
        tech_score += 4.0
        tech_details.append("ボリンジャーバンド: 上方バンドウォーク継続")
    elif bb_state == 'BANDWALK_DOWN':
        tech_score -= 4.0
        tech_details.append("ボリンジャーバンド: 下方バンドウォーク継続")

    # G. Price Action (Candle wicks / Engulfing)
    pa = tech_diag.get('price_action', {})
    if pa.get('score_bias', 0.0) != 0.0:
        tech_score += pa['score_bias']
        tech_details.append(f"ローソク足形状: {pa['pattern']}")

    # Clamp technical score to [-40, +40]
    tech_score = max(-40.0, min(40.0, tech_score))

    # 3. Factor 2: FedWatch Score (-30 to +30)
    fed_bias = fedwatch.get('us_bias_score', 0.0)  # -1.0 to +1.0
    fed_score = fed_bias * 30.0

    # 4. Factor 3: News Sentiment Score (-15 to +15)
    news_bias = news_summary.get('avg_score', 0.0)  # -1.0 to +1.0
    news_score = news_bias * 15.0

    # 5. Factor 4: Short-term Momentum (-15 to +15)
    momentum_score = 0.0
    if len(df) >= 7:
        past_6h_change = (curr_price - df['Close'].iloc[-7]) / pip_val
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
    drift_factor = 0.62 if is_strong_trend else 0.55
    estimated_pips_delta = norm_weight * expected_range_pips * drift_factor

    if direction == 'NEUTRAL':
        estimated_pips_delta = norm_weight * (expected_range_pips * 0.2)

    # Uncertainty / Standard error around estimate (± Pips)
    pips_margin = expected_range_pips * 0.42

    # Target price calculation
    target_price = curr_price + (estimated_pips_delta * pip_val)
    target_price_upper = curr_price + ((estimated_pips_delta + pips_margin) * pip_val)
    target_price_lower = curr_price + ((estimated_pips_delta - pips_margin) * pip_val)

    # Confidence calculation:
    # High if technical, fedwatch, and news point in the same direction, boosted by strong ADX trend
    factors_aligned = (
        (tech_score > 0 and fed_score >= 0 and news_score >= 0) or
        (tech_score < 0 and fed_score <= 0 and news_score <= 0)
    )
    base_conf = 62 + abs(norm_weight) * 30
    if factors_aligned:
        base_conf += 6
    if is_strong_trend:
        base_conf += 4
    confidence_pct = min(94, max(54, int(base_conf)))

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
    t_dir = '強気(買いシグナル)' if tech_score > 0 else ('弱気(売りシグナル)' if tech_score < 0 else '中立')
    reasons.append(f"【テクニカル総合 ({tech_score:+.1f}pt)】{tech_diag.get('regime_label', '')}。{t_dir}。")
    if tech_details:
        reasons.append("【テクニカル詳細根拠】" + " / ".join(tech_details[:3]))

    reasons.append(f"【FedWatch金利観測 ({fed_score:+.1f}pt)】次回FOMCスタンス: 『{fedwatch.get('stance', '中立')}』。")

    if abs(news_score) > 2.5:
        n_dir = 'ドル高・円安優勢' if news_score > 0 else 'ドル安・円高優勢'
        reasons.append(f"【最新ニュースセンチメント ({news_score:+.1f}pt)】為替速報バイアス: 『{n_dir}』。")

    reasons.append(f"【市場セッション & ATR】{session_name}。1h-ATR({latest_atr:.{summary['decimal_places']}f})基準で{target_hours}hの想定変動幅は約±{expected_range_pips:.1f} Pips。")

    target_time = base_time + timedelta(hours=target_hours)

    return {
        'target_hours': target_hours,
        'base_time': base_time,
        'target_time': target_time,
        'base_time_str': base_time.strftime('%H:%M JST (%m/%d)'),
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
        'tech_diag': tech_diag,
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
    tech_diag = pred.get('tech_diag', {})

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
以下の高度な実データ（一目均衡表、ADXトレンド強度、ストキャスティクス、CME FedWatch、最新ニュース速報、複合予測モデル結果）に基づき、
今後 {target_hours}時間後の {pair} の相場予測と詳細な戦術レポートをプロフェッショナルな日本語で作成してください。

【対象市場データ】
- 通貨ペア: {pair}
- 現在レート: {curr_price:.{summary['decimal_places']}f}
- 予測時間枠: {target_hours}時間後（目標時刻: {pred['target_time_str']}）
- 統計予測モデル結果: 方向={direction}, 想定変動Pips={pips:+.1f} Pips (レンジ: {pred['min_pips_delta']:+.1f}〜{pred['max_pips_delta']:+.1f} Pips), 信頼度={pred['confidence_pct']}%
- テクニカル相場レジーム: {tech_diag.get('regime_label', '未定')} ({tech_diag.get('regime_desc', '')})
- 一目均衡表: {tech_diag.get('ichimoku_pos_label', '未定')}, {tech_diag.get('tk_cross_label', '')}
- ストキャスティクス & RSI: {tech_diag.get('stoch_status', '')}
- CME FedWatch次回会合: {fedwatch.get('next_meeting_date')} (現在金利: {fedwatch.get('current_target')}, 据置確率={fedwatch.get('hold_prob')}%, 利下げ確率={fedwatch.get('cut_prob')}%, 利上げ確率={fedwatch.get('hike_prob')}%)
- 現在の市場セッション: {pred['session_name']}
- 直近のヘッドラインニュース:
{news_snippets}

【出力構成】
1. 【{target_hours}時間後 エグゼクティブ・サマリー】（結論、方向性、ターゲットレート）
2. 【ファンダメンタルズ & FedWatch分析】（米金利動向・日銀思惑が及ぼす影響）
3. 【テクニカル構造診断 (一目均衡表・ADX・オシレーター)】（トレンド継続性や上値下値の節目）
4. 【リスク管理 & トレーディング戦略】（推奨アクション、利確・損切りの目安Pips）
端的に、説得力のある論理的なトーンで記述してください。
"""
            response = model.generate_content(prompt)
            if response and response.text:
                return response.text
        except Exception:
            pass

    # Built-in Intelligent Macro-Strategist Report Generator
    report_lines = [
        f"### 🎯 【{target_hours}時間後 エグゼクティブ・サマリー】",
        f"- **分析起点**: {pred['base_time_str']} (現時点) ➡️ **到達目標時刻**: **{pred['target_time_str']}** ({target_hours}時間後)",
        f"- **予測方向**: **{direction}**",
        f"- **推定変動幅**: **{pips:+.1f} Pips** (想定レンジ: **{pred['min_pips_delta']:+.1f} Pips** 〜 **{pred['max_pips_delta']:+.1f} Pips**)",
        f"- **推定ターゲット価格**: **{pred['target_price']:.{summary['decimal_places']}f}** (到達見込み: **{pred['target_time_str']}**)",
        f"- **モデル総合信頼度**: **{pred['confidence_pct']}%**",
        "",
        "### 🏛️ 【FedWatch & 金融政策ファンダメンタルズ分析】",
        f"- 次回FOMC（予定: {fedwatch.get('next_meeting_date', '未定')}）における金利織り込みは、現状ターゲット **{fedwatch.get('current_target')}** に対し、**据え置き確率 {fedwatch.get('hold_prob')}%** / **利下げ確率 {fedwatch.get('cut_prob')}%** となっています。",
        f"- 金融政策スタンス: 『**{fedwatch.get('stance')}**』。{fedwatch.get('summary_text')}",
        "",
        "### 📐 【高度テクニカル構造診断 (一目均衡表・ADX・オシレーター)】",
        f"- **相場環境レジーム**: **{tech_diag.get('regime_label', '判定中')}** ({tech_diag.get('regime_desc', '')})",
        f"- **一目均衡表**: **{tech_diag.get('ichimoku_pos_label', '')}** / 転換線・基準線は **{tech_diag.get('tk_cross_label', '')}**",
        f"- **オシレーター**: ストキャスティクスは **{tech_diag.get('stoch_status', '')}**、ボリンジャーは **{tech_diag.get('bb_state_label', '')}**",
        f"- テクニカル寄与度は **{pred['scores']['technical']:+.1f} pt** で推移し、現在のトレンド・モメンタムと強く同期しています。",
        "",
        "### 💡 【推奨トレード戦略 & リスク管理】",
        f"- **推奨スタンス**: **{pred['recommended_action']}**",
        f"- **利確目標 (TP)**: 現在値から **{pred['take_profit_pips']} Pips** ({pred['tp_price']:.{summary['decimal_places']}f})",
        f"- **損切り目安 (SL)**: 現在値から **{pred['stop_loss_pips']} Pips** ({pred['sl_price']:.{summary['decimal_places']}f})",
        f"- **想定ボラティリティ**: 1時間足ATRから換算した5〜6時間の標準変動幅は **約 ±{pred['expected_volatility_pips']:.1f} Pips** です。"
    ]

    return "\n".join(report_lines)
