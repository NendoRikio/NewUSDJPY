"""
ForexAI Vision: USD/JPY & JPY/USD 5-6時間後 Pips予測アプリケーション
Streamlit Web Dashboard with Multi-Factor Intelligence (Chart, FedWatch, News, AI Engine).
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from data_fetcher import fetch_pair_data, PAIR_CONFIG
from technical import compute_technical_indicators, find_support_resistance, get_advanced_technical_summary
from fedwatch_fetcher import fetch_fedwatch_analysis
from news_fetcher import fetch_latest_fx_news
from brain_engine import compute_pips_prediction, generate_llm_insight

# 1. Page Configuration
st.set_page_config(
    page_title="ForexAI Vision - 5〜6時間後 Pips予測システム",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark fintech terminal theme
st.markdown("""
<style>
    /* Dark Theme Base */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #131b2e 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px 22px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.4);
    }
    
    .metric-title {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    
    .metric-value {
        color: #f8fafc;
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    /* Main Prediction Hero Card */
    .hero-card {
        background: linear-gradient(145deg, #0f172a 0%, #1e1b4b 100%);
        border: 2px solid #6366f1;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.25);
    }
    
    .hero-direction {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 8px 0;
    }
    
    .pips-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 8px;
        font-size: 1.3rem;
        font-weight: 700;
    }
    
    .badge-bullish {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid #10b981;
    }
    
    .badge-bearish {
        background-color: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1px solid #ef4444;
    }
    
    .badge-neutral {
        background-color: rgba(245, 158, 11, 0.2);
        color: #f59e0b;
        border: 1px solid #f59e0b;
    }
    
    /* News & Info Cards */
    .news-card {
        background-color: #111827;
        border-left: 4px solid #6366f1;
        padding: 12px 16px;
        margin-bottom: 10px;
        border-radius: 4px;
    }
    
    .news-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #f1f5f9;
        text-decoration: none;
    }
    .news-title:hover {
        color: #818cf8;
    }

    .tech-box {
        background: #111827;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 14px;
    }
</style>
""", unsafe_allow_html=True)

# 2. Sidebar Controls
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bullish.png", width=64)
    st.title("ForexAI Vision")
    st.caption("リアルタイム 5〜6時間後 Pips推論システム")
    st.markdown("---")
    
    pair_selection = st.selectbox(
        "💱 通貨ペア選択",
        options=["USD/JPY", "JPY/USD"],
        index=0,
        help="日本のFXで一般的なドル円は『USD/JPY』です。1円あたりの米ドルを見たい場合は『JPY/USD』を選択してください。"
    )
    
    target_hours = st.slider(
        "⏱️ 予測対象ホライズン",
        min_value=5.0,
        max_value=6.0,
        value=5.5,
        step=0.5,
        format="%.1f 時間後",
        help="何時間後の価格とPips変動幅を推定するか指定します。"
    )
    
    st.markdown("---")
    st.subheader("🤖 AI頭脳 (Gemini API)")
    gemini_key = st.text_input(
        "Gemini API キー (任意)",
        type="password",
        value="",
        placeholder="AIzaSy...",
        help="入力するとGemini 2.5によるディープなFXマクロレポートが生成されます。空欄でも内蔵のハイブリッドAIエンジンが完全機能します。"
    )
    
    st.markdown("---")
    refresh_button = st.button("🔄 最新データを取得・再推論", use_container_width=True, type="primary")
    now_jst_init = datetime.now(ZoneInfo('Asia/Tokyo'))
    st.caption(f"起動/分析時刻: {now_jst_init.strftime('%Y-%m-%d %H:%M:%S JST')}")
    st.info("💡 一目均衡表・ADX/DMI・FedWatch・為替速報をリアルタイム統合解析中")


# 3. Data Ingestion & Brain Computation
@st.cache_data(ttl=90)  # Cache raw market fetch for 90 seconds
def load_raw_market_data(pair: str):
    df, summary = fetch_pair_data(pair=pair, period='1mo', interval='1h')
    df = compute_technical_indicators(df)
    sr_levels = find_support_resistance(df)
    fedwatch = fetch_fedwatch_analysis()
    news_items, news_summary = fetch_latest_fx_news(limit=12)
    return df, summary, sr_levels, fedwatch, news_items, news_summary


if refresh_button:
    st.cache_data.clear()

with st.spinner("最新為替レート、FedWatch金利確率、高度テクニカル指標を取得し、AI頭脳で分析中..."):
    try:
        df, summary, sr_levels, fedwatch, news_items, news_summary = load_raw_market_data(pair_selection)
        
        # 起動時点（現時点）の日本時間を確実に基準として5〜6時間後をリアルタイム推論
        now_jst = datetime.now(ZoneInfo('Asia/Tokyo'))
        pred = compute_pips_prediction(
            df=df,
            summary=summary,
            fedwatch=fedwatch,
            news_summary=news_summary,
            target_hours=target_hours,
            pair=pair_selection,
            base_time=now_jst
        )
    except Exception as e:
        st.error(f"データ取得中にエラーが発生しました: {e}")
        st.stop()


# 4. Top Header & Real-time Rate Cards
col_h1, col_h2, col_h3, col_h4 = st.columns([1.5, 1, 1, 1.2])

with col_h1:
    pair_cfg = PAIR_CONFIG[pair_selection]
    st.markdown(f"### {pair_cfg['label']}")
    st.caption(f"ティッカー: `{summary['ticker']}` | {pair_cfg['description']}")

with col_h2:
    st.metric(
        label="現在レート (リアルタイム)",
        value=f"{summary['current_price']:.{summary['decimal_places']}f}",
        delta=f"{summary['change_pips']:+.1f} Pips ({summary['pct_change']:+.2f}%)"
    )

with col_h3:
    st.metric(
        label="24h レンジ",
        value=f"{summary['high_24h']:.{summary['decimal_places']}f}",
        delta=f"安値: {summary['low_24h']:.{summary['decimal_places']}f}",
        delta_color="off"
    )

with col_h4:
    st.metric(
        label=f"🎯 予想時刻 ({target_hours}h後)",
        value=pred['target_time'].strftime('%H:%M JST'),
        delta=f"現在: {pred['base_time'].strftime('%H:%M JST')}",
        delta_color="off"
    )

st.markdown("<br>", unsafe_allow_html=True)


# 5. HERO PREDICTION CARD (5-6時間後の推定Pips・方向・信頼度)
p_delta = pred['estimated_pips_delta']
badge_class = "badge-bullish" if pred['direction'] == 'UP' else ("badge-bearish" if pred['direction'] == 'DOWN' else "badge-neutral")

st.markdown(f"""
<div class="hero-card">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>
            <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-bottom: 6px;">
                <span style="background: rgba(99, 102, 241, 0.25); border: 1px solid #6366f1; padding: 3px 10px; border-radius: 6px; font-size: 0.85rem; color: #c7d2fe; font-weight: 600;">
                    ⏱️ 起動/分析時点: {pred['base_time_str']}
                </span>
                <span style="color: #818cf8; font-weight: bold;">➡️</span>
                <span style="background: rgba(16, 185, 129, 0.25); border: 1px solid #10b981; padding: 3px 10px; border-radius: 6px; font-size: 0.85rem; color: #a7f3d0; font-weight: 700;">
                    🎯 予想時刻: {pred['target_time_str']} ({target_hours}時間後)
                </span>
            </div>
            <div class="hero-direction" style="color: {pred['direction_color']};">
                {pred['direction_arrow']} {pred['direction_label']}
            </div>
        </div>
        <div style="text-align: right;">
            <span class="pips-badge {badge_class}">
                予想変動量: {p_delta:+.1f} Pips
            </span>
            <div style="color: #94a3b8; font-size: 0.9rem; margin-top: 6px;">
                想定幅: {pred['min_pips_delta']:+.1f} 〜 {pred['max_pips_delta']:+.1f} Pips
            </div>
        </div>
    </div>
    <hr style="border-color: rgba(99, 102, 241, 0.3); margin: 16px 0;">
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px;">
        <div>
            <div class="metric-title">🎯 予想ターゲット価格</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #38bdf8;">
                {pred['target_price']:.{summary['decimal_places']}f}
            </div>
            <div style="font-size: 0.8rem; color: #94a3b8;">
                上限 {pred['target_price_upper']:.{summary['decimal_places']}f} / 下限 {pred['target_price_lower']:.{summary['decimal_places']}f}
            </div>
        </div>
        <div>
            <div class="metric-title">🔒 モデル信頼度</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #a855f7;">
                {pred['confidence_pct']}%
            </div>
            <div style="font-size: 0.8rem; color: #94a3b8;">
                高度テクニカル・Fed・ニュースの一致度
            </div>
        </div>
        <div>
            <div class="metric-title">⚡ 推奨スタンス</div>
            <div style="font-size: 1.3rem; font-weight: 700; color: #facc15;">
                {pred['recommended_action']}
            </div>
            <div style="font-size: 0.8rem; color: #94a3b8;">
                TP目安: +{pred['take_profit_pips']} Pips / SL目安: -{pred['stop_loss_pips']} Pips
            </div>
        </div>
        <div>
            <div class="metric-title">📊 5-6h 想定ボラティリティ</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #34d399;">
                ±{pred['expected_volatility_pips']:.1f} Pips
            </div>
            <div style="font-size: 0.8rem; color: #94a3b8;">
                1h-ATR({df['ATR14'].iloc[-1]:.{summary['decimal_places']}f}) 時間拡散モデル
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# 6. Interactive Chart Section with 5-6h Prediction Cone
st.markdown("### 📈 チャート分析 & 5〜6時間後 予測コーン投影 (土日休場スキップ連動)")

# Filter bars for clean chart view (last 80 bars)
plot_df = df.iloc[-80:].copy()

# Detect missing bars (weekends and market holidays) to eliminate gaps and connect lines seamlessly
full_range = pd.date_range(start=plot_df.index[0], end=plot_df.index[-1], freq='1h')
missing_times = full_range.difference(plot_df.index)
break_values = missing_times.strftime('%Y-%m-%d %H:%M:%S').tolist()
rangebreaks_cfg = [dict(values=break_values, dvalue=3600*1000)] if len(break_values) > 0 else []

col_ctrl1, col_ctrl2 = st.columns([2, 1])
with col_ctrl1:
    chart_overlays = st.multiselect(
        "📊 チャート重ね合わせ指標",
        options=["一目均衡表 (雲・基準線)", "ボリンジャーバンド (±2σ)", "移動平均線 (SMA20/50)"],
        default=["一目均衡表 (雲・基準線)", "移動平均線 (SMA20/50)"],
        help="チャート上に表示したいテクニカル指標を選択できます。"
    )
with col_ctrl2:
    sub_indicator = st.radio(
        "📉 下段オシレーター選択",
        options=["RSI (14)", "Slow Stochastics", "MACD"],
        horizontal=True,
        index=0
    )

sub_title_2 = f"オシレーター: {sub_indicator}"
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.06,
    row_heights=[0.74, 0.26],
    subplot_titles=(f"{pair_selection} 1時間足 (ローソク足 + 連続推移 + 予測コーン)", sub_title_2)
)

# Candlestick
fig.add_trace(
    go.Candlestick(
        x=plot_df.index,
        open=plot_df['Open'],
        high=plot_df['High'],
        low=plot_df['Low'],
        close=plot_df['Close'],
        name='OHLC',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ),
    row=1, col=1
)

# Ichimoku Cloud Overlay
if "一目均衡表 (雲・基準線)" in chart_overlays:
    if 'Ichimoku_SpanA' in plot_df.columns and 'Ichimoku_SpanB' in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=plot_df['Ichimoku_SpanA'],
                line=dict(color='rgba(16, 185, 129, 0.4)', width=1),
                name='先行スパンA'
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=plot_df['Ichimoku_SpanB'],
                line=dict(color='rgba(239, 68, 68, 0.4)', width=1),
                fill='tonexty',
                fillcolor='rgba(99, 102, 241, 0.12)',
                name='一目 雲'
            ),
            row=1, col=1
        )
    if 'Ichimoku_Kijun' in plot_df.columns:
        fig.add_trace(
            go.Scatter(x=plot_df.index, y=plot_df['Ichimoku_Kijun'], line=dict(color='#fb923c', width=1.3), name='一目 基準線'),
            row=1, col=1
        )
    if 'Ichimoku_Tenkan' in plot_df.columns:
        fig.add_trace(
            go.Scatter(x=plot_df.index, y=plot_df['Ichimoku_Tenkan'], line=dict(color='#38bdf8', width=1.1, dash='dot'), name='一目 転換線'),
            row=1, col=1
        )

# Moving Averages Overlay
if "移動平均線 (SMA20/50)" in chart_overlays:
    if 'SMA20' in plot_df.columns:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['SMA20'], line=dict(color='#38bdf8', width=1.5), name='SMA 20'), row=1, col=1)
    if 'SMA50' in plot_df.columns:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['SMA50'], line=dict(color='#f59e0b', width=1.5), name='SMA 50'), row=1, col=1)

# Bollinger Bands Overlay
if "ボリンジャーバンド (±2σ)" in chart_overlays:
    if 'BB_Upper2' in plot_df.columns:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['BB_Upper2'], line=dict(color='rgba(148, 163, 184, 0.45)', width=1, dash='dot'), name='BB +2σ'), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['BB_Lower2'], line=dict(color='rgba(148, 163, 184, 0.45)', width=1, dash='dot'), name='BB -2σ'), row=1, col=1)

# Future Prediction Cone (Projection into 5.5 hours future from CURRENT LAUNCH TIME)
last_candle_time = plot_df.index[-1]
cone_start_time = pred['base_time'].replace(tzinfo=None)

# Safety check: if weekend / closed market (gap > 12 hours from last confirmed bar),
# anchor prediction cone cleanly to last confirmed candle to avoid weekend gap
time_since_last_bar = (cone_start_time - last_candle_time).total_seconds() / 3600
if time_since_last_bar > 12 or cone_start_time < last_candle_time:
    cone_start_time = last_candle_time

cone_end_time = cone_start_time + pd.Timedelta(hours=target_hours)
future_times = [cone_start_time, cone_end_time]

# Connect last confirmed candle to current real-time price if gap exists during active trading
if cone_start_time > last_candle_time:
    fig.add_trace(
        go.Scatter(
            x=[last_candle_time, cone_start_time],
            y=[plot_df['Close'].iloc[-1], summary['current_price']],
            mode='lines+markers',
            line=dict(color='#38bdf8', width=1.5, dash='dot'),
            marker=dict(size=[0, 7], color='#38bdf8'),
            name="リアルタイム現在値"
        ),
        row=1, col=1
    )

# Center target line (from launch time to target time)
fig.add_trace(
    go.Scatter(
        x=future_times,
        y=[summary['current_price'], pred['target_price']],
        mode='lines+markers',
        line=dict(color=pred['direction_color'], width=3, dash='dash'),
        marker=dict(size=[6, 9], color=pred['direction_color']),
        name=f"AI予測 ({p_delta:+.1f} Pips)"
    ),
    row=1, col=1
)

# Upper bound cone
fig.add_trace(
    go.Scatter(
        x=future_times,
        y=[summary['current_price'], pred['target_price_upper']],
        mode='lines',
        line=dict(color='rgba(99, 102, 241, 0.6)', width=1, dash='dot'),
        name="想定レンジ上限"
    ),
    row=1, col=1
)

# Lower bound cone with fill
fig.add_trace(
    go.Scatter(
        x=future_times,
        y=[summary['current_price'], pred['target_price_lower']],
        mode='lines',
        line=dict(color='rgba(99, 102, 241, 0.6)', width=1, dash='dot'),
        fill='tonexty',
        fillcolor='rgba(99, 102, 241, 0.15)',
        name=f"{target_hours}h後 予測コーン"
    ),
    row=1, col=1
)

# Subplot Indicator based on user selection
if sub_indicator == "RSI (14)":
    if 'RSI14' in plot_df.columns:
        fig.add_trace(
            go.Scatter(x=plot_df.index, y=plot_df['RSI14'], line=dict(color='#c084fc', width=1.5), name='RSI(14)'),
            row=2, col=1
        )
        fig.add_hline(y=70, line=dict(color='rgba(239, 68, 68, 0.6)', dash='dash', width=1), row=2, col=1)
        fig.add_hline(y=30, line=dict(color='rgba(16, 185, 129, 0.6)', dash='dash', width=1), row=2, col=1)
elif sub_indicator == "Slow Stochastics":
    if 'Stoch_K' in plot_df.columns and 'Stoch_D' in plot_df.columns:
        fig.add_trace(
            go.Scatter(x=plot_df.index, y=plot_df['Stoch_K'], line=dict(color='#38bdf8', width=1.5), name='%K (Slow)'),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=plot_df.index, y=plot_df['Stoch_D'], line=dict(color='#f59e0b', width=1.5), name='%D (Slow)'),
            row=2, col=1
        )
        fig.add_hline(y=80, line=dict(color='rgba(239, 68, 68, 0.6)', dash='dash', width=1), row=2, col=1)
        fig.add_hline(y=20, line=dict(color='rgba(16, 185, 129, 0.6)', dash='dash', width=1), row=2, col=1)
else:  # MACD
    if 'MACD' in plot_df.columns and 'MACD_Signal' in plot_df.columns:
        fig.add_trace(
            go.Scatter(x=plot_df.index, y=plot_df['MACD'], line=dict(color='#38bdf8', width=1.5), name='MACD'),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=plot_df.index, y=plot_df['MACD_Signal'], line=dict(color='#f59e0b', width=1.5), name='Signal'),
            row=2, col=1
        )
        colors = ['#26a69a' if val >= 0 else '#ef5350' for val in plot_df['MACD_Hist']]
        fig.add_trace(
            go.Bar(x=plot_df.index, y=plot_df['MACD_Hist'], marker_color=colors, name='Hist'),
            row=2, col=1
        )

fig.update_layout(
    template='plotly_dark',
    plot_bgcolor='#0f172a',
    paper_bgcolor='#0b0f19',
    height=580,
    margin=dict(l=40, r=40, t=30, b=30),
    xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

# Apply rangebreaks to completely eliminate weekend gaps and seamlessly connect from last week
fig.update_xaxes(
    rangebreaks=rangebreaks_cfg,
    range=[plot_df.index[-50], cone_end_time + pd.Timedelta(hours=1.5)],
    row=1, col=1
)
fig.update_xaxes(
    rangebreaks=rangebreaks_cfg,
    range=[plot_df.index[-50], cone_end_time + pd.Timedelta(hours=1.5)],
    row=2, col=1
)

st.plotly_chart(fig, use_container_width=True)


# 7. Deep Multi-Factor Tabs (Brain, FedWatch, News, Technicals)
tab_brain, tab_tech, tab_fed, tab_news = st.tabs([
    "🧠 かしこい頭脳 AI分析レポート",
    "📐 高度テクニカル分析 & レジーム",
    "🏛️ CME FedWatch 金利観測",
    "📰 最新為替ニュース & センチメント"
])

tech_diag = pred.get('tech_diag', {})

with tab_brain:
    col_b1, col_b2 = st.columns([1.8, 1.2])
    
    with col_b1:
        st.markdown("#### 📝 AIストラテジストによる総合相場診断")
        ai_insight = generate_llm_insight(pred, summary, fedwatch, news_items, api_key=gemini_key)
        st.markdown(ai_insight)
    
    with col_b2:
        st.markdown("#### ⚖️ 各ファクターの寄与度スコア")
        scores = pred['scores']
        
        score_df = pd.DataFrame([
            {"ファクター": "高度テクニカル総合 (一目/ADX/MA/オシレーター)", "寄与度": scores['technical'], "配分レンジ": "±40 pt"},
            {"ファクター": "FedWatch金融政策 (次回FOMC)", "寄与度": scores['fedwatch'], "配分レンジ": "±30 pt"},
            {"ファクター": "ニュースセンチメント (為替速報)", "寄与度": scores['news'], "配分レンジ": "±15 pt"},
            {"ファクター": "直近モメンタム (過去6時間推移)", "寄与度": scores['momentum'], "配分レンジ": "±15 pt"},
            {"ファクター": "総合USDバイアス", "寄与度": scores['composite_usd'], "配分レンジ": "±100 pt"}
        ])
        
        st.dataframe(score_df, use_container_width=True, hide_index=True)
        
        st.markdown("##### 🔍 推論の判断根拠サマリー")
        for reason in pred['reasons']:
            st.markdown(f"- {reason}")

with tab_tech:
    st.markdown("#### 📐 高度テクニカル精密分析 (トレンド・一目・オシレーター)")
    
    # Row 1: Market Regime & Ichimoku Diagnostics
    col_t1, col_t2 = st.columns(2)
    
    with col_t1:
        st.markdown(f"""
        <div class="tech-box">
            <div class="metric-title">🏛️ 相場環境レジーム & ADXトレンド強度</div>
            <div style="font-size: 1.3rem; font-weight: 700; color: {tech_diag.get('regime_color', '#38bdf8')}; margin-bottom: 6px;">
                {tech_diag.get('regime_label', '判定中')}
            </div>
            <div style="font-size: 0.85rem; color: #cbd5e1; margin-bottom: 8px;">
                {tech_diag.get('regime_desc', '')}
            </div>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; font-size: 0.85rem;">
                <div>ADX(14): <b>{tech_diag.get('adx', 0):.1f}</b></div>
                <div>+DI: <b style="color: #10b981;">{tech_diag.get('plus_di', 0):.1f}</b></div>
                <div>-DI: <b style="color: #ef4444;">{tech_diag.get('minus_di', 0):.1f}</b></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_t2:
        st.markdown(f"""
        <div class="tech-box">
            <div class="metric-title">☁️ 一目均衡表 (Ichimoku Kinko Hyo) 診断</div>
            <div style="font-size: 1.3rem; font-weight: 700; color: {tech_diag.get('ichimoku_color', '#f59e0b')}; margin-bottom: 6px;">
                {tech_diag.get('ichimoku_pos_label', '判定中')}
            </div>
            <div style="font-size: 0.85rem; color: #cbd5e1; margin-bottom: 8px;">
                シグナル: <b>{tech_diag.get('tk_cross_label', '')}</b>
            </div>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; font-size: 0.85rem;">
                <div>転換線: <b>{tech_diag.get('tenkan', 0):.{summary['decimal_places']}f}</b></div>
                <div>基準線: <b>{tech_diag.get('kijun', 0):.{summary['decimal_places']}f}</b></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Row 2: Oscillators, Price Action & S/R Levels
    col_t3, col_t4 = st.columns(2)
    latest_row = df.iloc[-1]
    
    with col_t3:
        st.markdown("##### ⚡ オシレーター & ボリンジャーバンド状態")
        st.write(f"- **RSI (14期間)**: **{latest_row.get('RSI14', 50):.1f}** ({'買われすぎ警戒' if latest_row.get('RSI14', 50) > 70 else ('売られすぎ反発期待' if latest_row.get('RSI14', 50) < 30 else '中立ゾーン')})")
        st.write(f"- **Slow Stochastics (14, 3, 3)**: **{tech_diag.get('stoch_status', '')}**")
        st.write(f"- **MACD**: **{latest_row.get('MACD', 0):.4f}** (Signal: {latest_row.get('MACD_Signal', 0):.4f} / Hist: {latest_row.get('MACD_Hist', 0):.4f})")
        st.write(f"- **ボリンジャーバンド**: **{tech_diag.get('bb_state_label', 'バンド内推移')}** (上限 {latest_row.get('BB_Upper2', 0):.{summary['decimal_places']}f} / 下限 {latest_row.get('BB_Lower2', 0):.{summary['decimal_places']}f})")
        
    with col_t4:
        st.markdown("##### 🛡️ プライスアクション & 水準（支持線・抵抗線）")
        pa = tech_diag.get('price_action', {})
        st.write(f"- **直近ローソク足プライスアクション**: **{pa.get('pattern', '通常')}** ({'強気' if pa.get('signal')=='BULLISH' else ('弱気' if pa.get('signal')=='BEARISH' else '中立')})")
        st.write(f"- **上値レジスタンス 1**: **{sr_levels['resistances'][0]:.{summary['decimal_places']}f}**" if sr_levels['resistances'] else "- レジスタンス: なし")
        if len(sr_levels['resistances']) > 1:
            st.write(f"- **上値レジスタンス 2**: **{sr_levels['resistances'][1]:.{summary['decimal_places']}f}**")
        st.write(f"- **下値サポート 1**: **{sr_levels['supports'][0]:.{summary['decimal_places']}f}**" if sr_levels['supports'] else "- サポート: なし")
        if len(sr_levels['supports']) > 1:
            st.write(f"- **下値サポート 2**: **{sr_levels['supports'][1]:.{summary['decimal_places']}f}**")
        st.write(f"- **1時間足 ATR**: **{latest_row.get('ATR14', 0):.{summary['decimal_places']}f}** (約 {latest_row.get('ATR14', 0)/summary['pip_value']:.1f} Pips)")

with tab_fed:
    st.markdown("#### 🏛️ CME FedWatch（次回FOMC 金利予想確率）")
    
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        st.metric("次回FOMC日程", fedwatch.get('next_meeting_date', '未定'), f"あと約 {fedwatch.get('days_to_meeting', '-')} 日")
    with col_f2:
        st.metric("現在のFF金利水準", fedwatch.get('current_target', '3.75%-4.00%'), f"EFFR: {fedwatch.get('current_effr')}%")
    with col_f3:
        st.metric("政策スタンス判定", fedwatch.get('stance', '中立'), f"ドルバイアス: {fedwatch.get('us_bias_score', 0):+.2f}")
    
    st.markdown("##### 📊 次回会合における政策金利シナリオ確率")
    p_hold = fedwatch.get('hold_prob', 0.0)
    p_cut = fedwatch.get('cut_prob', 0.0)
    p_hike = fedwatch.get('hike_prob', 0.0)
    
    st.write(f"**据え置き ({p_hold}%)**")
    st.progress(min(1.0, p_hold / 100.0))
    
    st.write(f"**利下げ ({p_cut}%)**")
    st.progress(min(1.0, p_cut / 100.0))
    
    st.write(f"**利上げ ({p_hike}%)**")
    st.progress(min(1.0, p_hike / 100.0))
    
    st.info(f"💡 **為替への影響解説**: {fedwatch.get('summary_text')}")
    
    if fedwatch.get('all_meetings'):
        st.markdown("##### 📅 今後のFOMC会合予定と金利織り込み")
        m_list = []
        for m in fedwatch['all_meetings']:
            probs_str = ", ".join([f"{k}: {v}%" for k, v in m.get('probabilities', {}).items()])
            m_list.append({"会合日": m['date'], "先物限月": m['contract'], "金利織り込み": probs_str})
        st.dataframe(pd.DataFrame(m_list), use_container_width=True, hide_index=True)

with tab_news:
    st.markdown("#### 📰 最新の為替・ドル円・FRB関連ニュース速報")
    st.caption(f"収集件数: {news_summary['total_news']}件 | 強気: {news_summary['bullish_count']}件 / 弱気: {news_summary['bearish_count']}件 / 中立: {news_summary['neutral_count']}件")
    
    st.info(f"**現在のニュース全体センチメント**: {news_summary['overall_sentiment']}")
    
    for item in news_items:
        sent_color = "#10b981" if "BULLISH" in item['sentiment'] else ("#ef4444" if "BEARISH" in item['sentiment'] else "#94a3b8")
        st.markdown(f"""
        <div class="news-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: #94a3b8; font-size: 0.8rem;">{item['source']} • {item['published']}</span>
                <span style="color: {sent_color}; font-size: 0.8rem; font-weight: 700;">{item['sentiment']}</span>
            </div>
            <a href="{item['link']}" target="_blank" class="news-title">{item['title']}</a>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.caption("⚠️ 免責事項: 本アプリは統計モデルおよびAIによる推論情報を提供するものであり、将来の利益を保証するものではありません。外国為替証拠金取引(FX)は元本損失リスクが伴うため、投資判断は自己責任で行ってください。")
