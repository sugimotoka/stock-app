import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="株シグナルアプリ", layout="wide")
st.title("📈 株売買シグナルアプリ")

ticker = st.sidebar.text_input("ティッカーシンボル", value="AAPL")

timeframe = st.sidebar.selectbox(
    "時間足",
    ["1時間足", "4時間足", "日足", "週足", "月足"],
    index=2
)

timeframe_map = {
    "1時間足":  {"interval": "1h",  "period": "60d"},
    "4時間足":  {"interval": "4h",  "period": "60d"},
    "日足":     {"interval": "1d",  "period": "1y"},
    "週足":     {"interval": "1wk", "period": "5y"},
    "月足":     {"interval": "1mo", "period": "10y"},
}

if ticker:
    try:
        info = yf.Ticker(ticker).info
        company_name = info.get("longName") or info.get("shortName") or ticker
        st.sidebar.markdown(f"<div style='background:#1a1a2e;padding:8px;border-radius:6px;color:white;font-size:13px'>🏢 {company_name}</div>", unsafe_allow_html=True)
    except:
        pass

if st.sidebar.button("分析開始"):
    params = timeframe_map[timeframe]

    with st.spinner("データ取得中..."):
        df = yf.download(ticker, interval=params["interval"], period=params["period"])
        df.columns = df.columns.get_level_values(0)
        try:
            info = yf.Ticker(ticker).info
            company_name = info.get("longName") or info.get("shortName") or ticker
        except:
            company_name = ticker

    if df.empty:
        st.error("データが取得できませんでした。ティッカーを確認してください。")
        st.stop()

    # 指標計算
    df['EMA21'] = ta.trend.ema_indicator(df['Close'], window=21)
    df['EMA50'] = ta.trend.ema_indicator(df['Close'], window=50)
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    bb = ta.volatility.BollingerBands(df['Close'])
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_lower'] = bb.bollinger_lband()
    df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    df.dropna(inplace=True)

    if len(df) < 4:
        st.warning(f"⚠️ {timeframe}はデータが少なすぎます。4時間足・日足をお試しください。")
        st.stop()

    # トレンド判定
    df['trend'] = 'neutral'
    df.loc[(df['EMA21'] > df['EMA50']) & (df['MACD'] > df['MACD_signal']), 'trend'] = 'buy'
    df.loc[(df['EMA21'] < df['EMA50']) & (df['MACD'] < df['MACD_signal']), 'trend'] = 'sell'

    # スコア計算
    def calc_score(row, prev_row):
        s = 0
        if row['EMA21'] > row['EMA50']: s += 1
        else: s -= 1
        if row['RSI'] < 30: s += 2
        elif row['RSI'] > 70: s -= 2
        if row['MACD'] > row['MACD_signal']: s += 1
        else: s -= 1
        if row['Close'] < row['BB_lower']: s += 1
        elif row['Close'] > row['BB_upper']: s -= 1
        if row['OBV'] > prev_row['OBV']: s += 1
        else: s -= 1
        return s

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]
    score_today = calc_score(latest, prev)
    score_yesterday = calc_score(prev, prev2)

    # 企業名・株価ヘッダー
    current_price = float(latest['Close'])
    prev_price = float(prev['Close'])
    price_change = current_price - prev_price
    price_pct = (price_change / prev_price) * 100
    price_color = "#00C851" if price_change >= 0 else "#ff4444"
    price_arrow = "▲" if price_change >= 0 else "▼"

    st.markdown(f"""
    <div style='background:#1a1a2e;padding:8px 15px;border-radius:8px;
    display:flex;align-items:center;gap:20px;margin-bottom:8px'>
    <span style='color:white;font-size:16px;font-weight:bold'>{company_name}</span>
    <span style='color:gray;font-size:13px'>{ticker} · {timeframe}</span>
    <span style='color:{price_color};font-size:18px;font-weight:bold'>
    {current_price:.2f} {price_arrow} {abs(price_change):.2f} ({price_pct:+.2f}%)</span>
    </div>""", unsafe_allow_html=True)

    # ===== メインチャート =====
    df_buy = df[df['trend'] == 'buy']
    df_sell = df[df['trend'] == 'sell']
    df_neutral = df[df['trend'] == 'neutral']

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.12, 0.18, 0.15],
                        subplot_titles=(
                            f"🟡買いトレンド　🔴売りトレンド　⚪中立　({timeframe})",
                            "トレンドバー", "RSI", "OBV"))

    if not df_buy.empty:
        fig.add_trace(go.Candlestick(
            x=df_buy.index, open=df_buy['Open'], high=df_buy['High'],
            low=df_buy['Low'], close=df_buy['Close'],
            increasing_line_color='gold', decreasing_line_color='gold',
            increasing_fillcolor='gold', decreasing_fillcolor='gold',
            name="🟡買いトレンド"), row=1, col=1)

    if not df_sell.empty:
        fig.add_trace(go.Candlestick(
            x=df_sell.index, open=df_sell['Open'], high=df_sell['High'],
            low=df_sell['Low'], close=df_sell['Close'],
            increasing_line_color='red', decreasing_line_color='red',
            increasing_fillcolor='red', decreasing_fillcolor='red',
            name="🔴売りトレンド"), row=1, col=1)

    if not df_neutral.empty:
        fig.add_trace(go.Candlestick(
            x=df_neutral.index, open=df_neutral['Open'], high=df_neutral['High'],
            low=df_neutral['Low'], close=df_neutral['Close'],
            increasing_line_color='gray', decreasing_line_color='gray',
            increasing_fillcolor='gray', decreasing_fillcolor='gray',
            name="⚪中立"), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['EMA21'],
                              line=dict(color='orange', width=1.5), name="EMA21"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'],
                              line=dict(color='cyan', width=1.5), name="EMA50"), row=1, col=1)

    trend_colors = ['gold' if t == 'buy' else 'red' if t == 'sell' else 'gray' for t in df['trend']]
    fig.add_trace(go.Bar(x=df.index, y=[1]*len(df),
                          marker_color=trend_colors, showlegend=False), row=2, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'],
                              line=dict(color='magenta', width=1.5), name="RSI"), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['OBV'],
                              line=dict(color='teal', width=1.5), name="OBV"), row=4, col=1)

    fig.update_layout(height=750, xaxis_rangeslider_visible=False, template="plotly_dark")
    fig.update_yaxes(showticklabels=False, row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # ===== シグナルまとめ =====
    col1, col2 = st.columns(2)

    with col1:
        if score_today >= 3: bg, label = "#00C851", "🟢🟢 強い買い"
        elif score_today >= 1: bg, label = "#00C851", "🟢 弱い買い"
        elif score_today <= -3: bg, label = "#ff4444", "🔴🔴 強い売り"
        elif score_today <= -1: bg, label = "#ff4444", "🔴 弱い売り"
        else: bg, label = "#555", "⚪ 中立"

        st.markdown(f"""
        <div style='background:{bg};padding:10px;border-radius:8px;
        text-align:center;margin-bottom:6px'>
        <b style='color:white;font-size:18px'>{label}</b><br>
        <span style='color:white;font-size:12px'>スコア：{score_today}（前回：{score_yesterday}）</span>
        </div>""", unsafe_allow_html=True)

        buy_triggers, sell_triggers = [], []
        if latest['MACD'] > latest['MACD_signal'] and prev['MACD'] <= prev['MACD_signal']:
            buy_triggers.append("MACDゴールデンクロス")
        if latest['MACD'] < latest['MACD_signal'] and prev['MACD'] >= prev['MACD_signal']:
            sell_triggers.append("MACDデッドクロス")
        if latest['RSI'] > 30 and prev['RSI'] <= 30:
            buy_triggers.append("RSI売られすぎから回復")
        if latest['RSI'] < 70 and prev['RSI'] >= 70:
            sell_triggers.append("RSI買われすぎから下落")
        if latest['EMA21'] > latest['EMA50'] and prev['EMA21'] <= prev['EMA50']:
            buy_triggers.append("EMAゴールデンクロス")
        if latest['EMA21'] < latest['EMA50'] and prev['EMA21'] >= prev['EMA50']:
            sell_triggers.append("EMAデッドクロス")
        if latest['OBV'] > prev['OBV'] and prev['OBV'] <= prev2['OBV']:
            buy_triggers.append("OBV上昇転換")
        if latest['OBV'] < prev['OBV'] and prev['OBV'] >= prev2['OBV']:
            sell_triggers.append("OBV下落転換")
        if score_today > 0 and score_yesterday <= 0:
            buy_triggers.append("総合スコアがプラス転換")
        if score_today < 0 and score_yesterday >= 0:
            sell_triggers.append("総合スコアがマイナス転換")

        if buy_triggers:
            t_html = "".join([f"<div style='font-size:12px'>✅ {t}</div>" for t in buy_triggers])
            st.markdown(f"<div style='background:#007E33;padding:8px;border-radius:6px;margin-bottom:4px'><b style='color:white'>🟢 買い転換</b>{t_html}</div>", unsafe_allow_html=True)
        if sell_triggers:
            t_html = "".join([f"<div style='font-size:12px'>❌ {t}</div>" for t in sell_triggers])
            st.markdown(f"<div style='background:#CC0000;padding:8px;border-radius:6px;margin-bottom:4px'><b style='color:white'>🔴 売り転換</b>{t_html}</div>", unsafe_allow_html=True)
        if not buy_triggers and not sell_triggers:
            st.markdown("<div style='background:#555;padding:8px;border-radius:6px'><b style='color:white'>⚪ 転換シグナルなし</b></div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<b style='color:white;font-size:14px'>🔍 各指標</b>", unsafe_allow_html=True)
        signals = []
        signals.append(("✅ EMA上昇" if latest['EMA21'] > latest['EMA50'] else "❌ EMA下降",
                        "buy" if latest['EMA21'] > latest['EMA50'] else "sell"))
        rsi_val = float(latest['RSI'])
        if rsi_val < 30: signals.append((f"✅ RSI売られすぎ({rsi_val:.0f})", "buy"))
        elif rsi_val > 70: signals.append((f"❌ RSI買われすぎ({rsi_val:.0f})", "sell"))
        else: signals.append((f"⚪ RSI中立({rsi_val:.0f})", "neutral"))
        signals.append(("✅ MACDゴールデン" if latest['MACD'] > latest['MACD_signal'] else "❌ MACDデッド",
                        "buy" if latest['MACD'] > latest['MACD_signal'] else "sell"))
        if latest['Close'] < latest['BB_lower']: signals.append(("✅ BB下限反発", "buy"))
        elif latest['Close'] > latest['BB_upper']: signals.append(("❌ BB上限過熱", "sell"))
        else: signals.append(("⚪ BB中央", "neutral"))
        signals.append(("✅ OBV上昇" if latest['OBV'] > prev['OBV'] else "❌ OBV下降",
                        "buy" if latest['OBV'] > prev['OBV'] else "sell"))
        atr_pct = float(latest['ATR'] / latest['Close']) * 100
        signals.append((f"⚠️ ATR {atr_pct:.1f}%({'高ボラ' if atr_pct > 3 else '普通'})", "neutral"))

        for text, kind in signals:
            bg = "#00C851" if kind == "buy" else "#ff4444" if kind == "sell" else "#444"
            st.markdown(f"<div style='background:{bg};padding:5px 10px;border-radius:5px;color:white;margin:2px 0;font-size:12px'>{text}</div>", unsafe_allow_html=True)