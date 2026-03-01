import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =========================
# 基本設定
# =========================
st.set_page_config(page_title="株シグナルアプリ", layout="wide")
st.title("📈 株売買シグナルアプリ")

# =========================
# ユーティリティ
# =========================
def is_jp_equity(ticker: str) -> bool:
    t = (ticker or "").upper().strip()
    return t.endswith(".T")  # 東証の簡易判定


@st.cache_data(ttl=300, show_spinner=False)
def cached_download(ticker: str, interval: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, interval=interval, period=period, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    df.index = pd.to_datetime(df.index)
    return df


@st.cache_data(ttl=300, show_spinner=False)
def fetch_4h_usual_from_1m(ticker: str, days: int = 7) -> pd.DataFrame:
    """米株など：1分足を4Hにresample"""
    days = max(1, min(int(days), 7))
    df_1m = yf.download(ticker, interval="1m", period=f"{days}d", progress=False)
    if df_1m is None or df_1m.empty:
        return pd.DataFrame()
    if isinstance(df_1m.columns, pd.MultiIndex):
        df_1m.columns = df_1m.columns.get_level_values(0)
    df_1m = df_1m.dropna()
    df_1m.index = pd.to_datetime(df_1m.index)

    df_4h = df_1m.resample("4H").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }).dropna()

    return df_4h


@st.cache_data(ttl=300, show_spinner=False)
def fetch_jp_4h_sessions_from_1m(ticker: str, days: int = 7) -> pd.DataFrame:
    """日本株：前場/後場を各1本にまとめる（2本/日）。市場外・夜はYahooが1分足を返さない場合あり。"""
    days = max(1, min(int(days), 7))
    df = yf.download(ticker, interval="1m", period=f"{days}d", progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    df.index = pd.to_datetime(df.index)

    morning = df.between_time("09:00", "11:30")
    afternoon = df.between_time("12:30", "15:00")

    def ohlcv(x):
        return pd.Series({
            "Open": x["Open"].iloc[0],
            "High": x["High"].max(),
            "Low": x["Low"].min(),
            "Close": x["Close"].iloc[-1],
            "Volume": x["Volume"].sum(),
        })

    if morning.empty and afternoon.empty:
        return pd.DataFrame()

    df_m = morning.groupby(morning.index.date).apply(ohlcv) if not morning.empty else pd.DataFrame()
    df_a = afternoon.groupby(afternoon.index.date).apply(ohlcv) if not afternoon.empty else pd.DataFrame()

    # 見た目の時刻ラベル
    if not df_m.empty:
        df_m.index = pd.to_datetime(df_m.index) + pd.Timedelta(hours=11, minutes=30)
    if not df_a.empty:
        df_a.index = pd.to_datetime(df_a.index) + pd.Timedelta(hours=15)

    df_4h = pd.concat([df_m, df_a]).sort_index().dropna()
    return df_4h


def make_pseudo_4h_from_daily(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    日本株用フォールバック：
    日足を「前場」「後場」相当に2分割した疑似4H（夜・休日でも必ず出す）
    """
    df_daily = df_daily.copy()
    if df_daily is None or df_daily.empty:
        return pd.DataFrame()

    if isinstance(df_daily.columns, pd.MultiIndex):
        df_daily.columns = df_daily.columns.get_level_values(0)

    df_daily = df_daily.dropna()
    df_daily.index = pd.to_datetime(df_daily.index)

    rows = []
    for idx, r in df_daily.iterrows():
        o = float(r["Open"])
        h = float(r["High"])
        l = float(r["Low"])
        c = float(r["Close"])
        v = float(r.get("Volume", 0.0))

        mid = (o + c) / 2.0

        # 前場（11:30）
        rows.append({
            "Datetime": idx + pd.Timedelta(hours=11, minutes=30),
            "Open": o,
            "High": h,
            "Low": l,
            "Close": mid,
            "Volume": v / 2.0
        })

        # 後場（15:00）
        rows.append({
            "Datetime": idx + pd.Timedelta(hours=15),
            "Open": mid,
            "High": h,
            "Low": l,
            "Close": c,
            "Volume": v / 2.0
        })

    df = pd.DataFrame(rows).set_index("Datetime")
    df = df.sort_index().dropna()
    return df


@st.cache_data(ttl=300, show_spinner=False)
def fetch_4h_auto(ticker: str, days: int = 7) -> pd.DataFrame:
    """
    4H自動：
    - 日本株（.T）: まず1分足→前場/後場を試す
      失敗したら 日足→疑似4H にフォールバック（夜・休日でも必ず表示）
    - それ以外: 1分足→通常4H
    """
    days = max(1, min(int(days), 7))

    if is_jp_equity(ticker):
        # ① まず本物（1分足→前場/後場）
        df = fetch_jp_4h_sessions_from_1m(ticker, days=days)
        if df is not None and not df.empty:
            return df

        # ② フォールバック（日足→疑似4H）
        df_daily = cached_download(ticker, interval="1d", period="60d")
        if df_daily is not None and not df_daily.empty:
            return make_pseudo_4h_from_daily(df_daily)

        return pd.DataFrame()

    # 米株など
    return fetch_4h_usual_from_1m(ticker, days=days)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_company_name_optional(ticker: str) -> str:
    """yfinance .info は重いので任意＆キャッシュ"""
    try:
        info = yf.Ticker(ticker).info
        return info.get("longName") or info.get("shortName") or ticker
    except:
        return ticker


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["EMA21"] = ta.trend.ema_indicator(df["Close"], window=21)
    df["EMA50"] = ta.trend.ema_indicator(df["Close"], window=50)
    df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
    macd = ta.trend.MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    bb = ta.volatility.BollingerBands(df["Close"])
    df["BB_upper"] = bb.bollinger_hband()
    df["BB_lower"] = bb.bollinger_lband()
    df["OBV"] = ta.volume.on_balance_volume(df["Close"], df["Volume"])
    df["ATR"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], window=14)
    df.dropna(inplace=True)
    return df


def add_daily_trend(df_daily: pd.DataFrame) -> pd.DataFrame:
    """日足長期トレンド（EMA50/EMA200 + MACD）"""
    df = df_daily.copy()
    df["EMA50"] = ta.trend.ema_indicator(df["Close"], window=50)
    df["EMA200"] = ta.trend.ema_indicator(df["Close"], window=200)
    macd = ta.trend.MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df.dropna(inplace=True)

    df["daily_trend"] = "neutral"
    df.loc[(df["EMA50"] > df["EMA200"]) & (df["MACD"] > df["MACD_signal"]), "daily_trend"] = "bull"
    df.loc[(df["EMA50"] < df["EMA200"]) & (df["MACD"] < df["MACD_signal"]), "daily_trend"] = "bear"
    return df


def calc_score(row, prev_row):
    s = 0
    # EMA
    s += 1 if row["EMA21"] > row["EMA50"] else -1

    # RSI
    if row["RSI"] < 30:
        s += 2
    elif row["RSI"] > 70:
        s -= 2

    # MACD
    s += 1 if row["MACD"] > row["MACD_signal"] else -1

    # BB
    if row["Close"] < row["BB_lower"]:
        s += 1
    elif row["Close"] > row["BB_upper"]:
        s -= 1

    # OBV（微差ノイズを抑える）
    obv_delta = row["OBV"] - prev_row["OBV"]
    thresh = abs(prev_row["OBV"]) * 0.001 if abs(prev_row["OBV"]) > 0 else 0
    if obv_delta > thresh:
        s += 1
    elif obv_delta < -thresh:
        s -= 1

    return s


def enforce_daily_filter(score: int, daily_trend: str):
    """日足トレンドに逆らうスコアは無効化"""
    if daily_trend == "bull" and score < 0:
        return 0, "⚠️ 日足が上向きのため、売りシグナルは無効化しました"
    if daily_trend == "bear" and score > 0:
        return 0, "⚠️ 日足が下向きのため、買いシグナルは無効化しました"
    return score, None


# =========================
# サイドバーUI
# =========================
st.sidebar.subheader("設定")

preset = st.sidebar.selectbox(
    "よく使う銘柄",
    ["（手入力）", "7203.T トヨタ", "6758.T ソニー", "9984.T SBグループ", "8306.T 三菱UFJ", "AAPL Apple", "MSFT Microsoft"],
    index=0
)

default_ticker = "7203.T"
ticker = st.sidebar.text_input(
    "ティッカーシンボル（例：7203.T / AAPL）",
    value=(preset.split()[0] if preset != "（手入力）" else default_ticker)
)

timeframe = st.sidebar.selectbox(
    "時間足",
    ["1時間足", "4時間足（自動）", "日足", "週足", "月足"],
    index=1
)

timeframe_map = {
    "1時間足": {"type": "yahoo", "interval": "1h", "period": "60d"},
    "4時間足（自動）": {"type": "4h_auto", "days": 7},
    "日足": {"type": "yahoo", "interval": "1d", "period": "1y"},
    "週足": {"type": "yahoo", "interval": "1wk", "period": "5y"},
    "月足": {"type": "yahoo", "interval": "1mo", "period": "10y"},
}

show_company = st.sidebar.toggle("会社名を表示（遅い場合OFF）", value=False)
auto_run = st.sidebar.toggle("入力変更で自動分析", value=False)

# ATRリスク設定
st.sidebar.subheader("ATRリスク管理")
sl_mult = st.sidebar.slider("損切り ATR倍率", 0.5, 3.0, 1.5, 0.1)
tp_mult = st.sidebar.slider("利確 ATR倍率", 1.0, 6.0, 2.5, 0.1)
show_risk_lines = st.sidebar.toggle("ATRライン（SL/TP）をチャート表示", value=True)

run = st.sidebar.button("分析開始") or auto_run

# =========================
# メイン
# =========================
if run and ticker:
    params = timeframe_map[timeframe]

    with st.spinner("データ取得中..."):
        if params["type"] == "4h_auto":
            df = fetch_4h_auto(ticker, days=params.get("days", 7))
        else:
            df = cached_download(ticker, interval=params["interval"], period=params["period"])

        # 日足（長期トレンド判定用）
        df_daily = cached_download(ticker, interval="1d", period="5y")

        # 会社名（任意）
        company_name = ticker
        if show_company:
            company_name = fetch_company_name_optional(ticker)

    if df is None or df.empty:
        st.error("データが取得できませんでした。ティッカーや時間足を確認してください。")
        st.stop()

    # 日足トレンド判定
    daily_trend = "neutral"
    if df_daily is not None and not df_daily.empty and len(df_daily) >= 250:
        df_daily_tr = add_daily_trend(df_daily)
        if not df_daily_tr.empty:
            daily_trend = df_daily_tr.iloc[-1]["daily_trend"]

    # 指標計算
    df = add_indicators(df)
    if len(df) < 4:
        st.warning("⚠️ データが少なすぎます。別の時間足をお試しください。")
        st.stop()

    # トレンド判定
    df["trend"] = "neutral"
    df.loc[(df["EMA21"] > df["EMA50"]) & (df["MACD"] > df["MACD_signal"]), "trend"] = "buy"
    df.loc[(df["EMA21"] < df["EMA50"]) & (df["MACD"] < df["MACD_signal"]), "trend"] = "sell"

    # スコア計算
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]

    score_today = calc_score(latest, prev)
    score_yesterday = calc_score(prev, prev2)

    # 日足フィルタ強制（逆行は0にする）
    score_today, filter_msg_today = enforce_daily_filter(score_today, daily_trend)
    score_yesterday, _ = enforce_daily_filter(score_yesterday, daily_trend)

    # bias（ATRライン表示用）
    bias = "neutral"
    if score_today >= 1:
        bias = "long"
    elif score_today <= -1:
        bias = "short"

    # ATRリスクライン
    atr = float(latest["ATR"])
    entry = float(latest["Close"])
    long_sl = entry - atr * sl_mult
    long_tp = entry + atr * tp_mult
    short_sl = entry + atr * sl_mult
    short_tp = entry - atr * tp_mult

    # 企業名・株価ヘッダー
    current_price = float(latest["Close"])
    prev_price = float(prev["Close"])
    price_change = current_price - prev_price
    price_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0
    price_color = "#00C851" if price_change >= 0 else "#ff4444"
    price_arrow = "▲" if price_change >= 0 else "▼"

    daily_trend_label = "🟢 日足：上向き" if daily_trend == "bull" else "🔴 日足：下向き" if daily_trend == "bear" else "⚪ 日足：中立"

    st.markdown(f"""
    <div style='background:#1a1a2e;padding:8px 15px;border-radius:8px;
    display:flex;align-items:center;gap:20px;margin-bottom:8px'>
    <span style='color:white;font-size:16px;font-weight:bold'>{company_name}</span>
    <span style='color:gray;font-size:13px'>{ticker} · {timeframe}</span>
    <span style='color:{price_color};font-size:18px;font-weight:bold'>
    {current_price:.2f} {price_arrow} {abs(price_change):.2f} ({price_pct:+.2f}%)</span>
    <span style='color:white;font-size:13px'>{daily_trend_label}</span>
    </div>""", unsafe_allow_html=True)

    # フィルタ警告
    if filter_msg_today:
        st.warning(filter_msg_today)

    # ===== メインチャート =====
    df_buy = df[df["trend"] == "buy"]
    df_sell = df[df["trend"] == "sell"]
    df_neutral = df[df["trend"] == "neutral"]

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.12, 0.18, 0.15],
                        subplot_titles=(
                            f"🟡買いトレンド　🔴売りトレンド　⚪中立　({timeframe})",
                            "トレンドバー", "RSI", "OBV"))

    if not df_buy.empty:
        fig.add_trace(go.Candlestick(
            x=df_buy.index, open=df_buy["Open"], high=df_buy["High"],
            low=df_buy["Low"], close=df_buy["Close"],
            increasing_line_color="gold", decreasing_line_color="gold",
            increasing_fillcolor="gold", decreasing_fillcolor="gold",
            name="🟡買いトレンド"), row=1, col=1)

    if not df_sell.empty:
        fig.add_trace(go.Candlestick(
            x=df_sell.index, open=df_sell["Open"], high=df_sell["High"],
            low=df_sell["Low"], close=df_sell["Close"],
            increasing_line_color="red", decreasing_line_color="red",
            increasing_fillcolor="red", decreasing_fillcolor="red",
            name="🔴売りトレンド"), row=1, col=1)

    if not df_neutral.empty:
        fig.add_trace(go.Candlestick(
            x=df_neutral.index, open=df_neutral["Open"], high=df_neutral["High"],
            low=df_neutral["Low"], close=df_neutral["Close"],
            increasing_line_color="gray", decreasing_line_color="gray",
            increasing_fillcolor="gray", decreasing_fillcolor="gray",
            name="⚪中立"), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["EMA21"],
                              line=dict(color="orange", width=1.5), name="EMA21"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"],
                              line=dict(color="cyan", width=1.5), name="EMA50"), row=1, col=1)

    # ATRライン（SL/TP）
    if show_risk_lines:
        if bias == "long":
            fig.add_hline(y=long_sl, line_dash="dot", row=1, col=1)
            fig.add_hline(y=long_tp, line_dash="dot", row=1, col=1)
        elif bias == "short":
            fig.add_hline(y=short_sl, line_dash="dot", row=1, col=1)
            fig.add_hline(y=short_tp, line_dash="dot", row=1, col=1)

    trend_colors = ["gold" if t == "buy" else "red" if t == "sell" else "gray" for t in df["trend"]]
    fig.add_trace(go.Bar(x=df.index, y=[1]*len(df),
                          marker_color=trend_colors, showlegend=False), row=2, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"],
                              line=dict(color="magenta", width=1.5), name="RSI"), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["OBV"],
                              line=dict(color="teal", width=1.5), name="OBV"), row=4, col=1)

    fig.update_layout(height=750, xaxis_rangeslider_visible=False, template="plotly_dark")
    fig.update_yaxes(showticklabels=False, row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # ===== シグナルまとめ =====
    col1, col2 = st.columns(2)

    with col1:
        if score_today >= 3:
            bg, label = "#00C851", "🟢🟢 強い買い"
        elif score_today >= 1:
            bg, label = "#00C851", "🟢 弱い買い"
        elif score_today <= -3:
            bg, label = "#ff4444", "🔴🔴 強い売り"
        elif score_today <= -1:
            bg, label = "#ff4444", "🔴 弱い売り"
        else:
            bg, label = "#555", "⚪ 中立"

        st.markdown(f"""
        <div style='background:{bg};padding:10px;border-radius:8px;
        text-align:center;margin-bottom:6px'>
        <b style='color:white;font-size:18px'>{label}</b><br>
        <span style='color:white;font-size:12px'>スコア：{score_today}（前回：{score_yesterday}）</span>
        </div>""", unsafe_allow_html=True)

        buy_triggers, sell_triggers = [], []

        if latest["MACD"] > latest["MACD_signal"] and prev["MACD"] <= prev["MACD_signal"]:
            buy_triggers.append("MACDゴールデンクロス")
        if latest["MACD"] < latest["MACD_signal"] and prev["MACD"] >= prev["MACD_signal"]:
            sell_triggers.append("MACDデッドクロス")

        if latest["RSI"] > 30 and prev["RSI"] <= 30:
            buy_triggers.append("RSI売られすぎから回復")
        if latest["RSI"] < 70 and prev["RSI"] >= 70:
            sell_triggers.append("RSI買われすぎから下落")

        if latest["EMA21"] > latest["EMA50"] and prev["EMA21"] <= prev["EMA50"]:
            buy_triggers.append("EMAゴールデンクロス")
        if latest["EMA21"] < latest["EMA50"] and prev["EMA21"] >= prev["EMA50"]:
            sell_triggers.append("EMAデッドクロス")

        if latest["OBV"] > prev["OBV"] and prev["OBV"] <= prev2["OBV"]:
            buy_triggers.append("OBV上昇転換")
        if latest["OBV"] < prev["OBV"] and prev["OBV"] >= prev2["OBV"]:
            sell_triggers.append("OBV下落転換")

        if score_today > 0 and score_yesterday <= 0:
            buy_triggers.append("総合スコアがプラス転換（フィルタ後）")
        if score_today < 0 and score_yesterday >= 0:
            sell_triggers.append("総合スコアがマイナス転換（フィルタ後）")

        if daily_trend == "bull":
            buy_triggers.append("日足が上向き")
        elif daily_trend == "bear":
            sell_triggers.append("日足が下向き")

        if buy_triggers:
            t_html = "".join([f"<div style='font-size:12px'>✅ {t}</div>" for t in buy_triggers])
            st.markdown(f"<div style='background:#007E33;padding:8px;border-radius:6px;margin-bottom:4px'><b style='color:white'>🟢 買い転換</b>{t_html}</div>", unsafe_allow_html=True)
        if sell_triggers:
            t_html = "".join([f"<div style='font-size:12px'>❌ {t}</div>" for t in sell_triggers])
            st.markdown(f"<div style='background:#CC0000;padding:8px;border-radius:6px;margin-bottom:4px'><b style='color:white'>🔴 売り転換</b>{t_html}</div>", unsafe_allow_html=True)
        if not buy_triggers and not sell_triggers:
            st.markdown("<div style='background:#555;padding:8px;border-radius:6px'><b style='color:white'>⚪ 転換シグナルなし</b></div>", unsafe_allow_html=True)

        st.markdown("<div style='margin-top:10px'><b style='color:white'>🎯 ATR目安（エントリー＝最新終値）</b></div>", unsafe_allow_html=True)
        if bias == "long":
            st.write(f"ロング想定：SL **{long_sl:.2f}** / TP **{long_tp:.2f}**（ATR={atr:.2f}）")
        elif bias == "short":
            st.write(f"ショート想定：SL **{short_sl:.2f}** / TP **{short_tp:.2f}**（ATR={atr:.2f}）")
        else:
            st.write(f"中立：参考（ATR={atr:.2f}）")
            st.write(f"ロング：SL {long_sl:.2f} / TP {long_tp:.2f}　｜　ショート：SL {short_sl:.2f} / TP {short_tp:.2f}")

    with col2:
        st.markdown("<b style='color:white;font-size:14px'>🔍 各指標</b>", unsafe_allow_html=True)
        signals = []

        signals.append(("✅ EMA上昇" if latest["EMA21"] > latest["EMA50"] else "❌ EMA下降",
                        "buy" if latest["EMA21"] > latest["EMA50"] else "sell"))

        rsi_val = float(latest["RSI"])
        if rsi_val < 30:
            signals.append((f"✅ RSI売られすぎ({rsi_val:.0f})", "buy"))
        elif rsi_val > 70:
            signals.append((f"❌ RSI買われすぎ({rsi_val:.0f})", "sell"))
        else:
            signals.append((f"⚪ RSI中立({rsi_val:.0f})", "neutral"))

        signals.append(("✅ MACDゴールデン" if latest["MACD"] > latest["MACD_signal"] else "❌ MACDデッド",
                        "buy" if latest["MACD"] > latest["MACD_signal"] else "sell"))

        if latest["Close"] < latest["BB_lower"]:
            signals.append(("✅ BB下限反発", "buy"))
        elif latest["Close"] > latest["BB_upper"]:
            signals.append(("❌ BB上限過熱", "sell"))
        else:
            signals.append(("⚪ BB中央", "neutral"))

        signals.append(("✅ OBV上昇" if latest["OBV"] > prev["OBV"] else "❌ OBV下降",
                        "buy" if latest["OBV"] > prev["OBV"] else "sell"))

        atr_pct = float(latest["ATR"] / latest["Close"]) * 100 if latest["Close"] != 0 else 0
        signals.append((f"⚠️ ATR {atr_pct:.1f}%（{'高ボラ' if atr_pct > 3 else '普通'}）", "neutral"))

        if daily_trend == "bull":
            signals.append(("🟢 日足トレンド上向き", "buy"))
        elif daily_trend == "bear":
            signals.append(("🔴 日足トレンド下向き", "sell"))
        else:
            signals.append(("⚪ 日足トレンド中立", "neutral"))

        for text, kind in signals:
            bg = "#00C851" if kind == "buy" else "#ff4444" if kind == "sell" else "#444"
            st.markdown(f"<div style='background:{bg};padding:5px 10px;border-radius:5px;color:white;margin:2px 0;font-size:12px'>{text}</div>", unsafe_allow_html=True)

else:
    st.info("左のサイドバーでティッカーと時間足を選んで「分析開始」を押してください（自動分析ONも可）。")
