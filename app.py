import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =========================
# 基本設定
# =========================
st.set_page_config(page_title="株売買シグナルアプリ", layout="wide")
st.title("📈 株売買シグナルアプリ")


# =========================
# ユーティリティ
# =========================
def is_jp_equity(ticker: str) -> bool:
    t = (ticker or "").upper().strip()
    return t.endswith(".T")


def parse_watchlist(text: str) -> list[str]:
    if not text:
        return []
    # , 改行 スペース を全部区切りにする
    raw = (
        text.replace("\n", ",")
            .replace(" ", ",")
            .replace("　", ",")
    )
    items = [x.strip().upper() for x in raw.split(",")]
    items = [x for x in items if x]
    # 重複排除（順序維持）
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


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
def fetch_4h_usual_from_1m(ticker: str, days: int = 30) -> pd.DataFrame:
    """米株など：1分足を4Hにresample（Yahooが1分足を返さない/少ない場合は空になり得る）"""
    days = max(1, min(int(days), 30))
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
def fetch_jp_4h_sessions_from_1m(ticker: str, days: int = 30) -> pd.DataFrame:
    """
    日本株：前場/後場を各1本にまとめる（2本/日）。
    Yahooが1分足を返さない場合は空になり得る。
    """
    days = max(1, min(int(days), 30))
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
            "Volume": float(x["Volume"].sum()) if "Volume" in x else 0.0,
        })

    if morning.empty and afternoon.empty:
        return pd.DataFrame()

    df_m = morning.groupby(morning.index.date).apply(ohlcv) if not morning.empty else pd.DataFrame()
    df_a = afternoon.groupby(afternoon.index.date).apply(ohlcv) if not afternoon.empty else pd.DataFrame()

    # 表示上の時刻ラベル（前場終わり / 引け）
    if not df_m.empty:
        df_m.index = pd.to_datetime(df_m.index) + pd.Timedelta(hours=11, minutes=30)
    if not df_a.empty:
        df_a.index = pd.to_datetime(df_a.index) + pd.Timedelta(hours=15)

    df_4h = pd.concat([df_m, df_a]).sort_index().dropna()
    return df_4h


def make_pseudo_4h_from_daily(df_daily: pd.DataFrame, parts_per_day: int = 4) -> pd.DataFrame:
    """
    フォールバック用「疑似4H」生成（表示を途切れさせないための近似）
    - parts_per_day=2 : 日本株想定（2本/日）
    - parts_per_day=4 : 米株想定（4本/日）
    """
    if df_daily is None or df_daily.empty:
        return pd.DataFrame()

    df = df_daily.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    df.index = pd.to_datetime(df.index)

    parts_per_day = int(parts_per_day)
    if parts_per_day not in (2, 4):
        parts_per_day = 4

    if parts_per_day == 2:
        time_offsets = [
            pd.Timedelta(hours=11, minutes=30),
            pd.Timedelta(hours=15),
        ]
    else:
        time_offsets = [
            pd.Timedelta(hours=10, minutes=30),
            pd.Timedelta(hours=14, minutes=30),
            pd.Timedelta(hours=18, minutes=30),
            pd.Timedelta(hours=22, minutes=30),
        ]

    rows = []
    for idx, r in df.iterrows():
        o = float(r["Open"])
        h = float(r["High"])
        l = float(r["Low"])
        c = float(r["Close"])
        v = float(r.get("Volume", 0.0))

        if parts_per_day == 2:
            closes = [(o + c) / 2.0, c]
            opens = [o, (o + c) / 2.0]
        else:
            p0 = o
            p1 = o + (c - o) * 0.25
            p2 = o + (c - o) * 0.50
            p3 = o + (c - o) * 0.75
            p4 = c
            opens = [p0, p1, p2, p3]
            closes = [p1, p2, p3, p4]

        vol_each = v / float(parts_per_day) if parts_per_day > 0 else 0.0

        for i in range(parts_per_day):
            rows.append({
                "Datetime": idx + time_offsets[i],
                "Open": float(opens[i]),
                "High": h,
                "Low": l,
                "Close": float(closes[i]),
                "Volume": vol_each
            })

    out = pd.DataFrame(rows).set_index("Datetime")
    out = out.sort_index().dropna()
    return out


@st.cache_data(ttl=300, show_spinner=False)
def fetch_company_name_optional(ticker: str) -> str:
    try:
        info = yf.Ticker(ticker).info
        return info.get("longName") or info.get("shortName") or ticker
    except:
        return ticker


@st.cache_data(ttl=300, show_spinner=False)
def fetch_4h_auto(ticker: str, days: int = 30) -> tuple[pd.DataFrame, str]:
    """
    4H自動（安定版）
    - 日本株(.T): 1分足→前場/後場（native） 失敗/不足→ 日足→疑似2本/日（pseudo）
    - 米株等:    1分足→4H（native）        失敗/不足→ 日足→疑似4本/日（pseudo）
    """
    days = max(1, min(int(days), 30))

    if is_jp_equity(ticker):
        df_native = fetch_jp_4h_sessions_from_1m(ticker, days=days)
        if df_native is not None and len(df_native) >= 4:
            return df_native, "native"

        df_daily = cached_download(ticker, interval="1d", period="180d")
        if df_daily is not None and not df_daily.empty:
            return make_pseudo_4h_from_daily(df_daily, parts_per_day=2), "pseudo"

        return pd.DataFrame(), "none"

    df_native = fetch_4h_usual_from_1m(ticker, days=days)
    if df_native is not None and len(df_native) >= 4:
        return df_native, "native"

    df_daily = cached_download(ticker, interval="1d", period="180d")
    if df_daily is not None and not df_daily.empty:
        return make_pseudo_4h_from_daily(df_daily, parts_per_day=4), "pseudo"

    return pd.DataFrame(), "none"


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
    s += 1 if row["EMA21"] > row["EMA50"] else -1

    if row["RSI"] < 30:
        s += 2
    elif row["RSI"] > 70:
        s -= 2

    s += 1 if row["MACD"] > row["MACD_signal"] else -1

    if row["Close"] < row["BB_lower"]:
        s += 1
    elif row["Close"] > row["BB_upper"]:
        s -= 1

    # OBV（微差ノイズ抑制）
    obv_delta = row["OBV"] - prev_row["OBV"]
    thresh = abs(prev_row["OBV"]) * 0.001 if abs(prev_row["OBV"]) > 0 else 0
    if obv_delta > thresh:
        s += 1
    elif obv_delta < -thresh:
        s -= 1

    return int(s)


def enforce_daily_filter(score: int, daily_trend: str):
    """日足トレンドに逆らうスコアは無効化（0へ）"""
    if daily_trend == "bull" and score < 0:
        return 0, "⚠️ 日足が上向きのため、売りシグナルは無効化しました"
    if daily_trend == "bear" and score > 0:
        return 0, "⚠️ 日足が下向きのため、買いシグナルは無効化しました"
    return score, None


def score_label(score: int) -> tuple[str, str]:
    """(表示ラベル, 色)"""
    if score >= 3:
        return "🟢🟢 強い買い", "#00C851"
    if score >= 1:
        return "🟢 弱い買い", "#00C851"
    if score <= -3:
        return "🔴🔴 強い売り", "#ff4444"
    if score <= -1:
        return "🔴 弱い売り", "#ff4444"
    return "⚪ 中立", "#666"


def daily_trend_label(dt: str) -> str:
    if dt == "bull":
        return "🟢 上向き"
    if dt == "bear":
        return "🔴 下向き"
    return "⚪ 中立"


# =========================
# 初期ウォッチリスト
# =========================
DEFAULT_WATCHLIST = "247A.T,2979.T,4344.T,4502.T,4519.T,5332.T,6178.T,6345.T,6758.T,7177.T,8058.T,8411.T,8424.T,8473.T,8985.T,9104.T,9434.T,9602.T,AAPL,NVDA,TSLA"


# =========================
# サイドバーUI
# =========================
st.sidebar.subheader("📌 一括スクリーニング")

watch_text = st.sidebar.text_area(
    "ウォッチリスト（カンマ or 改行区切り）",
    value=DEFAULT_WATCHLIST,
    height=120
)
watchlist = parse_watchlist(watch_text)

show_company = st.sidebar.toggle("会社名を表示（遅い場合OFF）", value=False)

st.sidebar.subheader("⏱️ 詳細表示")
detail_timeframe = st.sidebar.selectbox(
    "時間足（詳細チャート）",
    ["1時間足", "4時間足（自動）", "日足", "週足", "月足"],
    index=1
)

st.sidebar.subheader("🧯 ATRリスク管理（詳細チャート）")
sl_mult = st.sidebar.slider("損切り ATR倍率", 0.5, 3.0, 1.5, 0.1)
tp_mult = st.sidebar.slider("利確 ATR倍率", 1.0, 6.0, 2.5, 0.1)
show_risk_lines = st.sidebar.toggle("ATRライン（SL/TP）をチャート表示", value=True)

col_a, col_b = st.sidebar.columns(2)
run_screen = col_a.button("一括分析")
auto_run = col_b.toggle("自動更新", value=False)

if auto_run:
    run_screen = True


# =========================
# 一括スクリーニング
# =========================
@st.cache_data(ttl=300, show_spinner=False)
def screen_one(ticker: str) -> dict:
    """
    スクリーニング（固定で4H自動＋日足フィルタ）
    返り値は表用dict
    """
    # 4H
    df4h, mode = fetch_4h_auto(ticker, days=30)
    if df4h is None or df4h.empty:
        return {
            "Ticker": ticker,
            "Company": ticker,
            "4H_Mode": mode,
            "Daily": "N/A",
            "Price": None,
            "4H_Score": None,
            "Signal": "取得失敗",
            "ATR%": None,
            "Note": "4Hデータ取得失敗"
        }

    df4h = add_indicators(df4h)
    if len(df4h) < 4:
        return {
            "Ticker": ticker,
            "Company": ticker,
            "4H_Mode": mode,
            "Daily": "N/A",
            "Price": None,
            "4H_Score": None,
            "Signal": "データ不足",
            "ATR%": None,
            "Note": "4Hデータ不足"
        }

    # 日足トレンド
    df_daily = cached_download(ticker, interval="1d", period="5y")
    dt = "neutral"
    if df_daily is not None and not df_daily.empty and len(df_daily) >= 250:
        dft = add_daily_trend(df_daily)
        if not dft.empty:
            dt = str(dft.iloc[-1]["daily_trend"])

    latest = df4h.iloc[-1]
    prev = df4h.iloc[-2]

    score = calc_score(latest, prev)
    score_f, note = enforce_daily_filter(score, dt)

    sig, _ = score_label(score_f)
    price = float(latest["Close"])
    atr_pct = float(latest["ATR"] / latest["Close"]) * 100 if float(latest["Close"]) != 0 else 0.0

    company = ticker
    if show_company:
        company = fetch_company_name_optional(ticker)

    return {
        "Ticker": ticker,
        "Company": company,
        "4H_Mode": mode,
        "Daily": dt,
        "Price": round(price, 2),
        "4H_Score": int(score_f),
        "Signal": sig,
        "ATR%": round(atr_pct, 2),
        "Note": note or ""
    }


def run_screening(wl: list[str]) -> pd.DataFrame:
    rows = []
    progress = st.progress(0)
    total = max(1, len(wl))
    for i, t in enumerate(wl):
        rows.append(screen_one(t))
        progress.progress(int((i + 1) / total * 100))
    progress.empty()
    df = pd.DataFrame(rows)

    # 見やすい並び（買い強→売り強）
    # scoreがNoneの行は最後
    def sort_key(x):
        try:
            return int(x)
        except:
            return -9999

    df["_score_sort"] = df["4H_Score"].apply(lambda x: sort_key(x) if pd.notna(x) else -9999)
    df["_fail"] = df["4H_Score"].isna().astype(int)
    df = df.sort_values(by=["_fail", "_score_sort", "ATR%"], ascending=[True, False, False]).drop(columns=["_score_sort", "_fail"])
    return df


if "screen_df" not in st.session_state:
    st.session_state.screen_df = None
if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = None

if run_screen:
    if not watchlist:
        st.warning("ウォッチリストが空です。銘柄を入力してください。")
    else:
        with st.spinner("一括分析中...（初回は少し時間がかかることがあります）"):
            st.session_state.screen_df = run_screening(watchlist)
            if st.session_state.selected_ticker is None and not st.session_state.screen_df.empty:
                st.session_state.selected_ticker = st.session_state.screen_df.iloc[0]["Ticker"]


# =========================
# 結果表示（表 + 詳細連動）
# =========================
screen_df = st.session_state.screen_df

if screen_df is None:
    st.info("左の「一括分析」を押すと、ウォッチリストの結果が一覧表示されます。")
    st.stop()

st.subheader("📋 一括スクリーニング結果（4時間足自動 + 日足フィルタ）")
st.dataframe(
    screen_df[["Ticker", "Company", "Signal", "4H_Score", "Daily", "Price", "ATR%", "4H_Mode", "Note"]],
    use_container_width=True,
    hide_index=True
)

# 詳細表示する銘柄を選択（表クリックの代わりに確実なUI）
tickers_in_table = screen_df["Ticker"].tolist()
default_idx = 0
if st.session_state.selected_ticker in tickers_in_table:
    default_idx = tickers_in_table.index(st.session_state.selected_ticker)

sel = st.selectbox("🔎 詳細表示する銘柄", tickers_in_table, index=default_idx)
st.session_state.selected_ticker = sel


# =========================
# 詳細チャート（選択銘柄）
# =========================
timeframe_map = {
    "1時間足": {"type": "yahoo", "interval": "1h", "period": "60d"},
    "4時間足（自動）": {"type": "4h_auto", "days": 30},
    "日足": {"type": "yahoo", "interval": "1d", "period": "1y"},
    "週足": {"type": "yahoo", "interval": "1wk", "period": "5y"},
    "月足": {"type": "yahoo", "interval": "1mo", "period": "10y"},
}

ticker = st.session_state.selected_ticker
params = timeframe_map[detail_timeframe]

with st.spinner("詳細データ取得中..."):
    mode_label = ""
    if params["type"] == "4h_auto":
        df, mode = fetch_4h_auto(ticker, days=params.get("days", 30))
        if mode == "native":
            mode_label = "（4H: 本物）"
        elif mode == "pseudo":
            mode_label = "（4H: 疑似/日足フォールバック）"
        else:
            mode_label = "（4H: 取得失敗）"
    else:
        df = cached_download(ticker, interval=params["interval"], period=params["period"])
        mode_label = ""

    df_daily = cached_download(ticker, interval="1d", period="5y")
    company_name = fetch_company_name_optional(ticker) if show_company else ticker

if df is None or df.empty:
    st.error("詳細データが取得できませんでした。時間足を変えて再試行してください。")
    st.stop()

df = add_indicators(df)
if len(df) < 4:
    st.warning("⚠️ データが少なすぎます。別の時間足をお試しください。")
    st.stop()

# 日足トレンド
daily_trend = "neutral"
if df_daily is not None and not df_daily.empty and len(df_daily) >= 250:
    df_daily_tr = add_daily_trend(df_daily)
    if not df_daily_tr.empty:
        daily_trend = str(df_daily_tr.iloc[-1]["daily_trend"])

# スコア
latest = df.iloc[-1]
prev = df.iloc[-2]
prev2 = df.iloc[-3]
score_today = calc_score(latest, prev)
score_yesterday = calc_score(prev, prev2)

score_today, filter_msg = enforce_daily_filter(score_today, daily_trend)
score_yesterday, _ = enforce_daily_filter(score_yesterday, daily_trend)

# 価格表示
current_price = float(latest["Close"])
prev_price = float(prev["Close"])
price_change = current_price - prev_price
price_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0
price_color = "#00C851" if price_change >= 0 else "#ff4444"
price_arrow = "▲" if price_change >= 0 else "▼"

st.markdown(f"""
<div style='background:#1a1a2e;padding:8px 15px;border-radius:8px;
display:flex;align-items:center;gap:20px;margin-bottom:8px'>
<span style='color:white;font-size:16px;font-weight:bold'>{company_name}</span>
<span style='color:gray;font-size:13px'>{ticker} · {detail_timeframe} {mode_label}</span>
<span style='color:{price_color};font-size:18px;font-weight:bold'>
{current_price:.2f} {price_arrow} {abs(price_change):.2f} ({price_pct:+.2f}%)</span>
<span style='color:white;font-size:13px'>日足：{daily_trend_label(daily_trend)}</span>
</div>
""", unsafe_allow_html=True)

if filter_msg:
    st.warning(filter_msg)

# ATRライン
atr = float(latest["ATR"])
entry = float(latest["Close"])
long_sl = entry - atr * sl_mult
long_tp = entry + atr * tp_mult
short_sl = entry + atr * sl_mult
short_tp = entry - atr * tp_mult

bias = "neutral"
if score_today >= 1:
    bias = "long"
elif score_today <= -1:
    bias = "short"

# トレンド判定（表示用）
df["trend"] = "neutral"
df.loc[(df["EMA21"] > df["EMA50"]) & (df["MACD"] > df["MACD_signal"]), "trend"] = "buy"
df.loc[(df["EMA21"] < df["EMA50"]) & (df["MACD"] < df["MACD_signal"]), "trend"] = "sell"

df_buy = df[df["trend"] == "buy"]
df_sell = df[df["trend"] == "sell"]
df_neutral = df[df["trend"] == "neutral"]

fig = make_subplots(
    rows=4, cols=1, shared_xaxes=True,
    row_heights=[0.55, 0.12, 0.18, 0.15],
    subplot_titles=(
        f"🟡買いトレンド　🔴売りトレンド　⚪中立　({detail_timeframe})",
        "トレンドバー", "RSI", "OBV"
    )
)

if not df_buy.empty:
    fig.add_trace(go.Candlestick(
        x=df_buy.index, open=df_buy["Open"], high=df_buy["High"],
        low=df_buy["Low"], close=df_buy["Close"],
        increasing_line_color="gold", decreasing_line_color="gold",
        increasing_fillcolor="gold", decreasing_fillcolor="gold",
        name="🟡買いトレンド"
    ), row=1, col=1)

if not df_sell.empty:
    fig.add_trace(go.Candlestick(
        x=df_sell.index, open=df_sell["Open"], high=df_sell["High"],
        low=df_sell["Low"], close=df_sell["Close"],
        increasing_line_color="red", decreasing_line_color="red",
        increasing_fillcolor="red", decreasing_fillcolor="red",
        name="🔴売りトレンド"
    ), row=1, col=1)

if not df_neutral.empty:
    fig.add_trace(go.Candlestick(
        x=df_neutral.index, open=df_neutral["Open"], high=df_neutral["High"],
        low=df_neutral["Low"], close=df_neutral["Close"],
        increasing_line_color="gray", decreasing_line_color="gray",
        increasing_fillcolor="gray", decreasing_fillcolor="gray",
        name="⚪中立"
    ), row=1, col=1)

fig.add_trace(go.Scatter(x=df.index, y=df["EMA21"], line=dict(color="orange", width=1.5), name="EMA21"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], line=dict(color="cyan", width=1.5), name="EMA50"), row=1, col=1)

# ATRライン（SL/TP）
if show_risk_lines:
    if bias == "long":
        fig.add_hline(y=long_sl, line_dash="dot", row=1, col=1)
        fig.add_hline(y=long_tp, line_dash="dot", row=1, col=1)
    elif bias == "short":
        fig.add_hline(y=short_sl, line_dash="dot", row=1, col=1)
        fig.add_hline(y=short_tp, line_dash="dot", row=1, col=1)

trend_colors = ["gold" if t == "buy" else "red" if t == "sell" else "gray" for t in df["trend"]]
fig.add_trace(go.Bar(x=df.index, y=[1]*len(df), marker_color=trend_colors, showlegend=False), row=2, col=1)

fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], line=dict(color="magenta", width=1.5), name="RSI"), row=3, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

fig.add_trace(go.Scatter(x=df.index, y=df["OBV"], line=dict(color="teal", width=1.5), name="OBV"), row=4, col=1)

fig.update_layout(height=750, xaxis_rangeslider_visible=False, template="plotly_dark")
fig.update_yaxes(showticklabels=False, row=2, col=1)
st.plotly_chart(fig, use_container_width=True)

# ===== シグナルまとめ =====
st.subheader("🧭 シグナルまとめ")
col1, col2 = st.columns(2)

with col1:
    label, bg = score_label(score_today)
    st.markdown(f"""
    <div style='background:{bg};padding:10px;border-radius:8px;
    text-align:center;margin-bottom:6px'>
    <b style='color:white;font-size:18px'>{label}</b><br>
    <span style='color:white;font-size:12px'>スコア：{score_today}（前回：{score_yesterday}）</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<b style='color:white'>🎯 ATR目安（エントリー＝最新終値）</b>", unsafe_allow_html=True)
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

    atr_pct = float(latest["ATR"] / latest["Close"]) * 100 if float(latest["Close"]) != 0 else 0.0
    signals.append((f"⚠️ ATR {atr_pct:.1f}%（{'高ボラ' if atr_pct > 3 else '普通'}）", "neutral"))

    if daily_trend == "bull":
        signals.append(("🟢 日足トレンド上向き", "buy"))
    elif daily_trend == "bear":
        signals.append(("🔴 日足トレンド下向き", "sell"))
    else:
        signals.append(("⚪ 日足トレンド中立", "neutral"))

    for text, kind in signals:
        c = "#00C851" if kind == "buy" else "#ff4444" if kind == "sell" else "#444"
        st.markdown(
            f"<div style='background:{c};padding:5px 10px;border-radius:5px;color:white;margin:2px 0;font-size:12px'>{text}</div>",
            unsafe_allow_html=True
        )
