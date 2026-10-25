import time
import datetime as dt
import requests
import pandas as pd
import numpy as np
import pyotp
import streamlit as st
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from SmartApi import SmartConnect

st.set_page_config(page_title="Key Levels Scanner", layout="wide")

TF = {
    "1m": ("ONE_MINUTE", 20), "3m": ("THREE_MINUTE", 45),
    "5m": ("FIVE_MINUTE", 80), "15m": ("FIFTEEN_MINUTE", 150),
    "30m": ("THIRTY_MINUTE", 150), "1h": ("ONE_HOUR", 300),
    "1D": ("ONE_DAY", 900),
}
INDICES = {
    "NIFTY": ("NSE", "99926000"), "BANKNIFTY": ("NSE", "99926009"),
    "FINNIFTY": ("NSE", "99926037"), "SENSEX": ("BSE", "99919000"),
}
DEFAULT_STOCKS = ["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY",
                  "SBIN", "ITC", "LT", "AXISBANK", "KOTAKBANK"]
SCRIP_URL = ("https://margincalculator.angelbroking.com/"
             "OpenAPI_File/files/OpenAPIScripMaster.json")


@st.cache_resource(ttl=6 * 3600)
def login():
    s = st.secrets
    api = SmartConnect(api_key=s["API_KEY"])
    otp = pyotp.TOTP(s["TOTP_SECRET"]).now()
    r = api.generateSession(s["CLIENT_ID"], s["PIN"], otp)
    if not r.get("status"):
        raise RuntimeError(r.get("message", "Login fail"))
    return api


@st.cache_data(ttl=12 * 3600, show_spinner="Symbol list load ho rahi hai...")
def scrip_master():
    data = requests.get(SCRIP_URL, timeout=90).json()
    df = pd.DataFrame(data)[["token", "symbol", "name", "exch_seg"]]
    return df


@st.cache_data(ttl=45, show_spinner=False)
def get_candles(exch, token, tf):
    iv, days = TF[tf]
    now = dt.datetime.utcnow() + dt.timedelta(hours=5, minutes=30)
    frm = now - dt.timedelta(days=days)
    time.sleep(0.4)
    res = login().getCandleData({
        "exchange": exch, "symboltoken": token, "interval": iv,
        "fromdate": frm.strftime("%Y-%m-%d %H:%M"),
        "todate": now.strftime("%Y-%m-%d %H:%M"),
    })
    d = res.get("data") if isinstance(res, dict) else None
    if not d:
        return None
    df = pd.DataFrame(d, columns=["time", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    return df


def key_levels(df, plen=6, mem=10, nlev=4, atr_mult=0.25):
    h, l = df["high"].values, df["low"].values
    piv = []
    for i in range(plen, len(df) - plen):
        if h[i] == h[i - plen:i + plen + 1].max():
            piv.append((i, h[i]))
        if l[i] == l[i - plen:i + plen + 1].min():
            piv.append((i, l[i]))
    piv.sort()
    prices = np.array([p for _, p in piv[-mem:]])
    if len(prices) < 2:
        return None
    k = min(nlev, len(np.unique(np.round(prices, 2))))
    if k < 2:
        return None
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(prices.reshape(-1, 1))
    levels = []
    for c in range(k):
        pts = prices[km.labels_ == c]
        levels.append((float(pts.mean()), len(pts) / len(prices) * 100))
    levels.sort()
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1])
    return levels, atr * atr_mult


def build_targets(stocks, custom):
    t = {}
    for n, v in INDICES.items():
        if n in st.session_state.get("idx_pick", []):
            t[n] = v
    if stocks:
        m = scrip_master()
        eq = m[(m.exch_seg == "NSE") & (m.symbol.str.endswith("-EQ"))]
        for s in stocks:
            r = eq[eq.name == s]
            if len(r):
                t[s] = ("NSE", r.iloc[0].token)
    for sym, (ex, tok) in custom.items():
        t[sym] = (ex, tok)
    return t


st.title("📊 Key Levels Scanner")

try:
    login()
except Exception as e:
    st.error(f"Angel One login fail: {e}")
    st.stop()

with st.expander("⚙️ Settings", expanded=True):
    st.multiselect("Index", list(INDICES), default=["NIFTY", "BANKNIFTY"], key="idx_pick")
    m = scrip_master()
    names = sorted(m[(m.exch_seg == "NSE") & (m.symbol.str.endswith("-EQ"))].name.unique())
    stocks = st.multiselect("Stocks", names, default=DEFAULT_STOCKS)
    q = st.text_input("Futures/Options search (jaise NIFTY28OCT26 ya 22600CE)")
    custom = {}
    if q:
        f = m[m.exch_seg.isin(["NFO", "BFO"]) & m.symbol.str.contains(q.upper())].head(40)
        pick = st.multiselect("F&O chuno", f.symbol.tolist())
        for s in pick:
            r = f[f.symbol == s].iloc[0]
            custom[s] = (r.exch_seg, r.token)
    tfs = st.multiselect("Timeframe", list(TF), default=["5m", "15m", "1h", "1D"])
    auto = st.checkbox("Auto-refresh (60 sec)")

if auto:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=60000, key="ar")

go_scan = st.button("🔍 Scan karo", use_container_width=True, type="primary")

if go_scan or (auto and "rows" in st.session_state):
    targets = build_targets(stocks, custom)
    rows, bar = [], st.progress(0.0)
    total = max(len(targets) * len(tfs), 1)
    done = 0
    for name, (ex, tok) in targets.items():
        for tf in tfs:
            done += 1
            bar.progress(done / total, text=f"{name} {tf}")
            try:
                df = get_candles(ex, tok, tf)
                if df is None or len(df) < 30:
                    continue
                out = key_levels(df)
                if not out:
                    continue
                levels, band = out
                ltp = float(df["close"].iloc[-1])
                lo, hi = levels[0], levels[-1]
                if lo[0] - band <= ltp <= lo[0] + band:
                    z, lv = "BUY", lo
                elif hi[0] - band <= ltp <= hi[0] + band:
                    z, lv = "SELL", hi
                else:
                    continue
                rows.append({"Symbol": name, "TF": tf, "Zone": z,
                             "LTP": round(ltp, 2), "Level": round(lv[0], 2),
                             "Strength%": round(lv[1]),
                             "Dist%": round((ltp - lv[0]) / lv[0] * 100, 2),
                             "ex": ex, "tok": tok})
            except Exception:
                continue
    bar.empty()
    st.session_state["rows"] = rows

rows = st.session_state.get("rows")
if rows is not None:
    res = pd.DataFrame(rows)
    if res.empty:
        st.info("Abhi koi stock yellow/red zone mein nahi hai.")
    else:
        show = ["Symbol", "TF", "LTP", "Level", "Strength%", "Dist%"]
        st.success(f"🟢 BUY ZONE (yellow): {(res.Zone == 'BUY').sum()}")
        st.dataframe(res[res.Zone == "BUY"][show], use_container_width=True, hide_index=True)
        st.error(f"🔴 SELL ZONE (red): {(res.Zone == 'SELL').sum()}")
        st.dataframe(res[res.Zone == "SELL"][show], use_container_width=True, hide_index=True)

st.divider()
st.subheader("📈 Chart")
all_t = build_targets(stocks, custom)
if all_t:
    c1, c2 = st.columns(2)
    sym = c1.selectbox("Symbol", list(all_t))
    tf = c2.selectbox("Timeframe ", list(TF), index=2)
    ex, tok = all_t[sym]
    df = get_candles(ex, tok, tf)
    if df is not None and len(df) > 30:
        out = key_levels(df)
        view = df.tail(150)
        fig = go.Figure(go.Candlestick(
            x=view.time, open=view.open, high=view.high,
            low=view.low, close=view.close))
        if out:
            levels, band = out
            colors = ["#FFD700", "#1E90FF", "#FFA500", "#FF3B3B"]
            for i, (p, s) in enumerate(levels):
                c = colors[0] if i == 0 else colors[3] if i == len(levels) - 1 else colors[i]
                fig.add_hrect(y0=p - band, y1=p + band, fillcolor=c, opacity=0.2, line_width=0)
                fig.add_hline(y=p, line_color=c, line_width=2,
                              annotation_text=f"{s:.0f}%")
        fig.update_layout(template="plotly_dark", height=520,
                          xaxis_rangeslider_visible=False,
                          margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

st.caption("Sirf analysis tool hai. Trade se pehle khud confirm karein.")
