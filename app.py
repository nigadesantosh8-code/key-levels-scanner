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
    return pd.DataFrame(data)[["token", "symbol", "name", "exch_seg"]]


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


@st.cache_data(ttl=55, show_spinner=False)
def scan(targets, tfs):
    rows = []
    for name, ex, tok in targets:
        for tf in tfs:
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
                    z, lv = "🟢 BUY", lo
                elif hi[0] - band <= ltp <= hi[0] + band:
                    z, lv = "🔴 SELL", hi
                else:
                    continue
                rows.append({"Symbol": name, "TF": tf, "Zone": z,
                             "LTP": round(ltp, 2), "Level": round(lv[0], 2),
                             "Strength%": round(lv[1]),
                             "Dist%": round((ltp - lv[0]) / lv[0] * 100, 2),
                             "ex": ex, "tok": tok})
            except Exception:
                continue
    return rows


st.title("📊 Key Levels Scanner")

try:
    login()
except Exception as e:
    st.error(f"Angel One login fail: {e}")
    st.stop()

with st.expander("⚙️ Settings", expanded=False):
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
    auto = st.checkbox("Auto-refresh (60 sec)", value=True)

if auto:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=60000, key="ar")

if st.button("🔄 Abhi scan karo", use_container_width=True, type="primary"):
    scan.clear()

targets = build_targets(stocks, custom)
tgt = tuple((n, ex, tok) for n, (ex, tok) in targets.items())

with st.spinner("Scan ho raha hai..."):
    rows = scan(tgt, tuple(tfs))

now_ist = dt.datetime.utcnow() + dt.timedelta(hours=5, minutes=30)
st.caption(f"Last update: {now_ist.strftime('%H:%M:%S')}")

res = pd.DataFrame(rows)

# Timeframe buttons
opts = ["All"] + list(tfs)


def lab(x):
    if res.empty:
        return x
    n = len(res) if x == "All" else int((res.TF == x).sum())
    return f"{x} ({n})"


tf_pick = st.radio("Timeframe chuno", opts, format_func=lab, horizontal=True)
zone_pick = st.radio("Zone", ["Sab", "🟢 BUY", "🔴 SELL"], horizontal=True)

view = res.copy()
if not view.empty:
    if tf_pick != "All":
        view = view[view.TF == tf_pick]
    if zone_pick != "Sab":
        view = view[view.Zone == zone_pick]
    view = view.sort_values(["Zone", "Symbol"]).reset_index(drop=True)

if view.empty:
    st.info("Is selection mein koi stock yellow/red zone mein nahi hai.")
else:
    c1, c2 = st.columns(2)
    c1.metric("🟢 BUY zone", int((view.Zone == "🟢 BUY").sum()))
    c2.metric("🔴 SELL zone", int((view.Zone == "🔴 SELL").sum()))
    st.caption("👇 Chart kholne ke liye row par tap karo")
    cols = ["Symbol", "TF", "Zone", "LTP", "Level", "Strength%", "Dist%"]
    ev = st.dataframe(view[cols], use_container_width=True, hide_index=True,
                      on_select="rerun", selection_mode="single-row", key="tbl")
    if ev.selection.rows:
        r = view.iloc[ev.selection.rows[0]]
        new = (r.Symbol, r.ex, r.tok, r.TF)
        if st.session_state.get("sel") != new:
            st.session_state["sel"] = new
            st.session_state["chart_tf"] = r.TF

st.divider()
st.subheader("📈 Chart")

sel = st.session_state.get("sel")
if not sel and targets:
    n0 = list(targets)[0]
    sel = (n0, targets[n0][0], targets[n0][1], tfs[0] if tfs else "15m")
    st.session_state.setdefault("chart_tf", sel[3])

if sel:
    sym, ex, tok, _ = sel
    ctf = st.radio("Chart timeframe", list(TF), horizontal=True, key="chart_tf")
    st.markdown(f"**{sym}** · {ctf}")
    df = get_candles(ex, tok, ctf)
    if df is not None and len(df) > 30:
        out = key_levels(df)
        v = df.tail(150)
        fig = go.Figure(go.Candlestick(
            x=v.time, open=v.open, high=v.high, low=v.low, close=v.close))
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
    else:
        st.warning("Is timeframe ka data nahi mila.")

st.caption("Sirf analysis tool hai. Trade se pehle khud confirm karein.")
