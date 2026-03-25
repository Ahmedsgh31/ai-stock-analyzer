"""
AI-Powered Stock Market Analyzer  v4
──────────────────────────────────────
Price data   : Twelve Data (free)  →  yfinance fallback for Saudi
Fundamentals : Yahoo Finance quoteSummary API  (direct HTTPS, NOT yfinance .info)
               This bypasses the rate-limit that blocks yfinance on Streamlit Cloud.
Saudi stocks : yfinance .history() for price  (works even when .info is blocked)
"""

import random, json
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Stock Analyzer", page_icon="📈", layout="wide")
st.title("📈 AI-Powered Stock Market Analyzer")
st.markdown("---")

# ─────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────
def _sf(x):
    """Safe float – returns None on any failure / NaN / Inf."""
    try:
        if x is None: return None
        if isinstance(x, (int, float, np.number)):
            f = float(x)
            return None if (np.isnan(f) or np.isinf(f)) else f
        s = str(x).replace(",", "").strip()
        if s.lower() in ("", "none", "n/a", "null", "nan", "inf", "-inf"): return None
        return float(s)
    except Exception:
        return None

def _money(x):
    v = _sf(x)
    if v is None: return "N/A"
    a = abs(v)
    if a >= 1e12: return f"${v/1e12:.2f}T"
    if a >= 1e9:  return f"${v/1e9:.2f}B"
    if a >= 1e6:  return f"${v/1e6:.2f}M"
    if a >= 1e3:  return f"${v/1e3:.2f}K"
    return f"${v:.2f}"

def _pct(x):
    v = _sf(x)
    if v is None: return "N/A"
    # ratios like 0.0149 → 1.49%,  values like 1.49 already a %
    if abs(v) < 2.0: v *= 100
    return f"{v:.2f}%"

def _n(x, pre="", suf="", dec=2):
    v = _sf(x)
    if v is None: return "N/A"
    return f"{pre}{v:,.{dec}f}{suf}"

def _g(d: dict, *keys):
    for k in keys:
        v = _sf(d.get(k))
        if v is not None: return v
    return None

def _gs(d: dict, *keys):
    for k in keys:
        v = d.get(k)
        if v and str(v).strip().lower() not in ("", "none", "n/a", "null", "nan"):
            return str(v)
    return None

# ─────────────────────────────────────────────────────────
# Shared HTTP session  (rotated User-Agent to avoid blocks)
# ─────────────────────────────────────────────────────────
_UA_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
]

@st.cache_resource
def _sess() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": random.choice(_UA_POOL),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Origin": "https://finance.yahoo.com",
        "Referer": "https://finance.yahoo.com/",
    })
    return s

# ─────────────────────────────────────────────────────────
# Twelve Data  (price + quote — free plan ✅)
# ─────────────────────────────────────────────────────────
_TD_BASE = "https://api.twelvedata.com"

def _td_key():
    return st.secrets.get("TWELVEDATA_API_KEY") or None

def _td(endpoint: str, params: dict, timeout=25) -> dict:
    k = _td_key()
    if not k:
        return {"status": "error", "message": "No TWELVEDATA_API_KEY"}
    try:
        r = _sess().get(f"{_TD_BASE}/{endpoint.lstrip('/')}",
                        params={**params, "apikey": k}, timeout=timeout)
        return r.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}

@st.cache_data(ttl=3600)
def td_resolve(sym: str) -> dict | None:
    s = sym.strip().upper()
    is_sa = s.endswith(".SR") or (s.isdigit() and len(s) in (3, 4, 5))
    base  = s.replace(".SR", "") if s.endswith(".SR") else s

    items = (_td("symbol_search", {"symbol": base, "outputsize": 50}).get("data") or [])
    if not items:
        items = (_td("symbol_search", {"keywords": base, "outputsize": 50}).get("data") or [])
    if not items:
        return None

    def _score(it):
        sc   = 10 if (it.get("instrument_type") or "").lower() == "common stock" else 0
        sym2 = (it.get("symbol") or "").upper()
        ex   = (it.get("exchange") or "").lower()
        ctry = (it.get("country") or "").lower()
        cur  = (it.get("currency") or "").upper()
        if is_sa:
            if "saudi" in ctry or "tadawul" in ex: sc += 70
            if cur == "SAR":    sc += 30
            if sym2 == base:    sc += 40
        else:
            if sym2 == s:  sc += 60
            if ex in ("nasdaq", "nyse", "nyse american", "nyse arca"): sc += 20
        return sc

    return sorted(items, key=_score, reverse=True)[0]

@st.cache_data(ttl=600)
def td_ohlcv(symbol, exchange, interval, outputsize) -> tuple[pd.DataFrame, str]:
    p = {"symbol": symbol, "interval": interval,
         "outputsize": outputsize, "format": "JSON"}
    if exchange: p["exchange"] = exchange
    d = _td("time_series", p)
    if d.get("status") == "error":
        return pd.DataFrame(), d.get("message", "")
    vals = d.get("values") or []
    if not vals:
        return pd.DataFrame(), "empty response"
    df = pd.DataFrame(vals)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").set_index("datetime")
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.rename(columns=str.title)
    for col in ["Open", "High", "Low", "Close"]:
        if col not in df.columns: return pd.DataFrame(), f"Missing column {col}"
    if "Volume" not in df.columns: df["Volume"] = 0
    return df.dropna(subset=["Close"]), ""

@st.cache_data(ttl=60)
def td_quote(symbol, exchange) -> dict:
    p = {"symbol": symbol, "format": "JSON"}
    if exchange: p["exchange"] = exchange
    d = _td("quote", p)
    return {} if d.get("status") == "error" else d

# ─────────────────────────────────────────────────────────
# Yahoo Finance  — direct quoteSummary API
# Much more reliable than yfinance .info on Streamlit Cloud
# ─────────────────────────────────────────────────────────
_YF_MODULES = ",".join([
    "summaryProfile", "summaryDetail", "financialData",
    "defaultKeyStatistics", "incomeStatementHistory",
    "balanceSheetHistory", "cashflowStatementHistory",
    "recommendationTrend", "earningsTrend",
    "price",
])

@st.cache_data(ttl=900)
def yahoo_fundamentals(ticker: str) -> dict:
    """
    Call Yahoo Finance quoteSummary directly (no yfinance wrapper).
    Returns a flat dict of fundamental values, or empty dict on failure.
    """
    sym = ticker.strip().upper()
    result: dict = {}

    # ── Step 1: get a crumb + cookie (required by Yahoo since 2023) ──
    crumb = None
    try:
        cr = _sess().get(
            "https://query2.finance.yahoo.com/v1/test/getcrumb",
            timeout=10
        )
        if cr.status_code == 200 and cr.text.strip():
            crumb = cr.text.strip()
    except Exception:
        pass

    # ── Step 2: quoteSummary call ──
    url = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{sym}"
    params = {"modules": _YF_MODULES, "formatted": "false", "lang": "en-US"}
    if crumb:
        params["crumb"] = crumb

    try:
        resp = _sess().get(url, params=params, timeout=20)
        if resp.status_code != 200:
            # try query2 domain
            url2 = url.replace("query1.", "query2.")
            resp = _sess().get(url2, params=params, timeout=20)
        data = resp.json()
    except Exception:
        return result

    try:
        summary = data["quoteSummary"]["result"][0]
    except (KeyError, TypeError, IndexError):
        return result

    # ── Helper to safely pull a raw value from a module ──
    def _mv(module, *keys, default=None):
        m = summary.get(module) or {}
        for k in keys:
            v = m.get(k)
            if v is None: continue
            # Yahoo wraps values as {"raw": 1.23, "fmt": "1.23"}
            if isinstance(v, dict): v = v.get("raw", v.get("fmt"))
            fv = _sf(v)
            if fv is not None: return fv
        return default

    def _ms(module, *keys, default=None):
        m = summary.get(module) or {}
        for k in keys:
            v = m.get(k)
            if v is None: continue
            if isinstance(v, dict): v = v.get("fmt") or v.get("raw")
            sv = str(v).strip() if v is not None else ""
            if sv.lower() not in ("", "none", "n/a", "null", "nan"):
                return sv
        return default

    p   = summary.get("price") or {}
    sd  = summary.get("summaryDetail") or {}
    fd  = summary.get("financialData") or {}
    ks  = summary.get("defaultKeyStatistics") or {}
    sp  = summary.get("summaryProfile") or {}

    def _raw(d, k):
        v = d.get(k)
        if isinstance(v, dict): v = v.get("raw", v.get("fmt"))
        return _sf(v)
    def _raws(d, k):
        v = d.get(k)
        if isinstance(v, dict): v = v.get("fmt") or v.get("raw")
        sv = str(v).strip() if v is not None else ""
        return sv if sv.lower() not in ("","none","n/a","null","nan") else None

    # ── Identity ──
    result["longName"]    = _raws(p, "longName") or _raws(p, "shortName")
    result["shortName"]   = _raws(p, "shortName")
    result["exchange"]    = _raws(p, "exchangeName") or _raws(p, "exchange")
    result["currency"]    = _raws(p, "currency") or _raws(sd, "currency")
    result["sector"]      = _raws(sp, "sector")
    result["industry"]    = _raws(sp, "industry")

    # ── Live price ──
    result["currentPrice"]     = _raw(p, "regularMarketPrice")
    result["previousClose"]    = _raw(p, "regularMarketPreviousClose") or _raw(sd, "previousClose")
    result["open"]             = _raw(p, "regularMarketOpen")
    result["dayHigh"]          = _raw(p, "regularMarketDayHigh")
    result["dayLow"]           = _raw(p, "regularMarketDayLow")
    result["volume"]           = _raw(p, "regularMarketVolume")
    result["averageVolume"]    = _raw(sd, "averageVolume") or _raw(sd, "averageDailyVolume10Day")
    result["marketCap"]        = _raw(p, "marketCap")

    # ── Valuation ──
    result["trailingPE"]       = _raw(sd, "trailingPE")
    result["forwardPE"]        = _raw(sd, "forwardPE") or _raw(ks, "forwardPE")
    result["pegRatio"]         = _raw(ks, "pegRatio")
    result["priceToBook"]      = _raw(ks, "priceToBook")
    result["priceToSalesTrailing12Months"] = _raw(sd, "priceToSalesTrailing12Months")
    result["enterpriseValue"]  = _raw(ks, "enterpriseValue")
    result["enterpriseToEbitda"] = _raw(ks, "enterpriseToEbitda")
    result["beta"]             = _raw(sd, "beta") or _raw(ks, "beta")

    # ── Dividends ──
    result["dividendYield"]    = _raw(sd, "dividendYield") or _raw(sd, "trailingAnnualDividendYield")
    result["dividendRate"]     = _raw(sd, "dividendRate") or _raw(sd, "trailingAnnualDividendRate")
    result["payoutRatio"]      = _raw(sd, "payoutRatio")
    ex = _raws(sd, "exDividendDate")
    if ex:
        try:
            result["exDividendDate"] = pd.to_datetime(float(ex), unit="s").strftime("%Y-%m-%d")
        except Exception:
            result["exDividendDate"] = ex

    # ── Financials ──
    result["totalRevenue"]     = _raw(fd, "totalRevenue")
    result["revenueGrowth"]    = _raw(fd, "revenueGrowth")
    result["grossMargins"]     = _raw(fd, "grossMargins")
    result["profitMargins"]    = _raw(fd, "profitMargins") or _raw(ks, "profitMargins")
    result["operatingMargins"] = _raw(fd, "operatingMargins")
    result["ebitda"]           = _raw(fd, "ebitda")
    result["trailingEps"]      = _raw(ks, "trailingEps")
    result["forwardEps"]       = _raw(ks, "forwardEps")
    result["returnOnEquity"]   = _raw(fd, "returnOnEquity")
    result["returnOnAssets"]   = _raw(fd, "returnOnAssets")

    # ── Balance sheet ──
    result["totalCash"]        = _raw(fd, "totalCash")
    result["totalDebt"]        = _raw(fd, "totalDebt")
    result["debtToEquity"]     = _raw(fd, "debtToEquity")
    result["currentRatio"]     = _raw(fd, "currentRatio")
    result["quickRatio"]       = _raw(fd, "quickRatio")
    result["bookValue"]        = _raw(ks, "bookValue")
    result["totalAssets"]      = None   # not in standard modules — will pull below
    result["freeCashflow"]     = _raw(fd, "freeCashflow")

    # ── Balance sheet history for totalAssets ──
    try:
        bs_stmts = (summary.get("balanceSheetHistory") or {}).get("balanceSheetStatements") or []
        if bs_stmts:
            latest = bs_stmts[0]
            for k, rk in [("totalAssets","totalAssets"),
                           ("totalLiab","totalLiab"),
                           ("totalStockholderEquity","totalStockholderEquity")]:
                v = latest.get(rk)
                if isinstance(v, dict): v = v.get("raw")
                fv = _sf(v)
                if fv: result[k] = fv
    except Exception:
        pass

    # ── 52-week / shares ──
    result["fiftyTwoWeekHigh"]   = _raw(sd, "fiftyTwoWeekHigh")
    result["fiftyTwoWeekLow"]    = _raw(sd, "fiftyTwoWeekLow")
    result["sharesOutstanding"]  = _raw(ks, "sharesOutstanding") or _raw(p, "sharesOutstanding")
    result["floatShares"]        = _raw(ks, "floatShares")

    # ── Analyst ──
    result["targetMeanPrice"]    = _raw(fd, "targetMeanPrice")
    result["targetMedianPrice"]  = _raw(fd, "targetMedianPrice")
    result["recommendationKey"]  = _raws(fd, "recommendationKey")

    # Strip None values so callers can use `.get()` cleanly
    return {k: v for k, v in result.items() if v is not None}


# ─────────────────────────────────────────────────────────
# yfinance  — price history only  (reliable even when .info blocked)
# ─────────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def yf_history(yf_sym: str, period: str) -> pd.DataFrame:
    try:
        import yfinance as yf
        t  = yf.Ticker(yf_sym)
        df = t.history(period=period, interval="1d", auto_adjust=False)
        if df is None or df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        for c in ["Open","High","Low","Close"]:
            if c not in df.columns: return pd.DataFrame()
        if "Volume" not in df.columns: df["Volume"] = 0
        return df.dropna(subset=["Close"])
    except Exception:
        return pd.DataFrame()

# ─────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Search Stock")
    stock_symbol = st.text_input(
        "Enter Stock Symbol",
        placeholder="e.g. AAPL, TSLA, 2222.SR",
        help="US: AAPL TSLA NVDA  |  Saudi: 2222.SR 1120.SR",
    )
    _PERIODS_TD = {
        "1 Month":  ("1day",  35),  "3 Months": ("1day",  95),
        "6 Months": ("1day", 185),  "1 Year":   ("1day", 262),
        "2 Years":  ("1day", 524),  "5 Years":  ("1day",1310),
    }
    _PERIODS_YF = {
        "1 Month":"1mo","3 Months":"3mo","6 Months":"6mo",
        "1 Year":"1y","2 Years":"2y","5 Years":"5y",
    }
    sel_period = st.selectbox("Select Time Period", list(_PERIODS_TD.keys()), index=3)
    show_debug = st.checkbox("Show debug panels", value=False)
    go_btn     = st.button("🔍 Analyze Stock", type="primary", use_container_width=True)

if "result" not in st.session_state:
    st.session_state.result = None

# ─────────────────────────────────────────────────────────
# Analysis  (runs only on button press)
# ─────────────────────────────────────────────────────────
if go_btn:
    raw = (stock_symbol or "").strip().upper()
    if not raw:
        st.warning("⚠️ Enter a stock symbol first."); st.stop()

    is_saudi = raw.endswith(".SR") or (raw.isdigit() and len(raw) in (3, 4, 5))
    yf_sym   = (raw.replace(".SR","") + ".SR") if is_saudi else raw

    hist = pd.DataFrame()
    td_sym = td_ex = display_name = currency = None
    provider = "unknown"

    # ── 1. Resolve via Twelve Data symbol search ──
    if _td_key():
        with st.spinner("Resolving symbol…"):
            resolved = td_resolve(raw)
        if resolved:
            td_sym       = resolved.get("symbol") or raw
            td_ex        = resolved.get("exchange") or None
            display_name = resolved.get("instrument_name") or td_sym
            currency     = resolved.get("currency") or "N/A"

    # ── 2. Price: Twelve Data (preferred, free for US/global) ──
    if td_sym and _td_key():
        interval, outputsize = _PERIODS_TD[sel_period]
        with st.spinner(f"Loading price data (Twelve Data)…"):
            hist, td_err = td_ohlcv(td_sym, td_ex, interval, outputsize)
        if not hist.empty:
            provider = "twelvedata"

    # ── 3. Price fallback: yfinance  (Saudi / when TD fails) ──
    if hist.empty:
        with st.spinner(f"Loading price data (Yahoo Finance)…"):
            hist = yf_history(yf_sym, _PERIODS_YF[sel_period])
        if not hist.empty:
            provider = "yfinance"
            if not display_name: display_name = yf_sym
            if not currency:     currency = "SAR" if is_saudi else "USD"
            if not td_sym:       td_sym = raw

    if hist.empty:
        st.error(f"No price data found for **{raw}**.")
        if is_saudi:
            st.warning(
                "Saudi (Tadawul) price data via Twelve Data requires the **Pro plan**.\n"
                "Yahoo Finance fallback is also sometimes blocked on Streamlit Cloud IPs.\n\n"
                "💡 Try again in 1–2 minutes — Yahoo rate-limits are usually temporary."
            )
        else:
            st.info("Check the symbol and that TWELVEDATA_API_KEY is set in Secrets.")
        st.stop()

    # ── 4. Enrich display name from Yahoo if still missing ──
    if not display_name or display_name == raw:
        with st.spinner("Fetching company info…"):
            _tmp = yahoo_fundamentals(yf_sym)
        n = _gs(_tmp, "longName", "shortName")
        if n: display_name = n
        c = _gs(_tmp, "currency")
        if c and (not currency or currency == "N/A"): currency = c

    st.session_state.result = {
        "raw": raw,  "td_sym": td_sym,  "td_ex": td_ex,
        "yf_sym": yf_sym, "name": display_name or raw,
        "currency": currency or "N/A", "hist": hist,
        "provider": provider,
    }

# ─────────────────────────────────────────────────────────
# Render  (only when results exist)
# ─────────────────────────────────────────────────────────
res = st.session_state.result
if not res:
    st.info("👈 Enter a stock symbol in the sidebar and click **Analyze Stock**.")
    st.markdown("""
| Market  | Examples |
|---------|---------|
| NASDAQ  | `AAPL` `TSLA` `NVDA` `MSFT` `AMZN` |
| NYSE    | `JPM` `XOM` `KO` `BRK-B` |
| Saudi / Tadawul | `2222.SR` `1120.SR` `2010.SR` |

> Requires `TWELVEDATA_API_KEY` in Streamlit Secrets.
""")
    st.stop()

sym      = res["td_sym"] or res["raw"]
yf_sym   = res["yf_sym"]
td_ex    = res["td_ex"]
name     = res["name"]
currency = res["currency"]
hist     = res["hist"]
provider = res["provider"]

# ── Header ──
c1, c2, c3, c4 = st.columns([2,1,1,1])
with c1:
    st.subheader(name)
    st.caption(f"**{sym}** | Exchange: {td_ex or 'Auto'} | Currency: {currency} | Source: {provider}")

cs = hist["Close"].dropna()
with c2:
    if len(cs) >= 2:
        cp, pp = float(cs.iloc[-1]), float(cs.iloc[-2])
        d = cp - pp;  dp = d/pp*100 if pp else 0
        st.metric("Current Price", f"{cp:.2f}", f"{d:+.2f} ({dp:+.2f}%)")
    else:
        st.metric("Current Price", "N/A")
with c3: st.metric("Period High", f"{float(hist['High'].max()):.2f}")
with c4: st.metric("Period Low",  f"{float(hist['Low'].min()):.2f}")

st.markdown("---")
tab1, tab2, tab3 = st.tabs(["📊 Price Analysis", "💼 Financial Metrics", "🔮 AI Forecast"])

# ═══════════════════════════════════════
# TAB 1 — Price chart
# ═══════════════════════════════════════
with tab1:
    st.subheader("Historical Price & Volume")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=hist.index, open=hist["Open"], high=hist["High"],
        low=hist["Low"], close=hist["Close"], name="Price"))
    fig.add_trace(go.Bar(
        x=hist.index, y=hist["Volume"], name="Volume",
        yaxis="y2", opacity=0.25, marker_color="rgba(100,180,255,0.5)"))
    fig.update_layout(
        title=f"{sym} — {sel_period}",
        yaxis_title=f"Price ({currency})",
        yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False),
        xaxis_title="Date", height=600,
        hovermode="x unified", template="plotly_dark",
        xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════
# TAB 2 — Financial Metrics
# ═══════════════════════════════════════
with tab2:
    st.subheader("Financial Metrics")

    with st.spinner("Loading fundamentals from Yahoo Finance…"):
        fund = yahoo_fundamentals(yf_sym)

    # Twelve Data live quote (free, US only)
    lq = {}
    if _td_key() and provider == "twelvedata":
        lq = td_quote(res["td_sym"], td_ex)

    if show_debug:
        with st.expander("🔧 Yahoo quoteSummary (parsed)"):  st.json(fund)
        with st.expander("🔧 Twelve Data /quote"):           st.json(lq)

    # ── Live Quote ──
    st.markdown("### 📌 Live Quote")
    q1, q2, q3, q4 = st.columns(4)

    lp = _g(lq,"close","price") or _g(fund,"currentPrice")
    if lp is None and len(cs): lp = float(cs.iloc[-1])

    chg   = _sf(lq.get("change"))
    chgp  = _sf(lq.get("percent_change"))
    if chg is None:
        prev = _g(fund, "previousClose")
        if prev and lp:
            chg  = lp - prev
            chgp = chg/prev*100 if prev else None

    vol    = _g(lq,"volume")    or _g(fund,"volume")
    avgvol = _g(lq,"average_volume") or _g(fund,"averageVolume")
    op     = _g(lq,"open")     or _g(fund,"open")
    hd     = _g(lq,"high")     or _g(fund,"dayHigh")
    ld2    = _g(lq,"low")      or _g(fund,"dayLow")
    pc     = _g(lq,"previous_close") or _g(fund,"previousClose")

    delta_s = f"{chg:+.2f} ({chgp:+.2f}%)" if chg is not None and chgp is not None else None
    q1.metric("Last Price", f"{lp:.2f}" if lp else "N/A", delta_s)
    q2.metric("Open",       _n(op)  if op  else "N/A")
    q3.metric("Day High",   _n(hd)  if hd  else "N/A")
    q4.metric("Day Low",    _n(ld2) if ld2 else "N/A")

    q5, q6, q7, q8 = st.columns(4)
    q5.metric("Prev Close", _n(pc)  if pc  else "N/A")
    q6.metric("Volume",     f"{int(vol):,}"    if vol    else "N/A")
    q7.metric("Avg Volume", f"{int(avgvol):,}" if avgvol else "N/A")
    q8.metric("Currency",   fund.get("currency") or currency)

    # ── Company ──
    sect = fund.get("sector"); ind = fund.get("industry")
    if sect:
        st.caption(f"🏢 **{sect}** › {ind or ''}")

    st.markdown("---")

    # ── Valuation ──
    st.markdown("### 📊 Valuation")
    v1,v2,v3,v4 = st.columns(4)
    v5,v6,v7,v8 = st.columns(4)

    v1.metric("Market Cap",        _money(_g(fund,"marketCap")))
    v2.metric("P/E Ratio (TTM)",   _n(_g(fund,"trailingPE","forwardPE")) or "N/A")
    v3.metric("P/B Ratio",         _n(_g(fund,"priceToBook")) or "N/A")
    v4.metric("P/S Ratio",         _n(_g(fund,"priceToSalesTrailing12Months")) or "N/A")
    v5.metric("Enterprise Value",  _money(_g(fund,"enterpriseValue")))
    v6.metric("EV/EBITDA",         _n(_g(fund,"enterpriseToEbitda")) or "N/A")
    v7.metric("PEG Ratio",         _n(_g(fund,"pegRatio")) or "N/A")
    v8.metric("Beta",              _n(_g(fund,"beta")) or "N/A")

    st.markdown("---")

    # ── Dividends ──
    st.markdown("### 💰 Dividends")
    d1,d2,d3,d4 = st.columns(4)
    d1.metric("Dividend Yield",   _pct(_g(fund,"dividendYield"))   or "N/A")
    d2.metric("Dividend Rate",    _n(_g(fund,"dividendRate"),"$")  or "N/A")
    d3.metric("Payout Ratio",     _pct(_g(fund,"payoutRatio"))     or "N/A")
    d4.metric("Ex-Dividend Date", fund.get("exDividendDate","N/A"))

    st.markdown("---")

    # ── Financial Performance ──
    st.markdown("### 📈 Financial Performance")
    f1,f2,f3,f4 = st.columns(4)
    f5,f6,f7,f8 = st.columns(4)
    f1.metric("Revenue (TTM)",     _money(_g(fund,"totalRevenue")))
    f2.metric("Revenue Growth",    _pct(_g(fund,"revenueGrowth"))    or "N/A")
    f3.metric("Gross Margin",      _pct(_g(fund,"grossMargins"))     or "N/A")
    f4.metric("Net Profit Margin", _pct(_g(fund,"profitMargins"))    or "N/A")
    f5.metric("EBITDA",            _money(_g(fund,"ebitda")))
    f6.metric("EPS (TTM)",         _n(_g(fund,"trailingEps","forwardEps"),"$") or "N/A")
    f7.metric("Return on Equity",  _pct(_g(fund,"returnOnEquity"))   or "N/A")
    f8.metric("Return on Assets",  _pct(_g(fund,"returnOnAssets"))   or "N/A")

    st.markdown("---")

    # ── Balance Sheet ──
    st.markdown("### 🏦 Balance Sheet")
    b1,b2,b3,b4 = st.columns(4)
    b5,b6,b7,b8 = st.columns(4)
    b1.metric("Total Cash",       _money(_g(fund,"totalCash")))
    b2.metric("Total Debt",       _money(_g(fund,"totalDebt")))
    b3.metric("Debt / Equity",    _n(_g(fund,"debtToEquity"))  or "N/A")
    b4.metric("Current Ratio",    _n(_g(fund,"currentRatio"))  or "N/A")
    b5.metric("Quick Ratio",      _n(_g(fund,"quickRatio"))    or "N/A")
    b6.metric("Book Value/Share", _n(_g(fund,"bookValue"),"$") or "N/A")
    b7.metric("Total Assets",     _money(_g(fund,"totalAssets")))
    b8.metric("Free Cash Flow",   _money(_g(fund,"freeCashflow")))

    st.markdown("---")

    # ── 52-Week & Analyst ──
    st.markdown("### 📅 52-Week Range & Analyst Targets")
    w1,w2,w3,w4 = st.columns(4)
    w5,w6 = st.columns(2)

    hi52 = _g(fund,"fiftyTwoWeekHigh")
    lo52 = _g(fund,"fiftyTwoWeekLow")

    # Fallback from price history if Yahoo didn't return it
    if hi52 is None and not hist.empty:
        sub  = hist[hist.index >= hist.index.max()-pd.Timedelta(days=365)]
        if not sub.empty:
            hi52 = float(sub["High"].max())
            lo52 = float(sub["Low"].min())

    shares = _g(fund,"sharesOutstanding","floatShares")
    w1.metric("52-Week High",       f"{hi52:.2f}" if hi52 else "N/A")
    w2.metric("52-Week Low",        f"{lo52:.2f}" if lo52 else "N/A")
    w3.metric("Analyst Target",     _n(_g(fund,"targetMeanPrice"),"$") or "N/A")
    w4.metric("Analyst Rating",     (fund.get("recommendationKey","N/A") or "N/A").upper())
    w5.metric("Shares Outstanding", _money(shares).replace("$","") if shares else "N/A")

    # ── No-data warning ──
    has_data = any(_g(fund, k) is not None for k in
                   ["marketCap","trailingPE","totalRevenue","trailingEps"])
    if not has_data:
        st.warning(
            "⚠️ Yahoo Finance returned no fundamental data for this symbol.\n\n"
            "**Possible causes:**\n"
            "- Yahoo is temporarily rate-limiting this Streamlit Cloud IP\n"
            "- Saudi (Tadawul) stocks have limited fundamental coverage on Yahoo\n\n"
            "**Try:** Click **Analyze Stock** again — it usually works on the 2nd or 3rd attempt."
        )

# ═══════════════════════════════════════
# TAB 3 — AI Forecast
# ═══════════════════════════════════════
with tab3:
    st.subheader("🔮 AI-Powered Price Forecast")
    horizon = st.slider("Forecast horizon (days)", 7, 90, 30, 1)

    cdf = (hist.reset_index()
               .rename(columns={hist.index.name or "index": "Date"})
               [["Date","Close"]].dropna())

    if cdf.empty or cdf["Close"].nunique() < 10:
        st.warning("Not enough historical data for a forecast.")
    else:
        use_p = True
        try:
            from prophet import Prophet
        except Exception:
            use_p = False

        if use_p:
            try:
                st.info("Using Prophet forecasting ✅")
                dfp = cdf.rename(columns={"Date":"ds","Close":"y"})
                dfp["ds"] = pd.to_datetime(dfp["ds"])
                m = Prophet(daily_seasonality=False,
                            weekly_seasonality=True, yearly_seasonality=True)
                m.fit(dfp)
                fut = m.make_future_dataframe(periods=horizon)
                fc  = m.predict(fut)
                last_d = dfp["ds"].max()
                fig = go.Figure([
                    go.Scatter(x=dfp["ds"], y=dfp["y"],
                               mode="lines", name="Actual"),
                    go.Scatter(x=fc["ds"],  y=fc["yhat"],
                               mode="lines", name="Forecast"),
                    go.Scatter(x=fc["ds"],  y=fc["yhat_upper"],
                               mode="lines", line=dict(width=0),
                               showlegend=False, hoverinfo="skip"),
                    go.Scatter(x=fc["ds"],  y=fc["yhat_lower"],
                               mode="lines", fill="tonexty",
                               fillcolor="rgba(100,180,255,0.15)",
                               line=dict(width=0),
                               showlegend=False, hoverinfo="skip"),
                ])
                fig.update_layout(
                    title=f"{sym} — {horizon}-day Forecast",
                    xaxis_title="Date", yaxis_title=f"Price ({currency})",
                    height=550, hovermode="x unified", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
                tbl = fc[fc["ds"] > last_d][["ds","yhat","yhat_lower","yhat_upper"]]
                tbl.columns = ["Date","Forecast","Low CI","High CI"]
                st.dataframe(tbl.tail(30), use_container_width=True)
            except Exception as e:
                st.warning(f"Prophet error: {e}. Using linear fallback.")
                use_p = False

        if not use_p:
            st.info("Using linear trend forecast ✅")
            y    = cdf["Close"].values.astype(float)
            x    = np.arange(len(y), dtype=float)
            coef = np.polyfit(x, y, 1)
            tr   = np.poly1d(coef)
            xa   = np.arange(len(y) + horizon, dtype=float)
            yhat = tr(xa)
            ld   = pd.to_datetime(cdf["Date"].max())
            fd   = pd.date_range(ld, periods=horizon+1, freq="D")[1:]
            ad   = pd.concat([pd.to_datetime(cdf["Date"]),
                               pd.Series(fd)], ignore_index=True)
            fig  = go.Figure([
                go.Scatter(x=pd.to_datetime(cdf["Date"]), y=cdf["Close"],
                           mode="lines", name="Actual"),
                go.Scatter(x=ad, y=yhat, mode="lines",
                           name="Trend", line=dict(dash="dash", color="orange")),
            ])
            fig.update_layout(
                title=f"{sym} — {horizon}-day Trend Forecast",
                xaxis_title="Date", yaxis_title=f"Price ({currency})",
                height=550, hovermode="x unified", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

    st.caption("⚠️ Forecasts are experimental and for educational purposes only. Not financial advice.")

st.markdown("---")
st.caption("⚠️ Educational purposes only. Not financial advice.")

