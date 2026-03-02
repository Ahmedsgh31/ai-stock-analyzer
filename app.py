"""
AI-Powered Stock Market Analyzer
---------------------------------
Price data   : Twelve Data (free) for US/global  |  yfinance fallback for Saudi (.SR)
Fundamentals : yfinance (free) — works on Streamlit Cloud via fast_info + Ticker props
Saudi stocks : yfinance with randomised User-Agent to avoid Yahoo rate-limits
"""

import time, random
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(page_title="AI Stock Analyzer", page_icon="📈", layout="wide")
st.title("📈 AI-Powered Stock Market Analyzer")
st.markdown("---")

# ─────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────
def _safe_float(x):
    try:
        if x is None: return None
        if isinstance(x, (int, float, np.number)):
            f = float(x)
            return None if (np.isnan(f) or np.isinf(f)) else f
        s = str(x).replace(",", "").strip()
        if s in ("", "None", "N/A", "null", "nan", "inf", "-inf"): return None
        return float(s)
    except Exception:
        return None

def _money(x):
    v = _safe_float(x)
    if v is None: return "N/A"
    a = abs(v)
    if a >= 1e12: return f"${v/1e12:.2f}T"
    if a >= 1e9:  return f"${v/1e9:.2f}B"
    if a >= 1e6:  return f"${v/1e6:.2f}M"
    if a >= 1e3:  return f"${v/1e3:.2f}K"
    return f"${v:.2f}"

def _pct(x, already_pct=False):
    v = _safe_float(x)
    if v is None: return "N/A"
    if not already_pct and abs(v) < 2.0:
        v *= 100
    return f"{v:.2f}%"

def _num(x, pre="", suf="", dec=2):
    v = _safe_float(x)
    if v is None: return "N/A"
    return f"{pre}{v:,.{dec}f}{suf}"

# ─────────────────────────────────────────────
# HTTP session  (shared, randomised UA)
# ─────────────────────────────────────────────
_UAS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/123 Safari/537.36",
]

@st.cache_resource
def _session():
    s = requests.Session()
    s.headers.update({"User-Agent": random.choice(_UAS)})
    return s

# ─────────────────────────────────────────────
# Twelve Data  (price + quote — free plan OK)
# ─────────────────────────────────────────────
_TD = "https://api.twelvedata.com"

def _td_key():
    return st.secrets.get("TWELVEDATA_API_KEY") or None

def _td(endpoint, params, timeout=25):
    k = _td_key()
    if not k:
        return {"status": "error", "message": "No TWELVEDATA_API_KEY"}
    try:
        r = _session().get(f"{_TD}/{endpoint.lstrip('/')}", params={**params, "apikey": k}, timeout=timeout)
        return r.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}

@st.cache_data(ttl=3600)
def td_resolve(sym: str) -> dict | None:
    s = sym.strip().upper()
    is_sa = s.endswith(".SR") or (s.isdigit() and len(s) in (3,4,5))
    base  = s.replace(".SR","") if s.endswith(".SR") else s

    items = (_td("symbol_search", {"symbol": base, "outputsize": 50}).get("data") or [])
    if not items:
        items = (_td("symbol_search", {"keywords": base, "outputsize": 50}).get("data") or [])
    if not items:
        return None

    def score(it):
        sc = 0
        sym2 = (it.get("symbol") or "").upper()
        ex   = (it.get("exchange") or "").lower()
        ctry = (it.get("country") or "").lower()
        cur  = (it.get("currency") or "").upper()
        typ  = (it.get("instrument_type") or "").lower()
        if typ == "common stock": sc += 10
        if is_sa:
            if "saudi" in ctry or "tadawul" in ex: sc += 70
            if cur == "SAR": sc += 30
            if sym2 == base: sc += 40
        else:
            if sym2 == s: sc += 60
            if ex in ("nasdaq","nyse","nyse american","nyse arca"): sc += 20
        return sc

    return sorted(items, key=score, reverse=True)[0]

@st.cache_data(ttl=600)
def td_ohlcv(symbol, exchange, interval, outputsize) -> tuple[pd.DataFrame, str]:
    p = {"symbol": symbol, "interval": interval, "outputsize": outputsize, "format": "JSON"}
    if exchange: p["exchange"] = exchange
    d = _td("time_series", p)
    if d.get("status") == "error":
        return pd.DataFrame(), d.get("message","")
    vals = d.get("values") or []
    if not vals:
        return pd.DataFrame(), "empty"
    df = pd.DataFrame(vals)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").set_index("datetime")
    for c in ["open","high","low","close","volume"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.rename(columns=str.title)
    for col in ["Open","High","Low","Close"]:
        if col not in df.columns: return pd.DataFrame(), f"Missing {col}"
    df["Volume"] = df.get("Volume", 0)
    return df.dropna(subset=["Close"]), ""

@st.cache_data(ttl=60)
def td_quote(symbol, exchange) -> dict:
    p = {"symbol": symbol, "format": "JSON"}
    if exchange: p["exchange"] = exchange
    d = _td("quote", p)
    return {} if d.get("status") == "error" else d

# ─────────────────────────────────────────────
# yfinance  — fundamentals + Saudi price
# ─────────────────────────────────────────────
def _yf_import():
    try:
        import yfinance as yf
        return yf
    except ImportError:
        return None

@st.cache_data(ttl=600)
def yf_history(yf_sym: str, period: str) -> pd.DataFrame:
    yf = _yf_import()
    if not yf: return pd.DataFrame()
    try:
        t  = yf.Ticker(yf_sym)
        df = t.history(period=period, interval="1d", auto_adjust=False)
        if df is None or df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        for c in ["Open","High","Low","Close"]:
            if c not in df.columns: return pd.DataFrame()
        df["Volume"] = df.get("Volume", 0)
        return df.dropna(subset=["Close"])
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=900)
def yf_fundamentals(yf_sym: str) -> dict:
    """
    Fetch fundamentals using yfinance's individual properties, which are
    much more reliable than .info on Streamlit Cloud (fewer rate-limits).
    Falls back to .info if available.
    """
    yf = _yf_import()
    if not yf: return {}

    result = {}
    try:
        t = yf.Ticker(yf_sym)

        # ── fast_info (lightweight, almost never blocked) ──
        try:
            fi = t.fast_info
            for attr in ["market_cap","shares_outstanding","last_price",
                          "previous_close","fifty_two_week_high","fifty_two_week_low",
                          "year_high","year_low","currency"]:
                val = getattr(fi, attr, None)
                if val is not None:
                    result[attr] = val
        except Exception:
            pass

        # ── .info dict (may be blocked, but try anyway) ──
        try:
            info = t.info or {}
            # Only take keys absent from result so far
            wanted = [
                "longName","shortName","exchange","currency","sector","industry",
                "marketCap","sharesOutstanding","floatShares","beta",
                "trailingPE","forwardPE","pegRatio","priceToBook",
                "priceToSalesTrailing12Months","enterpriseValue","enterpriseToEbitda",
                "dividendYield","trailingAnnualDividendYield","dividendRate",
                "trailingAnnualDividendRate","payoutRatio","exDividendDate",
                "totalRevenue","revenueGrowth","grossMargins","profitMargins",
                "operatingMargins","ebitda","trailingEps","forwardEps",
                "returnOnEquity","returnOnAssets",
                "totalCash","totalDebt","debtToEquity","currentRatio",
                "quickRatio","bookValue","totalAssets","freeCashflow",
                "fiftyTwoWeekHigh","fiftyTwoWeekLow",
                "targetMeanPrice","targetMedianPrice","recommendationKey",
                "currentPrice","regularMarketPrice","previousClose",
                "volume","averageVolume","averageVolume10days",
                "open","regularMarketOpen","dayHigh","dayLow",
            ]
            for k in wanted:
                if k not in result and info.get(k) is not None:
                    result[k] = info[k]
        except Exception:
            pass

        # ── Income statement (quarterly, fast) ──
        try:
            qs = t.quarterly_income_stmt
            if qs is not None and not qs.empty:
                col = qs.columns[0]  # most recent quarter
                for row_key, out_key in [
                    ("Total Revenue","revenue_q"),
                    ("Gross Profit","gross_profit_q"),
                    ("EBITDA","ebitda_q"),
                    ("Net Income","net_income_q"),
                    ("Diluted EPS","eps_q"),
                ]:
                    if row_key in qs.index:
                        v = _safe_float(qs.loc[row_key, col])
                        if v is not None: result[out_key] = v
        except Exception:
            pass

        # ── Balance sheet ──
        try:
            bs = t.quarterly_balance_sheet
            if bs is not None and not bs.empty:
                col = bs.columns[0]
                for row_key, out_key in [
                    ("Cash And Cash Equivalents","cash_q"),
                    ("Total Debt","total_debt_q"),
                    ("Total Assets","total_assets_q"),
                    ("Stockholders Equity","equity_q"),
                ]:
                    if row_key in bs.index:
                        v = _safe_float(bs.loc[row_key, col])
                        if v is not None: result[out_key] = v
        except Exception:
            pass

    except Exception:
        pass

    return result

def _g(d: dict, *keys):
    """Get first non-None value from dict by trying multiple keys."""
    for k in keys:
        v = _safe_float(d.get(k))
        if v is not None: return v
    return None

def _gs(d: dict, *keys):
    """Get first non-None string from dict by trying multiple keys."""
    for k in keys:
        v = d.get(k)
        if v and str(v).strip() not in ("","None","N/A","null","nan"):
            return str(v)
    return None

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("Search Stock")
    stock_symbol = st.text_input("Enter Stock Symbol",
        placeholder="e.g. AAPL, TSLA, 2222.SR",
        help="US: AAPL TSLA NVDA  |  Saudi: 2222.SR 1120.SR")

    period_td = {
        "1 Month":  ("1day",35),   "3 Months":("1day",95),
        "6 Months": ("1day",185),  "1 Year":  ("1day",262),
        "2 Years":  ("1day",524),  "5 Years": ("1day",1310),
    }
    period_yf = {
        "1 Month":"1mo","3 Months":"3mo","6 Months":"6mo",
        "1 Year":"1y","2 Years":"2y","5 Years":"5y",
    }
    sel_period  = st.selectbox("Select Time Period", list(period_td.keys()), index=3)
    show_debug  = st.checkbox("Show debug panels", value=False)
    go_btn      = st.button("🔍 Analyze Stock", type="primary", use_container_width=True)

if "result" not in st.session_state:
    st.session_state.result = None

# ─────────────────────────────────────────────
# On button press
# ─────────────────────────────────────────────
if go_btn:
    raw = (stock_symbol or "").strip().upper()
    if not raw:
        st.warning("⚠️ Enter a stock symbol first."); st.stop()

    is_saudi = raw.endswith(".SR") or (raw.isdigit() and len(raw) in (3,4,5))
    yf_sym   = f"{raw.replace('.SR','')}.SR" if is_saudi else raw

    hist = pd.DataFrame()
    td_sym = td_ex = display_name = currency = None
    provider = "unknown"

    # ── Resolve via Twelve Data ──
    if _td_key():
        with st.spinner("Resolving symbol…"):
            resolved = td_resolve(raw)
        if resolved:
            td_sym  = resolved.get("symbol") or raw
            td_ex   = resolved.get("exchange") or None
            display_name = resolved.get("instrument_name") or td_sym
            currency     = resolved.get("currency") or "N/A"

    # ── Price: Twelve Data (preferred) ──
    if td_sym and _td_key():
        interval, outputsize = period_td[sel_period]
        with st.spinner(f"Loading price data (Twelve Data) for {td_sym}…"):
            hist, err = td_ohlcv(td_sym, td_ex, interval, outputsize)
        if not hist.empty:
            provider = "twelvedata"

    # ── Price fallback: yfinance (especially for Saudi) ──
    if hist.empty:
        yf = _yf_import()
        if yf:
            with st.spinner(f"Loading price data (Yahoo Finance) for {yf_sym}…"):
                hist = yf_history(yf_sym, period_yf[sel_period])
            if not hist.empty:
                provider = "yfinance"
                if not display_name: display_name = yf_sym
                if not currency:     currency = "SAR" if is_saudi else "USD"

    if hist.empty:
        st.error(f"No price data found for **{raw}**.")
        if is_saudi:
            st.warning(
                "Saudi (Tadawul) price data requires:\n"
                "- **Twelve Data Pro plan** for the Tadawul exchange, OR\n"
                "- Yahoo Finance (sometimes blocked on Streamlit Cloud)\n\n"
                "Try again in a few minutes if you believe it should work."
            )
        else:
            st.info("Check the ticker symbol and ensure TWELVEDATA_API_KEY is set.")
        st.stop()

    # Enrich display_name from yfinance if still missing
    if not display_name or display_name == raw:
        yf_fund = yf_fundamentals(yf_sym)
        n = _gs(yf_fund, "longName","shortName")
        if n: display_name = n
        if not currency or currency == "N/A":
            c = _gs(yf_fund, "currency")
            if c: currency = c

    st.session_state.result = {
        "raw": raw, "td_sym": td_sym, "td_ex": td_ex,
        "yf_sym": yf_sym, "name": display_name or raw,
        "currency": currency or "N/A", "hist": hist,
        "provider": provider, "is_saudi": is_saudi,
    }

# ─────────────────────────────────────────────
# Render
# ─────────────────────────────────────────────
res = st.session_state.result
if not res:
    st.info("👈 Enter a stock symbol in the sidebar and click **Analyze Stock**.")
    st.markdown("""
**Examples**
| Market | Symbol |
|--------|--------|
| NASDAQ | `AAPL` `TSLA` `NVDA` `MSFT` |
| NYSE   | `JPM` `XOM` `KO` |
| Saudi  | `2222.SR` `1120.SR` `2010.SR` |
""")
    st.stop()

# Unpack
sym      = res["td_sym"] or res["raw"]
yf_sym   = res["yf_sym"]
td_ex    = res["td_ex"]
name     = res["name"]
currency = res["currency"]
hist     = res["hist"]
provider = res["provider"]

# ── Header metrics ──
c1,c2,c3,c4 = st.columns([2,1,1,1])
with c1:
    st.subheader(name)
    st.caption(f"**{sym}** | Exchange: {td_ex or 'Auto'} | Currency: {currency} | Data: {provider}")

close_s = hist["Close"].dropna()
with c2:
    if len(close_s) >= 2:
        cp,pp = float(close_s.iloc[-1]), float(close_s.iloc[-2])
        d = cp-pp; dp = d/pp*100 if pp else 0
        st.metric("Current Price", f"{cp:.2f}", f"{d:+.2f} ({dp:+.2f}%)")
    else:
        st.metric("Current Price","N/A")
with c3: st.metric("Period High", f"{float(hist['High'].max()):.2f}")
with c4: st.metric("Period Low",  f"{float(hist['Low'].min()):.2f}")

st.markdown("---")
tab1, tab2, tab3 = st.tabs(["📊 Price Analysis","💼 Financial Metrics","🔮 AI Forecast"])

# ═══════════════════════════════════════════
# TAB 1 — Price chart
# ═══════════════════════════════════════════
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
        xaxis_title="Date", height=600, hovermode="x unified",
        template="plotly_dark", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════
# TAB 2 — Financial Metrics
# ═══════════════════════════════════════════
with tab2:
    st.subheader("Financial Metrics")

    with st.spinner("Loading fundamentals (Yahoo Finance)…"):
        fund = yf_fundamentals(yf_sym)

    # Also pull live quote from Twelve Data if available
    live_q = {}
    if res["td_sym"] and _td_key() and provider == "twelvedata":
        live_q = td_quote(res["td_sym"], td_ex)

    if show_debug:
        with st.expander("🔧 yfinance fundamentals"): st.json(fund)
        with st.expander("🔧 Twelve Data quote"):     st.json(live_q)

    # ── Live Quote ──
    st.markdown("### 📌 Live Quote")
    q1,q2,q3,q4 = st.columns(4)

    last_p = _g(live_q,"close","price") or _g(fund,"currentPrice","regularMarketPrice","last_price")
    if last_p is None and len(close_s): last_p = float(close_s.iloc[-1])

    chg    = _safe_float(live_q.get("change"))
    chg_p  = _safe_float(live_q.get("percent_change"))
    if chg is None:
        prev = _g(fund,"previousClose","previous_close","regularMarketPreviousClose")
        if prev and last_p:
            chg  = last_p - prev
            chg_p = chg/prev*100 if prev else None

    vol    = _g(live_q,"volume") or _g(fund,"volume")
    avgvol = _g(live_q,"average_volume") or _g(fund,"averageVolume","averageVolume10days")
    open_p = _g(live_q,"open") or _g(fund,"open","regularMarketOpen")
    hi_d   = _g(live_q,"high") or _g(fund,"dayHigh")
    lo_d   = _g(live_q,"low")  or _g(fund,"dayLow")
    prev_c = _g(live_q,"previous_close") or _g(fund,"previousClose","previous_close")

    delta = f"{chg:+.2f} ({chg_p:+.2f}%)" if chg is not None and chg_p is not None else None
    q1.metric("Last Price",  f"{last_p:.2f}" if last_p else "N/A", delta)
    q2.metric("Open",        _num(open_p) if open_p else "N/A")
    q3.metric("Day High",    _num(hi_d)   if hi_d   else "N/A")
    q4.metric("Day Low",     _num(lo_d)   if lo_d   else "N/A")

    q5,q6,q7,q8 = st.columns(4)
    q5.metric("Prev Close",  _num(prev_c)  if prev_c  else "N/A")
    q6.metric("Volume",      f"{int(vol):,}" if vol else "N/A")
    q7.metric("Avg Volume",  f"{int(avgvol):,}" if avgvol else "N/A")
    q8.metric("Currency",    currency)

    st.markdown("---")

    # ── Valuation ──
    st.markdown("### 📊 Valuation")
    v1,v2,v3,v4 = st.columns(4)
    v5,v6,v7,v8 = st.columns(4)

    mktcap   = _g(fund,"marketCap","market_cap")
    pe       = _g(fund,"trailingPE","forwardPE")
    pb       = _g(fund,"priceToBook")
    ps       = _g(fund,"priceToSalesTrailing12Months")
    ev       = _g(fund,"enterpriseValue")
    evebitda = _g(fund,"enterpriseToEbitda")
    peg      = _g(fund,"pegRatio")
    beta     = _g(fund,"beta")

    v1.metric("Market Cap",       _money(mktcap))
    v2.metric("P/E Ratio (TTM)",  _num(pe)      if pe      else "N/A")
    v3.metric("P/B Ratio",        _num(pb)      if pb      else "N/A")
    v4.metric("P/S Ratio",        _num(ps)      if ps      else "N/A")
    v5.metric("Enterprise Value", _money(ev))
    v6.metric("EV/EBITDA",        _num(evebitda)if evebitda else "N/A")
    v7.metric("PEG Ratio",        _num(peg)     if peg     else "N/A")
    v8.metric("Beta",             _num(beta)    if beta    else "N/A")

    st.markdown("---")

    # ── Dividends ──
    st.markdown("### 💰 Dividends")
    d1,d2,d3,d4 = st.columns(4)

    div_y  = _g(fund,"dividendYield","trailingAnnualDividendYield")
    div_r  = _g(fund,"dividendRate","trailingAnnualDividendRate")
    payout = _g(fund,"payoutRatio")

    ex_div = fund.get("exDividendDate")
    if ex_div:
        try:
            ex_div = pd.to_datetime(ex_div, unit="s").strftime("%Y-%m-%d") \
                     if isinstance(ex_div,(int,float)) else str(ex_div)[:10]
        except Exception:
            ex_div = str(ex_div)

    d1.metric("Dividend Yield",   _pct(div_y) if div_y else "N/A")
    d2.metric("Dividend Rate",    _num(div_r,"$") if div_r else "N/A")
    d3.metric("Payout Ratio",     _pct(payout) if payout else "N/A")
    d4.metric("Ex-Dividend Date", ex_div or "N/A")

    st.markdown("---")

    # ── Financial Performance ──
    st.markdown("### 📈 Financial Performance")
    f1,f2,f3,f4 = st.columns(4)
    f5,f6,f7,f8 = st.columns(4)

    rev     = _g(fund,"totalRevenue","revenue_q")
    rev_g   = _g(fund,"revenueGrowth")
    gm      = _g(fund,"grossMargins")
    nm      = _g(fund,"profitMargins")
    ebitda  = _g(fund,"ebitda","ebitda_q")
    eps     = _g(fund,"trailingEps","forwardEps","eps_q")
    roe     = _g(fund,"returnOnEquity")
    roa     = _g(fund,"returnOnAssets")

    f1.metric("Revenue (TTM)",     _money(rev))
    f2.metric("Revenue Growth",    _pct(rev_g)  if rev_g  else "N/A")
    f3.metric("Gross Margin",      _pct(gm)     if gm     else "N/A")
    f4.metric("Net Profit Margin", _pct(nm)     if nm     else "N/A")
    f5.metric("EBITDA",            _money(ebitda))
    f6.metric("EPS (TTM)",         _num(eps,"$") if eps   else "N/A")
    f7.metric("Return on Equity",  _pct(roe)    if roe    else "N/A")
    f8.metric("Return on Assets",  _pct(roa)    if roa    else "N/A")

    st.markdown("---")

    # ── Balance Sheet ──
    st.markdown("### 🏦 Balance Sheet")
    b1,b2,b3,b4 = st.columns(4)
    b5,b6,b7,b8 = st.columns(4)

    cash   = _g(fund,"totalCash","cash_q")
    debt   = _g(fund,"totalDebt","total_debt_q")
    de     = _g(fund,"debtToEquity")
    cr     = _g(fund,"currentRatio")
    qr     = _g(fund,"quickRatio")
    bv     = _g(fund,"bookValue")
    assets = _g(fund,"totalAssets","total_assets_q")
    fcf    = _g(fund,"freeCashflow")

    b1.metric("Total Cash",       _money(cash))
    b2.metric("Total Debt",       _money(debt))
    b3.metric("Debt / Equity",    _num(de)  if de  else "N/A")
    b4.metric("Current Ratio",    _num(cr)  if cr  else "N/A")
    b5.metric("Quick Ratio",      _num(qr)  if qr  else "N/A")
    b6.metric("Book Value/Share", _num(bv,"$") if bv else "N/A")
    b7.metric("Total Assets",     _money(assets))
    b8.metric("Free Cash Flow",   _money(fcf))

    st.markdown("---")

    # ── 52-Week & Analyst ──
    st.markdown("### 📅 52-Week Range & Analyst Targets")
    w1,w2,w3,w4 = st.columns(4)

    hi52   = _g(fund,"fiftyTwoWeekHigh","fifty_two_week_high","year_high")
    lo52   = _g(fund,"fiftyTwoWeekLow", "fifty_two_week_low", "year_low")
    target = _g(fund,"targetMeanPrice","targetMedianPrice")
    rec    = _gs(fund,"recommendationKey")
    shares = _g(fund,"sharesOutstanding","floatShares","shares_outstanding")

    # Fallback from price history
    if hi52 is None and not hist.empty:
        sub = hist[hist.index >= hist.index.max()-pd.Timedelta(days=365)]
        if not sub.empty:
            hi52 = float(sub["High"].max())
            lo52 = float(sub["Low"].min())

    w1.metric("52-Week High",       f"{hi52:.2f}"  if hi52   else "N/A")
    w2.metric("52-Week Low",        f"{lo52:.2f}"  if lo52   else "N/A")
    w3.metric("Analyst Target",     f"{target:.2f}"if target  else "N/A")
    w4.metric("Analyst Rating",     (rec.upper()   if rec     else "N/A"))

    w5,w6 = st.columns(2)
    w5.metric("Shares Outstanding", _money(shares).replace("$","") if shares else "N/A")
    sect = _gs(fund,"sector");  ind = _gs(fund,"industry")
    w6.metric("Sector / Industry",  f"{sect} / {ind}" if sect else "N/A")

    # No-data note
    if not any([mktcap, pe, rev, eps]):
        st.info(
            "ℹ️ Some or all fundamental data could not be loaded from Yahoo Finance. "
            "This can happen when Streamlit Cloud's IP is temporarily rate-limited by Yahoo. "
            "**Try clicking Analyze Stock again** — it usually succeeds on the 2nd attempt."
        )

# ═══════════════════════════════════════════
# TAB 3 — AI Forecast
# ═══════════════════════════════════════════
with tab3:
    st.subheader("🔮 AI-Powered Price Forecast")
    horizon = st.slider("Forecast horizon (days)", 7, 90, 30, 1)

    cdf = hist.reset_index().rename(columns={hist.index.name or "index":"Date"})[["Date","Close"]].dropna()

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
                m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
                m.fit(dfp)
                fut = m.make_future_dataframe(periods=horizon)
                fc  = m.predict(fut)
                last_d = dfp["ds"].max()
                fig = go.Figure([
                    go.Scatter(x=dfp["ds"], y=dfp["y"], mode="lines", name="Actual"),
                    go.Scatter(x=fc["ds"],  y=fc["yhat"], mode="lines", name="Forecast"),
                    go.Scatter(x=fc["ds"],  y=fc["yhat_upper"], mode="lines",
                               line=dict(width=0), showlegend=False, hoverinfo="skip"),
                    go.Scatter(x=fc["ds"],  y=fc["yhat_lower"], mode="lines",
                               fill="tonexty", fillcolor="rgba(100,180,255,0.15)",
                               line=dict(width=0), showlegend=False, hoverinfo="skip"),
                ])
                fig.update_layout(title=f"{sym} — {horizon}-day Forecast",
                    xaxis_title="Date", yaxis_title=f"Price ({currency})",
                    height=550, hovermode="x unified", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
                tbl = fc[fc["ds"]>last_d][["ds","yhat","yhat_lower","yhat_upper"]]
                tbl.columns=["Date","Forecast","Low CI","High CI"]
                st.dataframe(tbl.tail(30), use_container_width=True)
            except Exception as e:
                st.warning(f"Prophet error: {e}. Using linear fallback.")
                use_p = False

        if not use_p:
            st.info("Using linear trend forecast ✅")
            y = cdf["Close"].values.astype(float)
            x = np.arange(len(y), dtype=float)
            coef = np.polyfit(x, y, 1); tr = np.poly1d(coef)
            xa   = np.arange(len(y)+horizon, dtype=float)
            yhat = tr(xa)
            ld   = pd.to_datetime(cdf["Date"].max())
            fd   = pd.date_range(ld, periods=horizon+1, freq="D")[1:]
            ad   = pd.concat([pd.to_datetime(cdf["Date"]),pd.Series(fd)], ignore_index=True)
            fig  = go.Figure([
                go.Scatter(x=pd.to_datetime(cdf["Date"]),y=cdf["Close"],mode="lines",name="Actual"),
                go.Scatter(x=ad,y=yhat,mode="lines",name="Trend",
                           line=dict(dash="dash",color="orange")),
            ])
            fig.update_layout(title=f"{sym} — {horizon}-day Trend",
                xaxis_title="Date", yaxis_title=f"Price ({currency})",
                height=550, hovermode="x unified", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

    st.caption("⚠️ Forecasts are experimental. Not financial advice.")

st.markdown("---")
st.caption("⚠️ Educational purposes only. Not financial advice.")
