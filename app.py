"""
AI-Powered Stock Market Analyzer
══════════════════════════════════
Price data   → Twelve Data  (free plan, API key required)
Fundamentals → Alpha Vantage (free plan, 25 req/day, API key required)

Add both keys to Streamlit → Settings → Secrets:
    TWELVEDATA_API_KEY  = "your_key"
    ALPHAVANTAGE_API_KEY = "your_key"

Get free keys at:
  https://twelvedata.com         (price history, quote)
  https://www.alphavantage.co    (fundamentals)
"""

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

# ── Page config ──────────────────────────────────────────
st.set_page_config(page_title="AI Stock Analyzer", page_icon="📈", layout="wide")
st.title("📈 AI-Powered Stock Market Analyzer")
st.markdown("---")

# ── Helpers ──────────────────────────────────────────────
def _sf(x):
    try:
        if x is None: return None
        if isinstance(x, (int, float, np.number)):
            f = float(x)
            return None if (np.isnan(f) or np.isinf(f)) else f
        s = str(x).replace(",", "").strip()
        if s.lower() in ("", "none", "n/a", "null", "nan", "-", "inf"): return None
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
    if abs(v) < 2.0: v *= 100
    return f"{v:.2f}%"

def _num(x, pre="", suf="", dec=2):
    v = _sf(x)
    if v is None: return "N/A"
    return f"{pre}{v:,.{dec}f}{suf}"

def _pick(d, *keys):
    for k in keys:
        v = _sf(d.get(k))
        if v is not None: return v
    return None

def _picks(d, *keys):
    for k in keys:
        v = d.get(k)
        if v and str(v).strip().lower() not in ("", "none", "n/a", "null", "-"):
            return str(v).strip()
    return None

# ── Shared HTTP session ───────────────────────────────────
@st.cache_resource
def _sess():
    s = requests.Session()
    s.headers.update({"User-Agent": "StockAnalyzer/4.0 (research)"})
    return s

# ════════════════════════════════════════════════════════
# TWELVE DATA  ─  price + quote
# ════════════════════════════════════════════════════════
def _td_key():
    try: return st.secrets.get("TWELVEDATA_API_KEY") or None
    except Exception: return None

def _td(endpoint, params, timeout=20):
    k = _td_key()
    if not k: return {"status":"error","message":"No TWELVEDATA_API_KEY in Secrets"}
    try:
        r = _sess().get(f"https://api.twelvedata.com/{endpoint}",
                        params={**params, "apikey": k}, timeout=timeout)
        return r.json()
    except Exception as e:
        return {"status":"error","message":str(e)}

@st.cache_data(ttl=3600, show_spinner=False)
def td_resolve(sym: str) -> dict | None:
    base = sym.replace(".SR","") if sym.endswith(".SR") else sym
    is_sa = sym.endswith(".SR") or (sym.isdigit() and len(sym) in (3,4,5))
    items = (_td("symbol_search",{"symbol":base,"outputsize":50}).get("data") or [])
    if not items:
        items = (_td("symbol_search",{"keywords":base,"outputsize":50}).get("data") or [])
    if not items: return None

    def sc(it):
        s2  = (it.get("symbol","")).upper()
        ex  = (it.get("exchange","")).lower()
        cur = (it.get("currency","")).upper()
        ct  = (it.get("country","")).lower()
        sc2 = 10 if it.get("instrument_type","").lower()=="common stock" else 0
        if is_sa:
            if "saudi" in ct or "tadawul" in ex: sc2 += 70
            if cur=="SAR": sc2 += 30
            if s2==base: sc2 += 40
        else:
            if s2==sym: sc2 += 60
            if ex in ("nasdaq","nyse","nyse american","nyse arca"): sc2 += 20
        return sc2
    return sorted(items, key=sc, reverse=True)[0]

@st.cache_data(ttl=600, show_spinner=False)
def td_ohlcv(symbol, exchange, interval, outputsize):
    p = {"symbol":symbol,"interval":interval,
         "outputsize":outputsize,"format":"JSON"}
    if exchange: p["exchange"] = exchange
    d = _td("time_series", p)
    if d.get("status")=="error":
        return pd.DataFrame(), d.get("message","")
    vals = d.get("values") or []
    if not vals: return pd.DataFrame(), "empty response"
    df = pd.DataFrame(vals)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").set_index("datetime")
    for c in ["open","high","low","close","volume"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.rename(columns={"open":"Open","high":"High","low":"Low",
                             "close":"Close","volume":"Volume"})
    for col in ["Open","High","Low","Close"]:
        if col not in df.columns: return pd.DataFrame(), f"Missing {col}"
    if "Volume" not in df.columns: df["Volume"] = 0
    return df.dropna(subset=["Close"]), ""

@st.cache_data(ttl=60, show_spinner=False)
def td_quote(symbol, exchange):
    p = {"symbol":symbol,"format":"JSON"}
    if exchange: p["exchange"] = exchange
    d = _td("quote", p)
    return {} if d.get("status")=="error" else d

# ════════════════════════════════════════════════════════
# ALPHA VANTAGE  ─  fundamentals (free, no IP blocks)
# ════════════════════════════════════════════════════════
def _av_key():
    try: return st.secrets.get("ALPHAVANTAGE_API_KEY") or None
    except Exception: return None

def _av(function, symbol, extra=None, timeout=20):
    k = _av_key()
    if not k: return {}
    p = {"function": function, "symbol": symbol, "apikey": k}
    if extra: p.update(extra)
    try:
        r = _sess().get("https://www.alphavantage.co/query", params=p, timeout=timeout)
        d = r.json()
        # AV returns {"Information":"..."} when rate limited
        if "Information" in d or "Note" in d: return {}
        return d
    except Exception:
        return {}

@st.cache_data(ttl=3600, show_spinner=False)
def av_overview(symbol: str) -> dict:
    """OVERVIEW – company profile + valuation + dividends + financials."""
    return _av("OVERVIEW", symbol)

@st.cache_data(ttl=3600, show_spinner=False)
def av_income(symbol: str) -> dict:
    return _av("INCOME_STATEMENT", symbol)

@st.cache_data(ttl=3600, show_spinner=False)
def av_balance(symbol: str) -> dict:
    return _av("BALANCE_SHEET", symbol)

@st.cache_data(ttl=3600, show_spinner=False)
def av_cashflow(symbol: str) -> dict:
    return _av("CASH_FLOW", symbol)

@st.cache_data(ttl=60, show_spinner=False)
def av_quote(symbol: str) -> dict:
    """GLOBAL_QUOTE – live price from Alpha Vantage."""
    d = _av("GLOBAL_QUOTE", symbol)
    return d.get("Global Quote") or {}

# ════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════
with st.sidebar:
    st.header("Search Stock")
    stock_input = st.text_input("Enter Stock Symbol",
        placeholder="AAPL · TSLA · 2222.SR",
        help="US: AAPL TSLA NVDA | Saudi: 2222.SR 1120.SR")

    period_map = {
        "1 Month":  ("1day", 35),   "3 Months": ("1day", 95),
        "6 Months": ("1day", 185),  "1 Year":   ("1day", 262),
        "2 Years":  ("1day", 524),  "5 Years":  ("1day", 1310),
    }
    sel_period = st.selectbox("Time Period", list(period_map.keys()), index=3)
    show_debug = st.checkbox("Show debug info", value=False)
    go_btn     = st.button("🔍 Analyze Stock", type="primary", use_container_width=True)

    # Key status indicators
    st.markdown("---")
    st.markdown("**API Keys**")
    st.markdown("🟢 Twelve Data" if _td_key() else "🔴 Twelve Data (missing)")
    st.markdown("🟢 Alpha Vantage" if _av_key() else "🔴 Alpha Vantage (missing)")
    if not _td_key() or not _av_key():
        st.info(
            "Add keys in **Settings → Secrets**:\n"
            "```\nTWELVEDATA_API_KEY = \"...\"\n"
            "ALPHAVANTAGE_API_KEY = \"...\"\n```\n"
            "Both are **free** at twelvedata.com and alphavantage.co"
        )

if "result" not in st.session_state:
    st.session_state.result = None

# ════════════════════════════════════════════════════════
# ON BUTTON PRESS
# ════════════════════════════════════════════════════════
if go_btn:
    raw = (stock_input or "").strip().upper()
    if not raw:
        st.warning("⚠️ Enter a stock symbol."); st.stop()

    if not _td_key():
        st.error("❌ **TWELVEDATA_API_KEY** is missing from Secrets.\n\n"
                 "Get a free key at [twelvedata.com](https://twelvedata.com) "
                 "and add it to **Settings → Secrets**.")
        st.stop()

    # Resolve symbol
    with st.spinner("Resolving symbol…"):
        resolved = td_resolve(raw)

    if not resolved:
        st.error(f"Symbol **{raw}** not found in Twelve Data.")
        st.info("Check the ticker. Saudi format: `2222.SR`. US: `AAPL`, `TSLA`.")
        st.stop()

    td_sym  = resolved.get("symbol") or raw
    td_ex   = resolved.get("exchange") or None
    name    = resolved.get("instrument_name") or td_sym
    cur     = resolved.get("currency") or "USD"

    # Fetch price history
    interval, outputsize = period_map[sel_period]
    with st.spinner(f"Loading price data for {td_sym}…"):
        hist, err = td_ohlcv(td_sym, td_ex, interval, outputsize)

    if hist.empty:
        st.error(f"❌ No price data for **{td_sym}**.")
        st.caption(f"Twelve Data said: {err}")
        # Saudi stocks need Pro plan — show helpful message
        is_sa = raw.endswith(".SR") or (raw.isdigit() and len(raw) in (3,4,5))
        if is_sa:
            st.warning(
                "Saudi (Tadawul) price data requires a **Twelve Data Pro plan**.\n\n"
                "**Free alternatives for Saudi stocks:**\n"
                "- Use [investing.com](https://www.investing.com) or "
                "[mubasher.info](https://mubasher.info) manually\n"
                "- Upgrade Twelve Data at [twelvedata.com/pricing](https://twelvedata.com/pricing)"
            )
        else:
            st.info("This symbol may not be covered on the free plan, or the API limit was reached.")
        st.stop()

    # AV symbol: for US stocks use same ticker; Saudi not supported by AV
    is_saudi = raw.endswith(".SR") or (raw.isdigit() and len(raw) in (3,4,5))
    av_sym   = None if is_saudi else td_sym

    st.session_state.result = {
        "raw": raw, "td_sym": td_sym, "td_ex": td_ex,
        "av_sym": av_sym, "name": name, "currency": cur,
        "hist": hist, "is_saudi": is_saudi, "period": sel_period,
    }

# ════════════════════════════════════════════════════════
# RENDER
# ════════════════════════════════════════════════════════
res = st.session_state.result
if not res:
    st.info("👈 Enter a stock symbol and click **Analyze Stock**.")
    st.markdown("""
| Market | Examples |
|--------|---------|
| NASDAQ / NYSE | `AAPL` `TSLA` `NVDA` `MSFT` `AMZN` `GOOGL` `JPM` |
| Saudi Tadawul | `2222.SR` `1120.SR` `2010.SR` *(requires Pro plan)* |

**Required API keys** (both free):
- `TWELVEDATA_API_KEY` → price & charts → [twelvedata.com](https://twelvedata.com)
- `ALPHAVANTAGE_API_KEY` → fundamentals → [alphavantage.co](https://www.alphavantage.co)
""")
    st.stop()

raw      = res["raw"]
td_sym   = res["td_sym"]
td_ex    = res["td_ex"]
av_sym   = res["av_sym"]
name     = res["name"]
currency = res["currency"]
hist     = res["hist"]
is_saudi = res["is_saudi"]

# ── Fetch all data ────────────────────────────────────────
with st.spinner("Loading quote & fundamentals…"):
    live_q  = td_quote(td_sym, td_ex)
    ov      = av_overview(av_sym) if av_sym else {}
    av_q    = av_quote(av_sym)    if av_sym else {}

    # Only fetch detailed statements if we have the key and symbol
    if av_sym and _av_key():
        inc = av_income(av_sym)
        bal = av_balance(av_sym)
        cf  = av_cashflow(av_sym)
    else:
        inc = bal = cf = {}

if show_debug:
    with st.expander("🔧 TD quote"):     st.json(live_q)
    with st.expander("🔧 AV overview"):  st.json(ov)
    with st.expander("🔧 AV quote"):     st.json(av_q)

# Helper: get latest annual report row
def _latest(section_key, statement):
    reps = statement.get(section_key) or []
    return reps[0] if reps else {}

inc_r = _latest("annualReports", inc)
bal_r = _latest("annualReports", bal)
cf_r  = _latest("annualReports", cf)

# ── Header ────────────────────────────────────────────────
c1,c2,c3,c4 = st.columns([2,1,1,1])
with c1:
    st.subheader(name)
    st.caption(f"**{td_sym}** · {td_ex or '—'} · {currency}")

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

# ════════════════════════════════════════════════════════
# TAB 1 — Price Chart
# ════════════════════════════════════════════════════════
with tab1:
    st.subheader("Historical Price & Volume")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=hist.index, open=hist["Open"], high=hist["High"],
        low=hist["Low"], close=hist["Close"], name="Price",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ))
    if hist["Volume"].sum() > 0:
        fig.add_trace(go.Bar(
            x=hist.index, y=hist["Volume"], name="Volume",
            yaxis="y2", opacity=0.25, marker_color="rgba(100,180,255,0.5)"))
    fig.update_layout(
        title=f"{td_sym} — {res['period']}",
        yaxis_title=f"Price ({currency})",
        yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False),
        xaxis_title="Date", height=600, hovermode="x unified",
        template="plotly_dark", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════
# TAB 2 — Financial Metrics
# ════════════════════════════════════════════════════════
with tab2:
    st.subheader("Financial Metrics")

    if is_saudi:
        st.info(
            "ℹ️ Fundamental data via Alpha Vantage is not available for Saudi stocks. "
            "Price chart and live quote are shown above."
        )

    # ── Live Quote ──────────────────────────────────────
    st.markdown("### 📌 Live Quote")
    q1,q2,q3,q4 = st.columns(4)
    q5,q6,q7,q8 = st.columns(4)

    # TD quote fields
    td_price   = _sf(live_q.get("close") or live_q.get("price"))
    td_chg     = _sf(live_q.get("change"))
    td_chg_pct = _sf(live_q.get("percent_change"))
    td_open    = _sf(live_q.get("open"))
    td_hi      = _sf(live_q.get("high"))
    td_lo      = _sf(live_q.get("low"))
    td_prev    = _sf(live_q.get("previous_close"))
    td_vol     = _sf(live_q.get("volume"))
    td_avg_vol = _sf(live_q.get("average_volume"))

    # AV quote fields (supplements TD)
    av_price   = _sf(av_q.get("05. price"))
    av_chg     = _sf(av_q.get("09. change"))
    av_chg_pct_s = (av_q.get("10. change percent") or "").replace("%","")
    av_chg_pct = _sf(av_chg_pct_s)
    av_vol     = _sf(av_q.get("06. volume"))
    av_open    = _sf(av_q.get("02. open"))
    av_hi      = _sf(av_q.get("03. high"))
    av_lo      = _sf(av_q.get("04. low"))
    av_prev    = _sf(av_q.get("08. previous close"))

    price  = td_price  or av_price  or (float(close_s.iloc[-1]) if len(close_s) else None)
    chg    = td_chg    or av_chg
    chgpct = td_chg_pct or av_chg_pct
    opn    = td_open   or av_open
    hi_d   = td_hi     or av_hi
    lo_d   = td_lo     or av_lo
    prev   = td_prev   or av_prev
    vol    = td_vol    or av_vol
    avgvol = td_avg_vol

    delta = f"{chg:+.2f} ({chgpct:+.2f}%)" if chg is not None and chgpct is not None else None
    q1.metric("Last Price",  f"{price:.2f}"    if price  else "N/A", delta)
    q2.metric("Open",        _num(opn)          if opn    else "N/A")
    q3.metric("Day High",    _num(hi_d)         if hi_d   else "N/A")
    q4.metric("Day Low",     _num(lo_d)         if lo_d   else "N/A")
    q5.metric("Prev Close",  _num(prev)         if prev   else "N/A")
    q6.metric("Volume",      f"{int(vol):,}"    if vol    else "N/A")
    q7.metric("Avg Volume",  f"{int(avgvol):,}" if avgvol else "N/A")
    q8.metric("Currency",    currency)

    st.markdown("---")

    # ── Valuation (Alpha Vantage OVERVIEW) ──────────────
    st.markdown("### 📊 Valuation")
    v1,v2,v3,v4 = st.columns(4)
    v5,v6,v7,v8 = st.columns(4)

    mktcap   = _pick(ov,"MarketCapitalization")
    pe       = _pick(ov,"TrailingPE","ForwardPE")
    pb       = _pick(ov,"PriceToBookRatio")
    ps       = _pick(ov,"PriceToSalesRatioTTM")
    ev       = _pick(ov,"EVToEBITDA")  # AV gives EV/EBITDA directly
    peg      = _pick(ov,"PEGRatio")
    beta     = _pick(ov,"Beta")
    eps      = _pick(ov,"EPS")

    v1.metric("Market Cap",      _money(mktcap))
    v2.metric("P/E (TTM)",       _num(pe)    if pe    else "N/A")
    v3.metric("P/B Ratio",       _num(pb)    if pb    else "N/A")
    v4.metric("P/S Ratio",       _num(ps)    if ps    else "N/A")
    v5.metric("EV/EBITDA",       _num(ev)    if ev    else "N/A")
    v6.metric("PEG Ratio",       _num(peg)   if peg   else "N/A")
    v7.metric("Beta",            _num(beta)  if beta  else "N/A")
    v8.metric("EPS (TTM)",       _num(eps,"$") if eps else "N/A")

    st.markdown("---")

    # ── Dividends ───────────────────────────────────────
    st.markdown("### 💰 Dividends")
    d1,d2,d3,d4 = st.columns(4)

    div_y  = _pick(ov,"DividendYield")
    div_r  = _pick(ov,"DividendPerShare")
    payout = _pick(ov,"PayoutRatio")
    ex_div = _picks(ov,"ExDividendDate")

    d1.metric("Dividend Yield",   _pct(div_y) if div_y else "N/A")
    d2.metric("Dividend/Share",   _num(div_r,"$") if div_r else "N/A")
    d3.metric("Payout Ratio",     _pct(payout) if payout else "N/A")
    d4.metric("Ex-Dividend Date", ex_div or "N/A")

    st.markdown("---")

    # ── Financial Performance ────────────────────────────
    st.markdown("### 📈 Financial Performance")
    f1,f2,f3,f4 = st.columns(4)
    f5,f6,f7,f8 = st.columns(4)

    # OVERVIEW has trailing figures
    rev      = _pick(ov,"RevenueTTM")
    rev_g    = _pick(ov,"QuarterlyRevenueGrowthYOY")
    gm       = _pick(ov,"GrossProfitTTM")  # gross profit absolute; compute margin below
    nm       = _pick(ov,"ProfitMargin")
    om       = _pick(ov,"OperatingMarginTTM")
    ebitda   = _pick(ov,"EBITDA")
    roe      = _pick(ov,"ReturnOnEquityTTM")
    roa      = _pick(ov,"ReturnOnAssetsTTM")

    # Gross margin = GrossProfit / Revenue
    gm_pct = None
    if gm and rev and rev != 0:
        gm_pct = gm / rev

    f1.metric("Revenue (TTM)",     _money(rev))
    f2.metric("Revenue Growth",    _pct(rev_g) if rev_g else "N/A")
    f3.metric("Gross Margin",      _pct(gm_pct) if gm_pct else "N/A")
    f4.metric("Net Profit Margin", _pct(nm) if nm else "N/A")
    f5.metric("EBITDA",            _money(ebitda))
    f6.metric("Operating Margin",  _pct(om) if om else "N/A")
    f7.metric("Return on Equity",  _pct(roe) if roe else "N/A")
    f8.metric("Return on Assets",  _pct(roa) if roa else "N/A")

    st.markdown("---")

    # ── Balance Sheet (latest annual report) ────────────
    st.markdown("### 🏦 Balance Sheet")
    b1,b2,b3,b4 = st.columns(4)
    b5,b6,b7,b8 = st.columns(4)

    cash   = _pick(bal_r,"cashAndCashEquivalentsAtCarryingValue","cashAndShortTermInvestments")
    debt   = _pick(bal_r,"shortLongTermDebtTotal","longTermDebt")
    assets = _pick(bal_r,"totalAssets")
    equity = _pick(bal_r,"totalShareholderEquity")
    cr     = _pick(ov,"CurrentRatio")
    qr     = _pick(ov,"QuickRatio")
    de     = _pick(ov,"DebtToEquityRatio")
    bv     = _pick(ov,"BookValue")

    b1.metric("Total Cash",       _money(cash))
    b2.metric("Total Debt",       _money(debt))
    b3.metric("Total Assets",     _money(assets))
    b4.metric("Shareholders Eq.", _money(equity))
    b5.metric("Current Ratio",    _num(cr)     if cr  else "N/A")
    b6.metric("Quick Ratio",      _num(qr)     if qr  else "N/A")
    b7.metric("Debt / Equity",    _num(de)     if de  else "N/A")
    b8.metric("Book Value/Share", _num(bv,"$") if bv  else "N/A")

    st.markdown("---")

    # ── Cash Flow ────────────────────────────────────────
    st.markdown("### 💵 Cash Flow")
    c1f,c2f,c3f,c4f = st.columns(4)

    op_cf  = _pick(cf_r,"operatingCashflow")
    capex  = _pick(cf_r,"capitalExpenditures")
    fcf    = None
    if op_cf and capex:
        fcf = op_cf - abs(capex)
    div_cf = _pick(cf_r,"dividendPayout")

    c1f.metric("Operating CF",   _money(op_cf))
    c2f.metric("CapEx",          _money(capex))
    c3f.metric("Free Cash Flow", _money(fcf))
    c4f.metric("Dividends Paid", _money(div_cf))

    st.markdown("---")

    # ── 52-Week & Analyst ─────────────────────────────────
    st.markdown("### 📅 52-Week Range & Analyst Targets")
    w1,w2,w3,w4 = st.columns(4)
    w5,w6,w7,w8 = st.columns(4)

    hi52   = _pick(ov,"52WeekHigh")
    lo52   = _pick(ov,"52WeekLow")
    target = _pick(ov,"AnalystTargetPrice")
    shares = _pick(ov,"SharesOutstanding")
    sect   = _picks(ov,"Sector")
    ind    = _picks(ov,"Industry")
    emp    = _pick(ov,"FullTimeEmployees")
    exch   = _picks(ov,"Exchange")

    # Fallback 52w from price history
    if hi52 is None and not hist.empty:
        sub  = hist[hist.index >= hist.index.max()-pd.Timedelta(days=365)]
        if not sub.empty:
            hi52 = float(sub["High"].max())
            lo52 = float(sub["Low"].min())

    w1.metric("52-Week High",       f"{hi52:.2f}"   if hi52   else "N/A")
    w2.metric("52-Week Low",        f"{lo52:.2f}"   if lo52   else "N/A")
    w3.metric("Analyst Target",     f"{target:.2f}" if target else "N/A")
    w4.metric("Shares Outstanding", _money(shares).replace("$","") if shares else "N/A")
    w5.metric("Sector",             sect or "N/A")
    w6.metric("Industry",           ind  or "N/A")
    w7.metric("Exchange",           exch or td_ex or "N/A")
    w8.metric("Employees",          f"{int(emp):,}" if emp else "N/A")

    # No-data note
    if not any([mktcap, pe, rev, eps]) and not is_saudi:
        if not _av_key():
            st.warning(
                "⚠️ **ALPHAVANTAGE_API_KEY** is missing — fundamental data cannot be loaded.\n\n"
                "Get a free key at [alphavantage.co](https://www.alphavantage.co/support/#api-key) "
                "and add it to **Settings → Secrets**:\n"
                "```\nALPHAVANTAGE_API_KEY = \"your_key_here\"\n```"
            )
        else:
            st.info(
                "ℹ️ Fundamental data returned empty. "
                "Alpha Vantage free plan allows **25 requests/day**. "
                "You may have reached the daily limit — try again tomorrow, or "
                "upgrade at [alphavantage.co/premium](https://www.alphavantage.co/premium/)."
            )

# ════════════════════════════════════════════════════════
# TAB 3 — AI Forecast
# ════════════════════════════════════════════════════════
with tab3:
    st.subheader("🔮 AI-Powered Price Forecast")
    horizon = st.slider("Forecast horizon (days)", 7, 90, 30, 1)

    cdf = (hist.reset_index()
               .rename(columns={hist.index.name or "index": "Date"})
               [["Date","Close"]].dropna())

    if cdf.empty or cdf["Close"].nunique() < 10:
        st.warning("Not enough data to forecast.")
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
                dfp["ds"] = pd.to_datetime(dfp["ds"]).dt.tz_localize(None)
                m  = Prophet(daily_seasonality=False,
                             weekly_seasonality=True, yearly_seasonality=True)
                m.fit(dfp)
                fc = m.predict(m.make_future_dataframe(periods=horizon))
                ld = dfp["ds"].max()
                fig = go.Figure([
                    go.Scatter(x=dfp["ds"], y=dfp["y"], mode="lines", name="Actual"),
                    go.Scatter(x=fc["ds"],  y=fc["yhat"], mode="lines", name="Forecast"),
                    go.Scatter(x=fc["ds"],  y=fc["yhat_upper"], mode="lines",
                               line=dict(width=0), showlegend=False, hoverinfo="skip"),
                    go.Scatter(x=fc["ds"],  y=fc["yhat_lower"], mode="lines",
                               fill="tonexty", fillcolor="rgba(100,180,255,0.15)",
                               line=dict(width=0), showlegend=False, hoverinfo="skip"),
                ])
                fig.update_layout(title=f"{td_sym} — {horizon}-day Forecast",
                    xaxis_title="Date", yaxis_title=f"Price ({currency})",
                    height=550, hovermode="x unified", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
                tbl = fc[fc["ds"]>ld][["ds","yhat","yhat_lower","yhat_upper"]]
                tbl.columns = ["Date","Forecast","Low CI","High CI"]
                st.dataframe(tbl.tail(30), use_container_width=True)
            except Exception as e:
                st.warning(f"Prophet error: {e} — falling back to linear trend.")
                use_p = False

        if not use_p:
            st.info("Using linear trend forecast ✅")
            y    = cdf["Close"].values.astype(float)
            x    = np.arange(len(y), dtype=float)
            coef = np.polyfit(x, y, 1)
            tr   = np.poly1d(coef)
            xa   = np.arange(len(y)+horizon, dtype=float)
            yhat = tr(xa)
            ld   = pd.to_datetime(cdf["Date"].max())
            fd   = pd.date_range(ld, periods=horizon+1, freq="D")[1:]
            ad   = pd.concat([pd.to_datetime(cdf["Date"]),
                               pd.Series(fd)], ignore_index=True)
            fig  = go.Figure([
                go.Scatter(x=pd.to_datetime(cdf["Date"]), y=cdf["Close"],
                           mode="lines", name="Actual"),
                go.Scatter(x=ad, y=yhat, mode="lines", name="Trend",
                           line=dict(dash="dash", color="orange")),
            ])
            fig.update_layout(title=f"{td_sym} — {horizon}-day Trend Forecast",
                xaxis_title="Date", yaxis_title=f"Price ({currency})",
                height=550, hovermode="x unified", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

    st.caption("⚠️ Forecasts are experimental. Not financial advice.")

st.markdown("---")
st.caption("⚠️ Educational purposes only. Not financial advice.")
