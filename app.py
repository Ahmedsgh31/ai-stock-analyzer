"""
AI-Powered Stock Market Analyzer
──────────────────────────────────
PRIMARY  : yfinance  – price history + fundamentals (free, no API key needed)
OPTIONAL : Twelve Data – enhances live quote if TWELVEDATA_API_KEY is set
           (not required – app works 100% without it)
"""

import time, random, json
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

# ── Page config ──────────────────────────────────────────
st.set_page_config(page_title="AI Stock Analyzer", page_icon="📈", layout="wide")
st.title("📈 AI-Powered Stock Market Analyzer")
st.markdown("---")

# ── Formatting helpers ───────────────────────────────────
def _sf(x):
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
    if abs(v) < 2.0: v *= 100
    return f"{v:.2f}%"

def _num(x, pre="", suf="", dec=2):
    v = _sf(x)
    if v is None: return "N/A"
    return f"{pre}{v:,.{dec}f}{suf}"

def _pick(d: dict, *keys):
    """Return first non-None numeric value from dict."""
    for k in keys:
        v = _sf(d.get(k))
        if v is not None: return v
    return None

def _picks(d: dict, *keys):
    """Return first non-empty string value from dict."""
    for k in keys:
        v = d.get(k)
        if v and str(v).strip().lower() not in ("", "none", "n/a", "null", "nan"):
            return str(v).strip()
    return None

# ── yfinance import (always available) ───────────────────
@st.cache_resource
def _yf():
    import yfinance as yf
    return yf

# ── Randomised request session ───────────────────────────
_UAS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
]

@st.cache_resource
def _sess():
    s = requests.Session()
    s.headers.update({
        "User-Agent": random.choice(_UAS),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
    })
    return s

# ════════════════════════════════════════════════════════
# PRICE HISTORY  — yfinance (primary)
# ════════════════════════════════════════════════════════
@st.cache_data(ttl=600, show_spinner=False)
def get_price_history(yf_sym: str, period: str) -> pd.DataFrame:
    """
    Fetch OHLCV with three yfinance strategies so at least one succeeds
    even when Streamlit Cloud IPs are partially rate-limited.
    """
    yf = _yf()
    df = pd.DataFrame()

    # Strategy 1 – Ticker.history (most reliable on cloud)
    try:
        t  = yf.Ticker(yf_sym)
        df = t.history(period=period, interval="1d", auto_adjust=True)
    except Exception:
        pass

    # Strategy 2 – yf.download
    if df is None or df.empty:
        try:
            df = yf.download(
                yf_sym, period=period, interval="1d",
                auto_adjust=True, progress=False,
                threads=False, group_by="column",
            )
        except Exception:
            pass

    # Strategy 3 – short sleep then retry Ticker.history
    if df is None or df.empty:
        time.sleep(1.5)
        try:
            t  = yf.Ticker(yf_sym)
            df = t.history(period=period, interval="1d", auto_adjust=True)
        except Exception:
            return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # Flatten multi-index columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Normalise column names
    df.columns = [c.strip().title() for c in df.columns]

    for col in ["Open", "High", "Low", "Close"]:
        if col not in df.columns:
            return pd.DataFrame()
    if "Volume" not in df.columns:
        df["Volume"] = 0

    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.dropna(subset=["Close"])


# ════════════════════════════════════════════════════════
# FUNDAMENTALS  — Yahoo quoteSummary (direct HTTP)
# Bypasses yfinance .info rate-limits on Streamlit Cloud
# ════════════════════════════════════════════════════════
_YQ_URL = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/{}"
_YQ_MODULES = (
    "assetProfile,summaryDetail,financialData,defaultKeyStatistics,"
    "incomeStatementHistory,balanceSheetHistory,cashflowStatementHistory,"
    "earningsTrend,price"
)

@st.cache_data(ttl=900, show_spinner=False)
def get_fundamentals(yf_sym: str) -> dict:
    """
    Fetch fundamentals directly from Yahoo Finance quoteSummary API.
    Falls back to yfinance Ticker properties if HTTP call fails.
    """
    result = {}

    # ── Method 1: direct Yahoo Finance HTTP ──
    try:
        url  = _YQ_URL.format(yf_sym)
        resp = _sess().get(
            url,
            params={"modules": _YQ_MODULES, "formatted": "false",
                    "lang": "en-US", "region": "US"},
            timeout=20,
        )
        if resp.status_code == 200:
            data = resp.json()
            qs   = (data.get("quoteSummary") or {}).get("result") or []
            if qs:
                raw = qs[0]

                def _deep(d, *keys):
                    for k in keys:
                        v = d.get(k) if isinstance(d, dict) else None
                        if isinstance(v, dict):
                            # Yahoo wraps values as {"raw": 123, "fmt": "123"}
                            rv = v.get("raw")
                            if rv is not None:
                                result[k] = rv
                                break
                        elif v is not None:
                            result[k] = v
                            break

                # price module (most reliable for live data)
                price = raw.get("price") or {}
                for k in ["regularMarketPrice","regularMarketOpen",
                          "regularMarketDayHigh","regularMarketDayLow",
                          "regularMarketPreviousClose","regularMarketVolume",
                          "averageDailyVolume10Day","marketCap","currency",
                          "shortName","longName","exchangeName",
                          "regularMarketChange","regularMarketChangePercent"]:
                    v = price.get(k)
                    if isinstance(v, dict): v = v.get("raw")
                    if v is not None: result[k] = v

                # summaryDetail
                sd = raw.get("summaryDetail") or {}
                for k in ["trailingPE","forwardPE","dividendYield",
                          "dividendRate","exDividendDate","payoutRatio",
                          "beta","fiftyTwoWeekHigh","fiftyTwoWeekLow",
                          "averageVolume","volume","marketCap",
                          "priceToSalesTrailing12Months"]:
                    v = sd.get(k)
                    if isinstance(v, dict): v = v.get("raw")
                    if v is not None and k not in result: result[k] = v

                # financialData
                fd = raw.get("financialData") or {}
                for k in ["currentPrice","targetMeanPrice","targetMedianPrice",
                          "recommendationKey","totalRevenue","revenueGrowth",
                          "grossMargins","operatingMargins","profitMargins",
                          "ebitda","returnOnEquity","returnOnAssets",
                          "totalCash","totalDebt","debtToEquity",
                          "currentRatio","quickRatio","freeCashflow",
                          "earningsGrowth"]:
                    v = fd.get(k)
                    if isinstance(v, dict): v = v.get("raw")
                    if v is not None: result[k] = v

                # defaultKeyStatistics
                ks = raw.get("defaultKeyStatistics") or {}
                for k in ["trailingEps","forwardEps","pegRatio",
                          "priceToBook","enterpriseValue","enterpriseToEbitda",
                          "bookValue","sharesOutstanding","floatShares",
                          "shortRatio","heldPercentInsiders","heldPercentInstitutions"]:
                    v = ks.get(k)
                    if isinstance(v, dict): v = v.get("raw")
                    if v is not None: result[k] = v

                # assetProfile
                ap = raw.get("assetProfile") or {}
                for k in ["sector","industry","country","website",
                          "longBusinessSummary","fullTimeEmployees"]:
                    v = ap.get(k)
                    if v: result[k] = v

    except Exception:
        pass

    # ── Method 2: yfinance fallback (if HTTP gave us nothing) ──
    if not result:
        try:
            yf = _yf()
            t  = yf.Ticker(yf_sym)
            try:
                fi = t.fast_info
                for attr in ["market_cap","shares_outstanding","last_price",
                              "previous_close","fifty_two_week_high",
                              "fifty_two_week_low","currency"]:
                    v = getattr(fi, attr, None)
                    if v is not None: result[attr] = v
            except Exception:
                pass
            try:
                info = t.info or {}
                for k, v in info.items():
                    if k not in result and v is not None:
                        result[k] = v
            except Exception:
                pass
        except Exception:
            pass

    return result


# ════════════════════════════════════════════════════════
# TWELVE DATA  (optional — live quote enhancement only)
# ════════════════════════════════════════════════════════
def _td_key():
    try:
        return st.secrets.get("TWELVEDATA_API_KEY") or None
    except Exception:
        return None

@st.cache_data(ttl=60, show_spinner=False)
def td_quote_optional(symbol: str) -> dict:
    k = _td_key()
    if not k: return {}
    try:
        r = _sess().get(
            "https://api.twelvedata.com/quote",
            params={"symbol": symbol, "format": "JSON", "apikey": k},
            timeout=10,
        )
        d = r.json()
        return {} if d.get("status") == "error" else d
    except Exception:
        return {}

@st.cache_data(ttl=3600, show_spinner=False)
def td_resolve_optional(symbol: str) -> dict | None:
    k = _td_key()
    if not k: return None
    try:
        base  = symbol.replace(".SR","") if symbol.endswith(".SR") else symbol
        items = _sess().get(
            "https://api.twelvedata.com/symbol_search",
            params={"symbol": base, "outputsize": 30, "apikey": k},
            timeout=10,
        ).json().get("data") or []
        if not items: return None
        is_sa = symbol.endswith(".SR") or (symbol.isdigit() and len(symbol) in (3,4,5))
        def sc(it):
            s,ex,cur,ct = (it.get("symbol","").upper(), it.get("exchange","").lower(),
                           it.get("currency","").upper(), it.get("country","").lower())
            sc2 = 10 if it.get("instrument_type","").lower()=="common stock" else 0
            if is_sa:
                if "saudi" in ct or "tadawul" in ex: sc2 += 60
                if cur == "SAR": sc2 += 30
                if s == base: sc2 += 40
            else:
                if s == symbol: sc2 += 60
                if ex in ("nasdaq","nyse"): sc2 += 20
            return sc2
        return sorted(items, key=sc, reverse=True)[0]
    except Exception:
        return None


# ════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════
with st.sidebar:
    st.header("Search Stock")
    stock_input = st.text_input(
        "Enter Stock Symbol",
        placeholder="AAPL · TSLA · 2222.SR",
        help="US: AAPL TSLA NVDA MSFT | Saudi: 2222.SR 1120.SR 2010.SR",
    )
    period_opts = {
        "1 Month": "1mo", "3 Months": "3mo", "6 Months": "6mo",
        "1 Year": "1y",   "2 Years": "2y",   "5 Years": "5y",
    }
    sel_period  = st.selectbox("Time Period", list(period_opts.keys()), index=3)
    show_debug  = st.checkbox("Show debug info", value=False)
    go_btn      = st.button("🔍 Analyze Stock", type="primary", use_container_width=True)

if "result" not in st.session_state:
    st.session_state.result = None

# ════════════════════════════════════════════════════════
# ON BUTTON PRESS
# ════════════════════════════════════════════════════════
if go_btn:
    raw = (stock_input or "").strip().upper()
    if not raw:
        st.warning("⚠️ Please enter a stock symbol."); st.stop()

    # Build Yahoo symbol
    is_saudi = raw.endswith(".SR") or (raw.isdigit() and len(raw) in (3, 4, 5))
    yf_sym   = f"{raw.replace('.SR','')}.SR" if is_saudi else raw

    # ── Fetch price ──────────────────────────────────────
    with st.spinner(f"Loading price data for {yf_sym}…"):
        hist = get_price_history(yf_sym, period_opts[sel_period])

    if hist is None or hist.empty:
        st.error(f"❌ Could not load price data for **{yf_sym}**.")
        st.warning(
            "**Possible fixes:**\n"
            "1. Double-check the ticker symbol (e.g. `AAPL`, `TSLA`, `2222.SR`)\n"
            "2. Wait 30 seconds and click **Analyze Stock** again — "
            "Yahoo Finance occasionally rate-limits Streamlit Cloud IPs temporarily.\n"
            "3. Try a slightly different period (e.g. '6 Months' instead of '1 Year')."
        )
        st.stop()

    # ── Resolve display name (optional TD, fallback later) ──
    td_sym = td_ex = None
    resolved = td_resolve_optional(raw)
    if resolved:
        td_sym = resolved.get("symbol") or raw
        td_ex  = resolved.get("exchange") or None

    st.session_state.result = {
        "raw": raw, "yf_sym": yf_sym,
        "td_sym": td_sym, "td_ex": td_ex,
        "hist": hist, "is_saudi": is_saudi,
        "period": sel_period,
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
| NASDAQ / NYSE | `AAPL` `TSLA` `NVDA` `MSFT` `AMZN` `GOOGL` |
| Saudi Tadawul | `2222.SR` `1120.SR` `2010.SR` `1211.SR` |

> Works without any API key. Add `TWELVEDATA_API_KEY` to Secrets for enhanced live quotes.
""")
    st.stop()

yf_sym  = res["yf_sym"]
raw     = res["raw"]
td_sym  = res["td_sym"]
td_ex   = res["td_ex"]
hist    = res["hist"]

# ── Load fundamentals (always from Yahoo) ────────────────
with st.spinner("Loading fundamentals…"):
    fund = get_fundamentals(yf_sym)

# ── Optional TD quote ─────────────────────────────────────
td_q = td_quote_optional(td_sym or raw)

# ── Derive display name & currency ───────────────────────
name     = (_picks(fund, "longName","shortName") or
            (td_sym if td_sym else raw))
currency = (_picks(fund,"currency") or
            ("SAR" if res["is_saudi"] else "USD"))
exchange = (_picks(fund,"exchangeName") or td_ex or "—")

if show_debug:
    with st.expander("🔧 fund dict"):  st.json(fund)
    with st.expander("🔧 td_q dict"):  st.json(td_q)

# ── Header row ────────────────────────────────────────────
c1,c2,c3,c4 = st.columns([2,1,1,1])
with c1:
    st.subheader(name)
    st.caption(f"**{raw}** · {exchange} · {currency}")

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
# TAB 1 — Chart
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
            yaxis="y2", opacity=0.25,
            marker_color="rgba(100,180,255,0.5)",
        ))
    fig.update_layout(
        title=f"{raw} — {res['period']}",
        yaxis_title=f"Price ({currency})",
        yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False),
        xaxis_title="Date", height=600, hovermode="x unified",
        template="plotly_dark", xaxis_rangeslider_visible=False,
    )
    st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════
# TAB 2 — Financial Metrics
# ════════════════════════════════════════════════════════
with tab2:
    st.subheader("Financial Metrics")

    # ── Live Quote ──────────────────────────────────────
    st.markdown("### 📌 Live Quote")
    q1,q2,q3,q4 = st.columns(4)

    # Price: prefer TD quote, fallback to Yahoo fund, fallback to last close
    last_p = (_sf(td_q.get("close") or td_q.get("price")) or
              _pick(fund,"currentPrice","regularMarketPrice") or
              (float(close_s.iloc[-1]) if len(close_s) else None))
    chg    = (_sf(td_q.get("change")) or _pick(fund,"regularMarketChange"))
    chg_pc = (_sf(td_q.get("percent_change")) or
              _sf(fund.get("regularMarketChangePercent")))
    if chg_pc and abs(chg_pc) < 1:   # Yahoo returns as ratio e.g. -0.014
        chg_pc *= 100
    if chg is None and last_p:
        prev = _pick(fund,"regularMarketPreviousClose","previousClose")
        if prev:
            chg   = last_p - prev
            chg_pc = chg/prev*100

    vol    = _pick(fund,"regularMarketVolume","volume")
    avgvol = _pick(fund,"averageDailyVolume10Day","averageVolume")
    open_p = _pick(fund,"regularMarketOpen","open")
    hi_d   = _pick(fund,"regularMarketDayHigh","dayHigh")
    lo_d   = _pick(fund,"regularMarketDayLow","dayLow")
    prev_c = _pick(fund,"regularMarketPreviousClose","previousClose")

    delta  = f"{chg:+.2f} ({chg_pc:+.2f}%)" if chg is not None and chg_pc is not None else None
    q1.metric("Last Price", f"{last_p:.2f}" if last_p else "N/A", delta)
    q2.metric("Open",       _num(open_p) if open_p else "N/A")
    q3.metric("Day High",   _num(hi_d)   if hi_d   else "N/A")
    q4.metric("Day Low",    _num(lo_d)   if lo_d   else "N/A")

    q5,q6,q7,q8 = st.columns(4)
    q5.metric("Prev Close", _num(prev_c)       if prev_c  else "N/A")
    q6.metric("Volume",     f"{int(vol):,}"     if vol     else "N/A")
    q7.metric("Avg Volume", f"{int(avgvol):,}"  if avgvol  else "N/A")
    q8.metric("Currency",   currency)

    st.markdown("---")

    # ── Valuation ───────────────────────────────────────
    st.markdown("### 📊 Valuation")
    v1,v2,v3,v4 = st.columns(4)
    v5,v6,v7,v8 = st.columns(4)

    mktcap   = _pick(fund,"marketCap","market_cap")
    pe       = _pick(fund,"trailingPE","forwardPE")
    pb       = _pick(fund,"priceToBook")
    ps       = _pick(fund,"priceToSalesTrailing12Months")
    ev       = _pick(fund,"enterpriseValue")
    evebitda = _pick(fund,"enterpriseToEbitda")
    peg      = _pick(fund,"pegRatio")
    beta     = _pick(fund,"beta")

    v1.metric("Market Cap",      _money(mktcap))
    v2.metric("P/E (TTM)",       _num(pe)       if pe       else "N/A")
    v3.metric("P/B Ratio",       _num(pb)       if pb       else "N/A")
    v4.metric("P/S Ratio",       _num(ps)       if ps       else "N/A")
    v5.metric("Enterprise Value",_money(ev))
    v6.metric("EV/EBITDA",       _num(evebitda) if evebitda else "N/A")
    v7.metric("PEG Ratio",       _num(peg)      if peg      else "N/A")
    v8.metric("Beta",            _num(beta)     if beta     else "N/A")

    st.markdown("---")

    # ── Dividends ───────────────────────────────────────
    st.markdown("### 💰 Dividends")
    d1,d2,d3,d4 = st.columns(4)

    div_y  = _pick(fund,"dividendYield","trailingAnnualDividendYield")
    div_r  = _pick(fund,"dividendRate","trailingAnnualDividendRate")
    payout = _pick(fund,"payoutRatio")
    ex_div = fund.get("exDividendDate")
    if ex_div:
        try:
            ex_div = (pd.to_datetime(ex_div, unit="s").strftime("%Y-%m-%d")
                      if isinstance(ex_div,(int,float)) else str(ex_div)[:10])
        except Exception:
            ex_div = str(ex_div)

    d1.metric("Dividend Yield",   _pct(div_y)         if div_y  else "N/A")
    d2.metric("Dividend Rate",    _num(div_r,"$")      if div_r  else "N/A")
    d3.metric("Payout Ratio",     _pct(payout)         if payout else "N/A")
    d4.metric("Ex-Dividend Date", ex_div               or "N/A")

    st.markdown("---")

    # ── Financial Performance ────────────────────────────
    st.markdown("### 📈 Financial Performance")
    f1,f2,f3,f4 = st.columns(4)
    f5,f6,f7,f8 = st.columns(4)

    rev    = _pick(fund,"totalRevenue")
    rev_g  = _pick(fund,"revenueGrowth")
    gm     = _pick(fund,"grossMargins")
    nm     = _pick(fund,"profitMargins")
    ebitda = _pick(fund,"ebitda")
    eps    = _pick(fund,"trailingEps","forwardEps")
    roe    = _pick(fund,"returnOnEquity")
    roa    = _pick(fund,"returnOnAssets")

    f1.metric("Revenue (TTM)",     _money(rev))
    f2.metric("Revenue Growth",    _pct(rev_g) if rev_g else "N/A")
    f3.metric("Gross Margin",      _pct(gm)    if gm    else "N/A")
    f4.metric("Net Profit Margin", _pct(nm)    if nm    else "N/A")
    f5.metric("EBITDA",            _money(ebitda))
    f6.metric("EPS (TTM)",         _num(eps,"$") if eps  else "N/A")
    f7.metric("Return on Equity",  _pct(roe)   if roe   else "N/A")
    f8.metric("Return on Assets",  _pct(roa)   if roa   else "N/A")

    st.markdown("---")

    # ── Balance Sheet ────────────────────────────────────
    st.markdown("### 🏦 Balance Sheet")
    b1,b2,b3,b4 = st.columns(4)
    b5,b6,b7,b8 = st.columns(4)

    cash   = _pick(fund,"totalCash")
    debt   = _pick(fund,"totalDebt")
    de     = _pick(fund,"debtToEquity")
    cr     = _pick(fund,"currentRatio")
    qr     = _pick(fund,"quickRatio")
    bv     = _pick(fund,"bookValue")
    assets = _pick(fund,"totalAssets","total_assets_q")
    fcf    = _pick(fund,"freeCashflow")

    b1.metric("Total Cash",       _money(cash))
    b2.metric("Total Debt",       _money(debt))
    b3.metric("Debt / Equity",    _num(de)       if de    else "N/A")
    b4.metric("Current Ratio",    _num(cr)       if cr    else "N/A")
    b5.metric("Quick Ratio",      _num(qr)       if qr    else "N/A")
    b6.metric("Book Value/Share", _num(bv,"$")   if bv    else "N/A")
    b7.metric("Total Assets",     _money(assets))
    b8.metric("Free Cash Flow",   _money(fcf))

    st.markdown("---")

    # ── 52-Week & Targets ───────────────────────────────
    st.markdown("### 📅 52-Week Range & Analyst Targets")
    w1,w2,w3,w4 = st.columns(4)

    hi52   = _pick(fund,"fiftyTwoWeekHigh","fifty_two_week_high")
    lo52   = _pick(fund,"fiftyTwoWeekLow", "fifty_two_week_low")
    target = _pick(fund,"targetMeanPrice","targetMedianPrice")
    rec    = _picks(fund,"recommendationKey")
    shares = _pick(fund,"sharesOutstanding","floatShares","shares_outstanding")

    # Compute from price history as fallback
    if hi52 is None and not hist.empty:
        sub  = hist[hist.index >= hist.index.max()-pd.Timedelta(days=365)]
        if not sub.empty:
            hi52 = float(sub["High"].max())
            lo52 = float(sub["Low"].min())

    w1.metric("52-Week High",       f"{hi52:.2f}"   if hi52   else "N/A")
    w2.metric("52-Week Low",        f"{lo52:.2f}"   if lo52   else "N/A")
    w3.metric("Analyst Target",     f"{target:.2f}" if target else "N/A")
    w4.metric("Analyst Rating",     rec.upper()     if rec    else "N/A")

    w5,w6 = st.columns(2)
    w5.metric("Shares Outstanding", _money(shares).replace("$","") if shares else "N/A")
    sect = _picks(fund,"sector"); ind = _picks(fund,"industry")
    w6.metric("Sector / Industry",  f"{sect} / {ind}" if sect else "N/A")

    # ── Data coverage note ───────────────────────────────
    has_data = any([mktcap, pe, rev, eps, div_y, beta])
    if not has_data:
        st.warning(
            "⚠️ Fundamental data returned empty for this symbol.\n\n"
            "**Try one of these:**\n"
            "- Click **Analyze Stock** again (Yahoo sometimes rate-limits on first attempt)\n"
            "- Wait 30–60 seconds and retry\n"
            "- Saudi stocks may have limited fundamental coverage on Yahoo Finance"
        )


# ════════════════════════════════════════════════════════
# TAB 3 — AI Forecast
# ════════════════════════════════════════════════════════
with tab3:
    st.subheader("🔮 AI-Powered Price Forecast")
    horizon = st.slider("Forecast horizon (days)", 7, 90, 30, 1)

    cdf = (hist.reset_index()
               .rename(columns={hist.index.name or "index":"Date"})
               [["Date","Close"]]
               .dropna())

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
                m   = Prophet(daily_seasonality=False,
                              weekly_seasonality=True, yearly_seasonality=True)
                m.fit(dfp)
                fc  = m.predict(m.make_future_dataframe(periods=horizon))
                ld  = dfp["ds"].max()

                fig = go.Figure([
                    go.Scatter(x=dfp["ds"],y=dfp["y"],mode="lines",name="Actual"),
                    go.Scatter(x=fc["ds"], y=fc["yhat"],mode="lines",name="Forecast"),
                    go.Scatter(x=fc["ds"], y=fc["yhat_upper"],mode="lines",
                               line=dict(width=0),showlegend=False,hoverinfo="skip"),
                    go.Scatter(x=fc["ds"], y=fc["yhat_lower"],mode="lines",
                               fill="tonexty",fillcolor="rgba(100,180,255,0.15)",
                               line=dict(width=0),showlegend=False,hoverinfo="skip"),
                ])
                fig.update_layout(title=f"{raw} — {horizon}-day Forecast",
                    xaxis_title="Date",yaxis_title=f"Price ({currency})",
                    height=550,hovermode="x unified",template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

                tbl = fc[fc["ds"]>ld][["ds","yhat","yhat_lower","yhat_upper"]]
                tbl.columns = ["Date","Forecast","Low CI","High CI"]
                st.dataframe(tbl.tail(30), use_container_width=True)
            except Exception as e:
                st.warning(f"Prophet error: {e} — using linear fallback.")
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
                go.Scatter(x=pd.to_datetime(cdf["Date"]),y=cdf["Close"],
                           mode="lines",name="Actual"),
                go.Scatter(x=ad,y=yhat,mode="lines",name="Trend",
                           line=dict(dash="dash",color="orange")),
            ])
            fig.update_layout(title=f"{raw} — {horizon}-day Trend Forecast",
                xaxis_title="Date",yaxis_title=f"Price ({currency})",
                height=550,hovermode="x unified",template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

    st.caption("⚠️ Forecasts are experimental. Not financial advice.")

st.markdown("---")
st.caption("⚠️ Educational purposes only. Not financial advice.")
