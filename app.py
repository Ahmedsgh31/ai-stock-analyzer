import time
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf

# =============================
# Page config
# =============================
st.set_page_config(
    page_title="AI Stock Analyzer",
    page_icon="📈",
    layout="wide",
)

st.title("📈 AI-Powered Stock Market Analyzer")
st.markdown("---")

# =============================
# Helpers
# =============================
def _safe_float(x):
    try:
        if x is None:
            return None
        if isinstance(x, (int, float, np.number)):
            return float(x)
        return float(str(x).replace(",", "").strip())
    except Exception:
        return None

def _human_money(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    x = float(x)
    absx = abs(x)
    if absx >= 1e12:
        return f"${x/1e12:.2f}T"
    if absx >= 1e9:
        return f"${x/1e9:.2f}B"
    if absx >= 1e6:
        return f"${x/1e6:.2f}M"
    if absx >= 1e3:
        return f"${x/1e3:.2f}K"
    return f"${x:.2f}"

def _human_pct(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    if x > 1.5:
        return f"{x:.2f}%"
    return f"{x*100:.2f}%"

# =============================
# Twelve Data client
# =============================
TD_BASE = "https://api.twelvedata.com"

def td_key() -> str | None:
    return st.secrets.get("TWELVEDATA_API_KEY", None)

@st.cache_resource
def http_session():
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0 Safari/537.36"
            )
        }
    )
    return s

def td_get(endpoint: str, params: dict, timeout=20) -> dict:
    key = td_key()
    if not key:
        return {"status": "error", "message": "Missing TWELVEDATA_API_KEY in Streamlit Secrets."}

    p = dict(params)
    p["apikey"] = key
    url = f"{TD_BASE}/{endpoint.lstrip('/')}"
    r = http_session().get(url, params=p, timeout=timeout)

    try:
        return r.json()
    except Exception:
        return {"status": "error", "message": f"Non-JSON response: {r.text[:200]}"}

def td_symbol_search(user_symbol: str) -> dict | None:
    s = user_symbol.strip().upper()
    raw = s

    if s.endswith(".SR"):
        s = s.replace(".SR", "")

    data = td_get("symbol_search", {"symbol": s, "outputsize": 30})
    if data.get("status") == "error":
        return None
    items = data.get("data") or []

    if not items:
        data2 = td_get("symbol_search", {"keywords": raw, "outputsize": 30})
        if data2.get("status") == "error":
            return None
        items = data2.get("data") or []
        if not items:
            return None

    def score(it: dict) -> int:
        sc = 0
        ex = (it.get("exchange") or "").lower()
        country = (it.get("country") or "").lower()
        currency = (it.get("currency") or "").upper()
        sym = (it.get("symbol") or "").upper()

        if raw.endswith(".SR"):
            if "saudi" in country or "saudi" in ex or "tadawul" in ex:
                sc += 50
            if currency == "SAR":
                sc += 20
            if sym.startswith(s.replace(".SR", "")):
                sc += 10

        if sym == raw:
            sc += 40
        if sym == s:
            sc += 25
        return sc

    best = sorted(items, key=score, reverse=True)[0]
    return best

@st.cache_data(ttl=600)
def td_time_series(symbol: str, interval: str, outputsize: int = 260) -> tuple[pd.DataFrame, dict]:
    data = td_get("time_series", {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "format": "JSON",
    })

    if data.get("status") == "error":
        return pd.DataFrame(), data

    values = data.get("values") or []
    if not values:
        return pd.DataFrame(), data

    df = pd.DataFrame(values)
    if "datetime" not in df.columns:
        return pd.DataFrame(), data

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").set_index("datetime")

    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    })

    for col in ["Open", "High", "Low", "Close"]:
        if col not in df.columns:
            return pd.DataFrame(), data
    if "Volume" not in df.columns:
        df["Volume"] = 0

    df = df.dropna(subset=["Close"])
    return df, data

@st.cache_data(ttl=600)
def td_quote(symbol: str) -> dict:
    data = td_get("quote", {"symbol": symbol, "format": "JSON"})
    return {} if data.get("status") == "error" else data

@st.cache_data(ttl=600)
def td_statistics(symbol: str) -> dict:
    data = td_get("statistics", {"symbol": symbol, "format": "JSON"})
    return {} if data.get("status") == "error" else data

@st.cache_data(ttl=600)
def td_profile(symbol: str) -> dict:
    data = td_get("profile", {"symbol": symbol, "format": "JSON"})
    return {} if data.get("status") == "error" else data

# =============================
# Saudi stock via Stooq (free, no API key)
# =============================
@st.cache_data(ttl=600)
def stooq_price_history(symbol: str, period_days: int) -> pd.DataFrame:
    """
    Fetch OHLCV from Stooq for Saudi stocks.
    Stooq uses symbol format like '2222.sa' for Saudi.
    """
    # Convert 2222.SR -> 2222.sa for stooq
    stooq_sym = symbol.lower().replace(".sr", ".sa")
    
    end = pd.Timestamp.today()
    start = end - pd.Timedelta(days=period_days)
    
    url = (
        f"https://stooq.com/q/d/l/"
        f"?s={stooq_sym}"
        f"&d1={start.strftime('%Y%m%d')}"
        f"&d2={end.strftime('%Y%m%d')}"
        f"&i=d"
    )
    
    try:
        sess = http_session()
        r = sess.get(url, timeout=20)
        if r.status_code != 200 or "No data" in r.text or len(r.text) < 50:
            return pd.DataFrame()
        
        from io import StringIO
        df = pd.read_csv(StringIO(r.text))
        
        if df.empty or "Date" not in df.columns:
            return pd.DataFrame()
        
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").set_index("Date")
        
        # Stooq columns: Date, Open, High, Low, Close, Volume
        rename_map = {c: c.capitalize() for c in df.columns}
        df = df.rename(columns=rename_map)
        
        for col in ["Open", "High", "Low", "Close"]:
            if col not in df.columns:
                return pd.DataFrame()
        
        if "Volume" not in df.columns:
            df["Volume"] = 0
        
        df = df.dropna(subset=["Close"])
        return df
    except Exception as e:
        return pd.DataFrame()


# =============================
# Yahoo fallback for Saudi (yfinance)
# =============================
@st.cache_data(ttl=600)
def yf_price_history(symbol: str, period: str) -> pd.DataFrame:
    try:
        df = yf.download(
            symbol,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
            group_by="column",
        )
    except Exception:
        df = pd.DataFrame()

    if df is None or df.empty:
        try:
            df = yf.Ticker(symbol).history(period=period, interval="1d", auto_adjust=False)
        except Exception:
            return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    for c in ["Open", "High", "Low", "Close"]:
        if c not in df.columns:
            return pd.DataFrame()
    if "Volume" not in df.columns:
        df["Volume"] = 0

    return df.dropna(subset=["Close"])

@st.cache_data(ttl=600)
def yf_info(symbol: str) -> dict:
    try:
        t = yf.Ticker(symbol)
        fi = {}
        try:
            fi = dict(t.fast_info) if t.fast_info else {}
        except Exception:
            fi = {}
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}
        return {"fast_info": fi, "info": info}
    except Exception:
        return {"fast_info": {}, "info": {}}

# =============================
# Sidebar
# =============================
with st.sidebar:
    st.header("Search Stock")

    stock_symbol = st.text_input(
        "Enter Stock Symbol",
        placeholder="e.g., AAPL, TSLA, 2222.SR",
        help="US stocks: use AAPL/TSLA. Saudi: use 2222.SR format.",
    )

    period_options_td = {
        "1 Month": ("1day", 30),
        "3 Months": ("1day", 90),
        "6 Months": ("1day", 180),
        "1 Year": ("1day", 260),
        "2 Years": ("1day", 520),
        "5 Years": ("1day", 1300),
    }
    period_options_yf = {
        "1 Month": "1mo",
        "3 Months": "3mo",
        "6 Months": "6mo",
        "1 Year": "1y",
        "2 Years": "2y",
        "5 Years": "5y",
    }
    period_days_map = {
        "1 Month": 35,
        "3 Months": 100,
        "6 Months": 190,
        "1 Year": 370,
        "2 Years": 740,
        "5 Years": 1850,
    }

    selected_period = st.selectbox(
        "Select Time Period",
        options=list(period_options_td.keys()),
        index=3,
    )

    show_debug = st.checkbox("Show debug panels", value=False)
    search_button = st.button("🔍 Analyze Stock", type="primary", use_container_width=True)

# =============================
# State
# =============================
if "result" not in st.session_state:
    st.session_state.result = None

# =============================
# Run analysis
# =============================
if search_button:
    user_sym = (stock_symbol or "").strip().upper()

    if not user_sym:
        st.warning("⚠️ Please enter a stock symbol to analyze.")
        st.stop()

    is_saudi = user_sym.endswith(".SR") or (user_sym.isdigit() and len(user_sym) in (3, 4, 5))

    yahoo_sym = user_sym
    if user_sym.isdigit():
        yahoo_sym = f"{user_sym}.SR"

    resolved = None
    td_symbol = None
    td_meta = {}
    td_raw_ts = None

    hist_data = pd.DataFrame()
    provider = None
    display_name = user_sym
    exchange = "N/A"
    currency = "N/A"

    if is_saudi:
        # Try Stooq first, then fallback to yfinance
        provider = "yfinance"
        
        with st.spinner(f"Fetching Saudi price data (Stooq) for {yahoo_sym}..."):
            hist_data = stooq_price_history(yahoo_sym, period_days_map[selected_period])
        
        if not hist_data.empty:
            provider = "stooq"
        else:
            # Fallback to yfinance
            with st.spinner(f"Stooq failed, trying Yahoo Finance for {yahoo_sym}..."):
                hist_data = yf_price_history(yahoo_sym, period_options_yf[selected_period])

        if hist_data.empty:
            st.error(f"No price data returned for '{yahoo_sym}' from any source (Stooq or Yahoo).")
            st.info(
                "Both Stooq and Yahoo Finance failed for this Saudi symbol. "
                "This can happen due to rate-limits on Streamlit Cloud. "
                "Try again in a few minutes, or consider a paid data provider for reliable Tadawul access."
            )
            st.stop()

        display_name = yahoo_sym
        meta = yf_info(yahoo_sym)
        fi = meta.get("fast_info", {}) or {}
        info = meta.get("info", {}) or {}
        display_name = info.get("longName") or info.get("shortName") or yahoo_sym
        exchange = info.get("exchange") or "Tadawul"
        currency = info.get("currency") or "SAR"

    else:
        provider = "twelve"
        if not td_key():
            st.warning("⚠️ Missing TWELVEDATA_API_KEY. Add it in Streamlit → App Settings → Secrets, then reboot the app.")
            st.stop()

        with st.spinner("Resolving symbol (Twelve Data)..."):
            resolved = td_symbol_search(user_sym)

        if not resolved:
            st.error(f"Could not find symbol in Twelve Data: {user_sym}")
            st.info("Try another ticker (e.g., AAPL, TSLA).")
            st.stop()

        td_symbol = resolved.get("symbol") or user_sym
        display_name = resolved.get("instrument_name") or td_symbol
        exchange = resolved.get("exchange") or "N/A"
        currency = resolved.get("currency") or "N/A"

        interval, outputsize = period_options_td[selected_period]
        with st.spinner(f"Fetching price data (Twelve Data) for {td_symbol}..."):
            hist_data, td_raw_ts = td_time_series(td_symbol, interval=interval, outputsize=outputsize)

        if hist_data.empty:
            st.error(f"No price data returned for '{td_symbol}'.")
            st.info("This can happen if the instrument/market is not covered by your Twelve Data plan (free plan is limited).")
            st.stop()

        td_meta = {"resolved": resolved, "raw_time_series": td_raw_ts}

    st.session_state.result = {
        "provider": provider,
        "user_sym": user_sym,
        "symbol": (yahoo_sym if provider in ("yfinance", "stooq") else td_symbol),
        "name": display_name,
        "exchange": exchange,
        "currency": currency,
        "hist": hist_data,
        "td_meta": td_meta,
        "yahoo_sym": yahoo_sym,
        "is_saudi": is_saudi,
    }

# =============================
# Render
# =============================
res = st.session_state.result
if not res:
    st.info("👈 Enter a stock symbol in the sidebar, then click **Analyze Stock**.")
    st.markdown(
        """
### Notes
- **US stocks** work best via Twelve Data (requires API key in Secrets).
- **Saudi stocks**: uses Stooq first, then Yahoo Finance fallback. Format: `2222.SR`
"""
    )
    st.stop()

provider = res["provider"]
symbol = res["symbol"]
name = res["name"]
exchange = res["exchange"]
currency = res["currency"]
hist_data = res["hist"]
is_saudi = res.get("is_saudi", False)

if show_debug and provider == "twelve":
    with st.expander("🔧 Debug: Twelve Data symbol resolution + raw time_series", expanded=False):
        st.write(res.get("td_meta", {}))

# =============================
# Header
# =============================
col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

with col1:
    st.subheader(name)
    st.caption(f"Symbol: {symbol}  |  Provider: {provider}  |  Exchange: {exchange}  |  Currency: {currency}")

close = hist_data["Close"].dropna()

with col2:
    if len(close) >= 2:
        current_price = float(close.iloc[-1])
        prev_price = float(close.iloc[-2])
        delta = current_price - prev_price
        delta_pct = (delta / prev_price) if prev_price else 0.0
        st.metric("Current Price", f"{current_price:.2f}", f"{delta:+.2f} ({delta_pct*100:+.2f}%)")
    else:
        st.metric("Current Price", "N/A")

with col3:
    st.metric("Period High", f"{float(hist_data['High'].max()):.2f}")

with col4:
    st.metric("Period Low", f"{float(hist_data['Low'].min()):.2f}")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📊 Price Analysis", "💼 Financial Metrics", "🔮 AI Forecast"])

# =============================
# TAB 1: Price
# =============================
with tab1:
    st.subheader("Historical Price Trend")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=hist_data.index,
        open=hist_data["Open"],
        high=hist_data["High"],
        low=hist_data["Low"],
        close=hist_data["Close"],
        name="Price",
    ))
    fig.add_trace(go.Bar(
        x=hist_data.index,
        y=hist_data["Volume"],
        name="Volume",
        yaxis="y2",
        opacity=0.30,
    ))
    fig.update_layout(
        title=f"{symbol} Price & Volume",
        yaxis_title="Price",
        yaxis2=dict(title="Volume", overlaying="y", side="right"),
        xaxis_title="Date",
        height=600,
        hovermode="x unified",
        template="plotly_dark",
    )
    st.plotly_chart(fig, use_container_width=True)

# =============================
# TAB 2: Financial Metrics — richly populated
# =============================
with tab2:
    st.subheader("Financial Metrics")

    # ---- Gather data from yfinance (works for both US and Saudi) ----
    yf_data = yf_info(symbol)
    info = yf_data.get("info", {}) or {}
    fi = yf_data.get("fast_info", {}) or {}

    # Also get Twelve Data quote for live price/volume (US only)
    td_q = {}
    if provider == "twelve" and td_symbol:
        td_q = td_quote(td_symbol)
        if not isinstance(td_q, dict) or td_q.get("status") == "error":
            td_q = {}

    # Helper: get value from yf info or fallback
    def gv(keys, fallback=None):
        for k in (keys if isinstance(keys, list) else [keys]):
            v = info.get(k) or fi.get(k)
            if v is not None and v != "" and v != "N/A":
                try:
                    if isinstance(v, float) and np.isnan(v):
                        continue
                except Exception:
                    pass
                return v
        return fallback

    # ---- Live price row ----
    st.markdown("### 📌 Live Quote")
    q1, q2, q3, q4 = st.columns(4)

    # Price
    last_price = None
    if td_q:
        last_price = _safe_float(td_q.get("close") or td_q.get("price"))
    if last_price is None:
        last_price = gv(["currentPrice", "regularMarketPrice", "previousClose"])
        if last_price is None and len(close) > 0:
            last_price = float(close.iloc[-1])

    chg = _safe_float(td_q.get("change")) if td_q else None
    chg_pct = _safe_float(td_q.get("percent_change")) if td_q else None

    if chg is None:
        prev = gv(["previousClose", "regularMarketPreviousClose"])
        if prev and last_price:
            chg = last_price - float(prev)
            chg_pct = (chg / float(prev)) * 100 if float(prev) else None

    q1.metric("Last Price", f"{last_price:.2f}" if last_price else "N/A",
              f"{chg:+.2f} ({chg_pct:+.2f}%)" if chg is not None and chg_pct is not None else None)

    vol = _safe_float(td_q.get("volume")) if td_q else None
    if vol is None:
        vol = _safe_float(gv(["volume", "regularMarketVolume"]))

    avg_vol = _safe_float(td_q.get("average_volume")) if td_q else None
    if avg_vol is None:
        avg_vol = _safe_float(gv(["averageVolume", "averageDailyVolume10Day"]))

    q2.metric("Volume", f"{vol:,.0f}" if vol is not None else "N/A")
    q3.metric("Avg Volume (30d)", f"{avg_vol:,.0f}" if avg_vol is not None else "N/A")

    open_price = _safe_float(td_q.get("open")) if td_q else None
    if open_price is None:
        open_price = _safe_float(gv(["open", "regularMarketOpen"]))
    q4.metric("Open", f"{open_price:.2f}" if open_price is not None else "N/A")

    st.markdown("---")

    # ---- Valuation ----
    st.markdown("### 📊 Valuation")
    v1, v2, v3, v4 = st.columns(4)

    mktcap = _safe_float(gv(["marketCap"]))
    pe = _safe_float(gv(["trailingPE", "forwardPE"]))
    pb = _safe_float(gv(["priceToBook"]))
    ps = _safe_float(gv(["priceToSalesTrailing12Months"]))
    ev = _safe_float(gv(["enterpriseValue"]))
    evebitda = _safe_float(gv(["enterpriseToEbitda"]))
    peg = _safe_float(gv(["pegRatio"]))
    beta = _safe_float(gv(["beta"]))

    v1.metric("Market Cap", _human_money(mktcap) if mktcap else "N/A")
    v2.metric("P/E Ratio (TTM)", f"{pe:.2f}" if pe else "N/A")
    v3.metric("P/B Ratio", f"{pb:.2f}" if pb else "N/A")
    v4.metric("P/S Ratio", f"{ps:.2f}" if ps else "N/A")

    v5, v6, v7, v8 = st.columns(4)
    v5.metric("Enterprise Value", _human_money(ev) if ev else "N/A")
    v6.metric("EV/EBITDA", f"{evebitda:.2f}" if evebitda else "N/A")
    v7.metric("PEG Ratio", f"{peg:.2f}" if peg else "N/A")
    v8.metric("Beta", f"{beta:.2f}" if beta else "N/A")

    st.markdown("---")

    # ---- Dividends & Yield ----
    st.markdown("### 💰 Dividends & Returns")
    d1, d2, d3, d4 = st.columns(4)

    div_yield = _safe_float(gv(["dividendYield", "trailingAnnualDividendYield"]))
    div_rate = _safe_float(gv(["dividendRate", "trailingAnnualDividendRate"]))
    payout = _safe_float(gv(["payoutRatio"]))
    ex_div = gv(["exDividendDate"])

    d1.metric("Dividend Yield", _human_pct(div_yield) if div_yield else "N/A")
    d2.metric("Dividend Rate", f"{div_rate:.2f}" if div_rate else "N/A")
    d3.metric("Payout Ratio", _human_pct(payout) if payout else "N/A")
    if ex_div:
        try:
            ex_div_str = pd.to_datetime(ex_div, unit="s").strftime("%Y-%m-%d") if isinstance(ex_div, (int, float)) else str(ex_div)
        except Exception:
            ex_div_str = str(ex_div)
        d4.metric("Ex-Dividend Date", ex_div_str)
    else:
        d4.metric("Ex-Dividend Date", "N/A")

    st.markdown("---")

    # ---- Financials ----
    st.markdown("### 📈 Financial Performance")
    f1, f2, f3, f4 = st.columns(4)

    rev = _safe_float(gv(["totalRevenue", "revenue"]))
    rev_growth = _safe_float(gv(["revenueGrowth"]))
    gross_margin = _safe_float(gv(["grossMargins"]))
    profit_margin = _safe_float(gv(["profitMargins"]))
    ebitda = _safe_float(gv(["ebitda"]))
    eps = _safe_float(gv(["trailingEps", "epsTrailingTwelveMonths"]))
    roe = _safe_float(gv(["returnOnEquity"]))
    roa = _safe_float(gv(["returnOnAssets"]))

    f1.metric("Revenue (TTM)", _human_money(rev) if rev else "N/A")
    f2.metric("Revenue Growth", _human_pct(rev_growth) if rev_growth else "N/A")
    f3.metric("Gross Margin", _human_pct(gross_margin) if gross_margin else "N/A")
    f4.metric("Net Profit Margin", _human_pct(profit_margin) if profit_margin else "N/A")

    f5, f6, f7, f8 = st.columns(4)
    f5.metric("EBITDA", _human_money(ebitda) if ebitda else "N/A")
    f6.metric("EPS (TTM)", f"{eps:.2f}" if eps else "N/A")
    f7.metric("Return on Equity", _human_pct(roe) if roe else "N/A")
    f8.metric("Return on Assets", _human_pct(roa) if roa else "N/A")

    st.markdown("---")

    # ---- Balance Sheet ----
    st.markdown("### 🏦 Balance Sheet")
    b1, b2, b3, b4 = st.columns(4)

    total_cash = _safe_float(gv(["totalCash", "totalCashPerShare"]))
    total_debt = _safe_float(gv(["totalDebt"]))
    de_ratio = _safe_float(gv(["debtToEquity"]))
    current_ratio = _safe_float(gv(["currentRatio"]))
    quick_ratio = _safe_float(gv(["quickRatio"]))
    book_val = _safe_float(gv(["bookValue"]))
    total_assets = _safe_float(gv(["totalAssets"]))
    fcf = _safe_float(gv(["freeCashflow"]))

    b1.metric("Total Cash", _human_money(total_cash) if (total_cash and total_cash > 1000) else ("N/A"))
    b2.metric("Total Debt", _human_money(total_debt) if total_debt else "N/A")
    b3.metric("Debt/Equity", f"{de_ratio:.2f}" if de_ratio else "N/A")
    b4.metric("Current Ratio", f"{current_ratio:.2f}" if current_ratio else "N/A")

    b5, b6, b7, b8 = st.columns(4)
    b5.metric("Quick Ratio", f"{quick_ratio:.2f}" if quick_ratio else "N/A")
    b6.metric("Book Value/Share", f"{book_val:.2f}" if book_val else "N/A")
    b7.metric("Total Assets", _human_money(total_assets) if total_assets else "N/A")
    b8.metric("Free Cash Flow", _human_money(fcf) if fcf else "N/A")

    st.markdown("---")

    # ---- 52-Week Range ----
    st.markdown("### 📅 52-Week Range & Targets")
    w1, w2, w3, w4 = st.columns(4)

    high52 = _safe_float(gv(["fiftyTwoWeekHigh"]))
    low52 = _safe_float(gv(["fiftyTwoWeekLow"]))
    target = _safe_float(gv(["targetMeanPrice"]))
    recommendation = gv(["recommendationKey"])

    w1.metric("52-Week High", f"{high52:.2f}" if high52 else "N/A")
    w2.metric("52-Week Low", f"{low52:.2f}" if low52 else "N/A")
    w3.metric("Analyst Target", f"{target:.2f}" if target else "N/A")
    if recommendation:
        w4.metric("Analyst Rating", str(recommendation).upper())
    else:
        w4.metric("Analyst Rating", "N/A")

    # Show note if data is limited
    has_fundamentals = any([mktcap, pe, pb, div_yield, rev, eps])
    if not has_fundamentals:
        st.info(
            "ℹ️ Fundamental data (Market Cap, P/E, dividends, etc.) is not available for this symbol via yfinance. "
            "This is common for Saudi stocks on Streamlit Cloud where Yahoo may rate-limit requests. "
            "Price chart data may still be available above."
        )

    if show_debug:
        with st.expander("🔧 Debug: yfinance info dict", expanded=False):
            st.json(info)
        if td_q:
            with st.expander("🔧 Debug: Twelve Data quote", expanded=False):
                st.json(td_q)


# =============================
# TAB 3: Forecast
# =============================
with tab3:
    st.subheader("🔮 AI-Powered Price Forecast")
    horizon_days = st.slider("Forecast horizon (days)", 7, 90, 30, 1)

    close_df = hist_data.reset_index().copy()
    if close_df.columns[0] not in ["Date", "datetime"]:
        close_df = close_df.rename(columns={close_df.columns[0]: "Date"})
    else:
        close_df = close_df.rename(columns={close_df.columns[0]: "Date"})

    close_df = close_df[["Date", "Close"]].dropna().copy()

    if close_df.empty or close_df["Close"].nunique() < 10:
        st.warning("Not enough historical data to run a forecast.")
    else:
        use_prophet = True
        try:
            from prophet import Prophet
        except Exception:
            use_prophet = False

        if use_prophet:
            try:
                st.info("Using Prophet forecasting ✅")
                dfp = close_df.rename(columns={"Date": "ds", "Close": "y"})
                dfp["ds"] = pd.to_datetime(dfp["ds"])

                m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
                m.fit(dfp)

                future = m.make_future_dataframe(periods=horizon_days)
                fcst = m.predict(future)

                last_actual_date = dfp["ds"].max()

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=dfp["ds"], y=dfp["y"], mode="lines", name="Actual"))
                fig.add_trace(go.Scatter(x=fcst["ds"], y=fcst["yhat"], mode="lines", name="Forecast"))

                fig.add_trace(go.Scatter(
                    x=fcst["ds"], y=fcst["yhat_upper"],
                    mode="lines", line=dict(width=0),
                    showlegend=False, hoverinfo="skip"
                ))
                fig.add_trace(go.Scatter(
                    x=fcst["ds"], y=fcst["yhat_lower"],
                    mode="lines", fill="tonexty",
                    line=dict(width=0),
                    showlegend=False, hoverinfo="skip"
                ))

                fig.update_layout(
                    title=f"{symbol} Forecast (Next {horizon_days} days)",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    height=550,
                    hovermode="x unified",
                    template="plotly_dark",
                )
                st.plotly_chart(fig, use_container_width=True)

                future_only = fcst[fcst["ds"] > last_actual_date][["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
                future_only.columns = ["Date", "Forecast", "Low (CI)", "High (CI)"]
                st.dataframe(future_only.tail(min(30, len(future_only))), use_container_width=True)

            except Exception as e:
                st.warning("Prophet failed on this run. Falling back to a simple trend forecast.")
                st.caption(f"Prophet error: {e}")
                use_prophet = False

        if not use_prophet:
            st.info("Using simple trend forecast (fallback) ✅")
            y = close_df["Close"].values.astype(float)
            x = np.arange(len(y), dtype=float)

            coef = np.polyfit(x, y, 1)
            trend = np.poly1d(coef)

            x_future = np.arange(len(y) + horizon_days, dtype=float)
            yhat = trend(x_future)

            last_date = pd.to_datetime(close_df["Date"].max())
            future_dates = pd.date_range(start=last_date, periods=horizon_days + 1, freq="D")[1:]
            all_dates = pd.concat([pd.to_datetime(close_df["Date"]), pd.Series(future_dates)], ignore_index=True)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pd.to_datetime(close_df["Date"]), y=close_df["Close"], mode="lines", name="Actual"))
            fig.add_trace(go.Scatter(x=all_dates, y=yhat, mode="lines", name="Forecast (Trend)"))

            fig.update_layout(
                title=f"{symbol} Forecast (Trend) - Next {horizon_days} days",
                xaxis_title="Date",
                yaxis_title="Price",
                height=550,
                hovermode="x unified",
                template="plotly_dark",
            )
            st.plotly_chart(fig, use_container_width=True)

    st.caption("⚠️ Forecasts are experimental and for education only. Not financial advice.")

st.markdown("---")
st.caption("⚠️ Disclaimer: This tool is for educational purposes only. Not financial advice.")
