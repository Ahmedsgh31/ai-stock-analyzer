import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

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
            f = float(x)
            return None if np.isnan(f) else f
        s = str(x).replace(",", "").strip()
        if s in ("", "None", "N/A", "null", "nan"):
            return None
        return float(s)
    except Exception:
        return None

def _fmt(val, prefix="", suffix="", decimals=2):
    if val is None:
        return "N/A"
    return f"{prefix}{val:,.{decimals}f}{suffix}"

def _human_money(x):
    v = _safe_float(x)
    if v is None:
        return "N/A"
    av = abs(v)
    if av >= 1e12: return f"${v/1e12:.2f}T"
    if av >= 1e9:  return f"${v/1e9:.2f}B"
    if av >= 1e6:  return f"${v/1e6:.2f}M"
    if av >= 1e3:  return f"${v/1e3:.2f}K"
    return f"${v:.2f}"

def _human_pct(x):
    v = _safe_float(x)
    if v is None:
        return "N/A"
    # If already looks like a percentage (e.g. 1.49 meaning 1.49%), keep it
    # If it looks like a ratio (e.g. 0.0149 meaning 1.49%), multiply by 100
    if abs(v) <= 1.5:
        v = v * 100
    return f"{v:.2f}%"

# =============================
# Twelve Data API
# =============================
TD_BASE = "https://api.twelvedata.com"

def td_key() -> str | None:
    return st.secrets.get("TWELVEDATA_API_KEY", None)

@st.cache_resource
def http_session():
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 StockAnalyzer/2.0"})
    return s

def td_get(endpoint: str, params: dict, timeout=25) -> dict:
    key = td_key()
    if not key:
        return {"status": "error", "message": "Missing TWELVEDATA_API_KEY"}
    p = {**params, "apikey": key}
    url = f"{TD_BASE}/{endpoint.lstrip('/')}"
    try:
        r = http_session().get(url, params=p, timeout=timeout)
        return r.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}

# -------- Symbol resolution --------
@st.cache_data(ttl=3600)
def td_resolve_symbol(user_symbol: str) -> dict | None:
    s = user_symbol.strip().upper()
    is_saudi = s.endswith(".SR") or (s.isdigit() and len(s) in (3, 4, 5))
    base = s.replace(".SR", "") if s.endswith(".SR") else s

    # Search by symbol
    data = td_get("symbol_search", {"symbol": base, "outputsize": 50})
    items = data.get("data") or []

    # Fallback: keyword search
    if not items:
        data2 = td_get("symbol_search", {"keywords": base, "outputsize": 50})
        items = data2.get("data") or []

    if not items:
        return None

    def score(it: dict) -> int:
        sc = 0
        sym      = (it.get("symbol") or "").upper()
        ex       = (it.get("exchange") or "").lower()
        country  = (it.get("country") or "").lower()
        currency = (it.get("currency") or "").upper()
        itype    = (it.get("instrument_type") or "").lower()

        if itype == "common stock":
            sc += 10

        if is_saudi:
            if "saudi" in country or "saudi" in ex or "tadawul" in ex:
                sc += 60
            if currency == "SAR":
                sc += 30
            if sym == base or sym == s:
                sc += 40
        else:
            if sym == s:
                sc += 50
            if sym == base:
                sc += 30
            if ex in ("nasdaq", "nyse", "nyse american", "nyse arca"):
                sc += 20

        return sc

    ranked = sorted(items, key=score, reverse=True)
    return ranked[0] if ranked else None


# -------- Price history --------
@st.cache_data(ttl=600)
def td_time_series(symbol: str, exchange: str | None,
                   interval: str, outputsize: int) -> tuple[pd.DataFrame, str]:
    params = {"symbol": symbol, "interval": interval,
              "outputsize": outputsize, "format": "JSON"}
    if exchange:
        params["exchange"] = exchange

    data = td_get("time_series", params)
    if data.get("status") == "error":
        return pd.DataFrame(), data.get("message", "API error")

    values = data.get("values") or []
    if not values:
        return pd.DataFrame(), "No values returned"

    df = pd.DataFrame(values)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").set_index("datetime")
    for c in ["open","high","low","close","volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.rename(columns={"open":"Open","high":"High","low":"Low",
                             "close":"Close","volume":"Volume"})
    for col in ["Open","High","Low","Close"]:
        if col not in df.columns:
            return pd.DataFrame(), f"Missing column: {col}"
    if "Volume" not in df.columns:
        df["Volume"] = 0
    return df.dropna(subset=["Close"]), ""


# -------- Quote --------
@st.cache_data(ttl=60)
def td_quote(symbol: str, exchange: str | None) -> dict:
    p = {"symbol": symbol, "format": "JSON"}
    if exchange: p["exchange"] = exchange
    d = td_get("quote", p)
    return {} if d.get("status") == "error" else d


# -------- Statistics (fundamentals) --------
@st.cache_data(ttl=3600)
def td_statistics(symbol: str, exchange: str | None) -> dict:
    p = {"symbol": symbol, "format": "JSON"}
    if exchange: p["exchange"] = exchange
    d = td_get("statistics", p)
    if d.get("status") == "error":
        return {}
    return d


# =============================
# Stat extraction helper
# =============================
def _sv(stat_data: dict, *paths):
    """
    Search multiple dot-path keys across all nested sub-dicts in stat_data.
    Returns first non-None numeric value found.
    """
    # Flatten all sub-dicts to search across
    sub_dicts = [stat_data]
    for v in stat_data.values():
        if isinstance(v, dict):
            sub_dicts.append(v)
            for vv in v.values():
                if isinstance(vv, dict):
                    sub_dicts.append(vv)

    for path in paths:
        keys = path.split(".")
        for root in sub_dicts:
            v = root
            for k in keys:
                if not isinstance(v, dict): v = None; break
                v = v.get(k)
            val = _safe_float(v)
            if val is not None:
                return val
    return None


# =============================
# Sidebar
# =============================
with st.sidebar:
    st.header("Search Stock")
    stock_symbol = st.text_input(
        "Enter Stock Symbol",
        placeholder="e.g., AAPL, TSLA, 2222.SR",
        help="US: AAPL, TSLA | Saudi: 2222.SR, 1120.SR",
    )
    period_map = {
        "1 Month":  ("1day", 35),
        "3 Months": ("1day", 95),
        "6 Months": ("1day", 185),
        "1 Year":   ("1day", 262),
        "2 Years":  ("1day", 524),
        "5 Years":  ("1day", 1310),
    }
    selected_period = st.selectbox("Select Time Period",
                                   options=list(period_map.keys()), index=3)
    show_debug = st.checkbox("Show debug panels", value=False)
    search_button = st.button("🔍 Analyze Stock", type="primary", use_container_width=True)

# =============================
# Session state
# =============================
if "result" not in st.session_state:
    st.session_state.result = None

# =============================
# Analysis
# =============================
if search_button:
    user_sym = (stock_symbol or "").strip().upper()
    if not user_sym:
        st.warning("⚠️ Please enter a stock symbol.")
        st.stop()
    if not td_key():
        st.error("⚠️ TWELVEDATA_API_KEY missing from Streamlit Secrets.")
        st.stop()

    with st.spinner("Resolving symbol…"):
        resolved = td_resolve_symbol(user_sym)

    if not resolved:
        st.error(f"Symbol not found: **{user_sym}**")
        st.info("Check the ticker format. Saudi examples: 2222.SR | US examples: TSLA, AAPL")
        st.stop()

    td_symbol   = resolved.get("symbol") or user_sym
    td_exchange = resolved.get("exchange") or None
    display_name = resolved.get("instrument_name") or td_symbol
    currency    = resolved.get("currency") or "N/A"

    interval, outputsize = period_map[selected_period]
    with st.spinner(f"Loading price data for {td_symbol}…"):
        hist_data, err_msg = td_time_series(td_symbol, td_exchange, interval, outputsize)

    if hist_data.empty:
        st.error(f"No price data for **{td_symbol}** ({td_exchange or 'auto'})")
        st.caption(f"Twelve Data said: {err_msg}")
        st.info(
            "Free Twelve Data plan covers most US/global symbols. "
            "Saudi (Tadawul) availability depends on your plan tier."
        )
        st.stop()

    st.session_state.result = {
        "symbol": td_symbol, "exchange": td_exchange,
        "name": display_name, "currency": currency,
        "hist": hist_data, "resolved": resolved,
    }

# =============================
# Render
# =============================
res = st.session_state.result
if not res:
    st.info("👈 Enter a stock symbol in the sidebar and click **Analyze Stock**.")
    st.markdown("""
**Supported formats**
- US stocks: `AAPL`, `TSLA`, `MSFT`, `GOOGL`, `NVDA`
- Saudi / Tadawul: `2222.SR`, `1120.SR`, `2010.SR`

> Requires `TWELVEDATA_API_KEY` in Streamlit Secrets.
""")
    st.stop()

symbol    = res["symbol"]
exchange  = res["exchange"]
name      = res["name"]
currency  = res["currency"]
hist_data = res["hist"]

# ---- Header ----
col1, col2, col3, col4 = st.columns([2,1,1,1])
with col1:
    st.subheader(name)
    st.caption(f"**{symbol}** | Exchange: {exchange or 'Auto'} | Currency: {currency}")

close_s = hist_data["Close"].dropna()
with col2:
    if len(close_s) >= 2:
        cp = float(close_s.iloc[-1])
        pp = float(close_s.iloc[-2])
        d  = cp - pp
        dp = d / pp * 100 if pp else 0
        st.metric("Current Price", f"{cp:.2f}", f"{d:+.2f} ({dp:+.2f}%)")
    else:
        st.metric("Current Price", "N/A")
with col3:
    st.metric("Period High", f"{float(hist_data['High'].max()):.2f}")
with col4:
    st.metric("Period Low",  f"{float(hist_data['Low'].min()):.2f}")

st.markdown("---")
tab1, tab2, tab3 = st.tabs(["📊 Price Analysis", "💼 Financial Metrics", "🔮 AI Forecast"])

# ============================
# TAB 1 – Chart
# ============================
with tab1:
    st.subheader("Historical Price & Volume")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=hist_data.index,
        open=hist_data["Open"], high=hist_data["High"],
        low=hist_data["Low"],  close=hist_data["Close"],
        name="Price"))
    fig.add_trace(go.Bar(
        x=hist_data.index, y=hist_data["Volume"],
        name="Volume", yaxis="y2", opacity=0.25,
        marker_color="rgba(100,180,255,0.5)"))
    fig.update_layout(
        title=f"{symbol} — {selected_period}",
        yaxis_title=f"Price ({currency})",
        yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False),
        xaxis_title="Date", height=600,
        hovermode="x unified", template="plotly_dark",
        xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# ============================
# TAB 2 – Financial Metrics
# ============================
with tab2:
    st.subheader("Financial Metrics")

    with st.spinner("Loading fundamentals from Twelve Data…"):
        quote = td_quote(symbol, exchange)
        stats_raw = td_statistics(symbol, exchange)

    if show_debug:
        with st.expander("🔧 Raw /quote"):      st.json(quote)
        with st.expander("🔧 Raw /statistics"): st.json(stats_raw)

    # Twelve Data wraps fundamentals under "statistics" key
    stat_data = stats_raw.get("statistics") or stats_raw

    # ---- Live Quote ----
    st.markdown("### 📌 Live Quote")
    q1, q2, q3, q4 = st.columns(4)

    last_price = _safe_float(quote.get("close") or quote.get("price"))
    if last_price is None and len(close_s):
        last_price = float(close_s.iloc[-1])
    chg      = _safe_float(quote.get("change"))
    chg_pct  = _safe_float(quote.get("percent_change"))
    vol      = _safe_float(quote.get("volume"))
    avg_vol  = _safe_float(quote.get("average_volume"))
    open_p   = _safe_float(quote.get("open"))
    hi_day   = _safe_float(quote.get("high"))
    lo_day   = _safe_float(quote.get("low"))
    prev_cls = _safe_float(quote.get("previous_close"))

    delta_str = f"{chg:+.2f} ({chg_pct:+.2f}%)" if chg is not None and chg_pct is not None else None
    q1.metric("Last Price",  f"{last_price:.2f}" if last_price else "N/A", delta_str)
    q2.metric("Open",        f"{open_p:.2f}"     if open_p    else "N/A")
    q3.metric("Day High",    f"{hi_day:.2f}"     if hi_day    else "N/A")
    q4.metric("Day Low",     f"{lo_day:.2f}"     if lo_day    else "N/A")

    q5, q6, q7, q8 = st.columns(4)
    q5.metric("Prev Close",  f"{prev_cls:.2f}"   if prev_cls  else "N/A")
    q6.metric("Volume",      f"{vol:,.0f}"        if vol       else "N/A")
    q7.metric("Avg Volume",  f"{avg_vol:,.0f}"    if avg_vol   else "N/A")
    q8.metric("Currency",    currency)

    st.markdown("---")

    # ---- Valuation ----
    st.markdown("### 📊 Valuation")
    v1,v2,v3,v4 = st.columns(4)
    v5,v6,v7,v8 = st.columns(4)

    market_cap = _sv(stat_data,
        "market_capitalization",
        "highlights.market_capitalization",
        "valuations_metrics.market_capitalization")
    pe = _sv(stat_data,
        "pe_ratio","trailing_pe","forward_pe",
        "valuations_metrics.trailing_pe",
        "highlights.pe_ratio",
        "price_to_earnings_ttm")
    pb = _sv(stat_data,
        "pb_ratio","price_to_book_mrq",
        "valuations_metrics.price_to_book_mrq")
    ps = _sv(stat_data,
        "ps_ratio","price_to_sales_ttm",
        "valuations_metrics.price_to_sales_ttm")
    ev = _sv(stat_data,
        "enterprise_value",
        "valuations_metrics.enterprise_value",
        "highlights.enterprise_value")
    ev_ebitda = _sv(stat_data,
        "enterprise_to_ebitda","ev_to_ebitda",
        "valuations_metrics.enterprise_to_ebitda")
    peg  = _sv(stat_data, "peg_ratio","valuations_metrics.peg_ratio")
    beta = _sv(stat_data, "beta","five_year_monthly_beta",
               "stock_statistics.beta","stock_statistics.five_year_monthly_beta")

    v1.metric("Market Cap",      _human_money(market_cap))
    v2.metric("P/E Ratio (TTM)", _fmt(pe) if pe else "N/A")
    v3.metric("P/B Ratio",       _fmt(pb) if pb else "N/A")
    v4.metric("P/S Ratio",       _fmt(ps) if ps else "N/A")
    v5.metric("Enterprise Value",_human_money(ev))
    v6.metric("EV/EBITDA",       _fmt(ev_ebitda) if ev_ebitda else "N/A")
    v7.metric("PEG Ratio",       _fmt(peg)  if peg  else "N/A")
    v8.metric("Beta",            _fmt(beta) if beta else "N/A")

    st.markdown("---")

    # ---- Dividends ----
    st.markdown("### 💰 Dividends & Returns")
    d1,d2,d3,d4 = st.columns(4)

    div_yield = _sv(stat_data,
        "dividend_yield","forward_annual_dividend_yield",
        "dividends_and_splits.forward_annual_dividend_yield",
        "highlights.dividend_yield")
    div_rate  = _sv(stat_data,
        "forward_annual_dividend_rate","dividend_rate",
        "dividends_and_splits.forward_annual_dividend_rate")
    payout    = _sv(stat_data,
        "payout_ratio","dividends_and_splits.payout_ratio",
        "highlights.payout_ratio")

    # Ex-dividend date (string, not numeric)
    def _sstr(d, *keys):
        for k in keys:
            parts = k.split(".")
            v = d
            for p in parts:
                if not isinstance(v, dict): v = None; break
                v = v.get(p)
            if v and str(v).strip() not in ("", "None", "null", "N/A"):
                return str(v)
        return None

    ex_div = _sstr(stat_data,
        "ex_dividend_date",
        "dividends_and_splits.ex_dividend_date",
        "stock_statistics.ex_dividend_date")

    d1.metric("Dividend Yield",   _human_pct(div_yield) if div_yield else "N/A")
    d2.metric("Dividend Rate",    _fmt(div_rate, prefix="$") if div_rate else "N/A")
    d3.metric("Payout Ratio",     _human_pct(payout) if payout else "N/A")
    d4.metric("Ex-Dividend Date", ex_div or "N/A")

    st.markdown("---")

    # ---- Financial Performance ----
    st.markdown("### 📈 Financial Performance")
    f1,f2,f3,f4 = st.columns(4)
    f5,f6,f7,f8 = st.columns(4)

    revenue      = _sv(stat_data,
        "total_revenue","revenue_ttm",
        "financials.income_statement.total_revenue",
        "highlights.total_revenue")
    rev_growth   = _sv(stat_data,
        "quarterly_revenue_growth_yoy","revenue_growth",
        "financials.income_statement.quarterly_revenue_growth_yoy",
        "highlights.quarterly_revenue_growth_yoy")
    gross_margin = _sv(stat_data,
        "gross_profit_margin","gross_margin",
        "financials.income_statement.gross_profit_margin",
        "highlights.gross_profit_margin")
    net_margin   = _sv(stat_data,
        "net_profit_margin","profit_margins",
        "financials.income_statement.net_profit_margin",
        "highlights.profit_margin")
    ebitda       = _sv(stat_data,
        "ebitda","financials.income_statement.ebitda",
        "highlights.ebitda")
    eps          = _sv(stat_data,
        "diluted_eps_ttm","eps_ttm","eps",
        "financials.income_statement.diluted_eps_ttm",
        "highlights.diluted_eps_ttm")
    roe          = _sv(stat_data,
        "return_on_equity_ttm","return_on_equity",
        "financials.income_statement.return_on_equity_ttm",
        "highlights.return_on_equity_ttm")
    roa          = _sv(stat_data,
        "return_on_assets_ttm","return_on_assets",
        "financials.income_statement.return_on_assets_ttm",
        "highlights.return_on_assets_ttm")

    f1.metric("Revenue (TTM)",     _human_money(revenue))
    f2.metric("Revenue Growth",    _human_pct(rev_growth)   if rev_growth   else "N/A")
    f3.metric("Gross Margin",      _human_pct(gross_margin) if gross_margin else "N/A")
    f4.metric("Net Profit Margin", _human_pct(net_margin)   if net_margin   else "N/A")
    f5.metric("EBITDA",            _human_money(ebitda))
    f6.metric("EPS (TTM)",         _fmt(eps, prefix="$")    if eps          else "N/A")
    f7.metric("Return on Equity",  _human_pct(roe)          if roe          else "N/A")
    f8.metric("Return on Assets",  _human_pct(roa)          if roa          else "N/A")

    st.markdown("---")

    # ---- Balance Sheet ----
    st.markdown("### 🏦 Balance Sheet")
    b1,b2,b3,b4 = st.columns(4)
    b5,b6,b7,b8 = st.columns(4)

    total_cash    = _sv(stat_data,
        "total_cash_mrq","cash_and_equivalents",
        "financials.balance_sheet.total_cash_mrq")
    total_debt    = _sv(stat_data,
        "total_debt_mrq","total_debt",
        "financials.balance_sheet.total_debt_mrq")
    de_ratio      = _sv(stat_data,
        "total_debt_to_equity_mrq","debt_to_equity",
        "financials.balance_sheet.total_debt_to_equity_mrq")
    current_ratio = _sv(stat_data,
        "current_ratio_mrq","current_ratio",
        "financials.balance_sheet.current_ratio_mrq")
    quick_ratio   = _sv(stat_data,
        "quick_ratio","financials.balance_sheet.quick_ratio")
    book_val      = _sv(stat_data,
        "book_value_per_share_mrq","book_value",
        "financials.balance_sheet.book_value_per_share_mrq")
    total_assets  = _sv(stat_data,
        "total_assets_mrq","total_assets",
        "financials.balance_sheet.total_assets_mrq")
    fcf           = _sv(stat_data,
        "levered_free_cash_flow_ttm","free_cash_flow",
        "financials.cash_flow.levered_free_cash_flow_ttm")

    b1.metric("Total Cash",       _human_money(total_cash))
    b2.metric("Total Debt",       _human_money(total_debt))
    b3.metric("Debt / Equity",    _fmt(de_ratio)      if de_ratio      else "N/A")
    b4.metric("Current Ratio",    _fmt(current_ratio) if current_ratio else "N/A")
    b5.metric("Quick Ratio",      _fmt(quick_ratio)   if quick_ratio   else "N/A")
    b6.metric("Book Value/Share", _fmt(book_val, prefix="$") if book_val else "N/A")
    b7.metric("Total Assets",     _human_money(total_assets))
    b8.metric("Free Cash Flow",   _human_money(fcf))

    st.markdown("---")

    # ---- 52-Week & Targets ----
    st.markdown("### 📅 52-Week Range & Analyst Targets")
    w1,w2,w3,w4 = st.columns(4)

    high52  = _sv(stat_data,
        "fifty_two_week_high","52_week_high",
        "stock_statistics.fifty_two_week_high",
        "stock_price_summary.fifty_two_week_high")
    low52   = _sv(stat_data,
        "fifty_two_week_low","52_week_low",
        "stock_statistics.fifty_two_week_low",
        "stock_price_summary.fifty_two_week_low")
    target  = _sv(stat_data,
        "one_year_target_price","analyst_target_price",
        "highlights.one_year_target_price")
    shares  = _sv(stat_data,
        "shares_outstanding","float_shares",
        "stock_statistics.shares_outstanding",
        "highlights.shares_outstanding")

    # Fallback 52w from price history
    if high52 is None and not hist_data.empty:
        one_yr = hist_data.index.max() - pd.Timedelta(days=365)
        sub = hist_data[hist_data.index >= one_yr]
        if not sub.empty:
            high52 = float(sub["High"].max())
            low52  = float(sub["Low"].min())

    w1.metric("52-Week High",       f"{high52:.2f}" if high52 else "N/A")
    w2.metric("52-Week Low",        f"{low52:.2f}"  if low52  else "N/A")
    w3.metric("Analyst Target",     f"{target:.2f}" if target else "N/A")
    w4.metric("Shares Outstanding", _human_money(shares).replace("$","") if shares else "N/A")

    # ---- Info note ----
    has_fundamentals = any([market_cap, pe, revenue, eps, div_yield])
    if not has_fundamentals:
        st.warning("""
⚠️ **Fundamental data not available for this symbol.**

Possible reasons:
- **`/statistics` requires a paid Twelve Data plan** (Basic or above). The free plan only covers price data and the `/quote` endpoint.
- Saudi (Tadawul) fundamental data may not be available even on paid plans.

**What you can do:**
1. Upgrade your Twelve Data plan at [twelvedata.com/pricing](https://twelvedata.com/pricing)
2. The price chart (Tab 1) and price metrics above are fully functional on the free plan.
""")


# ============================
# TAB 3 – Forecast
# ============================
with tab3:
    st.subheader("🔮 AI-Powered Price Forecast")
    horizon_days = st.slider("Forecast horizon (days)", 7, 90, 30, 1)

    close_df = hist_data.reset_index().copy()
    close_df.columns = ["Date"] + list(close_df.columns[1:])
    close_df = close_df[["Date","Close"]].dropna()

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
                dfp = close_df.rename(columns={"Date":"ds","Close":"y"})
                dfp["ds"] = pd.to_datetime(dfp["ds"])
                m = Prophet(daily_seasonality=False,
                            weekly_seasonality=True, yearly_seasonality=True)
                m.fit(dfp)
                future = m.make_future_dataframe(periods=horizon_days)
                fcst   = m.predict(future)
                last_actual = dfp["ds"].max()

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=dfp["ds"], y=dfp["y"],
                                          mode="lines", name="Actual"))
                fig.add_trace(go.Scatter(x=fcst["ds"], y=fcst["yhat"],
                                          mode="lines", name="Forecast"))
                fig.add_trace(go.Scatter(x=fcst["ds"], y=fcst["yhat_upper"],
                                          mode="lines", line=dict(width=0),
                                          showlegend=False, hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=fcst["ds"], y=fcst["yhat_lower"],
                                          mode="lines", fill="tonexty",
                                          line=dict(width=0),
                                          fillcolor="rgba(100,180,255,0.15)",
                                          showlegend=False, hoverinfo="skip"))
                fig.update_layout(
                    title=f"{symbol} — {horizon_days}-day Forecast",
                    xaxis_title="Date", yaxis_title=f"Price ({currency})",
                    height=550, hovermode="x unified", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

                fut = fcst[fcst["ds"] > last_actual][["ds","yhat","yhat_lower","yhat_upper"]]
                fut.columns = ["Date","Forecast","Low (CI)","High (CI)"]
                st.dataframe(fut.tail(30), use_container_width=True)

            except Exception as e:
                st.warning(f"Prophet error: {e}. Falling back to trend forecast.")
                use_prophet = False

        if not use_prophet:
            st.info("Using linear trend forecast ✅")
            y = close_df["Close"].values.astype(float)
            x = np.arange(len(y), dtype=float)
            coef  = np.polyfit(x, y, 1)
            trend = np.poly1d(coef)
            x_all = np.arange(len(y) + horizon_days, dtype=float)
            yhat  = trend(x_all)
            last_date    = pd.to_datetime(close_df["Date"].max())
            future_dates = pd.date_range(start=last_date, periods=horizon_days+1, freq="D")[1:]
            all_dates    = pd.concat([pd.to_datetime(close_df["Date"]),
                                       pd.Series(future_dates)], ignore_index=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pd.to_datetime(close_df["Date"]),
                                      y=close_df["Close"], mode="lines", name="Actual"))
            fig.add_trace(go.Scatter(x=all_dates, y=yhat, mode="lines",
                                      name="Trend Forecast",
                                      line=dict(dash="dash", color="orange")))
            fig.update_layout(
                title=f"{symbol} — {horizon_days}-day Trend Forecast",
                xaxis_title="Date", yaxis_title=f"Price ({currency})",
                height=550, hovermode="x unified", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

    st.caption("⚠️ Forecasts are experimental and for educational purposes only. Not financial advice.")

st.markdown("---")
st.caption("⚠️ Disclaimer: Educational purposes only. Not financial advice.")
