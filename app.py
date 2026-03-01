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
    # some sources return already in %
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

    # Normalize Saudi-like suffix (Yahoo format)
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

        # If user typed .SR, try to prefer Saudi/Tadawul if it exists (often NOT in free plan)
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
# Yahoo fallback for Saudi (yfinance)
# =============================
@st.cache_data(ttl=600)
def yf_price_history(symbol: str, period: str) -> pd.DataFrame:
    # yfinance can fail on Streamlit Cloud due to Yahoo blocks/429
    df = yf.download(
        symbol,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
        group_by="column",
    )
    if df is None or df.empty:
        # alternate
        try:
            df = yf.Ticker(symbol).history(period=period, interval="1d", auto_adjust=False)
        except Exception:
            return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # ensure standard columns
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
        # fast_info is lighter than info
        fi = {}
        try:
            fi = dict(t.fast_info) if t.fast_info else {}
        except Exception:
            fi = {}
        info = {}
        # .info is heavy (often blocked). keep best-effort
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
        help="US stocks: use AAPL/TSLA. Saudi: try 2222.SR (fallback uses yfinance).",
    )

    period_options_td = {  # Twelve uses interval+outputsize
        "1 Month": ("1day", 30),
        "3 Months": ("1day", 90),
        "6 Months": ("1day", 180),
        "1 Year": ("1day", 260),
        "2 Years": ("1day", 520),
        "5 Years": ("1day", 1300),
    }
    period_options_yf = {  # Yahoo uses strings
        "1 Month": "1mo",
        "3 Months": "3mo",
        "6 Months": "6mo",
        "1 Year": "1y",
        "2 Years": "2y",
        "5 Years": "5y",
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
# Run analysis (only when button pressed)
# =============================
if search_button:
    user_sym = (stock_symbol or "").strip().upper()

    if not user_sym:
        st.warning("⚠️ Please enter a stock symbol to analyze.")
        st.stop()

    # Decide provider:
    # - Saudi (.SR or numeric-only like 2222) -> yfinance fallback (because Twelve free often has no Tadawul)
    is_saudi = user_sym.endswith(".SR") or (user_sym.isdigit() and len(user_sym) in (3, 4, 5))

    # Build a normalized yahoo symbol for Saudi numeric input
    yahoo_sym = user_sym
    if user_sym.isdigit():
        yahoo_sym = f"{user_sym}.SR"

    # Resolve Twelve symbol (for non-Saudi primarily)
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
        provider = "yfinance"
        with st.spinner(f"Fetching Saudi price data (Yahoo) for {yahoo_sym}..."):
            hist_data = yf_price_history(yahoo_sym, period_options_yf[selected_period])

        if hist_data.empty:
            st.error(f"No price data returned for '{yahoo_sym}'.")
            st.info(
                "Saudi market via Yahoo sometimes fails on Streamlit Cloud (rate-limit / blocked IP). "
                "If you need Tadawul reliably from the cloud, you usually need a paid data source that supports Tadawul."
            )
            st.stop()

        display_name = yahoo_sym
        # try get meta
        meta = yf_info(yahoo_sym)
        fi = meta.get("fast_info", {}) or {}
        info = meta.get("info", {}) or {}
        display_name = info.get("longName") or info.get("shortName") or yahoo_sym
        exchange = info.get("exchange") or "Tadawul (via Yahoo)"
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

    # save result
    st.session_state.result = {
        "provider": provider,
        "user_sym": user_sym,
        "symbol": (yahoo_sym if provider == "yfinance" else td_symbol),
        "name": display_name,
        "exchange": exchange,
        "currency": currency,
        "hist": hist_data,
        "td_meta": td_meta,
        "yahoo_sym": yahoo_sym,
    }

# =============================
# Render (only if we have results)
# =============================
res = st.session_state.result
if not res:
    st.info("👈 Enter a stock symbol in the sidebar, then click **Analyze Stock**.")
    st.markdown(
        """
### Notes
- **US stocks** work best via Twelve Data.
- **Saudi stocks**: this app tries **Yahoo (yfinance)** fallback for `.SR` tickers.  
  On Streamlit Cloud, Yahoo may block requests sometimes.
"""
    )
    st.stop()

provider = res["provider"]
symbol = res["symbol"]
name = res["name"]
exchange = res["exchange"]
currency = res["currency"]
hist_data = res["hist"]

# Debug panels
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
# TAB 2: Financial Metrics (Quote-first; works on Free)
# =============================
with tab2:
    st.subheader("Financial Snapshot (Twelve Data)")

    quote = td_quote(symbol)

    # لو ما رجع شيء
    if not isinstance(quote, dict) or quote.get("status") == "error" or len(quote) == 0:
        st.warning("Financial snapshot not available for this symbol on your current Twelve Data access.")
        if isinstance(quote, dict) and quote.get("message"):
            st.caption(quote.get("message"))
    else:
        last = _safe_float(quote.get("close") or quote.get("price"))
        chg = _safe_float(quote.get("change"))
        chg_pct = _safe_float(quote.get("percent_change"))
        vol = _safe_float(quote.get("volume"))
        avg_vol = _safe_float(quote.get("average_volume"))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Last Price", f"{last:.2f}" if last is not None else "N/A")
        if chg is not None and chg_pct is not None:
            c2.metric("Change", f"{chg:+.2f}", f"{chg_pct:+.2f}%")
        else:
            c2.metric("Change", "N/A")
        c3.metric("Volume", f"{vol:,.0f}" if vol is not None else "N/A")
        c4.metric("Avg Volume", f"{avg_vol:,.0f}" if avg_vol is not None else "N/A")

        st.markdown("### Raw quote (debug)")
        st.json(quote)

# =============================
# TAB 3: Forecast
# =============================
with tab3:
    st.subheader("🔮 AI-Powered Price Forecast")
    horizon_days = st.slider("Forecast horizon (days)", 7, 90, 30, 1)

    close_df = hist_data.reset_index().copy()
    # index name may be datetime or Date depending on source
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
