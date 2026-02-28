import time
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
    return f"{x*100:.2f}%"


def make_line_chart(series: pd.Series, title: str, y_format: str = "money"):
    fig = go.Figure()
    if series is None or series.empty:
        fig.update_layout(title=title, height=350, template="plotly_dark")
        return fig

    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines+markers", name=title))
    fig.update_layout(
        title=title,
        height=350,
        hovermode="x unified",
        template="plotly_dark",
        margin=dict(l=30, r=30, t=50, b=30),
    )

    if y_format == "money":
        fig.update_yaxes(tickprefix="$")
    elif y_format == "pct":
        fig.update_yaxes(tickformat=".2%")

    return fig


# =============================
# Twelve Data client
# =============================
TD_BASE = "https://api.twelvedata.com"

def td_key() -> str | None:
    # Streamlit secrets
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
        data = r.json()
    except Exception:
        return {"status": "error", "message": f"Non-JSON response: {r.text[:200]}"}

    return data


def td_symbol_search(user_symbol: str) -> dict | None:
    """
    Finds best match using Twelve Data symbol_search.
    Tries to map inputs like:
      - AAPL
      - TSLA
      - 2222.SR (Tadawul)  -> will search for '2222'
    """
    s = user_symbol.strip().upper()
    raw = s

    # Normalize Saudi-style suffix
    if s.endswith(".SR"):
        s = s.replace(".SR", "")

    # Symbol search
    data = td_get("symbol_search", {"symbol": s, "outputsize": 30})
    if data.get("status") == "error":
        return None

    items = data.get("data") or []
    if not items:
        # Try keywords if symbol exact failed
        data2 = td_get("symbol_search", {"keywords": raw, "outputsize": 30})
        if data2.get("status") == "error":
            return None
        items = data2.get("data") or []
        if not items:
            return None

    # Prefer Saudi Arabia for numbers like 2222
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

        # Generic preferences: exact symbol match
        if sym == raw:
            sc += 40
        if sym == s:
            sc += 25
        return sc

    best = sorted(items, key=score, reverse=True)[0]
    return best


@st.cache_data(ttl=600)
def td_time_series(symbol: str, interval: str, outputsize: int = 260) -> pd.DataFrame:
    """
    Fetch OHLCV from Twelve Data.
    """
    data = td_get("time_series", {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "format": "JSON",
    })

    if data.get("status") == "error":
        return pd.DataFrame()

    values = data.get("values") or []
    if not values:
        return pd.DataFrame()

    df = pd.DataFrame(values)
    # Expected columns: datetime, open, high, low, close, volume
    if "datetime" not in df.columns:
        return pd.DataFrame()

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").set_index("datetime")

    # Convert numeric
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Rename to match rest of app
    rename = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    df = df.rename(columns=rename)

    # Ensure columns exist
    for col in ["Open", "High", "Low", "Close"]:
        if col not in df.columns:
            return pd.DataFrame()
    if "Volume" not in df.columns:
        df["Volume"] = 0

    df = df.dropna(subset=["Close"])
    return df


@st.cache_data(ttl=600)
def td_quote(symbol: str) -> dict:
    data = td_get("quote", {"symbol": symbol, "format": "JSON"})
    if data.get("status") == "error":
        return {}
    return data


@st.cache_data(ttl=600)
def td_statistics(symbol: str) -> dict:
    data = td_get("statistics", {"symbol": symbol, "format": "JSON"})
    if data.get("status") == "error":
        return {}
    return data


@st.cache_data(ttl=600)
def td_profile(symbol: str) -> dict:
    data = td_get("profile", {"symbol": symbol, "format": "JSON"})
    if data.get("status") == "error":
        return {}
    return data


# =============================
# Sidebar
# =============================
with st.sidebar:
    st.header("Search Stock")

    stock_symbol = st.text_input(
        "Enter Stock Symbol",
        placeholder="e.g., AAPL, TSLA, 2222.SR",
        help="For Saudi (Tadawul) you can type like 2222.SR and the app will search via Twelve Data.",
    )

    period_options = {
        "1 Month": ("1day", 30),
        "3 Months": ("1day", 90),
        "6 Months": ("1day", 180),
        "1 Year": ("1day", 260),
        "2 Years": ("1day", 520),
        "5 Years": ("1day", 1300),
    }

    selected_period = st.selectbox(
        "Select Time Period",
        options=list(period_options.keys()),
        index=3,
    )

    search_button = st.button("🔍 Analyze Stock", type="primary", use_container_width=True)

# =============================
# Main
# =============================
if not td_key():
    st.warning("⚠️ Missing TWELVEDATA_API_KEY. Add it in Streamlit → App Settings → Secrets, then reboot the app.")

if search_button and stock_symbol:
    user_sym = stock_symbol.strip().upper()

    # Resolve symbol using symbol_search (helps for Saudi / different exchanges)
    with st.spinner("Resolving symbol..."):
        resolved = td_symbol_search(user_sym)
    
# DEBUG: show what Twelve Data found
with st.expander("🔧 Debug: Symbol resolution (Twelve Data)", expanded=False):
    st.write("Input:", stock_symbol)
    st.write("Resolved object:", resolved)

# Stop if not resolved
if not resolved:
    st.error(f"Could not find symbol: {user_sym}")
    st.stop()

# Resolve final symbol/info
symbol = resolved.get("symbol") or user_sym
name = resolved.get("instrument_name") or symbol
exchange = resolved.get("exchange") or "N/A"
currency = resolved.get("currency") or "N/A"

# Pick interval/outputsize from your period selector
interval, outputsize = period_options[selected_period]

# Fetch price data (and keep raw response for debugging)
raw_ts = None
with st.spinner(f"Fetching price data for {symbol}..."):
    raw_ts = td_get(
        "time_series",
        {
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "format": "JSON",
        },
    )
    hist_data = td_time_series(symbol, interval=interval, outputsize=outputsize)

with st.expander("🔧 Debug: time_series raw response", expanded=False):
    st.write(raw_ts)

# Stop if no price data
if hist_data is None or hist_data.empty:
    st.error(f"No price data returned for '{symbol}'.")
    st.info("If this is a Saudi ticker, it depends on Twelve Data coverage/plan for Tadawul.")
    st.stop()

# =============================
# Header
# =============================
col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

with col1:
    st.subheader(name)
    st.caption(f"Symbol: {symbol}  |  Exchange: {exchange}  |  Currency: {currency}")

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
    hi = float(hist_data["High"].max())
    st.metric("Period High", f"{hi:.2f}")

with col4:
    lo = float(hist_data["Low"].min())
    st.metric("Period Low", f"{lo:.2f}")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📊 Price Analysis", "💼 Financial Metrics", "🔮 AI Forecast"])

# =============================
# TAB 1: Price
# =============================
with tab1:
    st.subheader("Historical Price Trend")

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=hist_data.index,
            open=hist_data["Open"],
            high=hist_data["High"],
            low=hist_data["Low"],
            close=hist_data["Close"],
            name="Price",
        )
    )
    fig.add_trace(
        go.Bar(
            x=hist_data.index,
            y=hist_data["Volume"],
            name="Volume",
            yaxis="y2",
            opacity=0.30,
        )
    )
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

    st.subheader("Technical Indicators")
    c1, c2, c3 = st.columns(3)

    with c1:
        ma_20 = hist_data["Close"].rolling(window=20).mean().iloc[-1]
        ma_50 = hist_data["Close"].rolling(window=50).mean().iloc[-1]
        st.markdown("**Moving Averages**")
        st.write("20-Day MA:", f"{ma_20:.2f}" if not np.isnan(ma_20) else "N/A")
        st.write("50-Day MA:", f"{ma_50:.2f}" if not np.isnan(ma_50) else "N/A")

    with c2:
        returns = hist_data["Close"].pct_change()
        vol = returns.std() * np.sqrt(252)
        st.markdown("**Volatility**")
        st.write("Annual:", f"{vol*100:.2f}%")

    with c3:
        st.markdown("**Period Range**")
        st.write("High:", f"{hist_data['High'].max():.2f}")
        st.write("Low:", f"{hist_data['Low'].min():.2f}")
    # =============================
    # TAB 2: Financial metrics (Twelve Data)
    # =============================
    with tab2:
        st.subheader("Company Financial Position (Twelve Data)")

    quote = td_quote(symbol)
    stats = td_statistics(symbol)
    prof = td_profile(symbol)

    # Debug (important to see if plan blocks endpoints)
    with st.expander("🔧 Debug: quote / statistics / profile raw", expanded=False):
        st.write("quote:", quote)
        st.write("statistics:", stats)
        st.write("profile:", prof)

    # ---- Quote-based metrics (usually available) ----
    q_price = _safe_float(quote.get("close") or quote.get("price"))
    q_change = _safe_float(quote.get("change"))
    q_change_pct = _safe_float(quote.get("percent_change"))
    q_volume = _safe_float(quote.get("volume"))
    q_avg_volume = _safe_float(quote.get("average_volume"))
    q_high_52 = _safe_float(quote.get("fifty_two_week", {}).get("high") if isinstance(quote.get("fifty_two_week"), dict) else quote.get("fifty_two_week_high"))
    q_low_52  = _safe_float(quote.get("fifty_two_week", {}).get("low")  if isinstance(quote.get("fifty_two_week"), dict) else quote.get("fifty_two_week_low"))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last Price", f"{q_price:.2f}" if q_price is not None else "N/A")
    if q_change is not None and q_change_pct is not None:
        c2.metric("Change", f"{q_change:+.2f}", f"{q_change_pct:+.2f}%")
    else:
        c2.metric("Change", "N/A")
    c3.metric("Volume", f"{q_volume:,.0f}" if q_volume is not None else "N/A")
    c4.metric("Avg Volume", f"{q_avg_volume:,.0f}" if q_avg_volume is not None else "N/A")

    st.markdown("---")

    # ---- Statistics/Profile (optional, may be blocked by plan) ----
    # If stats/profile are blocked, show a clear note instead of N/A everywhere
    if (isinstance(stats, dict) and stats.get("status") == "error") or (isinstance(prof, dict) and prof.get("status") == "error"):
        st.warning("Statistics/Profile may be unavailable on your Twelve Data plan or for this market. Quote metrics above should still work.")
        # Show any message
        if isinstance(stats, dict) and stats.get("message"):
            st.caption(f"Statistics message: {stats.get('message')}")
        if isinstance(prof, dict) and prof.get("message"):
            st.caption(f"Profile message: {prof.get('message')}")
        st.stop()

    # Try to read extra fields if present
    market_cap = _safe_float(stats.get("market_cap"))
    pe_ratio = _safe_float(stats.get("pe_ratio") or stats.get("pe"))
    eps = _safe_float(stats.get("eps") or stats.get("eps_ttm"))
    div_yield = _safe_float(stats.get("dividend_yield"))

    s1, s2 = st.columns(2)
    with s1:
        st.markdown("### 📊 Extra Metrics (if available)")
        st.metric("Market Cap", _human_money(market_cap))
        st.metric("P/E", f"{pe_ratio:.2f}" if pe_ratio is not None else "N/A")
        st.metric("EPS", f"{eps:.2f}" if eps is not None else "N/A")
        st.metric("Dividend Yield", _human_pct(div_yield))

    with s2:
        st.markdown("### 📝 Company Info (if available)")
        st.write("Sector:", prof.get("sector", "N/A"))
        st.write("Industry:", prof.get("industry", "N/A"))
        st.write("Country:", prof.get("country", "N/A"))
        st.write("Website:", prof.get("website", "N/A"))

    desc = prof.get("description") or prof.get("long_description")
    if desc:
        with st.expander("📖 Business Summary"):
            st.write(desc)

    # =============================
    # TAB 3: Forecast
    # =============================
    with tab3:
        st.subheader("🔮 AI-Powered Price Forecast")
        horizon_days = st.slider("Forecast horizon (days)", 7, 90, 30, 1)

        close_df = hist_data.reset_index().rename(columns={"datetime": "Date"})
        close_df = close_df.rename(columns={close_df.columns[0]: "Date"})
        close_df = close_df[["Date", "Close"]].dropna().copy()

        if close_df.empty or close_df["Close"].nunique() < 10:
            st.warning("Not enough historical data to run a forecast.")
        else:
            # Prophet if available, otherwise fallback to trend
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

                    # Confidence band
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
                    st.warning("Prophet failed on this run. Falling back to a simple forecast.")
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

elif search_button:
    st.warning("⚠️ Please enter a stock symbol to analyze.")
else:
    st.info("👈 Enter a stock symbol in the sidebar to get started!")
    st.markdown(
        """
### 🎯 Features:
- **Price charts** (candlestick + volume)
- **Financial snapshot** (via Twelve Data: quote/statistics/profile)
- **AI forecast** (Prophet if available, otherwise trend fallback)

### 🌍 Notes:
- Yahoo/yfinance often fails on Streamlit Cloud (rate-limit / blocked IPs).
- This app uses **Twelve Data** as the primary provider to support **US + Tadawul (if covered by your plan)**.
"""
    )

st.markdown("---")
st.caption("⚠️ Disclaimer: This tool is for educational purposes only. Not financial advice.")
