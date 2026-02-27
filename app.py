import os
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
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
# Config / Secrets
# =============================
TD_KEY = ""
try:
    TD_KEY = st.secrets.get("TWELVEDATA_API_KEY", "")
except Exception:
    TD_KEY = ""
TD_KEY = TD_KEY or os.getenv("TWELVEDATA_API_KEY", "")

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


def _to_period_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    try:
        out.columns = pd.to_datetime(out.columns)
        out = out.sort_index(axis=1)
    except Exception:
        pass
    return out


def _retry(fn, tries=3, sleep_s=1.2):
    last = None
    for _ in range(tries):
        try:
            return fn()
        except Exception as e:
            last = e
            time.sleep(sleep_s)
    raise last


def series_from_stmt(df: pd.DataFrame, keys: list[str]) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)

    for k in keys:
        if k in df.index:
            s = df.loc[k].copy()
            try:
                s.index = pd.to_datetime(s.index)
                s = s.sort_index()
            except Exception:
                pass
            s = pd.to_numeric(s, errors="coerce")
            return s.dropna()
    return pd.Series(dtype=float)


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


@st.cache_resource
def get_http_session():
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


def normalize_symbol(user_symbol: str):
    """
    Returns:
      (symbol_for_provider, mic_code, is_saudi)
    User can type:
      - 2222.SR (Saudi) -> ("2222", "XSAU", True)
      - AAPL -> ("AAPL", None, False)
    """
    s = (user_symbol or "").strip().upper()
    if s.endswith(".SR"):
        base = s.replace(".SR", "").strip()
        # Twelve Data uses MIC code for exchange
        return base, "XSAU", True
    return s, None, False


def compute_fast_from_hist(hist: pd.DataFrame) -> dict:
    if hist is None or hist.empty:
        return {}
    out = {}
    try:
        out["yearHigh"] = float(hist["High"].max())
        out["yearLow"] = float(hist["Low"].min())
        out["lastClose"] = float(hist["Close"].dropna().iloc[-1])
    except Exception:
        pass
    return out


# =============================
# Twelve Data fetcher
# =============================
@st.cache_data(ttl=600)
def fetch_price_history_twelvedata(symbol: str, period: str, mic_code: str | None) -> pd.DataFrame:
    """
    Fetch daily OHLCV from Twelve Data time_series.
    period: "1mo","3mo","6mo","1y","2y","5y" -> mapped to outputsize (approx)
    """
    if not TD_KEY:
        return pd.DataFrame()

    output_map = {
        "1mo": 25,
        "3mo": 70,
        "6mo": 140,
        "1y": 260,
        "2y": 520,
        "5y": 1400,
    }
    outputsize = output_map.get(period, 260)

    params = {
        "symbol": symbol,
        "interval": "1day",
        "outputsize": str(outputsize),
        "apikey": TD_KEY,
        "format": "JSON",
    }
    if mic_code:
        params["mic_code"] = mic_code  # e.g., XSAU for Saudi Exchange

    url = "https://api.twelvedata.com/time_series"
    session = get_http_session()
    r = session.get(url, params=params, timeout=25)
    data = r.json()

    # Error handling
    if isinstance(data, dict) and data.get("status") == "error":
        return pd.DataFrame()

    values = (data or {}).get("values", [])
    if not values:
        return pd.DataFrame()

    df = pd.DataFrame(values)
    # Twelve Data returns strings
    # columns: datetime, open, high, low, close, volume
    rename = {
        "datetime": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    df = df.rename(columns=rename)

    if "Date" not in df.columns or "Close" not in df.columns:
        return pd.DataFrame()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Close"])
    return df


# =============================
# yfinance fetcher (US mainly)
# =============================
@st.cache_data(ttl=600)
def fetch_price_history_yfinance(symbol: str, period: str) -> pd.DataFrame:
    return yf.download(
        symbol,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
        group_by="column",
    )


@st.cache_data(ttl=600)
def fetch_company_info(symbol: str) -> dict:
    # heavy endpoint; may fail on Streamlit Cloud
    try:
        return yf.Ticker(symbol).info or {}
    except Exception:
        return {}


@st.cache_data(ttl=600)
def fetch_statements(symbol: str, quarterly: bool):
    t = yf.Ticker(symbol)

    def _get_all():
        if quarterly:
            inc = t.quarterly_income_stmt
            bal = t.quarterly_balance_sheet
            cfs = t.quarterly_cash_flow
        else:
            inc = t.income_stmt
            bal = t.balance_sheet
            cfs = t.cash_flow
        return _to_period_index(inc), _to_period_index(bal), _to_period_index(cfs)

    return _retry(_get_all, tries=2, sleep_s=1.5)


@st.cache_data(ttl=600)
def fetch_price_history(symbol_user: str, period: str) -> tuple[pd.DataFrame, dict]:
    """
    Returns: (hist_df, meta_dict)
    meta_dict includes:
      provider: "twelvedata" | "yfinance"
      mic_code / normalized_symbol
    """
    sym, mic, is_saudi = normalize_symbol(symbol_user)

    # Saudi: go Twelve Data first (Yahoo often blocked on Cloud)
    if is_saudi:
        df = fetch_price_history_twelvedata(sym, period, mic)
        return df, {"provider": "twelvedata", "mic_code": mic, "normalized_symbol": sym}

    # Non-Saudi: try yfinance first, fallback to Twelve Data
    try:
        df = fetch_price_history_yfinance(sym, period)
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df, {"provider": "yfinance", "mic_code": None, "normalized_symbol": sym}
    except Exception:
        pass

    df = fetch_price_history_twelvedata(sym, period, None)
    return df, {"provider": "twelvedata", "mic_code": None, "normalized_symbol": sym}


# =============================
# Sidebar
# =============================
with st.sidebar:
    st.header("Search Stock")

    stock_symbol = st.text_input(
        "Enter Stock Symbol",
        placeholder="e.g., AAPL, 2222.SR, TSLA",
        help="US: AAPL / TSLA. Saudi: add .SR مثل 2222.SR",
    )

    period_options = {
        "1 Month": "1mo",
        "3 Months": "3mo",
        "6 Months": "6mo",
        "1 Year": "1y",
        "2 Years": "2y",
        "5 Years": "5y",
    }

    selected_period = st.selectbox(
        "Select Time Period",
        options=list(period_options.keys()),
        index=3,
    )

    load_detailed = st.checkbox(
        "Load detailed company info (may hit Yahoo limits)",
        value=False,
        help="Uses yfinance .info (Yahoo) which can be blocked/rate-limited on Streamlit Cloud.",
    )

    search_button = st.button("🔍 Analyze Stock", type="primary", use_container_width=True)


# =============================
# Main
# =============================
if search_button and stock_symbol:
    user_symbol = stock_symbol.strip().upper()

    try:
        with st.spinner(f"Fetching price data for {user_symbol}..."):
            hist_data, meta = fetch_price_history(user_symbol, period_options[selected_period])

        if hist_data is None or hist_data.empty:
            st.error(f"No data returned for '{user_symbol}'.")
            if user_symbol.endswith(".SR") and not TD_KEY:
                st.warning("Saudi (.SR) يحتاج TWELVEDATA_API_KEY في Secrets.")
            st.info(
                "Probable causes:\n"
                "- Yahoo blocked / rate-limited on Streamlit Cloud\n"
                "- Missing Twelve Data API key (for .SR)\n\n"
                "✅ Fix:\n"
                "1) Add Twelve Data API key in Streamlit Secrets\n"
                "2) Reboot app\n"
            )
            st.stop()

        # Ensure expected columns exist
        for col in ["Open", "High", "Low", "Close"]:
            if col not in hist_data.columns:
                st.error(f"Price data missing required column: {col}")
                st.stop()
        if "Volume" not in hist_data.columns:
            hist_data["Volume"] = 0

        # Fast info from hist (provider-agnostic)
        fast = compute_fast_from_hist(hist_data)

        # Detailed info (still yfinance; optional)
        info = {}
        if load_detailed:
            with st.spinner("Loading detailed company info (Yahoo)..."):
                # For Saudi .SR: yfinance expects 2222.SR
                # For Twelve Data normalized symbol is different, so we use original user input.
                info = fetch_company_info(user_symbol) or {}

        # =============================
        # Header metrics
        # =============================
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

        with col1:
            company_name = (info or {}).get("longName") or user_symbol
            st.subheader(company_name)
            st.caption(f"Symbol: {user_symbol}  •  Data: {meta.get('provider','unknown')}")

        with col2:
            close = hist_data["Close"].dropna()
            if len(close) >= 2:
                current_price = float(close.iloc[-1])
                prev_price = float(close.iloc[-2])
                delta = current_price - prev_price
                delta_pct = (delta / prev_price) if prev_price else 0.0
                st.metric("Current Price", f"${current_price:.2f}", f"{delta:+.2f} ({delta_pct*100:+.2f}%)")
            else:
                st.metric("Current Price", "N/A")

        with col3:
            hi_52 = _safe_float(fast.get("yearHigh"))
            st.metric("52W High", f"${hi_52:.2f}" if hi_52 is not None else "N/A")

        with col4:
            lo_52 = _safe_float(fast.get("yearLow"))
            st.metric("52W Low", f"${lo_52:.2f}" if lo_52 is not None else "N/A")

        st.markdown("---")

        tab1, tab2, tab3 = st.tabs(["📊 Price Analysis", "💼 Financial Metrics", "🔮 AI Forecast"])

        # =============================
        # TAB 1: Price analysis
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
                title=f"{user_symbol} Price & Volume",
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
                st.write("20-Day MA:", f"${ma_20:.2f}" if not np.isnan(ma_20) else "N/A")
                st.write("50-Day MA:", f"${ma_50:.2f}" if not np.isnan(ma_50) else "N/A")

            with c2:
                returns = hist_data["Close"].pct_change()
                vol = returns.std() * np.sqrt(252)
                st.markdown("**Volatility**")
                st.write("Annual:", f"{vol*100:.2f}%")

            with c3:
                ph = float(hist_data["High"].max())
                pl = float(hist_data["Low"].min())
                st.markdown("**Period Range**")
                st.write("High:", f"${ph:.2f}")
                st.write("Low:", f"${pl:.2f}")

        # =============================
        # TAB 2: Financial Metrics + Trends
        # =============================
        with tab2:
            st.subheader("Company Financial Position")

            # Snapshot metrics (may be N/A if Yahoo info blocked)
            snap1, snap2 = st.columns(2)

            market_cap = _safe_float((info or {}).get("marketCap"))
            pe_ratio = _safe_float((info or {}).get("trailingPE")) or _safe_float((info or {}).get("forwardPE"))
            eps_ttm = _safe_float((info or {}).get("trailingEps"))
            div_yield = _safe_float((info or {}).get("dividendYield"))
            roe = _safe_float((info or {}).get("returnOnEquity"))
            roa = _safe_float((info or {}).get("returnOnAssets"))
            profit_margin = _safe_float((info or {}).get("profitMargins"))
            revenue = _safe_float((info or {}).get("totalRevenue"))

            with snap1:
                st.markdown("### 📊 Key Metrics (Yahoo)")
                st.metric("Market Cap", _human_money(market_cap))
                st.metric("P/E Ratio", f"{pe_ratio:.2f}" if pe_ratio is not None else "N/A")
                st.metric("EPS (TTM)", f"${eps_ttm:.2f}" if eps_ttm is not None else "N/A")
                st.metric("Dividend Yield", _human_pct(div_yield))

            with snap2:
                st.markdown("### 💰 Profitability (Yahoo)")
                st.metric("ROE", _human_pct(roe))
                st.metric("ROA", _human_pct(roa))
                st.metric("Profit Margin", _human_pct(profit_margin))
                st.metric("Revenue (TTM)", _human_money(revenue))

            st.info(
                "ملاحظة: Financial Metrics و Fundamentals هنا تعتمد على Yahoo عبر yfinance.\n"
                "إذا كانت N/A على Streamlit Cloud فهذا بسبب الحظر/الـ rate limit."
            )

            st.markdown("---")
            st.subheader("📈 Fundamentals Trends (Yahoo)")

            tc1, tc2, tc3 = st.columns([1, 1, 2])
            with tc1:
                freq = st.selectbox("Frequency", ["Annual", "Quarterly"], index=0)
            with tc2:
                show_table = st.checkbox("Show statements table", value=False)
            with tc3:
                st.caption("If empty: Yahoo blocked/rate-limited. Reboot app or try later.")

            quarterly = (freq == "Quarterly")

            try:
                with st.spinner("Loading financial statements (Yahoo)..."):
                    income, balance, cashflow = fetch_statements(user_symbol, quarterly=quarterly)

                if (income is None or income.empty) and (balance is None or balance.empty):
                    st.warning("No fundamentals returned (Yahoo blocked/empty).")
                    st.stop()

                eps_series = series_from_stmt(income, ["Diluted EPS", "Basic EPS", "Earnings Per Share"])
                rev_series = series_from_stmt(income, ["Total Revenue", "Revenue"])
                ni_series = series_from_stmt(income, ["Net Income", "Net Income Common Stockholders"])

                pm_series = pd.Series(dtype=float)
                if not rev_series.empty and not ni_series.empty:
                    pm_series = (ni_series / rev_series).replace([np.inf, -np.inf], np.nan).dropna()

                liab_series = series_from_stmt(balance, ["Total Liabilities Net Minority Interest", "Total Liabilities"])
                assets_series = series_from_stmt(balance, ["Total Assets"])
                debt_series = series_from_stmt(
                    balance,
                    ["Total Debt", "Long Term Debt", "Long Term Debt And Capital Lease Obligation"],
                )

                ocf_series = series_from_stmt(cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities"])
                capex_series = series_from_stmt(cashflow, ["Capital Expenditure", "Capital Expenditures"])
                fcf_series = pd.Series(dtype=float)
                if not ocf_series.empty and not capex_series.empty:
                    fcf_series = (ocf_series + capex_series).dropna()

                r1c1, r1c2 = st.columns(2)
                with r1c1:
                    st.plotly_chart(make_line_chart(eps_series, f"EPS Trend ({freq})", "money"), use_container_width=True)
                with r1c2:
                    st.plotly_chart(make_line_chart(pm_series, f"Profit Margin Trend ({freq})", "pct"), use_container_width=True)

                r2c1, r2c2 = st.columns(2)
                with r2c1:
                    st.plotly_chart(make_line_chart(rev_series, f"Revenue Trend ({freq})", "money"), use_container_width=True)
                with r2c2:
                    st.plotly_chart(make_line_chart(ni_series, f"Net Income Trend ({freq})", "money"), use_container_width=True)

                r3c1, r3c2 = st.columns(2)
                with r3c1:
                    st.plotly_chart(make_line_chart(liab_series, f"Total Liabilities ({freq})", "money"), use_container_width=True)
                with r3c2:
                    st.plotly_chart(make_line_chart(assets_series, f"Total Assets ({freq})", "money"), use_container_width=True)

                r4c1, r4c2 = st.columns(2)
                with r4c1:
                    st.plotly_chart(make_line_chart(debt_series, f"Debt Trend ({freq})", "money"), use_container_width=True)
                with r4c2:
                    st.plotly_chart(make_line_chart(fcf_series, f"Free Cash Flow ({freq})", "money"), use_container_width=True)

                if show_table:
                    st.markdown("#### Income Statement (selected lines)")
                    top_income = pd.DataFrame()
                    if income is not None and not income.empty:
                        keep = [i for i in ["Total Revenue", "Net Income", "Diluted EPS", "Basic EPS"] if i in income.index]
                        top_income = income.loc[keep] if keep else pd.DataFrame()
                    st.dataframe(top_income, use_container_width=True)

                    st.markdown("#### Balance Sheet (selected lines)")
                    top_bal = pd.DataFrame()
                    if balance is not None and not balance.empty:
                        keep = [i for i in ["Total Assets", "Total Liabilities", "Total Debt", "Cash And Cash Equivalents"] if i in balance.index]
                        top_bal = balance.loc[keep] if keep else pd.DataFrame()
                    st.dataframe(top_bal, use_container_width=True)

            except Exception as e:
                st.error(f"Could not load fundamentals (Yahoo): {e}")

        # =============================
        # TAB 3: AI Forecast
        # =============================
        with tab3:
            st.subheader("🔮 AI-Powered Price Forecast")

            horizon_days = st.slider("Forecast horizon (days)", 7, 90, 30, 1)

            close_df = hist_data.reset_index().copy()
            if "Date" not in close_df.columns:
                close_df = close_df.rename(columns={"index": "Date"})
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
                            title=f"{user_symbol} Forecast (Next {horizon_days} days)",
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
                        st.warning("Prophet failed. Falling back to a simple trend forecast.")
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
                        title=f"{user_symbol} Forecast (Trend) - Next {horizon_days} days",
                        xaxis_title="Date",
                        yaxis_title="Price",
                        height=550,
                        hovermode="x unified",
                        template="plotly_dark",
                    )
                    st.plotly_chart(fig, use_container_width=True)

            st.caption("⚠️ Forecasts are experimental and for education only. Not financial advice.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

elif search_button:
    st.warning("⚠️ Please enter a stock symbol to analyze.")

else:
    st.info("👈 Enter a stock symbol in the sidebar to get started!")
    st.markdown(
        """
### 🎯 Features:
- **Price charts** (candlestick + volume)
- **Financial snapshot** (Yahoo - optional)
- **Fundamentals trends** (Yahoo - may be blocked on cloud)
- **AI forecast** (Prophet if available, otherwise trend fallback)

### 🇸🇦 تداول (Saudi):
- اكتب: **2222.SR**, **1120.SR** ...
- الأسعار تُسحب من Twelve Data (MIC=XSAU) — يحتاج API KEY في Secrets
"""
    )

st.markdown("---")
st.caption("⚠️ Disclaimer: This tool is for educational purposes only. Not financial advice.")
