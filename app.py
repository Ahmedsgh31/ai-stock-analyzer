"""
AI-Powered Stock Market Analyzer — v2 (yfinance rebuild)
══════════════════════════════════════════════════════════
✅ NO API KEYS REQUIRED — uses yfinance (free, no limits)
✅ Saudi Tadawul stocks supported (e.g. 2222.SR, 1120.SR)
✅ Full fundamentals: P/E, P/B, EV/EBITDA, margins, cash flow
✅ Technical indicators: RSI, MACD, Bollinger Bands
✅ ARIMA forecast — statistically sound, no Prophet dependency issues
✅ AI narrative analysis via Claude/OpenAI (optional — add key to Secrets)

Optional AI narrative key (add to Streamlit → Settings → Secrets):
    ANTHROPIC_API_KEY = "your_key"   ← uses Claude
    OPENAI_API_KEY    = "your_key"   ← uses GPT-4o (fallback)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import requests

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="AI Stock Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 4px 0;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #cdd6f4;
        border-left: 3px solid #89b4fa;
        padding-left: 10px;
        margin: 20px 0 12px 0;
    }
    .positive { color: #a6e3a1; }
    .negative { color: #f38ba8; }
    .neutral  { color: #cdd6f4; }
    [data-testid="stMetricValue"] { font-size: 1.1rem !important; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════
def _sf(x):
    try:
        if x is None: return None
        if isinstance(x, (int, float, np.number)):
            f = float(x)
            return None if (np.isnan(f) or np.isinf(f)) else f
        s = str(x).replace(",", "").strip()
        if s.lower() in ("", "none", "n/a", "null", "nan", "-", "inf", "infinity"):
            return None
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

def _pct(x, already_pct=False):
    v = _sf(x)
    if v is None: return "N/A"
    if not already_pct and abs(v) <= 1.5:
        v *= 100
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

def _str(d, *keys):
    for k in keys:
        v = d.get(k)
        if v and str(v).strip().lower() not in ("", "none", "n/a", "null", "-"):
            return str(v).strip()
    return None

# ════════════════════════════════════════════════════════
# DATA LAYER — yfinance (free, no API key needed)
# ════════════════════════════════════════════════════════
@st.cache_data(ttl=300, show_spinner=False)
def get_ticker_data(symbol: str):
    """Fetch everything from yfinance in one call."""
    try:
        t = yf.Ticker(symbol)
        info = t.info or {}
        return t, info
    except Exception as e:
        return None, {}

@st.cache_data(ttl=300, show_spinner=False)
def get_history(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """Fetch OHLCV history."""
    try:
        t = yf.Ticker(symbol)
        df = t.history(period=period, interval=interval, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def get_financials(symbol: str):
    """Fetch income statement, balance sheet, cash flow."""
    try:
        t = yf.Ticker(symbol)
        return {
            "income":   t.financials,
            "balance":  t.balance_sheet,
            "cashflow": t.cashflow,
        }
    except Exception:
        return {"income": None, "balance": None, "cashflow": None}

def latest_val(df, *row_names):
    """Get the most recent value from a financials DataFrame."""
    if df is None or df.empty:
        return None
    for name in row_names:
        if name in df.index:
            row = df.loc[name].dropna()
            if not row.empty:
                return _sf(row.iloc[0])
    return None

# ════════════════════════════════════════════════════════
# TECHNICAL INDICATORS (pure numpy/pandas — no ta-lib needed)
# ════════════════════════════════════════════════════════
def calc_rsi(series: pd.Series, period=14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_g = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_l = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_f  = series.ewm(span=fast,   adjust=False).mean()
    ema_s  = series.ewm(span=slow,   adjust=False).mean()
    macd   = ema_f - ema_s
    sig    = macd.ewm(span=signal,   adjust=False).mean()
    hist   = macd - sig
    return macd, sig, hist

def calc_bbands(series: pd.Series, period=20, std=2):
    ma    = series.rolling(period).mean()
    sd    = series.rolling(period).std()
    upper = ma + std * sd
    lower = ma - std * sd
    return upper, ma, lower

def calc_sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()

# ════════════════════════════════════════════════════════
# ARIMA FORECAST
# ════════════════════════════════════════════════════════
def arima_forecast(series: pd.Series, horizon: int):
    """
    Fit ARIMA(1,1,1) on log-prices and return forecast + 95% CI.
    Falls back gracefully if statsmodels not available.
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
        log_s = np.log(series.values.astype(float))
        model = ARIMA(log_s, order=(1, 1, 1))
        fit   = model.fit()
        fc    = fit.get_forecast(steps=horizon)
        mean  = np.exp(fc.predicted_mean)
        ci    = np.exp(fc.conf_int(alpha=0.05))
        last  = series.index[-1]
        fdate = pd.date_range(last, periods=horizon + 1, freq="B")[1:]
        return pd.DataFrame({
            "Date":  fdate,
            "Forecast": mean,
            "Low CI":   ci[:, 0],
            "High CI":  ci[:, 1],
        })
    except ImportError:
        # statsmodels not installed — use linear trend with noise estimate
        return _linear_forecast(series, horizon)
    except Exception:
        return _linear_forecast(series, horizon)

def _linear_forecast(series: pd.Series, horizon: int):
    """Linear trend fallback with simple std-based CI."""
    y    = series.values.astype(float)
    x    = np.arange(len(y))
    coef = np.polyfit(x, y, 1)
    trend = np.poly1d(coef)
    xa   = np.arange(len(y), len(y) + horizon)
    yhat = trend(xa)
    std  = np.std(y[-30:]) if len(y) >= 30 else np.std(y)
    last = series.index[-1]
    fd   = pd.date_range(last, periods=horizon + 1, freq="B")[1:]
    return pd.DataFrame({
        "Date":    fd,
        "Forecast": yhat,
        "Low CI":   yhat - 1.96 * std,
        "High CI":  yhat + 1.96 * std,
    })

# ════════════════════════════════════════════════════════
# AI NARRATIVE (optional — needs ANTHROPIC_API_KEY or OPENAI_API_KEY)
# ════════════════════════════════════════════════════════
def get_ai_narrative(symbol, name, info, fc_df, rsi_val, macd_val):
    """Call Claude or GPT to generate a plain-English stock analysis."""
    anthropic_key = None
    openai_key    = None
    try:
        anthropic_key = st.secrets.get("ANTHROPIC_API_KEY")
        openai_key    = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        pass

    if not anthropic_key and not openai_key:
        return None

    pe      = _sf(info.get("trailingPE") or info.get("forwardPE"))
    rev     = _sf(info.get("totalRevenue"))
    margin  = _sf(info.get("profitMargins"))
    beta    = _sf(info.get("beta"))
    target  = _sf(info.get("targetMeanPrice"))
    price   = _sf(info.get("currentPrice") or info.get("regularMarketPrice"))
    sector  = info.get("sector", "Unknown")

    fc_end  = f"{fc_df['Forecast'].iloc[-1]:.2f}" if fc_df is not None and not fc_df.empty else "N/A"
    upside  = f"{((fc_df['Forecast'].iloc[-1]/price - 1)*100):+.1f}%" if (
        fc_df is not None and not fc_df.empty and price) else "N/A"

    prompt = f"""You are a CFA-level financial analyst. Analyze {name} ({symbol}) based on:

Key metrics:
- Sector: {sector}
- Current Price: {price}
- P/E Ratio: {_num(pe)}
- Revenue: {_money(rev)}
- Profit Margin: {_pct(margin)}
- Beta: {_num(beta)}
- Analyst Mean Target: {_num(target, "$")}
- RSI (14): {_num(rsi_val)}
- MACD signal: {"Bullish" if macd_val and macd_val > 0 else "Bearish" if macd_val else "N/A"}
- 30-day ARIMA forecast: {fc_end} ({upside} from current)

Write a concise 4-paragraph analysis:
1. Business snapshot and competitive position
2. Valuation assessment (cheap/fair/expensive vs sector norms)
3. Technical momentum reading (RSI, MACD, trend)
4. Key risks and opportunities — what should an investor watch?

Be direct and specific. Avoid generic disclaimers inside the analysis. End with one sentence summary verdict."""

    # Try Claude first
    if anthropic_key:
        try:
            r = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": anthropic_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 800,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=30,
            )
            data = r.json()
            return data["content"][0]["text"]
        except Exception:
            pass

    # Fallback to OpenAI
    if openai_key:
        try:
            r = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {openai_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-4o",
                    "max_tokens": 800,
                    "messages": [
                        {"role": "system", "content": "You are a CFA-level financial analyst."},
                        {"role": "user",   "content": prompt},
                    ],
                },
                timeout=30,
            )
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except Exception:
            pass

    return None

# ════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════
with st.sidebar:
    st.title("📈 Stock Analyzer")
    st.caption("Powered by yfinance — no API keys needed")
    st.markdown("---")

    stock_input = st.text_input(
        "Stock Symbol",
        placeholder="AAPL · TSLA · 2222.SR",
        help="US: AAPL TSLA NVDA MSFT | Saudi: 2222.SR 1120.SR 2010.SR | Gulf: EMAAR.AE",
    )

    period_map = {
        "1 Month":  ("1mo",  "1d"),
        "3 Months": ("3mo",  "1d"),
        "6 Months": ("6mo",  "1d"),
        "1 Year":   ("1y",   "1d"),
        "2 Years":  ("2y",   "1wk"),
        "5 Years":  ("5y",   "1wk"),
    }
    sel_period = st.selectbox("Time Period", list(period_map.keys()), index=3)

    st.markdown("---")
    st.markdown("**Examples**")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**US**\n- `AAPL`\n- `TSLA`\n- `NVDA`\n- `MSFT`\n- `AMZN`")
    with col_b:
        st.markdown("**Saudi**\n- `2222.SR`\n- `1120.SR`\n- `2010.SR`\n- `1180.SR`\n- `7010.SR`")

    st.markdown("---")
    # AI key status
    try:
        has_ai = bool(st.secrets.get("ANTHROPIC_API_KEY") or st.secrets.get("OPENAI_API_KEY"))
    except Exception:
        has_ai = False
    st.markdown("**AI Narrative**")
    if has_ai:
        st.markdown("🟢 AI analysis enabled")
    else:
        st.markdown("🟡 AI analysis disabled")
        with st.expander("Enable AI narrative"):
            st.markdown(
                "Add to **Settings → Secrets**:\n"
                "```\nANTHROPIC_API_KEY = \"sk-ant-...\"\n```\n"
                "or\n"
                "```\nOPENAI_API_KEY = \"sk-...\"\n```"
            )

    st.markdown("---")
    go_btn = st.button("🔍 Analyze Stock", type="primary", use_container_width=True)

# ════════════════════════════════════════════════════════
# LANDING PAGE
# ════════════════════════════════════════════════════════
if "result" not in st.session_state:
    st.session_state.result = None

if not go_btn and not st.session_state.result:
    st.title("AI-Powered Stock Market Analyzer")
    st.markdown("""
> **No API keys required.** Powered entirely by Yahoo Finance — free for everyone.

### Supported Markets
| Market | Examples | Notes |
|--------|----------|-------|
| NASDAQ / NYSE | `AAPL` `TSLA` `NVDA` `MSFT` `GOOGL` | Full data |
| Saudi Tadawul | `2222.SR` `1120.SR` `2010.SR` | Full data ✅ |
| Gulf / GCC | `EMAAR.AE` `FAB.AE` | Partial |
| Global | Most major exchanges | Varies |

### What you get
- 📊 **Interactive price chart** — candlestick + volume + moving averages
- 💼 **Full fundamentals** — P/E, P/B, EV/EBITDA, margins, cash flow, balance sheet
- 📉 **Technical indicators** — RSI, MACD, Bollinger Bands
- 🔮 **ARIMA forecast** — statistically sound 7–90 day outlook with confidence intervals
- 🤖 **AI narrative** — CFA-level plain-English analysis (optional, add API key)
""")
    st.stop()

# ════════════════════════════════════════════════════════
# ON BUTTON PRESS — fetch data
# ════════════════════════════════════════════════════════
if go_btn:
    raw = (stock_input or "").strip().upper()
    if not raw:
        st.warning("⚠️ Enter a stock symbol first.")
        st.stop()

    period_yf, interval_yf = period_map[sel_period]

    with st.spinner(f"Loading data for **{raw}**…"):
        ticker, info = get_ticker_data(raw)
        hist = get_history(raw, period_yf, interval_yf)

    if hist.empty or ticker is None:
        st.error(f"❌ No data found for **{raw}**")
        st.info(
            "**Tips:**\n"
            "- Saudi stocks use `.SR` suffix: `2222.SR` not `2222`\n"
            "- Check the symbol at [finance.yahoo.com](https://finance.yahoo.com)\n"
            "- Some smaller stocks may not be available"
        )
        st.stop()

    # Validate we got real price data
    if "Close" not in hist.columns or hist["Close"].dropna().empty:
        st.error(f"❌ Price data unavailable for **{raw}**")
        st.stop()

    st.session_state.result = {
        "symbol":   raw,
        "info":     info,
        "hist":     hist,
        "period":   sel_period,
        "interval": interval_yf,
    }

# ════════════════════════════════════════════════════════
# RENDER
# ════════════════════════════════════════════════════════
res = st.session_state.result
if not res:
    st.stop()

symbol   = res["symbol"]
info     = res["info"]
hist     = res["hist"]
currency = info.get("currency", "USD")
name     = info.get("longName") or info.get("shortName") or symbol
exchange = info.get("exchange") or info.get("fullExchangeName") or ""

# ── Header ────────────────────────────────────────────────
close_s   = hist["Close"].dropna()
cur_price = _sf(info.get("currentPrice") or info.get("regularMarketPrice")) or (
    float(close_s.iloc[-1]) if not close_s.empty else None)
prev_close = _sf(info.get("previousClose") or info.get("regularMarketPreviousClose"))

h1, h2, h3, h4, h5 = st.columns([3, 1.5, 1.5, 1.5, 1.5])
with h1:
    st.subheader(name)
    st.caption(f"**{symbol}** · {exchange} · {currency}")

with h2:
    if cur_price and prev_close:
        chg  = cur_price - prev_close
        chgp = chg / prev_close * 100
        st.metric("Price", f"{cur_price:.2f}", f"{chg:+.2f} ({chgp:+.2f}%)")
    elif cur_price:
        st.metric("Price", f"{cur_price:.2f}")
    else:
        st.metric("Price", "N/A")

with h3:
    hi52 = _sf(info.get("fiftyTwoWeekHigh"))
    st.metric("52W High", f"{hi52:.2f}" if hi52 else "N/A")
with h4:
    lo52 = _sf(info.get("fiftyTwoWeekLow"))
    st.metric("52W Low", f"{lo52:.2f}" if lo52 else "N/A")
with h5:
    mktcap = _sf(info.get("marketCap"))
    st.metric("Market Cap", _money(mktcap))

st.markdown("---")

# ── Tabs ─────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Price & Technicals",
    "💼 Fundamentals",
    "🔮 Forecast",
    "🤖 AI Analysis",
])

# ════════════════════════════════════════════════════════
# TAB 1 — Price Chart + Technical Indicators
# ════════════════════════════════════════════════════════
with tab1:
    close = hist["Close"].dropna()

    # Calculate indicators
    sma20  = calc_sma(close, 20)
    sma50  = calc_sma(close, 50)
    sma200 = calc_sma(close, 200)
    rsi    = calc_rsi(close)
    macd_line, macd_sig, macd_hist_vals = calc_macd(close)
    bb_up, bb_mid, bb_low = calc_bbands(close)

    # Current RSI and MACD values for AI
    rsi_current  = float(rsi.dropna().iloc[-1])  if not rsi.dropna().empty  else None
    macd_current = float(macd_hist_vals.dropna().iloc[-1]) if not macd_hist_vals.dropna().empty else None

    # Store for AI tab
    st.session_state["rsi_current"]  = rsi_current
    st.session_state["macd_current"] = macd_current

    # ── Main price chart ─────────────────────────────────
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=[f"{symbol} Price", "RSI (14)", "MACD"],
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=hist.index,
        open=hist["Open"], high=hist["High"],
        low=hist["Low"],   close=hist["Close"],
        name="Price",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
    ), row=1, col=1)

    # Bollinger Bands
    fig.add_trace(go.Scatter(
        x=hist.index, y=bb_up, mode="lines",
        line=dict(color="rgba(100,180,255,0.3)", width=1),
        name="BB Upper", showlegend=True,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=hist.index, y=bb_low, mode="lines",
        fill="tonexty", fillcolor="rgba(100,180,255,0.06)",
        line=dict(color="rgba(100,180,255,0.3)", width=1),
        name="BB Lower",
    ), row=1, col=1)

    # Moving averages
    for sma, label, color in [
        (sma20,  "SMA 20",  "#f9e2af"),
        (sma50,  "SMA 50",  "#89b4fa"),
        (sma200, "SMA 200", "#cba6f7"),
    ]:
        if sma.dropna().shape[0] > 5:
            fig.add_trace(go.Scatter(
                x=hist.index, y=sma, mode="lines",
                line=dict(width=1.2, color=color),
                name=label,
            ), row=1, col=1)

    # Volume bar
    if "Volume" in hist.columns and hist["Volume"].sum() > 0:
        colors = ["#26a69a" if c >= o else "#ef5350"
                  for c, o in zip(hist["Close"], hist["Open"])]
        fig.add_trace(go.Bar(
            x=hist.index, y=hist["Volume"],
            name="Volume", marker_color=colors,
            opacity=0.4, showlegend=False,
        ), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(
        x=hist.index, y=rsi, mode="lines",
        line=dict(color="#89b4fa", width=1.5), name="RSI",
    ), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#f38ba8",
                  line_width=1, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#a6e3a1",
                  line_width=1, row=2, col=1)
    fig.add_hrect(y0=30, y1=70, fillcolor="rgba(137,180,250,0.05)",
                  line_width=0, row=2, col=1)

    # MACD
    macd_colors = ["#26a69a" if v >= 0 else "#ef5350"
                   for v in macd_hist_vals.fillna(0)]
    fig.add_trace(go.Bar(
        x=hist.index, y=macd_hist_vals,
        marker_color=macd_colors, name="MACD Hist",
        opacity=0.7,
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=hist.index, y=macd_line, mode="lines",
        line=dict(color="#89b4fa", width=1.2), name="MACD",
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=hist.index, y=macd_sig, mode="lines",
        line=dict(color="#f9e2af", width=1.2), name="Signal",
    ), row=3, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=700,
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        margin=dict(l=40, r=20, t=40, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right",  x=1,
            font=dict(size=11),
        ),
    )
    fig.update_yaxes(title_text=currency, row=1, col=1)
    fig.update_yaxes(title_text="RSI",    row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD",   row=3, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # Technical summary cards
    st.markdown('<div class="section-header">Technical Summary</div>', unsafe_allow_html=True)
    tc1, tc2, tc3, tc4, tc5 = st.columns(5)

    rsi_label = "Overbought" if (rsi_current or 50) > 70 else (
                "Oversold"   if (rsi_current or 50) < 30 else "Neutral")
    rsi_color = "negative" if (rsi_current or 50) > 70 else (
                "positive"  if (rsi_current or 50) < 30 else "neutral")

    macd_label = "Bullish" if (macd_current or 0) > 0 else "Bearish"
    macd_color = "positive" if (macd_current or 0) > 0 else "negative"

    above_200 = bool(cur_price and not sma200.dropna().empty and
                     cur_price > float(sma200.dropna().iloc[-1]))

    tc1.metric("RSI (14)",   f"{rsi_current:.1f}" if rsi_current else "N/A",
               delta=rsi_label)
    tc2.metric("MACD",       f"{macd_current:.3f}" if macd_current else "N/A",
               delta=macd_label)
    tc3.metric("SMA 20",     f"{float(sma20.dropna().iloc[-1]):.2f}"  if not sma20.dropna().empty  else "N/A")
    tc4.metric("SMA 50",     f"{float(sma50.dropna().iloc[-1]):.2f}"  if not sma50.dropna().empty  else "N/A")
    tc5.metric("Above SMA200", "✅ Yes" if above_200 else "❌ No")

# ════════════════════════════════════════════════════════
# TAB 2 — Fundamentals (all from yfinance .info — works for Saudi too)
# ════════════════════════════════════════════════════════
with tab2:
    fins = get_financials(symbol)
    inc  = fins["income"]
    bal  = fins["balance"]
    cf   = fins["cashflow"]

    # ── Valuation ────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Valuation</div>', unsafe_allow_html=True)
    v1, v2, v3, v4 = st.columns(4)
    v5, v6, v7, v8 = st.columns(4)

    pe       = _sf(info.get("trailingPE")    or info.get("forwardPE"))
    fwd_pe   = _sf(info.get("forwardPE"))
    pb       = _sf(info.get("priceToBook"))
    ps       = _sf(info.get("priceToSalesTrailing12Months"))
    ev_ebit  = _sf(info.get("enterpriseToEbitda"))
    ev_rev   = _sf(info.get("enterpriseToRevenue"))
    peg      = _sf(info.get("pegRatio"))
    beta     = _sf(info.get("beta"))
    eps      = _sf(info.get("trailingEps"))
    fwd_eps  = _sf(info.get("forwardEps"))

    v1.metric("P/E (Trailing)",   _num(pe)       if pe      else "N/A")
    v2.metric("P/E (Forward)",    _num(fwd_pe)   if fwd_pe  else "N/A")
    v3.metric("P/B Ratio",        _num(pb)       if pb      else "N/A")
    v4.metric("P/S Ratio",        _num(ps)       if ps      else "N/A")
    v5.metric("EV/EBITDA",        _num(ev_ebit)  if ev_ebit else "N/A")
    v6.metric("EV/Revenue",       _num(ev_rev)   if ev_rev  else "N/A")
    v7.metric("PEG Ratio",        _num(peg)      if peg     else "N/A")
    v8.metric("Beta",             _num(beta)     if beta    else "N/A")

    st.markdown("---")

    # ── Profitability ────────────────────────────────────
    st.markdown('<div class="section-header">📈 Profitability & Growth</div>', unsafe_allow_html=True)
    p1, p2, p3, p4 = st.columns(4)
    p5, p6, p7, p8 = st.columns(4)

    rev        = _sf(info.get("totalRevenue"))
    rev_g      = _sf(info.get("revenueGrowth"))
    gross_m    = _sf(info.get("grossMargins"))
    op_m       = _sf(info.get("operatingMargins"))
    net_m      = _sf(info.get("profitMargins"))
    ebitda     = _sf(info.get("ebitda"))
    roe        = _sf(info.get("returnOnEquity"))
    roa        = _sf(info.get("returnOnAssets"))

    p1.metric("Revenue (TTM)",     _money(rev))
    p2.metric("Revenue Growth",    _pct(rev_g)   if rev_g   else "N/A")
    p3.metric("Gross Margin",      _pct(gross_m) if gross_m else "N/A")
    p4.metric("Operating Margin",  _pct(op_m)    if op_m    else "N/A")
    p5.metric("Net Margin",        _pct(net_m)   if net_m   else "N/A")
    p6.metric("EBITDA",            _money(ebitda))
    p7.metric("Return on Equity",  _pct(roe)     if roe     else "N/A")
    p8.metric("Return on Assets",  _pct(roa)     if roa     else "N/A")

    st.markdown("---")

    # ── EPS & Dividends ─────────────────────────────────
    st.markdown('<div class="section-header">💰 EPS & Dividends</div>', unsafe_allow_html=True)
    e1, e2, e3, e4 = st.columns(4)
    e5, e6, e7, e8 = st.columns(4)

    div_y    = _sf(info.get("dividendYield"))
    div_r    = _sf(info.get("dividendRate"))
    payout   = _sf(info.get("payoutRatio"))
    ex_div   = _str(info, "exDividendDate")
    target_p = _sf(info.get("targetMeanPrice"))
    target_h = _sf(info.get("targetHighPrice"))
    target_l = _sf(info.get("targetLowPrice"))
    rec      = _str(info, "recommendationKey")

    e1.metric("EPS (TTM)",         _num(eps,    "$") if eps     else "N/A")
    e2.metric("EPS (Forward)",     _num(fwd_eps,"$") if fwd_eps else "N/A")
    e3.metric("Dividend Yield",    _pct(div_y)       if div_y   else "N/A")
    e4.metric("Dividend/Share",    _num(div_r,  "$") if div_r   else "N/A")
    e5.metric("Payout Ratio",      _pct(payout)      if payout  else "N/A")
    e6.metric("Analyst Target",    _num(target_p,"$")if target_p else "N/A")
    e7.metric("Target Range",
              f"${target_l:.2f}–${target_h:.2f}" if target_l and target_h else "N/A")
    e8.metric("Recommendation",    (rec or "N/A").replace("_", " ").title())

    st.markdown("---")

    # ── Balance Sheet ────────────────────────────────────
    st.markdown('<div class="section-header">🏦 Balance Sheet</div>', unsafe_allow_html=True)
    b1, b2, b3, b4 = st.columns(4)
    b5, b6, b7, b8 = st.columns(4)

    cash       = _sf(info.get("totalCash"))
    debt       = _sf(info.get("totalDebt"))
    assets     = latest_val(bal, "Total Assets")
    equity_v   = _sf(info.get("bookValue"))
    shares     = _sf(info.get("sharesOutstanding"))
    cr         = _sf(info.get("currentRatio"))
    qr         = _sf(info.get("quickRatio"))
    de         = _sf(info.get("debtToEquity"))

    b1.metric("Total Cash",        _money(cash))
    b2.metric("Total Debt",        _money(debt))
    b3.metric("Book Value/Share",  _num(equity_v, "$") if equity_v else "N/A")
    b4.metric("Shares Outstanding",_money(shares).replace("$","") if shares else "N/A")
    b5.metric("Current Ratio",     _num(cr)    if cr  else "N/A")
    b6.metric("Quick Ratio",       _num(qr)    if qr  else "N/A")
    b7.metric("Debt/Equity",       _num(de)    if de  else "N/A")
    b8.metric("Cash/Share",        _num(_sf(info.get("totalCashPerShare")), "$")
              if info.get("totalCashPerShare") else "N/A")

    st.markdown("---")

    # ── Cash Flow ────────────────────────────────────────
    st.markdown('<div class="section-header">💵 Cash Flow</div>', unsafe_allow_html=True)
    c1f, c2f, c3f, c4f = st.columns(4)

    op_cf  = _sf(info.get("operatingCashflow"))
    fcf    = _sf(info.get("freeCashflow"))
    capex  = (op_cf - fcf) if (op_cf and fcf) else None

    # Fallback from statements
    if op_cf is None:
        op_cf = latest_val(cf, "Operating Cash Flow", "Cash Flow From Continuing Operating Activities")
    if fcf is None:
        fcf = latest_val(cf, "Free Cash Flow")

    c1f.metric("Operating CF",    _money(op_cf))
    c2f.metric("Free Cash Flow",  _money(fcf))
    c3f.metric("CapEx",           _money(capex))
    c4f.metric("FCF Yield",
               _pct(_sf(fcf) / mktcap) if (fcf and mktcap) else "N/A")

    st.markdown("---")

    # ── Company Info ─────────────────────────────────────
    st.markdown('<div class="section-header">🏢 Company Profile</div>', unsafe_allow_html=True)
    i1, i2, i3, i4 = st.columns(4)

    sector   = _str(info, "sector")
    industry = _str(info, "industry")
    country  = _str(info, "country")
    emp      = _sf(info.get("fullTimeEmployees"))
    website  = _str(info, "website")
    hq_city  = _str(info, "city")

    i1.metric("Sector",    sector   or "N/A")
    i2.metric("Industry",  industry or "N/A")
    i3.metric("Country",   country  or "N/A")
    i4.metric("Employees", f"{int(emp):,}" if emp else "N/A")

    if info.get("longBusinessSummary"):
        with st.expander("Business Summary"):
            st.write(info["longBusinessSummary"])

# ════════════════════════════════════════════════════════
# TAB 3 — ARIMA Forecast
# ════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🔮 Price Forecast")
    st.caption("Uses ARIMA(1,1,1) on log-prices. Confidence intervals shown. Not financial advice.")

    close_clean = hist["Close"].dropna()

    col_h, col_m = st.columns([3, 1])
    with col_h:
        horizon = st.slider("Forecast horizon (trading days)", 7, 90, 30, 1)
    with col_m:
        show_ci = st.checkbox("Show confidence interval", value=True)

    if len(close_clean) < 30:
        st.warning("Not enough data for a reliable forecast. Try a longer time period.")
    else:
        with st.spinner("Running ARIMA model…"):
            fc_df = arima_forecast(close_clean, horizon)
            st.session_state["fc_df"] = fc_df

        last_date = close_clean.index[-1]

        fig2 = go.Figure()

        # Historical
        fig2.add_trace(go.Scatter(
            x=close_clean.index, y=close_clean,
            mode="lines", name="Actual",
            line=dict(color="#89b4fa", width=1.5),
        ))

        # Confidence interval fill
        if show_ci:
            fig2.add_trace(go.Scatter(
                x=pd.concat([fc_df["Date"], fc_df["Date"][::-1]]),
                y=pd.concat([fc_df["High CI"], fc_df["Low CI"][::-1]]),
                fill="toself",
                fillcolor="rgba(166,227,161,0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                name="95% CI",
                showlegend=True,
            ))

        # Forecast line
        fig2.add_trace(go.Scatter(
            x=fc_df["Date"], y=fc_df["Forecast"],
            mode="lines", name="Forecast",
            line=dict(color="#a6e3a1", width=2, dash="dash"),
        ))

        # Vertical marker at today
        fig2.add_vline(
            x=last_date, line_dash="dot",
            line_color="rgba(255,255,255,0.3)", line_width=1,
        )

        fig2.update_layout(
            template="plotly_dark",
            height=500,
            hovermode="x unified",
            xaxis_title="Date",
            yaxis_title=f"Price ({currency})",
            margin=dict(l=40, r=20, t=20, b=40),
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Summary metrics
        fc_end    = float(fc_df["Forecast"].iloc[-1])
        fc_low    = float(fc_df["Low CI"].iloc[-1])
        fc_high   = float(fc_df["High CI"].iloc[-1])
        cur       = float(close_clean.iloc[-1])
        upside    = (fc_end / cur - 1) * 100

        fm1, fm2, fm3, fm4 = st.columns(4)
        fm1.metric("Current Price",    f"{cur:.2f} {currency}")
        fm2.metric(f"{horizon}d Target", f"{fc_end:.2f}",
                   delta=f"{upside:+.1f}%",
                   delta_color="normal")
        fm3.metric("CI Low",           f"{fc_low:.2f}")
        fm4.metric("CI High",          f"{fc_high:.2f}")

        # Forecast table
        with st.expander("Forecast table"):
            tbl = fc_df.copy()
            tbl["Date"] = tbl["Date"].dt.strftime("%Y-%m-%d")
            for col in ["Forecast", "Low CI", "High CI"]:
                tbl[col] = tbl[col].round(2)
            st.dataframe(tbl, use_container_width=True)

    st.warning("⚠️ Forecasts are statistical models based on historical price patterns only. "
               "They do not account for news, earnings, or market conditions. Not financial advice.")

# ════════════════════════════════════════════════════════
# TAB 4 — AI Narrative
# ════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 🤖 AI Investment Analysis")

    rsi_c  = st.session_state.get("rsi_current")
    macd_c = st.session_state.get("macd_current")
    fc_d   = st.session_state.get("fc_df")

    has_ai_key = False
    try:
        has_ai_key = bool(
            st.secrets.get("ANTHROPIC_API_KEY") or
            st.secrets.get("OPENAI_API_KEY")
        )
    except Exception:
        pass

    if not has_ai_key:
        st.info(
            "**AI narrative requires an API key.**\n\n"
            "Add one of these to **Settings → Secrets**:\n"
            "```\nANTHROPIC_API_KEY = \"sk-ant-...\"\n```\n"
            "or\n"
            "```\nOPENAI_API_KEY = \"sk-...\"\n```\n\n"
            "Once added, the AI will generate a CFA-level analysis of the stock "
            "combining fundamentals, technicals, and the forecast above."
        )
    else:
        if st.button("🤖 Generate AI Analysis", type="primary"):
            with st.spinner("Generating CFA-level analysis…"):
                narrative = get_ai_narrative(
                    symbol, name, info, fc_d, rsi_c, macd_c
                )
            if narrative:
                st.markdown(narrative)
                st.caption("⚠️ AI-generated analysis. Not financial advice. "
                           "Always do your own research.")
            else:
                st.error("AI analysis failed. Check your API key in Secrets.")

st.markdown("---")
st.caption("⚠️ Educational purposes only. Not financial advice. Data from Yahoo Finance.")
