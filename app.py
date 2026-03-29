"""
AI-Powered Stock Market Analyzer — v3
══════════════════════════════════════
✅ NO API KEYS REQUIRED for stock data (yfinance)
✅ Saudi Tadawul stocks fully supported (2222.SR, 1120.SR, etc.)
✅ Full fundamentals: P/E, EV/EBITDA, margins, cash flow, balance sheet
✅ Technical indicators: RSI, MACD, Bollinger Bands, SMA
✅ ARIMA forecast with 95% confidence intervals
✅ AI narrative always shown (add key to Secrets to activate)

Add to Streamlit → Settings → Secrets to enable AI analysis:
    ANTHROPIC_API_KEY = "sk-ant-..."   ← Claude (recommended)
    OPENAI_API_KEY    = "sk-..."       ← GPT-4o (fallback)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import time

st.set_page_config(
    page_title="AI Stock Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.1rem !important; }
[data-testid="stMetricDelta"] { font-size: 0.8rem !important; }
.section-hdr {
    font-size: 1rem; font-weight: 600;
    border-left: 3px solid #89b4fa;
    padding-left: 10px; margin: 18px 0 10px 0;
    color: #cdd6f4;
}
</style>
""", unsafe_allow_html=True)

def hdr(text):
    st.markdown(f'<div class="section-hdr">{text}</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════
def _sf(x):
    try:
        if x is None: return None
        if isinstance(x, (int, float, np.integer, np.floating)):
            f = float(x)
            return None if (np.isnan(f) or np.isinf(f) or f == 0) else f
        s = str(x).replace(",", "").strip()
        if s.lower() in ("", "none", "n/a", "null", "nan", "-", "inf", "0"):
            return None
        f = float(s)
        return None if (np.isnan(f) or np.isinf(f)) else f
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

def _str(d, *keys):
    for k in keys:
        v = d.get(k) if isinstance(d, dict) else None
        if v and str(v).strip().lower() not in ("", "none", "n/a", "null", "-", "nan"):
            return str(v).strip()
    return None

# ════════════════════════════════════════════════════════
# DATA — yfinance with robust fallback
# ════════════════════════════════════════════════════════
@st.cache_data(ttl=300, show_spinner=False)
def fetch_history(symbol: str, period: str, interval: str) -> pd.DataFrame:
    import yfinance as yf
    periods_to_try = {
        "1mo":  ["1mo", "3mo"],
        "3mo":  ["3mo", "6mo"],
        "6mo":  ["6mo", "1y"],
        "1y":   ["1y",  "2y"],
        "2y":   ["2y",  "5y"],
        "5y":   ["5y",  "max"],
    }.get(period, [period])

    for p in periods_to_try:
        try:
            df = yf.download(
                symbol, period=p, interval=interval,
                auto_adjust=True, progress=False,
                timeout=30
            )
            if df is not None and not df.empty and len(df) > 3:
                df.index = pd.to_datetime(df.index).tz_localize(None)
                # yfinance >= 0.2.40 returns MultiIndex columns — flatten
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0] for c in df.columns]
                df = df.dropna(subset=["Close"])
                if len(df) > 3:
                    return df
        except Exception:
            time.sleep(1)
    return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)
def fetch_info(symbol: str) -> dict:
    import yfinance as yf
    try:
        t    = yf.Ticker(symbol)
        info = t.info
        if info and isinstance(info, dict) and len(info) > 5:
            return info
    except Exception:
        pass
    # fast_info fallback
    try:
        t  = yf.Ticker(symbol)
        fi = t.fast_info
        return {
            "currentPrice":        getattr(fi, "last_price",     None),
            "previousClose":       getattr(fi, "previous_close", None),
            "fiftyTwoWeekHigh":    getattr(fi, "year_high",      None),
            "fiftyTwoWeekLow":     getattr(fi, "year_low",       None),
            "marketCap":           getattr(fi, "market_cap",     None),
            "currency":            getattr(fi, "currency",
                                           "SAR" if ".SR" in symbol else "USD"),
            "shortName": symbol,
        }
    except Exception:
        pass
    return {}

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_financials(symbol: str) -> dict:
    import yfinance as yf
    out = {"income": None, "balance": None, "cashflow": None}
    try:
        t = yf.Ticker(symbol)
        out["income"]   = t.financials
        out["balance"]  = t.balance_sheet
        out["cashflow"] = t.cashflow
    except Exception:
        pass
    return out

def latest_val(df, *names):
    if df is None or (hasattr(df, "empty") and df.empty):
        return None
    for n in names:
        try:
            if n in df.index:
                row = df.loc[n].dropna()
                if not row.empty:
                    v = _sf(row.iloc[0])
                    if v: return v
        except Exception:
            continue
    return None

# ════════════════════════════════════════════════════════
# TECHNICAL INDICATORS
# ════════════════════════════════════════════════════════
def calc_rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(alpha=1/p, min_periods=p).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1/p, min_periods=p).mean()
    rs = g / l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_macd(s, fast=12, slow=26, sig=9):
    m = s.ewm(span=fast, adjust=False).mean() - s.ewm(span=slow, adjust=False).mean()
    signal = m.ewm(span=sig, adjust=False).mean()
    return m, signal, m - signal

def calc_bb(s, p=20, n=2):
    ma = s.rolling(p).mean()
    sd = s.rolling(p).std()
    return ma + n*sd, ma, ma - n*sd

# ════════════════════════════════════════════════════════
# FORECAST
# ════════════════════════════════════════════════════════
def ensemble_forecast(series: pd.Series, horizon: int) -> pd.DataFrame:
    """
    Ensemble forecast combining 3 models:
      1. ARIMA(1,1,1)  — short-term autocorrelation
      2. Exponential Smoothing (Holt) — captures trend drift
      3. Monte Carlo simulation — captures real historical volatility

    Final forecast = equal-weight average of all three.
    CI is derived from Monte Carlo percentiles (5th / 95th).
    This avoids the flat-line problem of pure ARIMA on low-drift series.
    """
    y     = series.values.astype(float)
    last  = y[-1]
    dates = pd.bdate_range(series.index[-1], periods=horizon + 1)[1:]

    # ── Model 1: ARIMA on log-prices ─────────────────────
    arima_fc = None
    try:
        from statsmodels.tsa.arima.model import ARIMA
        log_s    = np.log(y)
        fit      = ARIMA(log_s, order=(1, 1, 1)).fit()
        raw      = fit.get_forecast(steps=horizon).predicted_mean
        arima_fc = np.exp(raw)
    except Exception:
        arima_fc = np.full(horizon, last)

    # ── Model 2: Holt's Linear Trend (double exp smoothing) ──
    holt_fc = None
    try:
        from statsmodels.tsa.holtwinters import Holt
        fit2    = Holt(y, initialization_method="estimated").fit(
                      optimized=True, remove_bias=True)
        holt_fc = fit2.forecast(horizon)
    except Exception:
        # Manual fallback: use recent slope
        recent  = y[-min(20, len(y)):]
        slope   = (recent[-1] - recent[0]) / len(recent)
        holt_fc = np.array([last + slope * (i + 1) for i in range(horizon)])

    # ── Model 3: Monte Carlo (log-normal random walk) ────
    # Use last 60 days of log-returns for vol estimation
    log_rets  = np.diff(np.log(y[-min(60, len(y)):]))
    mu_daily  = float(np.mean(log_rets))
    sig_daily = float(np.std(log_rets))

    np.random.seed(42)
    n_sims    = 500
    sim_paths = np.zeros((n_sims, horizon))
    for i in range(n_sims):
        shocks          = np.random.normal(mu_daily, sig_daily, horizon)
        log_path        = np.cumsum(shocks)
        sim_paths[i]    = last * np.exp(log_path)

    mc_mean  = sim_paths.mean(axis=0)
    ci_low   = np.percentile(sim_paths,  5, axis=0)
    ci_high  = np.percentile(sim_paths, 95, axis=0)

    # ── Ensemble: equal-weight average ───────────────────
    ensemble = (arima_fc + holt_fc + mc_mean) / 3.0

    return pd.DataFrame({
        "Date":     dates,
        "Forecast": ensemble,
        "ARIMA":    arima_fc,
        "Holt":     holt_fc,
        "MonteCarlo": mc_mean,
        "Low CI":   ci_low,
        "High CI":  ci_high,
    })

# ════════════════════════════════════════════════════════
# AI NARRATIVE
# ════════════════════════════════════════════════════════
def get_api_keys():
    try:
        return {
            "anthropic": st.secrets.get("ANTHROPIC_API_KEY") or None,
            "openai":    st.secrets.get("OPENAI_API_KEY")    or None,
            "gemini":    st.secrets.get("GEMINI_API_KEY")    or None,
        }
    except Exception:
        return {"anthropic": None, "openai": None, "gemini": None}

def _active_provider(keys: dict) -> str:
    """Return the name of the first available provider."""
    if keys["anthropic"]: return "Claude"
    if keys["openai"]:    return "GPT-4o"
    if keys["gemini"]:    return "Gemini"
    return None

def _build_prompt(symbol, name, info, fc_df, rsi_val, macd_val, currency):
    pe      = _num(_sf(info.get("trailingPE")  or info.get("forwardPE")))
    fpe     = _num(_sf(info.get("forwardPE")))
    rev     = _money(_sf(info.get("totalRevenue")))
    margin  = _pct(_sf(info.get("profitMargins")))
    gm      = _pct(_sf(info.get("grossMargins")))
    roe     = _pct(_sf(info.get("returnOnEquity")))
    de      = _num(_sf(info.get("debtToEquity")))
    ev_eb   = _num(_sf(info.get("enterpriseToEbitda")))
    pb      = _num(_sf(info.get("priceToBook")))
    fcf     = _money(_sf(info.get("freeCashflow")))
    mktcap  = _money(_sf(info.get("marketCap")))
    target  = _num(_sf(info.get("targetMeanPrice")), pre=f"{currency} ")
    rec     = (_str(info, "recommendationKey") or "N/A").replace("_"," ").title()
    price   = _sf(info.get("currentPrice") or info.get("regularMarketPrice"))
    sector  = info.get("sector", "Unknown")
    country = info.get("country", "")
    fc_end  = f"{fc_df['Forecast'].iloc[-1]:.2f}" if (fc_df is not None and not fc_df.empty) else "N/A"
    upside  = (f"{((fc_df['Forecast'].iloc[-1]/price - 1)*100):+.1f}%"
               if (fc_df is not None and not fc_df.empty and price and price > 0) else "N/A")
    rsi_txt  = f"{rsi_val:.1f}" if rsi_val else "N/A"
    macd_txt = ("Bullish momentum" if (macd_val and macd_val > 0)
                else "Bearish momentum" if macd_val else "N/A")
    return f"""You are a senior CFA charterholder and portfolio manager. Provide a professional investment analysis of {name} ({symbol}).

COMPANY: {country} | Sector: {sector} | Currency: {currency}
Market Cap: {mktcap} | Price: {price} {currency} | Analyst Target: {target} | Consensus: {rec}

VALUATION: P/E(TTM)={pe} | P/E(Fwd)={fpe} | P/B={pb} | EV/EBITDA={ev_eb}
FINANCIALS: Revenue={rev} | Gross Margin={gm} | Net Margin={margin} | ROE={roe} | D/E={de} | FCF={fcf}
TECHNICALS: RSI(14)={rsi_txt} | MACD={macd_txt}
FORECAST: Ensemble target={fc_end} {currency} | Implied move={upside}

Write exactly 4 paragraphs:
1. BUSINESS OVERVIEW: What does this company do and what is its competitive position?
2. VALUATION VERDICT: Is it cheap, fair, or expensive vs sector norms? Be specific with numbers.
3. TECHNICAL & MOMENTUM: What do RSI, MACD, and the price forecast suggest?
4. RISKS & OPPORTUNITIES: 2 key risks and 2 key catalysts an investor must watch.

End with one bold **VERDICT** sentence. Be direct and analytical."""

def get_ai_narrative(symbol, name, info, fc_df, rsi_val, macd_val, currency):
    keys = get_api_keys()
    if not any(keys.values()):
        return None, "no_key"

    prompt = _build_prompt(symbol, name, info, fc_df, rsi_val, macd_val, currency)

    # ── 1. Claude ────────────────────────────────────────
    if keys["anthropic"]:
        try:
            r = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": keys["anthropic"],
                         "anthropic-version": "2023-06-01",
                         "content-type": "application/json"},
                json={"model": "claude-sonnet-4-20250514", "max_tokens": 900,
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=45,
            )
            d = r.json()
            if "content" in d and d["content"]:
                return d["content"][0]["text"], "Claude"
            err = d.get("error", {}).get("message", "unknown")
            # If out of credits, fall through to next provider
            if "credit" not in err.lower() and "balance" not in err.lower():
                return None, f"claude_error: {err}"
            # else fall through silently
        except Exception as e:
            return None, f"claude_exception: {e}"

    # ── 2. OpenAI ────────────────────────────────────────
    if keys["openai"]:
        try:
            r = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {keys['openai']}",
                         "Content-Type": "application/json"},
                json={"model": "gpt-4o", "max_tokens": 900,
                      "messages": [
                          {"role": "system", "content": "You are a senior CFA charterholder."},
                          {"role": "user",   "content": prompt}]},
                timeout=45,
            )
            d = r.json()
            if "choices" in d:
                return d["choices"][0]["message"]["content"], "GPT-4o"
            return None, f"openai_error: {d.get('error',{}).get('message','unknown')}"
        except Exception as e:
            return None, f"openai_exception: {e}"

    # ── 3. Google Gemini (FREE tier) ─────────────────────
    if keys["gemini"]:
        # Try models in order — 1.5-flash has most reliable free quota
        gemini_models = [
            "gemini-1.5-flash",
            "gemini-1.5-flash-latest",
            "gemini-2.0-flash-lite",
        ]
        last_err = "unknown"
        for model in gemini_models:
            try:
                r = requests.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/"
                    f"{model}:generateContent?key={keys['gemini']}",
                    headers={"Content-Type": "application/json"},
                    json={
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "maxOutputTokens": 900,
                            "temperature": 0.4,
                        },
                    },
                    timeout=45,
                )
                d = r.json()
                if "candidates" in d and d["candidates"]:
                    text = d["candidates"][0]["content"]["parts"][0]["text"]
                    return text, f"Gemini ({model})"
                last_err = d.get("error", {}).get("message", str(d))
                # If quota exceeded on this model, try next
                if "quota" in last_err.lower() or "limit" in last_err.lower():
                    continue
                return None, f"gemini_error: {last_err}"
            except Exception as e:
                last_err = str(e)
                continue
        return None, f"gemini_quota: {last_err[:200]}"

# ════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════
with st.sidebar:
    st.title("📈 Stock Analyzer")
    st.caption("Yahoo Finance · No API keys needed")
    st.markdown("---")

    stock_input = st.text_input(
        "Stock Symbol",
        placeholder="AAPL  ·  TSLA  ·  2222.SR",
        help="US: AAPL TSLA NVDA | Saudi: 2222.SR 1120.SR 2010.SR",
    )

    period_opts = {
        "1 Month":  ("1mo",  "1d"),
        "3 Months": ("3mo",  "1d"),
        "6 Months": ("6mo",  "1d"),
        "1 Year":   ("1y",   "1d"),
        "2 Years":  ("2y",   "1wk"),
        "5 Years":  ("5y",   "1wk"),
    }
    sel_period = st.selectbox("Time Period", list(period_opts.keys()), index=3)

    st.markdown("---")
    st.markdown("**Examples**")
    ca, cb = st.columns(2)
    ca.markdown("🇺🇸 **US**\n\n`AAPL`\n\n`TSLA`\n\n`NVDA`\n\n`MSFT`\n\n`AMZN`")
    cb.markdown("🇸🇦 **Saudi**\n\n`2222.SR`\n\n`1120.SR`\n\n`2010.SR`\n\n`1180.SR`\n\n`7010.SR`")

    st.markdown("---")
    keys     = get_api_keys()
    provider = _active_provider(keys)
    if provider:
        st.success(f"🤖 AI: **{provider}** ready")
    else:
        st.warning("🤖 AI: add an API key")
        with st.expander("Free option — Google Gemini"):
            st.markdown(
                "1. Go to [aistudio.google.com](https://aistudio.google.com/app/apikey)\n"
                "2. Click **Get API key** → free, no credit card\n"
                "3. Add to **Settings → Secrets**:\n"
            )
            st.code('GEMINI_API_KEY = "AIza..."', language="toml")
        with st.expander("Paid options"):
            st.code('ANTHROPIC_API_KEY = "sk-ant-..."', language="toml")
            st.code('OPENAI_API_KEY = "sk-..."', language="toml")

    st.markdown("---")
    go_btn = st.button("🔍 Analyze Stock", type="primary", use_container_width=True)

# ════════════════════════════════════════════════════════
# LANDING
# ════════════════════════════════════════════════════════
if "result" not in st.session_state:
    st.session_state.result = None

if not go_btn and not st.session_state.result:
    st.title("📈 AI-Powered Stock Market Analyzer")
    st.markdown("""
> **Zero API keys required** for data. Powered by Yahoo Finance — free for everyone.

| Feature | Details |
|---------|---------|
| 📊 Price chart | Candlestick + Volume + Bollinger Bands + Moving Averages |
| 📉 Technicals | RSI + MACD (multi-panel interactive chart) |
| 💼 Fundamentals | P/E, EV/EBITDA, margins, balance sheet, cash flow |
| 🔮 Forecast | Ensemble: ARIMA + Holt Trend + Monte Carlo with 5–95% confidence band |
| 🤖 AI Analysis | CFA-level narrative (add API key to activate) |

🇺🇸 US Markets · 🇸🇦 Saudi Tadawul · 🌍 Most global exchanges

*Enter a symbol in the sidebar and click **Analyze Stock**.*
""")
    st.stop()

# ════════════════════════════════════════════════════════
# FETCH
# ════════════════════════════════════════════════════════
if go_btn:
    raw = (stock_input or "").strip().upper()
    if not raw:
        st.warning("⚠️ Enter a stock symbol.")
        st.stop()

    period_yf, interval_yf = period_opts[sel_period]
    prog = st.progress(0, text=f"Fetching price data for {raw}…")
    hist = fetch_history(raw, period_yf, interval_yf)
    prog.progress(50, text="Fetching company info…")
    info = fetch_info(raw)
    prog.progress(100, text="Ready!")
    prog.empty()

    if hist.empty:
        st.error(f"❌ No data found for **{raw}**")
        st.markdown("""
**Why this happens:**
- Yahoo Finance rate limit — **wait 30 seconds and try again**
- Wrong symbol format — Saudi stocks need `.SR`: use `2222.SR` not `2222`
- Symbol not on Yahoo Finance — check at [finance.yahoo.com](https://finance.yahoo.com)

**Verify your symbol:** search it at [finance.yahoo.com](https://finance.yahoo.com) first.
""")
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
currency = info.get("currency") or ("SAR" if ".SR" in symbol else "USD")
name     = info.get("longName") or info.get("shortName") or symbol
exchange = info.get("exchange") or info.get("fullExchangeName") or ""
close_s  = hist["Close"].dropna()

cur_price  = (_sf(info.get("currentPrice") or info.get("regularMarketPrice"))
              or (float(close_s.iloc[-1]) if not close_s.empty else None))
prev_close = (_sf(info.get("previousClose") or info.get("regularMarketPreviousClose"))
              or (float(close_s.iloc[-2]) if len(close_s) >= 2 else None))

# Header
h1, h2, h3, h4, h5 = st.columns([3, 1.5, 1.5, 1.5, 1.5])
with h1:
    st.subheader(name)
    st.caption(f"**{symbol}** · {exchange} · {currency}")
with h2:
    if cur_price and prev_close and prev_close > 0:
        chg = cur_price - prev_close
        pct = chg / prev_close * 100
        st.metric("Price", f"{cur_price:.2f}", f"{chg:+.2f} ({pct:+.2f}%)")
    elif cur_price:
        st.metric("Price", f"{cur_price:.2f}")
    else:
        st.metric("Price", "N/A")
with h3:
    hi52 = _sf(info.get("fiftyTwoWeekHigh")) or float(hist["High"].max())
    st.metric("52W High", f"{hi52:.2f}" if hi52 else "N/A")
with h4:
    lo52 = _sf(info.get("fiftyTwoWeekLow")) or float(hist["Low"].min())
    st.metric("52W Low",  f"{lo52:.2f}" if lo52 else "N/A")
with h5:
    st.metric("Market Cap", _money(_sf(info.get("marketCap"))))

st.markdown("---")
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Price & Technicals", "💼 Fundamentals", "🔮 Forecast", "🤖 AI Analysis"
])

# ════════════════════════════════════════════════════════
# TAB 1
# ════════════════════════════════════════════════════════
with tab1:
    close  = close_s
    sma20  = close.rolling(20).mean()
    sma50  = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    rsi    = calc_rsi(close)
    macd_l, macd_s, macd_h = calc_macd(close)
    bb_up, bb_mid, bb_dn   = calc_bb(close)

    rsi_now  = float(rsi.dropna().iloc[-1])    if not rsi.dropna().empty    else None
    macd_now = float(macd_h.dropna().iloc[-1]) if not macd_h.dropna().empty else None
    st.session_state["rsi_now"]  = rsi_now
    st.session_state["macd_now"] = macd_now

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.60, 0.20, 0.20], vertical_spacing=0.04,
        subplot_titles=[f"{symbol} · {res['period']}", "RSI (14)", "MACD"],
    )
    fig.add_trace(go.Candlestick(
        x=hist.index, open=hist["Open"], high=hist["High"],
        low=hist["Low"], close=hist["Close"], name="Price",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=bb_up, mode="lines",
        line=dict(color="rgba(100,180,255,0.35)", width=1), name="BB Upper"), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=bb_dn, mode="lines",
        fill="tonexty", fillcolor="rgba(100,180,255,0.07)",
        line=dict(color="rgba(100,180,255,0.35)", width=1), name="BB Lower"), row=1, col=1)
    for sma, lbl, col in [(sma20, "SMA 20", "#f9e2af"),
                           (sma50, "SMA 50", "#89b4fa"),
                           (sma200,"SMA 200","#cba6f7")]:
        if sma.dropna().shape[0] > 5:
            fig.add_trace(go.Scatter(x=hist.index, y=sma, mode="lines",
                line=dict(width=1.2, color=col), name=lbl), row=1, col=1)
    if "Volume" in hist.columns and hist["Volume"].sum() > 0:
        vc = ["#26a69a" if c >= o else "#ef5350"
              for c, o in zip(hist["Close"], hist["Open"])]
        fig.add_trace(go.Bar(x=hist.index, y=hist["Volume"],
            marker_color=vc, opacity=0.35, showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=rsi, mode="lines",
        line=dict(color="#89b4fa", width=1.5), name="RSI"), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#f38ba8", line_width=1, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#a6e3a1", line_width=1, row=2, col=1)
    mc = ["#26a69a" if v >= 0 else "#ef5350" for v in macd_h.fillna(0)]
    fig.add_trace(go.Bar(x=hist.index, y=macd_h, marker_color=mc,
        opacity=0.7, name="MACD Hist"), row=3, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=macd_l, mode="lines",
        line=dict(color="#89b4fa", width=1.2), name="MACD"), row=3, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=macd_s, mode="lines",
        line=dict(color="#f9e2af", width=1.2), name="Signal"), row=3, col=1)
    fig.update_layout(template="plotly_dark", height=680, hovermode="x unified",
        xaxis_rangeslider_visible=False, margin=dict(l=40, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=11)))
    fig.update_yaxes(title_text=currency, row=1, col=1)
    fig.update_yaxes(title_text="RSI",    row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD",   row=3, col=1)
    st.plotly_chart(fig, use_container_width=True)

    hdr("Technical Summary")
    tc1, tc2, tc3, tc4, tc5 = st.columns(5)
    rsi_lbl  = "Overbought ⚠️" if (rsi_now or 50) > 70 else (
               "Oversold 💡"   if (rsi_now or 50) < 30 else "Neutral")
    macd_lbl = "Bullish 📈" if (macd_now or 0) > 0 else "Bearish 📉"
    s200v    = float(sma200.dropna().iloc[-1]) if not sma200.dropna().empty else None
    tc1.metric("RSI (14)",    f"{rsi_now:.1f}"  if rsi_now  else "N/A", rsi_lbl)
    tc2.metric("MACD",        f"{macd_now:.3f}" if macd_now else "N/A", macd_lbl)
    tc3.metric("SMA 20",      f"{float(sma20.dropna().iloc[-1]):.2f}"  if not sma20.dropna().empty  else "N/A")
    tc4.metric("SMA 50",      f"{float(sma50.dropna().iloc[-1]):.2f}"  if not sma50.dropna().empty  else "N/A")
    tc5.metric("Above SMA 200", "✅ Yes" if (cur_price and s200v and cur_price > s200v) else "❌ No")

# ════════════════════════════════════════════════════════
# TAB 2
# ════════════════════════════════════════════════════════
with tab2:
    fins = fetch_financials(symbol)
    cf   = fins["cashflow"]

    hdr("📊 Valuation")
    v1,v2,v3,v4 = st.columns(4)
    v5,v6,v7,v8 = st.columns(4)
    v1.metric("P/E (Trailing)", _num(_sf(info.get("trailingPE"))))
    v2.metric("P/E (Forward)",  _num(_sf(info.get("forwardPE"))))
    v3.metric("P/B Ratio",      _num(_sf(info.get("priceToBook"))))
    v4.metric("P/S Ratio",      _num(_sf(info.get("priceToSalesTrailing12Months"))))
    v5.metric("EV/EBITDA",      _num(_sf(info.get("enterpriseToEbitda"))))
    v6.metric("EV/Revenue",     _num(_sf(info.get("enterpriseToRevenue"))))
    v7.metric("PEG Ratio",      _num(_sf(info.get("pegRatio"))))
    v8.metric("Beta",           _num(_sf(info.get("beta"))))

    st.markdown("---")
    hdr("📈 Profitability & Growth")
    p1,p2,p3,p4 = st.columns(4)
    p5,p6,p7,p8 = st.columns(4)
    p1.metric("Revenue (TTM)",    _money(_sf(info.get("totalRevenue"))))
    p2.metric("Revenue Growth",   _pct(_sf(info.get("revenueGrowth"))))
    p3.metric("Gross Margin",     _pct(_sf(info.get("grossMargins"))))
    p4.metric("Operating Margin", _pct(_sf(info.get("operatingMargins"))))
    p5.metric("Net Margin",       _pct(_sf(info.get("profitMargins"))))
    p6.metric("EBITDA",           _money(_sf(info.get("ebitda"))))
    p7.metric("Return on Equity", _pct(_sf(info.get("returnOnEquity"))))
    p8.metric("Return on Assets", _pct(_sf(info.get("returnOnAssets"))))

    st.markdown("---")
    hdr("💰 EPS, Dividends & Analyst Targets")
    e1,e2,e3,e4 = st.columns(4)
    e5,e6,e7,e8 = st.columns(4)
    e1.metric("EPS (TTM)",         _num(_sf(info.get("trailingEps")), pre=f"{currency} "))
    e2.metric("EPS (Forward)",     _num(_sf(info.get("forwardEps")),  pre=f"{currency} "))
    e3.metric("Dividend Yield",    _pct(_sf(info.get("dividendYield"))))
    e4.metric("Dividend/Share",    _num(_sf(info.get("dividendRate")), pre=f"{currency} "))
    e5.metric("Payout Ratio",      _pct(_sf(info.get("payoutRatio"))))
    e6.metric("Analyst Target",    _num(_sf(info.get("targetMeanPrice")), pre=f"{currency} "))
    rec_raw = _str(info, "recommendationKey") or "N/A"
    e7.metric("Recommendation",    rec_raw.replace("_"," ").title())
    e8.metric("# Analyst Opinions",_num(_sf(info.get("numberOfAnalystOpinions")), dec=0))

    st.markdown("---")
    hdr("🏦 Balance Sheet")
    b1,b2,b3,b4 = st.columns(4)
    b5,b6,b7,b8 = st.columns(4)
    b1.metric("Total Cash",        _money(_sf(info.get("totalCash"))))
    b2.metric("Total Debt",        _money(_sf(info.get("totalDebt"))))
    b3.metric("Book Value/Share",  _num(_sf(info.get("bookValue")), pre=f"{currency} "))
    b4.metric("Shares Outstanding",_money(_sf(info.get("sharesOutstanding"))).replace("$",""))
    b5.metric("Current Ratio",     _num(_sf(info.get("currentRatio"))))
    b6.metric("Quick Ratio",       _num(_sf(info.get("quickRatio"))))
    b7.metric("Debt / Equity",     _num(_sf(info.get("debtToEquity"))))
    b8.metric("Cash/Share",        _num(_sf(info.get("totalCashPerShare")), pre=f"{currency} "))

    st.markdown("---")
    hdr("💵 Cash Flow")
    c1f,c2f,c3f,c4f = st.columns(4)
    op_cf   = _sf(info.get("operatingCashflow")) or latest_val(
        cf, "Operating Cash Flow", "Cash Flow From Continuing Operating Activities")
    fcf_v   = _sf(info.get("freeCashflow")) or latest_val(cf, "Free Cash Flow")
    mktcap_v = _sf(info.get("marketCap"))
    cash_v   = _sf(info.get("totalCash"))
    debt_v   = _sf(info.get("totalDebt"))
    c1f.metric("Operating CF",    _money(op_cf))
    c2f.metric("Free Cash Flow",  _money(fcf_v))
    c3f.metric("FCF Yield",       _pct(fcf_v / mktcap_v) if (fcf_v and mktcap_v) else "N/A")
    c4f.metric("Cash/Debt",       _num(cash_v / debt_v, dec=2) if (cash_v and debt_v) else "N/A")

    st.markdown("---")
    hdr("🏢 Company Profile")
    i1,i2,i3,i4 = st.columns(4)
    i1.metric("Sector",   _str(info,"sector")   or "N/A")
    i2.metric("Industry", _str(info,"industry") or "N/A")
    i3.metric("Country",  _str(info,"country")  or "N/A")
    emp = _sf(info.get("fullTimeEmployees"))
    i4.metric("Employees", f"{int(emp):,}" if emp else "N/A")
    if info.get("longBusinessSummary"):
        with st.expander("Business Description"):
            st.write(info["longBusinessSummary"])

# ════════════════════════════════════════════════════════
# TAB 3
# ════════════════════════════════════════════════════════
with tab3:
    hdr("🔮 Ensemble Price Forecast")
    st.caption(
        "3-model ensemble: ARIMA (autocorrelation) + Holt Trend (drift) + "
        "Monte Carlo simulation (volatility). CI = 5th–95th percentile of 500 simulations."
    )

    fh_c, ci_c, ind_c = st.columns([3, 1, 1])
    with fh_c:
        horizon = st.slider("Forecast horizon (trading days)", 7, 90, 30)
    with ci_c:
        show_ci = st.checkbox("Show CI band", value=True)
    with ind_c:
        show_models = st.checkbox("Show models", value=False)

    if len(close_s) < 20:
        st.warning("Need at least 20 data points. Select a longer time period.")
    else:
        with st.spinner("Running ensemble model (ARIMA + Holt + Monte Carlo)…"):
            fc_df = ensemble_forecast(close_s, horizon)
            st.session_state["fc_df"] = fc_df

        fig2 = go.Figure()

        # Actual price
        fig2.add_trace(go.Scatter(
            x=close_s.index, y=close_s, mode="lines",
            name="Actual", line=dict(color="#89b4fa", width=1.8),
        ))

        # CI band (Monte Carlo 5–95th percentile)
        if show_ci and fc_df is not None:
            fig2.add_trace(go.Scatter(
                x=list(fc_df["Date"]) + list(fc_df["Date"])[::-1],
                y=list(fc_df["High CI"]) + list(fc_df["Low CI"])[::-1],
                fill="toself", fillcolor="rgba(166,227,161,0.12)",
                line=dict(color="rgba(0,0,0,0)"),
                name="5–95% Range", hoverinfo="skip",
            ))

        # Individual model lines (optional)
        if show_models and fc_df is not None:
            fig2.add_trace(go.Scatter(
                x=fc_df["Date"], y=fc_df["ARIMA"], mode="lines",
                name="ARIMA", line=dict(color="#f38ba8", width=1.2, dash="dot"),
            ))
            fig2.add_trace(go.Scatter(
                x=fc_df["Date"], y=fc_df["Holt"], mode="lines",
                name="Holt Trend", line=dict(color="#f9e2af", width=1.2, dash="dot"),
            ))
            fig2.add_trace(go.Scatter(
                x=fc_df["Date"], y=fc_df["MonteCarlo"], mode="lines",
                name="Monte Carlo", line=dict(color="#cba6f7", width=1.2, dash="dot"),
            ))

        # Ensemble (main forecast line)
        if fc_df is not None:
            fig2.add_trace(go.Scatter(
                x=fc_df["Date"], y=fc_df["Forecast"], mode="lines",
                name="Ensemble Forecast",
                line=dict(color="#a6e3a1", width=2.5, dash="dash"),
            ))

        # Today marker
        fig2.add_vline(
            x=close_s.index[-1],
            line_dash="dot", line_color="rgba(255,255,255,0.3)", line_width=1,
        )
        fig2.add_annotation(
            x=close_s.index[-1], y=float(close_s.iloc[-1]),
            text="Today", showarrow=True, arrowhead=2,
            font=dict(color="#cdd6f4", size=11),
            arrowcolor="#cdd6f4", ax=40, ay=-30,
        )

        fig2.update_layout(
            template="plotly_dark", height=500,
            hovermode="x unified",
            xaxis_title="Date", yaxis_title=f"Price ({currency})",
            margin=dict(l=40, r=20, t=20, b=40),
            legend=dict(orientation="h", yanchor="bottom",
                        y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig2, use_container_width=True)

        if fc_df is not None and not fc_df.empty:
            cur    = float(close_s.iloc[-1])
            fc_end = float(fc_df["Forecast"].iloc[-1])
            up     = (fc_end / cur - 1) * 100
            ci_lo  = float(fc_df["Low CI"].iloc[-1])
            ci_hi  = float(fc_df["High CI"].iloc[-1])
            ci_rng = ((ci_hi - ci_lo) / cur) * 100

            fm1, fm2, fm3, fm4, fm5 = st.columns(5)
            fm1.metric("Current Price",        f"{cur:.2f} {currency}")
            fm2.metric(f"{horizon}d Forecast", f"{fc_end:.2f}",
                       delta=f"{up:+.1f}%")
            fm3.metric("5th Pct (bear)",  f"{ci_lo:.2f}",
                       delta=f"{((ci_lo/cur)-1)*100:+.1f}%")
            fm4.metric("95th Pct (bull)", f"{ci_hi:.2f}",
                       delta=f"{((ci_hi/cur)-1)*100:+.1f}%")
            fm5.metric("Uncertainty Range", f"±{ci_rng/2:.1f}%")

            # Model breakdown
            st.markdown("**Model breakdown at forecast end:**")
            mb1, mb2, mb3 = st.columns(3)
            mb1.metric("ARIMA",       f"{float(fc_df['ARIMA'].iloc[-1]):.2f}",
                       delta=f"{((float(fc_df['ARIMA'].iloc[-1])/cur)-1)*100:+.1f}%")
            mb2.metric("Holt Trend",  f"{float(fc_df['Holt'].iloc[-1]):.2f}",
                       delta=f"{((float(fc_df['Holt'].iloc[-1])/cur)-1)*100:+.1f}%")
            mb3.metric("Monte Carlo", f"{float(fc_df['MonteCarlo'].iloc[-1]):.2f}",
                       delta=f"{((float(fc_df['MonteCarlo'].iloc[-1])/cur)-1)*100:+.1f}%")

            with st.expander("Full forecast table"):
                tbl = fc_df[["Date","Forecast","ARIMA","Holt",
                              "MonteCarlo","Low CI","High CI"]].copy()
                tbl["Date"] = tbl["Date"].dt.strftime("%Y-%m-%d")
                for c in tbl.columns[1:]: tbl[c] = tbl[c].round(2)
                st.dataframe(tbl, use_container_width=True, hide_index=True)

    st.warning(
        "⚠️ Statistical models based on historical price patterns only. "
        "They do not account for earnings, news, or macro events. Not financial advice."
    )

# ════════════════════════════════════════════════════════
# TAB 4 — AI (always shown, graceful if no key)
# ════════════════════════════════════════════════════════
with tab4:
    hdr("🤖 AI Investment Analysis")
    keys     = get_api_keys()
    provider = _active_provider(keys)

    if not provider:
        st.markdown("""
### Enable AI Analysis — Free Option Available

This tab generates a **CFA-level 4-paragraph analysis** combining fundamentals,
technicals, and the ensemble forecast.

---
#### 🆓 Option 1 — Google Gemini (FREE · no credit card needed)
1. Go to **[aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)**
2. Sign in with Google → click **Get API key** → **Create API key**
3. Copy the key (starts with `AIza...`)
4. In your Streamlit app → ⚙️ **Settings → Secrets**, add:
```toml
GEMINI_API_KEY = "AIzaSy..."
```
✅ Free tier: 1,500 requests/day — plenty for personal use.

---
#### 💳 Option 2 — Anthropic Claude (best quality · ~$5 credit needed)
```toml
ANTHROPIC_API_KEY = "sk-ant-..."
```
Get key: [console.anthropic.com](https://console.anthropic.com)

---
#### 💳 Option 3 — OpenAI GPT-4o
```toml
OPENAI_API_KEY = "sk-..."
```
Get key: [platform.openai.com](https://platform.openai.com)
""")
    else:
        st.success(f"✅ Powered by **{provider}**")
        st.markdown(
            "Generates a 4-paragraph CFA-level analysis: "
            "business overview · valuation verdict · technical momentum · risks & catalysts."
        )
        if st.button(f"🤖 Generate {provider} Analysis", type="primary"):
            with st.spinner(f"Generating analysis with {provider}…"):
                narrative, status = get_ai_narrative(
                    symbol, name, info,
                    st.session_state.get("fc_df"),
                    st.session_state.get("rsi_now"),
                    st.session_state.get("macd_now"),
                    currency,
                )
            if narrative:
                st.markdown("---")
                st.markdown(narrative)
                st.markdown("---")
                st.caption(
                    f"Generated by {provider} · "
                    "Educational purposes only · Not financial advice"
                )
            else:
                if "quota" in status.lower() or "limit" in status.lower():
                    st.error(
                        "⏳ **Gemini quota temporarily exceeded.**\n\n"
                        "This usually resolves in a few seconds. "
                        "**Wait 10 seconds and click Generate again.**\n\n"
                        f"Detail: `{status[:150]}`"
                    )
                else:
                    st.error(
                        f"AI call failed: `{status}`\n\n"
                        "Check your API key is correct and has not exceeded its quota."
                    )

st.markdown("---")
st.caption("Data: Yahoo Finance · Educational purposes only · Not financial advice")
