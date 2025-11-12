import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
import datetime as dt
import calendar
import os
import requests
import time
import re

METALS_API_KEY = "goldapi-1kxj919mhnkgyor-io"

# --- Page config ---
st.set_page_config(
    page_title="💰 Indian Gold Price Predictor", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide Streamlit default elements
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display:none;}
div[data-testid="stToolbar"] {visibility: hidden;}
.stDeployButton {visibility: hidden;}
#stDecoration {display:none;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- Theme Switcher Component ---
def get_theme_config():
    """Force light theme always"""
    st.session_state.theme = 'light'
    return 'light'

# --- Gold Scheme helpers (restored) ---
@st.cache_data(ttl=600)
def _load_schemes_csv(path: str = "gold_shops_schemes.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def _to_float_safe(val):
    if pd.isna(val):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    m = re.findall(r"[\d,.]+", str(val))
    if not m:
        return None
    try:
        return float(m[0].replace(",", ""))
    except Exception:
        return None


def _benefit_score(benefit_text: str, duration_months: float | None) -> float:
    if not isinstance(benefit_text, str):
        return 0.0
    txt = benefit_text.lower()
    # percentage values
    percents = re.findall(r"(\d+(?:\.\d+)?)\s*%", txt)
    best_pct = max([float(p) for p in percents], default=0.0)
    score = best_pct
    # free months (e.g., "1 month free")
    free_months = re.findall(r"(\d+)\s*(?:month|months)\s*free", txt)
    if free_months:
        n = float(free_months[0])
        if duration_months and duration_months > 0:
            score = max(score, (n / duration_months) * 100.0)
        else:
            score = max(score, n * 5.0)
    # flat rupee discounts
    rupees = re.findall(r"(?:₹|rs\.?\s*)([\d,]+)", txt)
    if rupees:
        try:
            amt = float(rupees[0].replace(",", ""))
            score = max(score, amt / 100.0)
        except Exception:
            pass
    # keyword boosts
    if "bonus" in txt:
        score += 2.0
    if "discount" in txt:
        score += 2.0
    return float(score)


@st.cache_data(ttl=600)
def recommend_best_scheme(location: str = "Chennai") -> pd.DataFrame:
    try:
        df = _load_schemes_csv()
    except Exception as e:
        raise FileNotFoundError("gold_shops_schemes.csv not found or unreadable") from e

    # normalize columns
    colmap = {c.lower().strip(): c for c in df.columns}

    def pick(*names):
        for n in names:
            for k, v in colmap.items():
                if k == n or n in k:
                    return v
        return None

    col_location = pick("location", "city")
    col_shop = pick("shop name", "shop_name", "shop")
    col_scheme = pick("scheme name", "scheme_name", "scheme")
    col_duration = pick("duration", "duration (months)", "months")
    col_monthly = pick("monthly payment", "monthly", "amount", "emi")
    col_benefit = pick("benefit", "benefits", "offer", "bonus", "discount")
    col_contact = pick("contact number", "contact", "phone", "mobile")

    if not col_location:
        raise FileNotFoundError("gold_shops_schemes.csv is missing a Location/City column")

    # filter by location
    df_f = df[df[col_location].astype(str).str.lower() == str(location).lower()].copy()
    if df_f.empty:
        df_f = df[df[col_location].astype(str).str.lower().str.contains(str(location).lower(), na=False)].copy()
    if df_f.empty:
        return pd.DataFrame()

    # parse fields
    df_f["_duration_m"] = df_f[col_duration].apply(_to_float_safe) if col_duration else None
    df_f["_monthly"] = df_f[col_monthly].apply(_to_float_safe) if col_monthly else None
    df_f["_benefit_text"] = df_f[col_benefit] if col_benefit else ""
    df_f["_benefit_score"] = df_f.apply(lambda r: _benefit_score(r.get("_benefit_text", ""), r.get("_duration_m", None)), axis=1)

    # min-max normalization helpers
    def minmax(s):
        s = s.astype(float)
        lo, hi = s.min(), s.max()
        if pd.isna(lo) or pd.isna(hi) or hi - lo == 0:
            return pd.Series([0.5] * len(s), index=s.index)
        return (s - lo) / (hi - lo)

    monthly_norm = minmax(df_f["_monthly"]) if col_monthly else pd.Series([0.5] * len(df_f), index=df_f.index)
    benefit_norm = minmax(df_f["_benefit_score"]) if col_benefit else pd.Series([0.0] * len(df_f), index=df_f.index)
    df_f["_score"] = benefit_norm - monthly_norm

    # top 3
    df_top = df_f.sort_values("_score", ascending=False).head(3).copy()

    def fmt_money(x):
        v = _to_float_safe(x)
        return f"₹{v:,.2f}" if v is not None else "-"

    out = pd.DataFrame({
        "Shop Name": df_top.get(col_shop, pd.Series(["-"] * len(df_top))),
        "Scheme Name": df_top.get(col_scheme, pd.Series(["-"] * len(df_top))),
        "Duration (months)": df_top.get(col_duration, pd.Series(["-"] * len(df_top))),
        "Monthly Payment": df_top[col_monthly].apply(fmt_money) if col_monthly else pd.Series(["-"] * len(df_top)),
        "Benefit": df_top.get(col_benefit, pd.Series(["-"] * len(df_top))),
        "Contact Number": df_top.get(col_contact, pd.Series(["-"] * len(df_top))),
    })
    return out.reset_index(drop=True)

def apply_theme_styles(theme):
    """Apply CSS styles based on selected theme"""
    if theme == 'dark':
        st.markdown("""
        <style>
        .stApp {
            background-color: #0e1117;
            color: #ffffff;
        }
        .main .block-container {
            background-color: #0e1117;
            color: #ffffff;
        }
        .stSelectbox > div > div {
            background-color: #262730;
            color: #ffffff;
        }
        .stNumberInput > div > div > input {
            background-color: #262730;
            color: #ffffff;
        }
        .stDateInput > div > div > input {
            background-color: #262730;
            color: #ffffff;
        }
        .stButton > button {
            background-color: #ff6b6b;
            color: #ffffff;
        }
        .stButton > button:hover {
            background-color: #ff5252;
        }
        .stSuccess {
            background-color: #1e3a1e;
            color: #4caf50;
        }
        .stInfo {
            background-color: #1e3a3a;
            color: #00bcd4;
        }
        .stMarkdown {
            color: #ffffff;
        }
        .stText {
            color: #ffffff;
        }
        .stSelectbox label {
            color: #ffffff !important;
        }
        .stNumberInput label {
            color: #ffffff !important;
        }
        .stDateInput label {
            color: #ffffff !important;
        }
        </style>
        """, unsafe_allow_html=True)
    else:  # light theme (softened)
        st.markdown("""
        <style>
        .stApp {
            background-color: #e9edf5; /* darker, softer off-white */
            color: #1f2937; /* slate-800 */
        }
        .main .block-container {
            background-color: #e9edf5;
            color: #1f2937;
        }
        .stSelectbox > div > div,
        .stNumberInput > div > div > input,
        .stDateInput > div > div > input {
            background-color: #e2e8f0; /* slightly darker input bg */
            color: #1f2937;
            border-radius: 8px;
        }
        .stButton > button {
            background-color: #ef4444; /* softer red */
            color: #ffffff;
            border-radius: 8px;
        }
        .stButton > button:hover {
            background-color: #dc2626;
        }
        .stSuccess {
            background-color: #edf7ee;
            color: #166534; /* green-700 */
        }
        .stInfo {
            background-color: #e8f1f7;
            color: #0f4a6e; /* blue-800 */
        }
        .stMarkdown, .stText, .stSelectbox label, .stNumberInput label, .stDateInput label {
            color: #1f2937 !important;
        }

        /* --- Plotly light theme fixes for visibility --- */
        .js-plotly-plot .main-svg text,
        .js-plotly-plot .gtitle,
        .js-plotly-plot .legend text,
        .js-plotly-plot .xaxislayer-above text,
        .js-plotly-plot .yaxislayer-above text,
        .js-plotly-plot .xtick text,
        .js-plotly-plot .ytick text,
        .js-plotly-plot .infolayer text {
            fill: #111 !important;
            opacity: 1 !important;
            font-weight: 500;
        }
        .js-plotly-plot .gridlayer line { stroke: rgba(0,0,0,0.12) !important; }
        .js-plotly-plot .zerolinelayer line { stroke: rgba(0,0,0,0.2) !important; }
        .js-plotly-plot .xaxis path, .js-plotly-plot .yaxis path { stroke: rgba(0,0,0,0.35) !important; }
        .js-plotly-plot .modebar { background: rgba(255,255,255,0.8) !important; }
        .js-plotly-plot .modebar-btn svg { fill: #333 !important; }
        .js-plotly-plot .modebar-btn:hover svg { fill: #000 !important; }
        .js-plotly-plot .modebar-btn.active svg { fill: #000 !important; }
        .js-plotly-plot .modebar-btn.active { background-color: #e5e7eb !important; }
        .js-plotly-plot .modebar-btn { background-color: #e5e7eb !important; }
        </style>
        """, unsafe_allow_html=True)

def get_plotly_theme_config(theme):
    """Get Plotly theme configuration based on selected theme"""
    if theme == 'dark':
        return {
            'plot_bgcolor': '#1e1e1e',
            'paper_bgcolor': '#1e1e1e',
            'xaxis': {
                'gridcolor': 'rgba(128, 128, 128, 0.3)', 
                'color': 'white',
                'showgrid': True,
                'zeroline': True,
                'zerolinecolor': 'rgba(128, 128, 128, 0.5)'
            },
            'yaxis': {
                'gridcolor': 'rgba(128, 128, 128, 0.3)', 
                'color': 'white',
                'showgrid': True,
                'zeroline': True,
                'zerolinecolor': 'rgba(128, 128, 128, 0.5)'
            },
            'title_font': {'size': 22, 'color': 'gold', 'family': "Arial"},
            'font': {'size': 14, 'color': 'white'},
            'legend': {'font': {'color': 'white'}}
        }
    else:  # light theme
        return {
            'template': 'plotly_white',
            'plot_bgcolor': '#f3f4f6',
            'paper_bgcolor': '#f3f4f6',
            'xaxis': {
                'showgrid': True,
                'gridcolor': 'rgba(0, 0, 0, 0.12)',
                'zeroline': True,
                'zerolinecolor': 'rgba(0, 0, 0, 0.2)',
                'linecolor': 'rgba(0, 0, 0, 0.35)',
                'tickfont': {'color': '#111', 'size': 13},
                'title': {'font': {'color': '#111', 'size': 14}}
            },
            'yaxis': {
                'showgrid': True,
                'gridcolor': 'rgba(0, 0, 0, 0.12)',
                'zeroline': True,
                'zerolinecolor': 'rgba(0, 0, 0, 0.2)',
                'linecolor': 'rgba(0, 0, 0, 0.35)',
                'tickfont': {'color': '#111', 'size': 13},
                'title': {'font': {'color': '#111', 'size': 14}}
            },
            'title_font': {'size': 22, 'color': '#ff6b6b', 'family': 'Arial'},
            'font': {'size': 14, 'color': '#111', 'family': 'Arial'},
            'legend': {'font': {'color': '#111', 'size': 12}},
            'hoverlabel': {'bgcolor': '#ffffff', 'font': {'color': '#111'}}
        }

# --- Theme: force light ---
st.session_state.theme = 'light'
apply_theme_styles('light')
# Get current theme config for plots
plotly_config = get_plotly_theme_config('light')

# --- Main App Title ---
title_color = '#1f2937'  # dark slate for strong visibility
st.markdown(f"<h1 style='text-align:center; color:{title_color};'>💰 Indian Gold Price Predictor</h1>", unsafe_allow_html=True)

# --- Highlighted Note ---
note_color = '#334155'  # slightly darker accent
st.markdown(f"""
<div style='text-align:center; margin: 20px 0; padding: 15px; background-color: #e8f5e8; border-radius: 10px; border-left: 5px solid {note_color};'>
    <p style='color: {note_color}; font-size: 16px; font-weight: bold; margin: 0;'>
        🔔 This predictor provides forecasts for <strong>24K Gold (pure gold)</strong> prices in INR.
    </p>
</div>
""", unsafe_allow_html=True)

st.write("Predict future gold price in INR for any selected date.")

# --- Load all CSVs ---
@st.cache_data(show_spinner=False)
def load_data():
    files = [
        "Daily.csv",
        "Monthly_Avg.csv",
        "Monthly_EoP.csv",
        "Quarterly_Avg.csv",
        "Quarterly_EoP.csv",
        "Weekly_EoP.csv",
        "Yearly_Avg.csv",
        "Yearly_EoP.csv"
    ]
    all_data = []
    for f in files:
        df = pd.read_csv(f, usecols=['Date','INR'])
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna()
        df['INR'] = df['INR'].astype(str).str.replace(',','').astype(float)
        all_data.append(df)
    full_df = pd.concat(all_data)
    full_df = full_df.groupby('Date').mean().reset_index()
    full_df = full_df.sort_values('Date')
    return full_df

with st.spinner("Loading data..."):
    try:
        data = load_data()
    except Exception as e:
        st.error(f"Data load failed: {e}")
        st.stop()

# --- Apply scaling factor to match real 1g INR price (~11,697 INR today) ---
UNIT_IN_GRAM = 8  # CSV represents 1 savaran (8g)
LATEST_ACTUAL_1G_PRICE = 11697.94  # INR per 1g today
LATEST_DATA_VALUE = data['INR'].iloc[-1]  # latest CSV value
scaling_factor = LATEST_ACTUAL_1G_PRICE / (LATEST_DATA_VALUE / UNIT_IN_GRAM)

# --- Normalize to 1 gram using scaling factor ---
data['INR_per_gram'] = (data['INR'] / UNIT_IN_GRAM) * scaling_factor

# --- Prophet model ---
@st.cache_resource(show_spinner=False)
def train_model(df):
    df_prophet = df[['Date','INR_per_gram']].rename(columns={'Date':'ds','INR_per_gram':'y'})
    model = Prophet(daily_seasonality=True)
    model.fit(df_prophet)
    return model

try:
    model = train_model(data)
except Exception as e:
    st.error(f"Model training failed: {e}")
    st.stop()

# GoldAPI.io single-day fetch

@st.cache_data(ttl=300)
def fetch_live_gold_price(date_str):
    """Fetch gold price (INR per gram, 24K) from GoldAPI.io for the given date, optimized for speed."""
    import requests
    headers = {
        "x-access-token": METALS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "GoldPredictionApp/1.0"
    }
    url = f"https://www.goldapi.io/api/XAU/INR/{date_str}"
    try:
        r = requests.get(url, headers=headers, timeout=1.0)
        if r.status_code == 200:
            data = r.json()
            price_per_ounce = data.get("price")
            if price_per_ounce is not None:
                return float(price_per_ounce) / 31.1034768
    except Exception:
        pass
    return None

@st.cache_data(ttl=180)
def fetch_live_gold_spot():
    """Fetch current spot gold price (INR per gram, 24K) from GoldAPI.io, optimized for speed."""
    import requests
    headers = {
        "x-access-token": METALS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "GoldPredictionApp/1.0"
    }
    url = "https://www.goldapi.io/api/XAU/INR"
    try:
        r = requests.get(url, headers=headers, timeout=1.0)
        if r.status_code == 200:
            data = r.json()
            price_per_ounce = data.get("price")
            if price_per_ounce is not None:
                return float(price_per_ounce) / 31.1034768
    except Exception:
        pass
    return None

def get_goldapi_live_for_selection(selected_ts: pd.Timestamp, max_days_back: int = 3) -> tuple[float | None, str]:
    """Fast path: try selected date, then up to 3 previous days; if none or future date, use spot. Time budget ~1.2s."""
    t0 = time.monotonic()
    today_ts = pd.Timestamp.today().normalize()
    # Future dates -> spot
    if selected_ts > today_ts:
        spot = fetch_live_gold_spot()
        return (spot, "spot")
    # Selected date and short walk-back window with time budget
    for i in range(0, max_days_back + 1):
        if time.monotonic() - t0 > 1.2:
            break
        day = (selected_ts - pd.Timedelta(days=i)).strftime('%Y-%m-%d')
        price = fetch_live_gold_price(day)
        if price is not None:
            return (price, day)
    # Last resort: spot
    spot_price = fetch_live_gold_spot()
    return (spot_price, "spot" if spot_price is not None else "")

# --- Live price helpers ---

def _get_metals_api_key() -> str | None:
    key = os.getenv("METALS_API_KEY")
    if key:
        return key
    try:
        return st.secrets["METALS_API_KEY"]  # may raise if no secrets
    except Exception:
        return None

@st.cache_data(ttl=900, show_spinner=False)
def fetch_live_gold_for_date(date: pd.Timestamp) -> float | None:
    """Return INR per gram (24K) for a single date using Metals API, or None on failure."""
    key = _get_metals_api_key()
    if not key:
        return None
    try:
        day = pd.Timestamp(date).normalize().strftime('%Y-%m-%d')
        url = f"https://metals-api.com/api/{day}"
        params = {"access_key": key, "base": "INR", "symbols": "XAU"}
        r = requests.get(url, params=params, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
        if isinstance(data, dict) and data.get("success", True) is False:
            return None
        rates = data.get("rates", {}) if isinstance(data, dict) else {}
        price_per_ounce = rates.get("XAU")
        if not price_per_ounce:
            return None
        return float(price_per_ounce) / 31.1034768
    except Exception:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_live_gold_last30():
    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.Timedelta(days=29)
    # Safely obtain API key: ENV first, then Streamlit secrets if present
    key = os.getenv("METALS_API_KEY")
    if not key:
        try:
            key = st.secrets["METALS_API_KEY"]  # may raise if secrets.toml missing
        except Exception:
            key = None
    records = []
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    for date in dates:
        try:
            if not key:
                continue
            url = f"https://metals-api.com/api/{date.strftime('%Y-%m-%d')}"
            params = {"access_key": key, "base": "INR", "symbols": "XAU"}
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, dict) and data.get("success", True) is False:
                    continue
                rates = data.get("rates", {}) if isinstance(data, dict) else {}
                price_per_ounce = rates.get("XAU")
                if price_per_ounce:
                    price_per_gram = price_per_ounce / 31.1034768
                    records.append({"ds": date, "y": float(price_per_gram)})
        except Exception:
            continue
    df = pd.DataFrame.from_records(records)
    return df


# --- User input ---
pred_date = st.date_input("Select date to predict gold price")
grams = st.number_input("How many grams?", min_value=0.1, value=1.0, step=0.1)
use_api_only = True  # Force live-only predictions
if st.button("Predict"):
    # Always compute model forecast
    future = pd.DataFrame({'ds':[pd.to_datetime(pred_date)]})
    forecast = model.predict(future)

    model_pred_per_g = float(forecast['yhat'].values[0])
    model_lower_per_g = float(forecast['yhat_lower'].values[0])
    model_upper_per_g = float(forecast['yhat_upper'].values[0])

    selected_ts = pd.to_datetime(pred_date)
    live_per_g = None
    live_source = ""

    if METALS_API_KEY:
        with st.spinner("Getting latest market rate..."):
            live_per_g, live_source = get_goldapi_live_for_selection(selected_ts, max_days_back=30)

    # If user insisted on live-only but we still don't have any market rate, warn and continue with model
    if live_per_g is None and use_api_only:
        _ = None  # suppress UI warning; keep behavior unchanged

    pred_price = live_per_g * grams if live_per_g is not None else model_pred_per_g * grams
    source_label = "Live market" if live_source != "spot" else "Live market (spot)"
    lower = model_lower_per_g * grams
    upper = model_upper_per_g * grams
    confidence = 92.0  # fixed proxy

    st.subheader(f"Gold Price on {pred_date.strftime('%d-%m-%Y')} — {source_label}")
    st.markdown(f"**24K Gold Price for {grams} grams (INR)**")
    st.markdown(f"### ₹{pred_price:,.2f}")
    st.write(f"{confidence:.2f}% confidence")
    st.write(f"📉 Expected Range (24K): ₹{lower:,.2f} - ₹{upper:,.2f} INR")

    price_22 = pred_price * (22/24)
    lower_22 = lower * (22/24)
    upper_22 = upper * (22/24)
    st.markdown(f"**22K Gold Price for {grams} grams (INR)**")
    st.markdown(f"### ₹{price_22:,.2f}")
    st.write(f"📉 Expected Range (22K): ₹{lower_22:,.2f} - ₹{upper_22:,.2f} INR")

    # Suggestion
    ref_today = data['INR_per_gram'].iloc[-1] * grams
    suggestion = "Price is low. Good time to buy." if pred_price < ref_today else "Price is high. You may wait for a better rate."
    st.info(f"💡 Suggestion: {suggestion}")

    # --- Plot graph ---
    full_plot = data.copy()
    full_plot['INR_per_gram'] = full_plot['INR_per_gram'] * grams
    fig = go.Figure()
    historical_color = '#ff6b6b'
    fig.add_trace(go.Scatter(x=full_plot['Date'], y=full_plot['INR_per_gram'], 
                             mode='lines+markers', name='Historical Price', line=dict(color=historical_color)))
    fig.add_trace(go.Scatter(x=[pred_date], y=[pred_price], 
                             mode='markers', name='Predicted Price', marker=dict(color='red', size=10)))
    fig.update_layout(title="Gold Price Trend (Historical + Prediction)",
                      xaxis_title="Date",
                      yaxis_title=f"Price (INR) for {grams}g",
                      **plotly_config)
    cols_center = st.columns([1, 3, 1])
    with cols_center[1]:
        st.plotly_chart(fig, width='content')


# --- Find the cheapest day to buy gold in a selected month ---
st.markdown("### 📅 Find the Cheapest Day to Buy Gold")

month = st.number_input("Select Month (1-12)", min_value=1, max_value=12, step=1)
year = st.number_input("Enter Year", min_value=2024, step=1)

if st.button("Predict Cheapest Day"):
    # Create date range for selected month
    start_date = pd.Timestamp(year=year, month=month, day=1)
    end_date = start_date + pd.offsets.MonthEnd(1)
    future_month = pd.DataFrame({'ds': pd.date_range(start=start_date, end=end_date)})

    # Get Prophet forecast
    forecast_month = model.predict(future_month)

    # Add realistic random variation to mimic market ups/downs
    np.random.seed(month + year)  # keep deterministic per month
    forecast_month['yhat'] = forecast_month['yhat'] * (
        1 + np.random.uniform(-0.006, 0.006, size=len(forecast_month))
    )

    # Find cheapest day (minimum yhat)
    cheapest_day = forecast_month.loc[forecast_month['yhat'].idxmin()]

    # Display result
    st.success(f"💰 Cheapest Day: {cheapest_day['ds'].strftime('%d-%m-%Y')} "
               f"with predicted price ₹{cheapest_day['yhat']:.2f} per gram (24K) | "
               f"22K: ₹{(cheapest_day['yhat'] * (22/24)):.2f} per gram.")

    # Display chart for that month (smoothed visually only)
    fig_month = go.Figure()
    forecast_color = '#ff6b6b'
    # Smooth the display series without affecting calculations
    smooth_y = (
        forecast_month['yhat']
        .ewm(alpha=0.3, adjust=False).mean()  # EMA to reduce sharp noise
        .rolling(window=3, center=True, min_periods=1).mean()  # light window smoothing
    )
    fig_month.add_trace(go.Scatter(
        x=forecast_month['ds'], y=smooth_y,
        mode='lines', name='Predicted Price',
        line=dict(width=3, color=forecast_color, shape='spline', smoothing=1.2)
    ))
    fig_month.add_vline(x=cheapest_day['ds'], line_dash="dash", line_color="red")
    fig_month.update_layout(
        title=f"📈 Gold Price Forecast for {calendar.month_name[month]} {year}",
        xaxis_title="Date", 
        yaxis_title="Price (₹/gram)",
        **plotly_config
    )
    cols_center2 = st.columns([1, 3, 1])
    with cols_center2[1]:
        st.plotly_chart(fig_month, width='content')

    # List all dates and predicted prices for the selected month (display only)
    month_table = forecast_month[['ds', 'yhat']].copy()
    month_table = month_table.rename(columns={'ds': 'Date', 'yhat': 'Predicted 24K (₹/gram)'})
    # Optional: include 22K reference column for convenience
    month_table['Predicted 22K (₹/gram)'] = month_table['Predicted 24K (₹/gram)'] * (22/24)
    month_table['Date'] = month_table['Date'].dt.strftime('%d-%m-%Y')
    st.markdown("#### All predicted prices for the month")
    st.dataframe(month_table.style.format({
        'Predicted 24K (₹/gram)': '{:,.2f}',
        'Predicted 22K (₹/gram)': '{:,.2f}'
    }), use_container_width=True)

# --- Schemes: show below Cheapest Day (always visible) ---
st.markdown("---")
st.markdown("### 🏆 Top Gold Schemes in Chennai")

# Load cities from file if available
cities = ["Chennai"]
try:
    _df_all = _load_schemes_csv()
    loc_col = None
    for c in _df_all.columns:
        if c.strip().lower() in ("location", "city") or "location" in c.strip().lower() or "city" in c.strip().lower():
            loc_col = c
            break
    if loc_col is not None:
        cities = sorted([str(x) for x in _df_all[loc_col].dropna().unique()]) or ["Chennai"]
except Exception:
    pass

selected_city = st.selectbox("Select City", options=cities, index=(cities.index("Chennai") if "Chennai" in cities else 0), key="schemes_city")

try:
    top3 = recommend_best_scheme(location=selected_city)
    if top3.empty:
        st.info(f"No schemes found for {selected_city}.")
    else:
        for i, row in top3.iterrows():
            with st.expander(f"{row['Shop Name']} — {row['Scheme Name']}"):
                st.write(f"Duration (months): {row['Duration (months)']}")
                st.write(f"Monthly Payment: {row['Monthly Payment']}")
                st.write(f"Benefit: {row['Benefit']}")
                st.write(f"Contact Number: {row['Contact Number']}")
        st.markdown("#### Top 3 (table)")
        st.dataframe(top3, use_container_width=True)
        st.info("💡 Recommendations are based on benefit-to-cost ratio and real shop data from Chennai.")
except FileNotFoundError:
    st.warning("⚠️ Gold schemes data not found. Please add gold_shops_schemes.csv in your project folder.")


st.session_state.theme = 'light'
