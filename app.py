"""Market Regime Detector -- Streamlit entry point."""

import numpy as np
import pandas as pd
import streamlit as st

from data import fetch_ohlcv
from features import compute_features
from model import (
    compute_transition_matrix,
    decode_states,
    fit_hmm,
    label_states,
)
from plots import (
    make_feature_charts,
    make_price_chart,
    make_regime_stats_html,
    make_regime_timeline,
    make_transition_heatmap,
)
from utils import REGIME_COLORS

# -- Page config
st.set_page_config(
    page_title="Market Regime Detector",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Terminal styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&display=swap');

    .stApp {
        background-color: #000000;
    }
    section[data-testid="stSidebar"] {
        background-color: #0a0a0a;
        border-right: 1px solid #1a1a1a;
    }
    .stButton > button {
        background-color: #ff8c00;
        color: #000000;
        font-family: 'JetBrains Mono', Consolas, monospace;
        font-weight: 700;
        letter-spacing: 1px;
        border: none;
        border-radius: 2px;
    }
    .stButton > button:hover {
        background-color: #e07b00;
        color: #000000;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# -- Sidebar controls
st.sidebar.title("Market Regime Detector")

ticker = st.sidebar.text_input("Ticker", value="SPY", max_chars=10).upper().strip()
n_states = st.sidebar.radio("Number of states", [2, 3], index=1, horizontal=True)
period = st.sidebar.selectbox("Lookback period", ["1y", "2y", "5y", "10y"], index=2)
analyze = st.sidebar.button("Analyze", type="primary", use_container_width=True)

with st.sidebar.expander("About"):
    st.write(
        "Detect stock market regimes using Gaussian Hidden Markov Models. "
        "The model identifies bull, bear, and volatile periods from price data, "
        "then visualizes the decoded state sequence and transition dynamics."
    )


# -- Cached analysis function
@st.cache_data(ttl=3600, show_spinner=False)
def run_analysis(ticker: str, n_states: int, period: str) -> dict:
    """
    Full pipeline: fetch, features, HMM fit, decode, label.

    Returns a result dict consumed by all dashboard panels.
    """
    # 1. Fetch data
    df = fetch_ohlcv(ticker, period)

    # 2. Engineer features
    features_df, scaler = compute_features(df)

    # 3. Align price/returns to feature dates (trimmed by rolling window)
    aligned_prices = df["Close"].squeeze().loc[features_df.index]
    raw_returns = np.log(aligned_prices / aligned_prices.shift(1)).dropna()

    # Re-align everything to raw_returns index (drops one more row)
    common_idx = raw_returns.index
    features_df = features_df.loc[common_idx]
    aligned_prices = aligned_prices.loc[common_idx]

    # 4. Fit HMM
    feat_array = features_df.values
    model = fit_hmm(feat_array, n_states)

    # 5. Decode states
    state_seq = decode_states(model, feat_array)

    # 6. Label states
    state_stats = label_states(model, raw_returns, state_seq, n_states)

    # 7. Transition matrix
    trans_matrix = compute_transition_matrix(model)

    # 8. Current regime info
    current_state = int(state_seq[-1])
    current_label = state_stats[current_state]["label"]
    streak = 1
    for i in range(len(state_seq) - 2, -1, -1):
        if state_seq[i] == current_state:
            streak += 1
        else:
            break
    streak_start = common_idx[-streak]

    # 9. Build result dict
    dates_list = [d.strftime("%Y-%m-%d") for d in common_idx]

    return {
        "ticker": ticker,
        "n_states": n_states,
        "dates": dates_list,
        "prices": aligned_prices.tolist(),
        "states": state_seq.tolist(),
        "state_labels": {i: state_stats[i]["label"] for i in state_stats},
        "state_stats": state_stats,
        "transition_matrix": trans_matrix.tolist(),
        "features": {
            "dates": dates_list,
            "log_returns": features_df["log_returns"].tolist(),
            "rolling_vol": features_df["rolling_vol"].tolist(),
            "momentum": features_df["momentum"].tolist(),
        },
        "current_state": current_state,
        "current_label": current_label,
        "current_streak_days": streak,
        "streak_start_date": streak_start.strftime("%Y-%m-%d"),
        "model_log_likelihood": float(model.score(feat_array)),
    }


# -- Main area
if analyze:
    try:
        with st.spinner("Fetching data and fitting model..."):
            result = run_analysis(ticker, n_states, period)
        st.session_state["result"] = result
    except (ValueError, RuntimeError) as e:
        st.error(str(e))
        st.session_state.pop("result", None)

result = st.session_state.get("result")

if result is None:
    st.info(
        "Enter a ticker in the sidebar and click Analyze to get started. "
        "Try SPY, QQQ, GLD, or BTC-USD."
    )
    st.stop()

# -- Panel 1: Current regime banner
label = result["current_label"]
colors = REGIME_COLORS[label]
streak = result["current_streak_days"]
start_date = result["streak_start_date"]
ll = result["model_log_likelihood"]

st.markdown(f"""
<div style="
  background: {colors['bg']};
  border-left: 4px solid {colors['border']};
  padding: 16px 20px;
  border-radius: 2px;
  margin-bottom: 12px;
  font-family: 'JetBrains Mono', Consolas, monospace;
">
  <span style="font-size:22px; font-weight:700; color:{colors['text']}; letter-spacing:2px">
    {label.upper()}
  </span>
  <span style="font-size:13px; color:#666666; margin-left:12px; font-family:'JetBrains Mono',Consolas,monospace">
    Active for {streak} days / Since {start_date} / Log-likelihood: {ll:.1f}
  </span>
</div>
""", unsafe_allow_html=True)

# -- Panel 2: Price chart + Regime stats
col1, col2 = st.columns([3, 1])

with col1:
    st.plotly_chart(make_price_chart(result), use_container_width=True)

with col2:
    st.markdown(make_regime_stats_html(result), unsafe_allow_html=True)

# -- Panel 3: Regime timeline
st.plotly_chart(make_regime_timeline(result), use_container_width=True)

# -- Panel 4: Transition matrix + Feature charts
col3, col4 = st.columns([1, 2])

with col3:
    st.plotly_chart(make_transition_heatmap(result), use_container_width=True)

with col4:
    st.plotly_chart(make_feature_charts(result), use_container_width=True)

# -- Footer
st.markdown("""
<div style="
    text-align: right;
    padding: 20px 10px 10px 0;
    font-size: 11px;
    color: #444444;
    font-family: 'JetBrains Mono', Consolas, monospace;
">Created by Sanjay Krishnan</div>
""", unsafe_allow_html=True)
