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


# -- Analysis pipeline with progress tracking
def run_analysis_with_progress(ticker: str, n_states: int, period: str) -> dict:
    """Full pipeline with progress bar updates."""
    progress = st.progress(0, text="Fetching price data...")

    df = fetch_ohlcv(ticker, period)
    progress.progress(20, text="Engineering features...")

    features_df, scaler = compute_features(df)
    aligned_prices = df["Close"].squeeze().loc[features_df.index]
    raw_returns = np.log(aligned_prices / aligned_prices.shift(1)).dropna()
    common_idx = raw_returns.index
    features_df = features_df.loc[common_idx]
    aligned_prices = aligned_prices.loc[common_idx]
    progress.progress(35, text="Fitting Hidden Markov Model...")

    feat_array = features_df.values
    model = fit_hmm(feat_array, n_states)
    progress.progress(75, text="Decoding state sequence...")

    state_seq = decode_states(model, feat_array)
    progress.progress(85, text="Labeling regimes...")

    state_stats = label_states(model, raw_returns, state_seq, n_states)
    trans_matrix = compute_transition_matrix(model)
    progress.progress(95, text="Building dashboard...")

    current_state = int(state_seq[-1])
    current_label = state_stats[current_state]["label"]
    streak = 1
    for i in range(len(state_seq) - 2, -1, -1):
        if state_seq[i] == current_state:
            streak += 1
        else:
            break
    streak_start = common_idx[-streak]
    dates_list = [d.strftime("%Y-%m-%d") for d in common_idx]

    progress.progress(100, text="Done.")
    progress.empty()

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
        result = run_analysis_with_progress(ticker, n_states, period)
        st.session_state["result"] = result
    except (ValueError, RuntimeError) as e:
        st.error(str(e))
        st.session_state.pop("result", None)

result = st.session_state.get("result")

if result is None:
    st.markdown("""
    <div style="
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 60vh;
        text-align: center;
        font-family: 'JetBrains Mono', Consolas, monospace;
    ">
        <div style="font-size: 40px; font-weight: 700; color: #ff8c00; letter-spacing: 6px; margin-bottom: 12px">
            MRD
        </div>
        <div style="font-size: 13px; color: #666666; letter-spacing: 2px; margin-bottom: 32px">
            MARKET REGIME DETECTOR
        </div>
        <div style="font-size: 12px; color: #444444; max-width: 480px; line-height: 1.8; margin-bottom: 16px">
            Identifies bull, bear, and volatile market regimes using
            Gaussian Hidden Markov Models trained on historical price data.
            Works with any stock, ETF, or crypto ticker.
        </div>
        <div style="font-size: 11px; color: #333333; margin-top: 8px">
            Enter a ticker in the sidebar and click Analyze to begin.
        </div>
    </div>
    """, unsafe_allow_html=True)
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
