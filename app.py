"""Market Regime Detector -- Streamlit entry point."""

import time

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


def _info_box(text: str) -> str:
    """Render a styled info box HTML string."""
    return (
        f'<div style="border-left:3px solid #ff8c00; padding:12px 14px; margin:8px 0; '
        f'background:#0a0a0a; font-family:JetBrains Mono,Consolas,monospace; '
        f'font-size:12px; color:#cccccc; line-height:1.7; border-radius:2px">{text}</div>'
    )

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
_dot_cycle = [".  ", ".. ", "..."]
_dot_index = 0


_step_descriptions = {
    "Fetching price data": "Downloading historical OHLCV data from Yahoo Finance",
    "Engineering features": "Computing log returns, rolling volatility, and momentum spread",
    "Fitting Hidden Markov Model": "Training a Gaussian HMM with full covariance across multiple random seeds",
    "Decoding state sequence": "Running the Viterbi algorithm to find the most likely regime path",
    "Labeling regimes": "Classifying each hidden state as bull, bear, or volatile based on return statistics",
    "Building dashboard": "Generating charts and computing transition probabilities",
}


def _step(progress, caption, pct, pause=0.0):
    """Update progress bar and animate dots in the caption area."""
    global _dot_index
    desc = _step_descriptions.get(caption, "")
    caption_area.markdown(
        f'<div style="font-family:JetBrains Mono,Consolas,monospace; text-align:center; padding:16px 0">'
        f'<div style="font-size:18px; color:#ffffff">{caption}{_dot_cycle[_dot_index]}</div>'
        f'<div style="font-size:12px; color:#999999; margin-top:8px">{desc}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    _dot_index = (_dot_index + 1) % 3
    progress.progress(pct)
    if pause:
        time.sleep(pause)


def run_analysis_with_progress(ticker: str, n_states: int, period: str) -> dict:
    """Full pipeline with progress bar updates."""
    global _dot_index
    _dot_index = 0

    progress = st.progress(0)
    _step(progress, "Fetching price data", 0, 0.3)

    df = fetch_ohlcv(ticker, period)
    _step(progress, "Engineering features", 20, 0.2)

    features_df, scaler = compute_features(df)
    aligned_prices = df["Close"].squeeze().loc[features_df.index]
    raw_returns = np.log(aligned_prices / aligned_prices.shift(1)).dropna()
    common_idx = raw_returns.index
    features_df = features_df.loc[common_idx]
    aligned_prices = aligned_prices.loc[common_idx]
    _step(progress, "Fitting Hidden Markov Model", 35, 0.3)

    feat_array = features_df.values
    # Animate dots during HMM fitting (the slow part)
    import threading
    fit_result = [None]
    fit_error = [None]

    def _fit():
        try:
            fit_result[0] = fit_hmm(feat_array, n_states)
        except Exception as e:
            fit_error[0] = e

    t = threading.Thread(target=_fit)
    t.start()
    pct = 35
    while t.is_alive():
        pct = min(pct + 2, 72)
        _step(progress, "Fitting Hidden Markov Model", pct, 0.4)
    t.join()

    if fit_error[0]:
        raise fit_error[0]
    model = fit_result[0]

    _step(progress, "Decoding state sequence", 75, 0.2)
    state_seq = decode_states(model, feat_array)

    _step(progress, "Labeling regimes", 85, 0.2)
    state_stats = label_states(model, raw_returns, state_seq, n_states)
    trans_matrix = compute_transition_matrix(model)

    _step(progress, "Building dashboard", 95, 0.2)

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

    progress.progress(100)
    caption_area.markdown(
        '<div style="font-size:18px; color:#00d26a; font-family:JetBrains Mono,Consolas,monospace; '
        'text-align:center; padding:16px 0">Done</div>',
        unsafe_allow_html=True,
    )
    time.sleep(0.4)
    progress.empty()
    caption_area.empty()

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
caption_area = st.empty()

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
        <div style="font-size: 13px; color: #cccccc; letter-spacing: 2px; margin-bottom: 32px">
            MARKET REGIME DETECTOR
        </div>
        <div style="font-size: 12px; color: #bbbbbb; max-width: 480px; line-height: 1.8; margin-bottom: 16px">
            Identifies bull, bear, and volatile market regimes using
            Gaussian Hidden Markov Models trained on historical price data.
            Works with any stock, ETF, or crypto ticker.
        </div>
        <div style="font-size: 11px; color: #999999; margin-top: 8px">
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
  <span style="font-size:13px; color:#cccccc; margin-left:12px; font-family:'JetBrains Mono',Consolas,monospace">
    Active for {streak} days / Since {start_date} / Log-likelihood: {ll:.1f}
  </span>
</div>
""", unsafe_allow_html=True)

# -- Build dynamic info text
tk = result["ticker"]
ss = result["state_stats"]
n = result["n_states"]
bull_s = [s for s in ss.values() if s["label"] == "bull"][0]
bear_s = [s for s in ss.values() if s["label"] == "bear"][0]
bull_pct = f"{bull_s['pct_time']:.0%}"
bear_pct = f"{bear_s['pct_time']:.0%}"
bull_ret = f"{bull_s['mean_return_annualized']:+.1%}"
bear_ret = f"{bear_s['mean_return_annualized']:+.1%}"
bull_vol = f"{bull_s['volatility_annualized']:.1%}"
bear_vol = f"{bear_s['volatility_annualized']:.1%}"

trans = result["transition_matrix"]
# Find bull and bear indices
bull_idx = [i for i, s in ss.items() if s["label"] == "bull"][0]
bear_idx = [i for i, s in ss.items() if s["label"] == "bear"][0]
bull_persist = f"{trans[bull_idx][bull_idx]:.0%}"
bear_persist = f"{trans[bear_idx][bear_idx]:.0%}"

# -- Panel 2: Price chart + Regime stats
col1, col2 = st.columns([3, 1])

with col1:
    st.plotly_chart(make_price_chart(result), use_container_width=True)
    st.markdown(_info_box(
        f"{tk} spent {bull_pct} of the period in a bull regime (annualized return {bull_ret}) "
        f"and {bear_pct} in a bear regime ({bear_ret}). "
        f"The colored bands show which regime the model assigned to each trading day. "
        f"Use the 6M/1Y/2Y/ALL buttons to zoom into specific periods."
    ), unsafe_allow_html=True)

with col2:
    st.markdown(make_regime_stats_html(result), unsafe_allow_html=True)

# -- Panel 3: Regime timeline
st.plotly_chart(make_regime_timeline(result), use_container_width=True)
st.markdown(_info_box(
    f"Each color represents the regime the HMM detected on that day. "
    f"Long unbroken stretches mean the model is confident in a sustained regime. "
    f"Frequent color changes suggest {tk} was cycling between states rapidly."
), unsafe_allow_html=True)

# -- Panel 4: Transition matrix + Feature charts
col3, col4 = st.columns([1, 2])

with col3:
    st.plotly_chart(make_transition_heatmap(result), use_container_width=True)
    st.markdown(_info_box(
        f"The probability of {tk} staying in a bull regime day-to-day is {bull_persist}, "
        f"and {bear_persist} for bear. Higher diagonal values mean regimes tend to persist "
        f"rather than flip. Off-diagonal values show how likely a regime switch is on any given day."
    ), unsafe_allow_html=True)

with col4:
    st.plotly_chart(make_feature_charts(result), use_container_width=True)
    st.markdown(_info_box(
        f"These are the three features the HMM uses to classify {tk} into regimes. "
        f"Log returns capture daily price moves. Rolling volatility (21-day) measures "
        f"how turbulent the market has been recently. Momentum spread (5d vs 21d average) "
        f"shows whether short-term trend is above or below the longer-term trend."
    ), unsafe_allow_html=True)

# -- Footer
st.markdown("""
<div style="
    text-align: right;
    padding: 20px 10px 10px 0;
    font-size: 11px;
    color: #999999;
    font-family: 'JetBrains Mono', Consolas, monospace;
">Created by Sanjay Krishnan</div>
""", unsafe_allow_html=True)
