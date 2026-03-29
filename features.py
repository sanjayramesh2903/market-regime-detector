"""Feature engineering for HMM regime detection."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def compute_features(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Compute three features from OHLCV data and standardize them.

    Feature 1 -- Log returns:
        r_t = log(Close_t / Close_{t-1})
        Log returns are approximately normally distributed, matching the
        Gaussian emission assumption of the HMM.

    Feature 2 -- 21-day rolling volatility (annualized):
        vol_t = std(r_{t-20} ... r_t) * sqrt(252)
        Window = 21 trading days ~ 1 month. Captures regime-level risk.

    Feature 3 -- Momentum spread (5/21 return spread):
        mom_t = mean(r_{t-4}...r_t) - mean(r_{t-20}...r_t)
        Short-term minus medium-term mean return. Positive in trending
        regimes, near zero in choppy ones.

    The first 21 rows are dropped (NaN from rolling windows).
    All features are standardized with sklearn StandardScaler before HMM fitting.

    Returns:
        (feature_df, scaler) where feature_df has columns
        ['log_returns', 'rolling_vol', 'momentum'] and the same DatetimeIndex
        (trimmed), and scaler is the fitted StandardScaler.
    """
    close = df["Close"].squeeze()

    # Feature 1: log returns
    log_returns = np.log(close / close.shift(1))

    # Feature 2: 21-day rolling volatility, annualized
    rolling_vol = log_returns.rolling(window=21).std() * np.sqrt(252)

    # Feature 3: momentum spread (5-day mean - 21-day mean of returns)
    momentum = log_returns.rolling(window=5).mean() - log_returns.rolling(window=21).mean()

    features = pd.DataFrame(
        {"log_returns": log_returns, "rolling_vol": rolling_vol, "momentum": momentum},
        index=df.index,
    )

    # Drop first 21 rows that contain NaN from rolling windows
    features = features.dropna()

    # Standardize
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(features.values)
    features_scaled = pd.DataFrame(
        scaled_values, index=features.index, columns=features.columns
    )

    return features_scaled, scaler
