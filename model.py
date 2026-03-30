"""HMM fitting, Viterbi decoding, and state labeling.

Uses a pure numpy/scipy Gaussian HMM implementation so that hmmlearn
(which requires C compilation) is not needed.
"""

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal


class SimpleGaussianHMM:
    """Gaussian HMM with full covariance, trained via EM."""

    def __init__(self, n_components, n_iter=200, random_state=42):
        self.n_components = n_components
        self.n_iter = n_iter
        self.random_state = random_state

        self.startprob_ = None
        self.transmat_ = None
        self.means_ = None
        self.covars_ = None
        self._score = None

    def fit(self, X):
        rng = np.random.RandomState(self.random_state)
        T, D = X.shape
        K = self.n_components

        # --- Initialization via K-Means++ style ---
        indices = rng.choice(T, size=K, replace=False)
        means = X[indices].copy()
        covars = np.array([np.eye(D) for _ in range(K)])
        startprob = np.ones(K) / K
        transmat = np.ones((K, K)) / K + rng.dirichlet(np.ones(K), size=K) * 0.1
        transmat = transmat / transmat.sum(axis=1, keepdims=True)

        log_likelihood_prev = -np.inf

        for iteration in range(self.n_iter):
            # --- E-step: forward-backward ---
            # Compute emission log-probs
            log_emit = np.zeros((T, K))
            for k in range(K):
                try:
                    log_emit[:, k] = multivariate_normal.logpdf(
                        X, mean=means[k], cov=covars[k], allow_singular=True
                    )
                except Exception:
                    log_emit[:, k] = -1e10

            # Forward pass (log-space)
            log_alpha = np.zeros((T, K))
            log_alpha[0] = np.log(startprob + 1e-300) + log_emit[0]

            log_trans = np.log(transmat + 1e-300)
            for t in range(1, T):
                for j in range(K):
                    log_alpha[t, j] = (
                        _logsumexp(log_alpha[t - 1] + log_trans[:, j])
                        + log_emit[t, j]
                    )

            log_likelihood = _logsumexp(log_alpha[-1])

            # Backward pass (log-space)
            log_beta = np.zeros((T, K))
            for t in range(T - 2, -1, -1):
                for i in range(K):
                    log_beta[t, i] = _logsumexp(
                        log_trans[i, :] + log_emit[t + 1] + log_beta[t + 1]
                    )

            # Posterior (gamma)
            log_gamma = log_alpha + log_beta
            log_gamma -= _logsumexp_2d(log_gamma)
            gamma = np.exp(log_gamma)

            # Xi (transition posteriors)
            xi = np.zeros((K, K))
            for t in range(T - 1):
                for i in range(K):
                    for j in range(K):
                        xi[i, j] += np.exp(
                            log_alpha[t, i]
                            + log_trans[i, j]
                            + log_emit[t + 1, j]
                            + log_beta[t + 1, j]
                            - log_likelihood
                        )

            # --- M-step ---
            startprob = gamma[0] / gamma[0].sum()

            transmat = xi / (xi.sum(axis=1, keepdims=True) + 1e-300)

            for k in range(K):
                wk = gamma[:, k]
                wk_sum = wk.sum() + 1e-300
                means[k] = (wk[:, None] * X).sum(axis=0) / wk_sum
                diff = X - means[k]
                covars[k] = (diff.T @ (diff * wk[:, None])) / wk_sum
                covars[k] += np.eye(D) * 1e-6  # regularization

            # Convergence check
            if abs(log_likelihood - log_likelihood_prev) < 1e-4:
                break
            log_likelihood_prev = log_likelihood

        self.startprob_ = startprob
        self.transmat_ = transmat
        self.means_ = means
        self.covars_ = covars
        self._score = log_likelihood

    def score(self, X):
        """Return log-likelihood of data under fitted model."""
        if self._score is not None:
            return self._score
        # Recompute via forward algorithm
        T, D = X.shape
        K = self.n_components
        log_emit = np.zeros((T, K))
        for k in range(K):
            log_emit[:, k] = multivariate_normal.logpdf(
                X, mean=self.means_[k], cov=self.covars_[k], allow_singular=True
            )
        log_alpha = np.zeros((T, K))
        log_alpha[0] = np.log(self.startprob_ + 1e-300) + log_emit[0]
        log_trans = np.log(self.transmat_ + 1e-300)
        for t in range(1, T):
            for j in range(K):
                log_alpha[t, j] = (
                    _logsumexp(log_alpha[t - 1] + log_trans[:, j])
                    + log_emit[t, j]
                )
        return _logsumexp(log_alpha[-1])

    def predict(self, X):
        """Viterbi decoding — most likely state sequence."""
        T, D = X.shape
        K = self.n_components

        log_emit = np.zeros((T, K))
        for k in range(K):
            log_emit[:, k] = multivariate_normal.logpdf(
                X, mean=self.means_[k], cov=self.covars_[k], allow_singular=True
            )

        log_trans = np.log(self.transmat_ + 1e-300)
        viterbi = np.zeros((T, K))
        backptr = np.zeros((T, K), dtype=int)

        viterbi[0] = np.log(self.startprob_ + 1e-300) + log_emit[0]

        for t in range(1, T):
            for j in range(K):
                scores = viterbi[t - 1] + log_trans[:, j]
                backptr[t, j] = np.argmax(scores)
                viterbi[t, j] = scores[backptr[t, j]] + log_emit[t, j]

        # Backtrace
        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(viterbi[-1])
        for t in range(T - 2, -1, -1):
            states[t] = backptr[t + 1, states[t + 1]]

        return states


def _logsumexp(x):
    """Numerically stable log-sum-exp for 1D array."""
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m)))


def _logsumexp_2d(x):
    """Row-wise logsumexp for 2D array, returns column vector."""
    m = np.max(x, axis=1, keepdims=True)
    return m + np.log(np.sum(np.exp(x - m), axis=1, keepdims=True))


def fit_hmm(features: np.ndarray, n_states: int = 3) -> SimpleGaussianHMM:
    """
    Fit a GaussianHMM with full covariance.

    Runs 5 fits with different random seeds and keeps the model with the
    highest log-likelihood.
    """
    best_model = None
    best_score = -np.inf

    for seed in range(5):
        model = SimpleGaussianHMM(
            n_components=n_states,
            n_iter=200,
            random_state=42 + seed,
        )
        try:
            model.fit(features)
            score = model.score(features)
            if score > best_score:
                best_score = score
                best_model = model
        except Exception:
            continue

    if best_model is None:
        raise RuntimeError("Model failed to converge after 5 attempts")

    return best_model


def decode_states(model: SimpleGaussianHMM, features: np.ndarray) -> np.ndarray:
    """Run Viterbi decoding to find the most likely state sequence."""
    return model.predict(features)


def smooth_states(states: np.ndarray, min_duration: int = 5) -> np.ndarray:
    """
    Merge short regime runs into neighbors.

    Repeats until no run is shorter than min_duration.
    """
    if min_duration <= 1:
        return states.copy()

    out = states.copy()

    while True:
        # Run-length encode
        runs = []
        start = 0
        for i in range(1, len(out)):
            if out[i] != out[start]:
                runs.append((start, i - start, int(out[start])))
                start = i
        runs.append((start, len(out) - start, int(out[start])))

        # Find first short run
        short_idx = None
        for idx, (_, length, _) in enumerate(runs):
            if length < min_duration:
                short_idx = idx
                break

        if short_idx is None:
            break

        # Merge into the longer neighbor
        s, length, _ = runs[short_idx]
        if short_idx == 0:
            merge_val = runs[1][2]
        elif short_idx == len(runs) - 1:
            merge_val = runs[-2][2]
        else:
            prev_len = runs[short_idx - 1][1]
            next_len = runs[short_idx + 1][1]
            merge_val = runs[short_idx - 1][2] if prev_len >= next_len else runs[short_idx + 1][2]

        out[s:s + length] = merge_val

    return out


def label_states(
    model: SimpleGaussianHMM,
    raw_returns: pd.Series,
    state_seq: np.ndarray,
    n_states: int,
) -> dict:
    """
    Auto-label each state as 'bull', 'bear', or 'volatile' based on
    mean return per state.
    """
    stats = {}
    for i in range(n_states):
        mask = state_seq == i
        returns_i = raw_returns[mask]
        mean_ret = returns_i.mean() if len(returns_i) > 0 else 0.0
        std_ret = returns_i.std() if len(returns_i) > 1 else 0.0
        pct = mask.sum() / len(state_seq)

        # Average consecutive run length
        runs = []
        count = 0
        for s in state_seq:
            if s == i:
                count += 1
            else:
                if count > 0:
                    runs.append(count)
                count = 0
        if count > 0:
            runs.append(count)
        avg_dur = np.mean(runs) if runs else 0.0

        ann_ret = mean_ret * 252
        ann_vol = std_ret * np.sqrt(252)
        sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0

        stats[i] = {
            "mean_return": mean_ret,
            "mean_return_annualized": ann_ret,
            "volatility_annualized": ann_vol,
            "sharpe": sharpe,
            "pct_time": pct,
            "avg_duration_days": avg_dur,
        }

    # Assign labels by mean return ranking
    sorted_states = sorted(stats.keys(), key=lambda k: stats[k]["mean_return"])

    if n_states == 2:
        stats[sorted_states[0]]["label"] = "bear"
        stats[sorted_states[1]]["label"] = "bull"
    else:
        stats[sorted_states[0]]["label"] = "bear"
        stats[sorted_states[-1]]["label"] = "bull"
        for s in sorted_states[1:-1]:
            stats[s]["label"] = "volatile"

    return stats


def compute_transition_matrix(model: SimpleGaussianHMM) -> np.ndarray:
    """Return the fitted transition matrix."""
    return model.transmat_
