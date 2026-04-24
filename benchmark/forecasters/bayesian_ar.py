"""BayesianARForecaster -- conjugate Gaussian prior on AR coefficients (ridge Bayes).

Fits AR(p) as linear regression on lags with a zero-mean Gaussian prior on
coefficients (diagonal precision ``prior_precision``).  The posterior mean of
the coefficients equals the ridge estimator

    (X'X + λ I)^{-1} X'y,

which is fast enough for expanding-window benchmarks.  Multi-step forecasts
use the posterior mean recursively (plug-in), not full predictive simulation.

Full MCMC (e.g. PyMC) per origin is intentionally not used here: the benchmark
runner refits at every origin, so sampling would be prohibitively slow.
"""

from __future__ import annotations

import numpy as np

from benchmark.forecasters.base import Forecaster


def _build_lag_design(history: np.ndarray, p: int, include_intercept: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return design matrix X and target y for AR(p) on *history* (1-D)."""
    y = np.asarray(history, dtype=np.float64).ravel()
    n = len(y)
    if n <= p:
        raise ValueError("history too short for AR order")
    # Rows t = p .. n-1: predict y[t] from lags y[t-1],...,y[t-p]
    m = n - p
    X = np.empty((m, p + int(include_intercept)), dtype=np.float64)
    if include_intercept:
        X[:, 0] = 1.0
        off = 1
    else:
        off = 0
    for j in range(p):
        # column j: y[t-j-1] for row index i = t - p, t = p+i -> y[p+i-j-1]
        X[:, off + j] = y[p - 1 + np.arange(m) - j]
    y_tgt = y[p:].copy()
    return X, y_tgt


class BayesianARForecaster(Forecaster):
    """AR(p) with Gaussian prior on coefficients; posterior mean for point forecasts.

    Parameters:
        p: Autoregressive order (number of lags).
        prior_precision: Diagonal prior precision λ (ridge penalty on ||φ||²).
            Larger values shrink coefficients harder toward zero.
        include_intercept: If True, prepend a constant column with the same prior.
    """

    def __init__(
        self,
        p: int = 2,
        prior_precision: float = 1.0,
        *,
        include_intercept: bool = False,
    ) -> None:
        if p < 1:
            raise ValueError("p must be at least 1")
        if prior_precision < 0:
            raise ValueError("prior_precision must be non-negative")
        self.p = p
        self.prior_precision = float(prior_precision)
        self.include_intercept = include_intercept
        self.name = f"BayesAR(p={p},λ={self.prior_precision})"
        self._mu: np.ndarray | None = None
        self._last_y: np.ndarray | None = None

    def fit(self, history: np.ndarray) -> None:
        y = np.asarray(history, dtype=np.float64).ravel()
        self._last_y = y.copy()
        d = self.p + int(self.include_intercept)
        if len(y) <= self.p:
            self._mu = None
            return
        X, y_tgt = _build_lag_design(y, self.p, self.include_intercept)
        xtx = X.T @ X
        rhs = X.T @ y_tgt
        reg = self.prior_precision * np.eye(d, dtype=np.float64)
        try:
            self._mu = np.linalg.solve(xtx + reg, rhs)
        except np.linalg.LinAlgError:
            self._mu = None

    def predict(self, horizon: int) -> np.ndarray:
        if self._mu is None or self._last_y is None:
            return np.full(horizon, np.nan, dtype=np.float64)
        mu = self._mu
        p = self.p
        # Extended series: known history then forecasts
        ext = np.concatenate([self._last_y, np.zeros(horizon, dtype=np.float64)])
        n_hist = len(self._last_y)
        out = np.empty(horizon, dtype=np.float64)
        for h in range(horizon):
            t = n_hist + h  # index we are predicting (0-based in ext)
            # lags ext[t-1], ..., ext[t-p]
            if self.include_intercept:
                x = np.empty(p + 1, dtype=np.float64)
                x[0] = 1.0
                x[1:] = ext[t - 1 : t - p - 1 : -1][:p]
            else:
                x = ext[t - 1 : t - p - 1 : -1][:p].copy()
            pred = float(x @ mu)
            out[h] = pred
            ext[t] = pred
        return out
