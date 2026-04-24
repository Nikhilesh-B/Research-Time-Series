"""BayesianARForecaster -- Gaussian prior on AR coefficients (ridge or Minnesota).

Fits AR(p) as linear regression on lags.  Two conjugate prior modes (same
posterior-mean linear solve per ``fit``):

* **ridge:** zero-mean prior with precision ``prior_precision`` * I — posterior
  mean equals ridge regression :math:`(X'X + \\lambda I)^{-1} X'y`.

* **minnesota:** diagonal precision that increases with lag (tighter prior on
  higher lags), optionally with prior mean on the first AR coefficient toward a
  random walk (:math:`\\phi_1 \\approx 1`).  Still closed-form; no MCMC.

Multi-step forecasts use the posterior mean recursively (plug-in).  Full MCMC
per origin is intentionally not used here.
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
        X[:, off + j] = y[p - 1 + np.arange(m) - j]
    y_tgt = y[p:].copy()
    return X, y_tgt


def _prior_precision_matrix(
    p: int,
    include_intercept: bool,
    prior_mode: str,
    prior_precision: float,
    minnesota_lag_decay_exponent: float,
) -> np.ndarray:
    """Return diagonal prior precision Σ0^{-1} (shape (d,d))."""
    d = p + int(include_intercept)
    if prior_mode == "ridge":
        return prior_precision * np.eye(d, dtype=np.float64)
    # Minnesota: per AR lag j (1..p), precision ∝ λ * j^exponent
    if minnesota_lag_decay_exponent < 0:
        raise ValueError("minnesota_lag_decay_exponent must be non-negative")
    diag = np.zeros(d, dtype=np.float64)
    if include_intercept:
        diag[0] = prior_precision
        lag_slice = slice(1, None)
    else:
        lag_slice = slice(0, None)
    j = np.arange(1, p + 1, dtype=np.float64)
    diag[lag_slice] = prior_precision * (j ** float(minnesota_lag_decay_exponent))
    return np.diag(diag)


def _prior_mean_vector(
    p: int,
    include_intercept: bool,
    prior_mode: str,
    minnesota_center_rw: bool,
) -> np.ndarray:
    """Return prior mean μ0 (length d)."""
    d = p + int(include_intercept)
    mu0 = np.zeros(d, dtype=np.float64)
    if prior_mode == "minnesota" and minnesota_center_rw:
        if include_intercept:
            mu0[1] = 1.0  # first AR lag
        else:
            mu0[0] = 1.0
    return mu0


class BayesianARForecaster(Forecaster):
    """AR(p) with Gaussian prior; posterior mean gives point forecasts.

    Parameters:
        p: Autoregressive order (number of lags).
        prior_precision: Overall tightness λ (ridge: uniform diagonal precision;
            Minnesota: scales the lag-specific diagonal precisions).
        prior_mode: ``"ridge"`` (default) or ``"minnesota"``.
        minnesota_lag_decay_exponent: For Minnesota only: precision on lag j is
            ``prior_precision * j**exponent`` (larger exponent = relatively
            tighter on higher lags).  Ignored in ridge mode.
        minnesota_center_rw: For Minnesota only: if True, prior mean pulls the
            first AR coefficient toward 1 (random-walk centering).  If False,
            prior mean is zero (decaying diagonal only).
        include_intercept: If True, prepend intercept; ridge uses same λ on it;
            Minnesota uses ``prior_precision`` on the intercept only.
    """

    def __init__(
        self,
        p: int = 2,
        prior_precision: float = 1.0,
        *,
        prior_mode: str = "ridge",
        minnesota_lag_decay_exponent: float = 2.0,
        minnesota_center_rw: bool = True,
        include_intercept: bool = False,
    ) -> None:
        if p < 1:
            raise ValueError("p must be at least 1")
        if prior_precision < 0:
            raise ValueError("prior_precision must be non-negative")
        if prior_mode not in ("ridge", "minnesota"):
            raise ValueError('prior_mode must be "ridge" or "minnesota"')
        self.p = p
        self.prior_precision = float(prior_precision)
        self.prior_mode = prior_mode
        self.minnesota_lag_decay_exponent = float(minnesota_lag_decay_exponent)
        self.minnesota_center_rw = bool(minnesota_center_rw)
        self.include_intercept = include_intercept
        self.name = self._make_name()
        self._mu: np.ndarray | None = None
        self._last_y: np.ndarray | None = None

    def _make_name(self) -> str:
        lam = self.prior_precision
        if self.prior_mode == "ridge":
            return f"BayesAR(p={self.p},ridge,λ={lam})"
        rw = "RW" if self.minnesota_center_rw else "0"
        dec = self.minnesota_lag_decay_exponent
        return f"BayesAR(p={self.p},MN,λ={lam},dec={dec},μ={rw})"

    def fit(self, history: np.ndarray) -> None:
        y = np.asarray(history, dtype=np.float64).ravel()
        self._last_y = y.copy()
        d = self.p + int(self.include_intercept)
        if len(y) <= self.p:
            self._mu = None
            return
        X, y_tgt = _build_lag_design(y, self.p, self.include_intercept)
        xtx = X.T @ X
        sigma_inv = _prior_precision_matrix(
            self.p,
            self.include_intercept,
            self.prior_mode,
            self.prior_precision,
            self.minnesota_lag_decay_exponent,
        )
        mu0 = _prior_mean_vector(
            self.p,
            self.include_intercept,
            self.prior_mode,
            self.minnesota_center_rw,
        )
        rhs = X.T @ y_tgt + sigma_inv @ mu0
        try:
            self._mu = np.linalg.solve(xtx + sigma_inv, rhs)
        except np.linalg.LinAlgError:
            self._mu = None

    def predict(self, horizon: int) -> np.ndarray:
        if self._mu is None or self._last_y is None:
            return np.full(horizon, np.nan, dtype=np.float64)
        mu = self._mu
        p = self.p
        ext = np.concatenate([self._last_y, np.zeros(horizon, dtype=np.float64)])
        n_hist = len(self._last_y)
        out = np.empty(horizon, dtype=np.float64)
        for h in range(horizon):
            t = n_hist + h
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
