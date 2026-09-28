"""Significance tools for geophysical time series and composites: false
discovery rate, regression with AR(1) errors, the variance inflation of a
mean of autocorrelated values, event-block bootstrap composites, and
permutation / bootstrap tests for correlations and classifiers.

Numerical notes that matter in practice:
- FALSE DISCOVERY RATE, NOT PER-POINT 5 %. A map or section of p-values
  tested point by point at 5 % shades 5 % of a pure-noise field, in
  spatially coherent blobs that look like signal. Benjamini-Hochberg (as
  recommended for field significance by Wilks 2016, BAMS, with
  alpha_FDR = 0.10) controls the expected share of false discoveries and
  stays valid under the positive spatial correlation of geophysical fields.
  NaN p-values (masked points) are excluded from the count, not treated as
  1: a land mask should not make the ocean test stricter.
- AR(1) ERRORS (prais_winsten). Monthly or daily anomalies are persistent
  (lag-1 autocorrelation 0.8-0.95 for stratospheric age of air), so OLS
  standard errors are too small by a factor ~sqrt((1+rho)/(1-rho)). Prais-
  Winsten GLS keeps the first observation (scaled by sqrt(1 - rho^2))
  instead of dropping it as Cochrane-Orcutt does, and `start` marks rows
  that begin a new contiguous segment -- after an excluded volcanic window,
  say -- which are scaled like the first row instead of being differenced
  against a row that is not their predecessor.
- BARTLETT, NOT AR(1), FOR THE NOISE OF A MEAN (acf_inflation). The
  variance of an n-day mean is var/n x (1 + 2 sum_k (1 - k/n) rho_k). An
  AR(1) model fills that sum from rho_1 alone, which fails for oscillating
  series: a daily index with rho_1 = 0.59 but a ~6-day oscillation
  (rho_3 = -0.44, rho_6 = +0.31) has an AR(1) inflation of 3.9 and a
  Bartlett-window inflation (L = 10) of ~1.3 -- the AR(1) model overstates
  the noise of a 30-day mean about threefold. The Bartlett sum uses the
  measured autocorrelations with weights 1 - k/(L+1).
- RESAMPLE EVENTS, NOT DAYS (event_bootstrap). Days of one event (an MJO
  phase passage, an SSW) are not independent; a day bootstrap understates
  the composite's spread by the square root of the event length. Resampling
  whole events with replacement, each carrying all of its days, keeps the
  within-event correlation. The composite is the mean over resampled DAYS
  (events weighted by their length), as the composite itself is.
- FREEDMAN-LANE (partial_corr_test). To test x against y given covariates
  Z, permute the residuals of the REDUCED model (y on Z), add them back to
  its fitted values, and recompute the partial correlation. Permuting the
  residuals of x and correlating them with those of y (Kennedy 1995)
  ignores that the permuted residuals are no longer orthogonal to Z and is
  anti-conservative in small samples with correlated covariates.

Consolidated from the significance code of the stratosphere, MJO and
age-of-air products on https://scorvec.com/stratosphere.html and
https://scorvec.com/mjo.html.

Requires: numpy, scipy (distributions, ranks).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["fdr_adjust", "benjamini_hochberg", "prais_winsten", "PraisWinsten",
           "segment_starts", "acf_inflation", "label_events", "event_bootstrap",
           "pearson", "residualize", "partial_corr_test", "compare_r", "auc"]


# --------------------------------------------------------------------------
# False discovery rate
# --------------------------------------------------------------------------
def fdr_adjust(p) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values), any shape; NaN stays
    NaN and is excluded from the count. Same as
    scipy.stats.false_discovery_control(p, method="bh") on the finite p."""
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    q = p[ok]
    n = q.size
    if n == 0:
        return out
    o = np.argsort(q)
    adj = q[o] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    res = np.empty(n)
    res[o] = np.clip(adj, 0, 1)
    out[ok] = res
    return out


def benjamini_hochberg(p, alpha: float = 0.10) -> np.ndarray:
    """Boolean mask (p's shape) of discoveries at false discovery rate alpha:
    every finite p at or below the largest p_(i) with p_(i) <= alpha i / n."""
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    out = np.zeros(p.shape, bool)
    s = np.sort(p[ok])
    n = s.size
    if n == 0:
        return out
    below = s <= alpha * np.arange(1, n + 1) / n
    if below.any():
        out = ok & (p <= s[np.flatnonzero(below).max()])
    return out


# --------------------------------------------------------------------------
# Regression with AR(1) errors
# --------------------------------------------------------------------------
@dataclass
class PraisWinsten:
    """beta (k, m), se (k, m), rho (m,), dof; m = number of series."""
    beta: np.ndarray
    se: np.ndarray
    rho: np.ndarray
    dof: int


def segment_starts(keep) -> np.ndarray:
    """For the retained rows of a boolean mask over a regular time axis:
    True where the previous time step was not retained (a new segment)."""
    idx = np.flatnonzero(np.asarray(keep, bool))
    return np.r_[True, np.diff(idx) > 1] if idx.size else np.zeros(0, bool)


def prais_winsten(y, X, start=None, iters: int = 2, rho_max: float = 0.97) -> PraisWinsten:
    """GLS with AR(1) errors (Prais-Winsten), iterated `iters` times.

    y     : (n,) or (n, m) -- m series sharing one design
    X     : (n, k) design matrix (include the intercept column yourself)
    start : (n,) bool, rows that begin a contiguous segment (row 0 always
            does); they are scaled by sqrt(1 - rho^2) like the first row
            instead of being quasi-differenced. See segment_starts.
    rho is estimated from residual pairs within segments only and clipped to
    +-rho_max. se is the GLS standard error with dof = n - k."""
    y = np.asarray(y, float)
    one = y.ndim == 1
    Y = y[:, None] if one else y
    X = np.asarray(X, float)
    n, k = X.shape
    st = np.zeros(n, bool) if start is None else np.asarray(start, bool).copy()
    st[0] = True
    pair = ~st[1:]
    nx = np.flatnonzero(~st)
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    rho = np.zeros(Y.shape[1])
    se = np.zeros_like(beta)
    for _ in range(max(1, iters)):
        r = Y - X @ beta
        num = (r[1:][pair] * r[:-1][pair]).sum(0)
        den = (r[:-1][pair] ** 2).sum(0)
        rho = np.clip(np.divide(num, den, out=np.zeros_like(num), where=den > 0), -rho_max, rho_max)
        beta = np.empty((k, Y.shape[1]))
        se = np.empty((k, Y.shape[1]))
        for j in range(Y.shape[1]):
            c = np.sqrt(1 - rho[j] ** 2)
            ys = np.empty(n)
            Xs = np.empty_like(X)
            ys[st] = c * Y[st, j]
            Xs[st] = c * X[st]
            ys[nx] = Y[nx, j] - rho[j] * Y[nx - 1, j]
            Xs[nx] = X[nx] - rho[j] * X[nx - 1]
            b = np.linalg.lstsq(Xs, ys, rcond=None)[0]
            e = ys - Xs @ b
            s2 = (e @ e) / (n - k)
            beta[:, j] = b
            se[:, j] = np.sqrt(np.diag(s2 * np.linalg.inv(Xs.T @ Xs)))
    if one:
        return PraisWinsten(beta[:, 0], se[:, 0], rho[:1], n - k)
    return PraisWinsten(beta, se, rho, n - k)


# --------------------------------------------------------------------------
# Autocorrelation
# --------------------------------------------------------------------------
def acf_inflation(res, times=None, lags: int = 10, clip=(1.0, None)) -> np.ndarray:
    """Variance inflation of a mean of autocorrelated values,
    1 + 2 sum_{k=1..L} (1 - k/(L+1)) rho_k (Bartlett window, L = lags),
    per series along axis 0 of `res`.

    times : optional datetime64-like array (one per row); the rows are then
            placed on a daily grid with NaN gaps, so the lags are in days
            even when some days are missing. None: rows are consecutive.
    clip  : (lo, hi) bounds on the result (None = open)."""
    R = np.asarray(res, float)
    if times is not None:
        d = np.asarray(times, dtype="datetime64[D]")
        k = (d - d.min()).astype(int)
        full = np.full((k.max() + 1,) + R.shape[1:], np.nan)
        full[k] = R
        R = full
    R = R - np.nanmean(R, 0)
    v = np.nanmean(R * R, 0)
    infl = np.ones(R.shape[1:])
    for k in range(1, lags + 1):
        rk = np.nanmean(R[k:] * R[:-k], 0) / v
        infl = infl + 2 * (1 - k / (lags + 1)) * np.nan_to_num(rk)
    lo, hi = clip
    return np.clip(infl, lo, hi) if (lo is not None or hi is not None) else infl


# --------------------------------------------------------------------------
# Composites
# --------------------------------------------------------------------------
def label_events(key, active) -> np.ndarray:
    """Event id per time step: a run of consecutive active steps with the
    same key (e.g. the MJO phase) is one event; -1 where inactive."""
    key = np.asarray(key)
    active = np.asarray(active, bool)
    ev = np.full(key.shape[0], -1)
    k = -1
    for i in range(key.shape[0]):
        if active[i]:
            if i == 0 or not active[i - 1] or key[i] != key[i - 1]:
                k += 1
            ev[i] = k
    return ev


def event_bootstrap(X, events, nboot: int = 1000, rng=None, return_samples: bool = False):
    """Composite mean of X (ndays, ...) over its rows and its standard error
    from an EVENT bootstrap: events (one id per row) are resampled with
    replacement and carry all their rows. NaN in X is skipped per cell.

    Returns (mean, se), plus the (nboot, ...) bootstrap means if
    return_samples. se is NaN with fewer than two events."""
    X = np.asarray(X, float)
    ev = np.asarray(events)
    if ev.shape[0] != X.shape[0]:
        raise ValueError("one event id per row of X")
    rng = np.random.default_rng(rng)
    flat = X.reshape(X.shape[0], -1)
    ok = np.isfinite(flat)
    with np.errstate(invalid="ignore"):
        mean = np.where(ok.any(0), np.nansum(flat, 0) / ok.sum(0), np.nan)
    ue, inv = np.unique(ev, return_inverse=True)
    ne = ue.size
    shape = X.shape[1:]
    if ne < 2:
        nan = np.full(shape, np.nan)
        return (mean.reshape(shape), nan) + ((np.full((0,) + shape, np.nan),) if return_samples else ())
    S = np.zeros((ne, flat.shape[1]))
    C = np.zeros((ne, flat.shape[1]))
    np.add.at(S, inv, np.where(ok, flat, 0.0))
    np.add.at(C, inv, ok.astype(float))
    R = rng.integers(0, ne, size=(nboot, ne))
    W = np.zeros((nboot, ne))
    np.add.at(W, (np.repeat(np.arange(nboot), ne), R.ravel()), 1.0)
    den = W @ C
    with np.errstate(invalid="ignore", divide="ignore"):
        bm = (W @ S) / np.where(den > 0, den, np.nan)
    se = np.nanstd(bm, axis=0)
    out = (mean.reshape(shape), se.reshape(shape))
    return out + ((bm.reshape((nboot,) + shape),) if return_samples else ())


# --------------------------------------------------------------------------
# Correlation tests
# --------------------------------------------------------------------------
def pearson(x, y) -> float:
    x = np.asarray(x, float) - np.mean(x)
    y = np.asarray(y, float) - np.mean(y)
    return float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum()))


def _design(covs, n):
    cols = [np.ones(n)] + [np.asarray(c, float) for c in (covs or [])]
    return np.column_stack(cols)


def residualize(y, covs):
    """Residual of y after OLS on an intercept and the covariates."""
    y = np.asarray(y, float)
    Z = _design(covs, y.size)
    return y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]


def partial_corr_test(x, y, covs=(), nperm: int = 9999, nboot: int = 9999, rng=None) -> dict:
    """Partial correlation of x with y given covariates, with

      p_perm : two-sided Freedman-Lane permutation p (reduced-model
               residuals of y permuted, added back to its fitted values)
      ci     : 95 % bootstrap interval (rows resampled, covariates refit)
      p_F    : nested-OLS F-test p for adding x to the covariates
      dR2    : the R^2 that x adds.
    """
    from scipy import stats
    rng = np.random.default_rng(rng)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = x.size
    covs = [np.asarray(c, float) for c in covs]
    Z = _design(covs, n)
    Hz = Z @ np.linalg.pinv(Z)                           # hat matrix of the reduced model
    rx = x - Hz @ x
    yfit = Hz @ y
    ry = y - yfit
    r = pearson(rx, ry)
    count = 0
    for _ in range(nperm):
        ys = yfit + rng.permutation(ry)
        rs = pearson(rx, ys - Hz @ ys)
        count += abs(rs) >= abs(r) - 1e-12
    p_perm = (count + 1) / (nperm + 1)
    bs = []
    for _ in range(nboot):
        i = rng.integers(0, n, n)
        a = residualize(x[i], [c[i] for c in covs])
        b = residualize(y[i], [c[i] for c in covs])
        if a.std() > 0 and b.std() > 0:
            bs.append(pearson(a, b))
    k = len(covs)
    rss0 = (ry ** 2).sum()
    rss1 = (residualize(y, covs + [x]) ** 2).sum()
    F = (rss0 - rss1) / (rss1 / (n - k - 2))
    return {"r": r, "p_perm": float(p_perm),
            "ci": (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))),
            "p_F": float(stats.f.sf(F, 1, n - k - 2)),
            "dR2": float((rss0 - rss1) / ((y - y.mean()) ** 2).sum()), "n": n}


def compare_r(x1, x2, y, nboot: int = 9999, rng=None) -> dict:
    """Is x1 a better linear predictor of y than x2? d = |r(x1, y)| -
    |r(x2, y)| with a paired bootstrap (the same resampled rows for both, so
    their correlation is respected): 95 % interval and p = share of
    resamples with d <= 0 (one-sided, x1 not better)."""
    rng = np.random.default_rng(rng)
    x1, x2, y = (np.asarray(a, float) for a in (x1, x2, y))
    d0 = abs(pearson(x1, y)) - abs(pearson(x2, y))
    n = y.size
    ds = np.empty(nboot)
    for b in range(nboot):
        i = rng.integers(0, n, n)
        ds[b] = abs(pearson(x1[i], y[i])) - abs(pearson(x2[i], y[i]))
    ds = ds[np.isfinite(ds)]
    return {"diff_abs_r": float(d0),
            "ci": (float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))),
            "p_not_better": float(np.mean(ds <= 0))}


def auc(score, label) -> float:
    """Area under the ROC curve of `score` for binary `label` (1 = event):
    the Mann-Whitney probability that an event outscores a non-event, ties
    counted half (average ranks)."""
    from scipy.stats import rankdata
    score = np.asarray(score, float)
    label = np.asarray(label).astype(bool)
    n1 = label.sum()
    n0 = label.size - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
