"""Forecast verification: CRPS (ensemble, fair and Gaussian), rank histograms and PIT, the Brier score with its
decomposition and reliability curve, the Diebold-Mariano test of equal skill, and strict pairing of dated series.

Numerical notes that matter in practice:
- THE PLAIN ENSEMBLE CRPS PENALISES SMALL ENSEMBLES. crps_ensemble scores the m-member empirical distribution,
  E|X - y| - 1/2 E|X - X'|. For members drawn from the true distribution its expectation is the true CRPS plus
  E|X - X'| / (2m): +20 % for a 5-member, +2 % for a 50-member ensemble of a Gaussian, so it ranks a big ensemble
  ahead of an equally good small one. crps_fair (Ferro 2014) divides the spread term by m(m - 1) instead of m^2 and
  is unbiased for the CRPS of the distribution the members are drawn from, at any m >= 2 (tested from m = 2 to 50).
  Use it to compare systems of different sizes; use the plain one to score the ensemble as issued.
- WEIGHTED MIXTURES (a multi-model blend). With member weights and group labels (one group per model), crps_fair
  estimates the CRPS of the mixture sum_k W_k F_k: cross-model pairs are already unbiased, within-model pairs are
  rescaled by W_k^2 / (W_k^2 - sum w_i^2), the weighted form of m/(m-1). A one-member group is a point mass (a
  deterministic model) and needs no correction.
- RANKS NEED TIE-BREAKING. Precipitation (zeros), rounded temperatures and bounded variables tie the observation with
  members; counting ties as "below" piles the histogram into one end and reads as bias. Ranks, the rank histogram and
  the PIT here spread a tie uniformly over the tied positions (seedable).
- THE BRIER DECOMPOSITION IS EXACT ONLY WITH ITS TWO EXTRA TERMS. Murphy's BS = REL - RES + UNC holds when the
  forecasts take one value per bin; binning continuous probabilities adds the within-bin variance and covariance
  (Stephenson et al. 2008), returned as 'wbv' and 'wbc' so that bs == rel - res + unc + wbv - wbc to rounding.
- DIEBOLD-MARIANO NEEDS A HAC VARIANCE. Daily loss differentials are autocorrelated (a persistent regime favours one
  model for a week); the naive t-test then rejects equal skill far too often (35 % at a nominal 5 % in the test
  suite's AR(1) case). The variance here is Newey-West (Bartlett weights, autocovariances over n, so never negative)
  after AR(1) prewhitening (Andrews & Monahan 1992), with the Andrews (1991) plug-in bandwidth, at least h - 1 lags
  for h-step forecasts, and the Harvey-Leybourne-Newbold (1997) small-sample factor with Student-t(n-1) p-values.
  Without the prewhitening the Bartlett estimate is biased low for persistent differentials and the test still
  over-rejects (8-10 %). quantmet.stats.acf_inflation is the same Bartlett sum but with lag-wise 1/(n-k) means, which
  can go negative in short samples; hence a separate HAC here.
- PAIR BEFORE YOU COMPARE. Two dated series that stop on different days give "the last N days" two different meanings;
  a difference or ratio taken over the window then charges one leg for days the other has not published, always in
  the direction that makes the newer quantity look low. ``paired`` keeps only dates where BOTH legs are finite and
  returns how many days that is, to be shown with the result ("6/7 d"). Missing is NaN, never 0.

References: Gneiting & Raftery (2007, J. Amer. Stat. Assoc. 102, 359-378); Hersbach (2000, Wea. Forecasting 15,
559-570); Ferro (2014, Q. J. R. Meteorol. Soc. 140, 1917-1923); Gneiting, Raftery, Westveld & Goldman (2005, Mon.
Wea. Rev. 133, 1098-1118); Hamill (2001, Mon. Wea. Rev. 129, 550-560); Murphy (1973, J. Appl. Meteor. 12, 595-600);
Stephenson, Coelho & Jolliffe (2008, Wea. Forecasting 23, 752-757); Diebold & Mariano (1995, J. Bus. Econ. Stat. 13,
253-263); Harvey, Leybourne & Newbold (1997, Int. J. Forecasting 13, 281-291); Newey & West (1987, Econometrica 55,
703-708); Andrews (1991, Econometrica 59, 817-858); Andrews & Monahan (1992, Econometrica 60, 953-966).

Requires: numpy, scipy (normal and t distributions).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "crps_ensemble",
    "crps_fair",
    "crps_gaussian",
    "ranks",
    "rank_histogram",
    "pit",
    "brier_score",
    "brier_decomposition",
    "reliability_curve",
    "diebold_mariano",
    "DieboldMariano",
    "paired",
    "Paired",
]


# --------------------------------------------------------------------------
# CRPS
# --------------------------------------------------------------------------
def _members_last(ens, obs, axis, weights):
    x = np.moveaxis(np.asarray(ens, float), axis, -1)
    y = np.asarray(obs, float)
    x, y = np.broadcast_arrays(x, y[..., None])
    y = y[..., 0]
    m = x.shape[-1]
    w = np.ones(m) if weights is None else np.asarray(weights, float)
    if w.shape != (m,) or (w < 0).any():
        raise ValueError("weights: one non-negative weight per member")
    w = np.broadcast_to(w, x.shape).copy()
    bad = ~np.isfinite(x)
    w[bad] = 0.0
    x = np.where(bad, 0.0, x)
    return x, y, w


def _pair_sum(x, w):
    """sum_ij w_i w_j |x_i - x_j| along the last axis, O(m log m) via the sorted cumulative-weight identity
    sum_ij w_i w_j |x_i - x_j| = 2 sum_i w_(i) x_(i) (2 C_i - w_(i) - W), C_i the cumulative weight, W the total."""
    o = np.argsort(x, axis=-1)
    xs = np.take_along_axis(x, o, -1)
    ws = np.take_along_axis(w, o, -1)
    c = np.cumsum(ws, -1)
    tot = c[..., -1:]
    return 2.0 * np.sum(ws * xs * (2.0 * c - ws - tot), -1)


def crps_ensemble(ens, obs, axis: int = -1, weights=None):
    """CRPS of the (weighted) empirical ensemble distribution, E|X - y| - 1/2 E|X - X'| (Gneiting & Raftery 2007),
    exact for the discrete distribution, in O(m log m).

    ens     : (..., m, ...) members along `axis`; obs broadcasts against the other axes
    weights : optional (m,) member weights (normalised per case); NaN members get zero weight, all-NaN -> NaN.
    A one-member ensemble scores its absolute error."""
    x, y, w = _members_last(ens, obs, axis, weights)
    W = w.sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        t1 = np.sum(w * np.abs(x - y[..., None]), -1) / W
        t2 = _pair_sum(x, w) / W**2
        out = t1 - 0.5 * t2
    return np.where((W > 0) & np.isfinite(y), out, np.nan)


def crps_fair(ens, obs, axis: int = -1, weights=None, groups=None):
    """Fair CRPS (Ferro 2014): E|X - y| - 1/2 E|X - X'| with the spread term estimated without bias,
        (1/m) sum_i |x_i - y| - 1/(2 m (m - 1)) sum_ij |x_i - x_j|,
    unbiased for the CRPS of the distribution the members are drawn from, whatever m (>= 2).

    weights : optional (m,) member weights; members of one group are then treated as exchangeable draws with
              unequal weights (a weighted U-statistic, factor W^2 / (W^2 - sum w^2)).
    groups  : optional (m,) labels, one per source distribution (e.g. per model of a multi-model blend): the target is
              the mixture sum_k W_k F_k, cross-group pairs enter as they are and within-group pairs are rescaled by
              W_k^2 / (W_k^2 - sum_{i in k} w_i^2). A one-member group is a point mass (no correction).
    NaN members get zero weight; a group left with one valid member is then treated as a point mass."""
    x, y, w = _members_last(ens, obs, axis, weights)
    W = w.sum(-1)
    lab = np.zeros(x.shape[-1], int) if groups is None else np.asarray(groups)
    if lab.shape != (x.shape[-1],):
        raise ValueError("groups: one label per member")
    with np.errstate(invalid="ignore", divide="ignore"):
        t1 = np.sum(w * np.abs(x - y[..., None]), -1) / W
        spread = _pair_sum(x, w)
        for g in np.unique(lab):
            sel = lab == g
            wg = w[..., sel]
            within = _pair_sum(x[..., sel], wg)
            Wg = wg.sum(-1)
            S2 = (wg**2).sum(-1)
            den = Wg**2 - S2
            corr = np.where(den > 1e-12 * np.maximum(Wg**2, 1e-300), Wg**2 / np.where(den > 0, den, 1.0) - 1.0, 0.0)
            spread = spread + within * corr
        out = t1 - 0.5 * spread / W**2
    return np.where((W > 0) & np.isfinite(y), out, np.nan)


def crps_gaussian(mu, sigma, obs):
    """Closed-form CRPS of N(mu, sigma^2) (Gneiting et al. 2005, the EMOS objective):
    sigma [z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi)],  z = (y - mu)/sigma;  |y - mu| at sigma = 0."""
    from scipy.stats import norm

    mu, sigma, y = np.broadcast_arrays(np.asarray(mu, float), np.asarray(sigma, float), np.asarray(obs, float))
    if (sigma < 0).any():
        raise ValueError("sigma must be non-negative")
    s = np.where(sigma > 0, sigma, 1.0)
    z = (y - mu) / s
    c = s * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    return np.where(sigma > 0, c, np.abs(y - mu))


# --------------------------------------------------------------------------
# Ranks and PIT
# --------------------------------------------------------------------------
def ranks(ens, obs, axis: int = -1, rng=None):
    """Rank of the observation among the m members, 0..m, ties spread uniformly over the tied positions
    (rank = #below + U{0..#tied}). Cases with any non-finite member or observation get -1."""
    rng = np.random.default_rng(rng)
    x = np.moveaxis(np.asarray(ens, float), axis, -1)
    y = np.asarray(obs, float)
    x, yb = np.broadcast_arrays(x, y[..., None])
    y = yb[..., 0]
    below = (x < y[..., None]).sum(-1)
    tied = (x == y[..., None]).sum(-1)
    r = below + np.floor(rng.random(y.shape) * (tied + 1)).astype(int)
    ok = np.isfinite(x).all(-1) & np.isfinite(y)
    return np.where(ok, r, -1)


def rank_histogram(ens, obs, axis: int = -1, rng=None) -> np.ndarray:
    """Counts of the observation's rank, (m + 1,), over all valid cases (see ranks). Flat for a calibrated ensemble
    (Hamill 2001); U-shaped = under-dispersive, dome = over-dispersive, sloped = biased."""
    r = ranks(ens, obs, axis, rng)
    m = np.asarray(ens).shape[axis]
    return np.bincount(r[r >= 0].ravel(), minlength=m + 1)


def pit(ens, obs, axis: int = -1, rng=None):
    """Randomised probability integral transform of an ensemble, u = (rank + V)/(m + 1), V ~ U(0, 1): exactly U(0, 1)
    for a calibrated ensemble, ties included. NaN for invalid cases. For a parametric forecast the PIT is simply its
    CDF at the observation (e.g. scipy.stats.norm.cdf(obs, mu, sigma))."""
    rng = np.random.default_rng(rng)
    r = ranks(ens, obs, axis, rng)
    m = np.asarray(ens).shape[axis]
    u = (r + rng.random(r.shape)) / (m + 1)
    return np.where(r >= 0, u, np.nan)


# --------------------------------------------------------------------------
# Brier score
# --------------------------------------------------------------------------
def _po(p, o):
    p = np.asarray(p, float).ravel()
    o = np.asarray(o, float).ravel()
    if p.shape != o.shape:
        raise ValueError("p and o must have the same size")
    ok = np.isfinite(p) & np.isfinite(o)
    p, o = p[ok], o[ok]
    if ((p < 0) | (p > 1)).any() or not np.isin(o, (0.0, 1.0)).all():
        raise ValueError("p in [0, 1], o in {0, 1}")
    return p, o


def _bins(p, bins):
    edges = np.linspace(0.0, 1.0, int(bins) + 1) if np.ndim(bins) == 0 else np.asarray(bins, float)
    k = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, edges.size - 2)
    return edges, k


def brier_score(p, o) -> float:
    """Mean of (p - o)^2 over the finite pairs; p probabilities, o 0/1 outcomes."""
    p, o = _po(p, o)
    return float(np.mean((p - o) ** 2)) if p.size else float("nan")


def brier_decomposition(p, o, bins=10) -> dict:
    """Murphy (1973) decomposition with the within-bin terms of Stephenson et al. (2008):
        bs = reliability - resolution + uncertainty + wbv - wbc        (exact)
    reliability = sum n_k (pbar_k - obar_k)^2 / N, resolution = sum n_k (obar_k - obar)^2 / N,
    uncertainty = obar (1 - obar), wbv = within-bin forecast variance, wbc = 2 x within-bin covariance of p and o.
    bins: number of equal-width bins on [0, 1] or explicit edges (last bin closed). Also returns n."""
    p, o = _po(p, o)
    n = p.size
    edges, k = _bins(p, bins)
    nb = edges.size - 1
    cnt = np.bincount(k, minlength=nb).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        pk = np.bincount(k, p, nb) / cnt
        ok_ = np.bincount(k, o, nb) / cnt
    has = cnt > 0
    obar = o.mean()
    rel = np.sum(cnt[has] * (pk[has] - ok_[has]) ** 2) / n
    res = np.sum(cnt[has] * (ok_[has] - obar) ** 2) / n
    unc = obar * (1 - obar)
    wbv = np.sum((p - pk[k]) ** 2) / n
    wbc = 2.0 * np.sum((p - pk[k]) * (o - ok_[k])) / n
    return {
        "bs": float(np.mean((p - o) ** 2)),
        "reliability": float(rel),
        "resolution": float(res),
        "uncertainty": float(unc),
        "wbv": float(wbv),
        "wbc": float(wbc),
        "n": int(n),
    }


def reliability_curve(p, o, bins=10) -> dict:
    """Reliability diagram data: per bin the mean forecast probability ('forecast'), the observed frequency
    ('observed') and the number of forecasts ('count'); empty bins are NaN. 'edges' are the bin edges."""
    p, o = _po(p, o)
    edges, k = _bins(p, bins)
    nb = edges.size - 1
    cnt = np.bincount(k, minlength=nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.bincount(k, p, nb) / cnt
        ob = np.bincount(k, o, nb) / cnt
    return {"edges": edges, "forecast": f, "observed": ob, "count": cnt}


# --------------------------------------------------------------------------
# Diebold-Mariano
# --------------------------------------------------------------------------
@dataclass
class DieboldMariano:
    """statistic (HLN-corrected unless hln=False), pvalue, mean_diff (mean of loss_a - loss_b: negative = a better),
    lags (HAC truncation), n, and lrv, the long-run variance of the differential."""

    statistic: float
    pvalue: float
    mean_diff: float
    lags: int
    n: int
    lrv: float


def _hac(u, lags):
    """Newey-West long-run variance of a centred series: gamma_0 + 2 sum_j (1 - j/(L+1)) gamma_j, gamma_j over n."""
    n = u.size
    v = u @ u / n
    for j in range(1, lags + 1):
        v += 2.0 * (1.0 - j / (lags + 1.0)) * (u[j:] @ u[:-j]) / n
    return v


def _andrews_lags(u):
    """Andrews (1991) AR(1) plug-in bandwidth for the Bartlett kernel, floor(1.1447 (alpha n)^(1/3))."""
    den = u @ u
    rho = float(np.clip((u[1:] @ u[:-1]) / den, -0.97, 0.97)) if den > 0 else 0.0
    alpha = 4 * rho**2 / ((1 - rho) ** 2 * (1 + rho) ** 2)
    return int(np.floor(1.1447 * (alpha * u.size) ** (1.0 / 3.0)))


def diebold_mariano(
    loss_a, loss_b, h: int = 1, lags=None, prewhiten: bool = True, hln: bool = True, alternative: str = "two-sided"
) -> DieboldMariano:
    """Diebold-Mariano (1995) test of equal expected loss for two forecasts of the same targets.

    loss_a, loss_b : (n,) losses on the SAME cases in time order (align them with `paired` first); non-finite values
                     raise rather than being dropped, because dropping them would shift the lags.
    h              : forecast horizon in time steps; the HAC uses at least h - 1 lags.
    lags           : Bartlett (Newey-West) truncation; None = the Andrews (1991) AR(1) plug-in,
                     floor(1.1447 (alpha n)^(1/3)), alpha = 4 rho^2 / ((1 - rho)^2 (1 + rho)^2).
    prewhiten      : AR(1) prewhitening (Andrews & Monahan 1992): fit d_t = phi d_t-1 + u_t (phi clipped to
                     +-0.97), take the HAC variance of u and recolour it by 1/(1 - phi)^2. Without it the Bartlett
                     estimate is biased low for persistent differentials and the test over-rejects (about 8-10 % at a
                     nominal 5 % for a lag-1 autocorrelation of 0.64 and n = 200-500; ~5-6 % with it).
    hln            : Harvey-Leybourne-Newbold (1997) factor sqrt((n + 1 - 2h + h(h - 1)/n) / n) and Student-t(n - 1)
                     p-values; False gives the original statistic with normal p-values.
    alternative    : "two-sided", "less" (a has lower loss), "greater"."""
    from scipy import stats

    a = np.asarray(loss_a, float).ravel()
    b = np.asarray(loss_b, float).ravel()
    if a.shape != b.shape:
        raise ValueError("loss_a and loss_b must be paired (same length)")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("non-finite losses: pair the series first (verify.paired)")
    d = a - b
    n = d.size
    if n < 3:
        raise ValueError("need at least 3 paired cases")
    u = d - d.mean()
    phi = 0.0
    if prewhiten:
        den = u[:-1] @ u[:-1]
        phi = float(np.clip((u[1:] @ u[:-1]) / den, -0.97, 0.97)) if den > 0 else 0.0
        e = u[1:] - phi * u[:-1]
        e = e - e.mean()
    else:
        e = u
    if lags is None:
        lags = _andrews_lags(e)
    lags = int(min(max(lags, h - 1, 0), e.size - 1))
    lrv = _hac(e, lags) / (1.0 - phi) ** 2
    dbar = float(d.mean())
    stat = dbar / np.sqrt(lrv / n) if lrv > 0 else (0.0 if dbar == 0 else np.sign(dbar) * np.inf)
    if hln:
        stat *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
        dist = stats.t(n - 1)
    else:
        dist = stats.norm
    if alternative == "two-sided":
        p = 2 * dist.sf(abs(stat))
    elif alternative == "less":
        p = dist.cdf(stat)
    elif alternative == "greater":
        p = dist.sf(stat)
    else:
        raise ValueError("alternative is 'two-sided', 'less' or 'greater'")
    return DieboldMariano(float(stat), float(p), dbar, lags, n, float(lrv))


# --------------------------------------------------------------------------
# Pairing dated series
# --------------------------------------------------------------------------
@dataclass
class Paired:
    """dates (datetime64) where both legs are finite, the two aligned value arrays, n = len(dates), and window = the
    number of dates the comparison was meant to cover (calendar steps from start to end when both are given, else the
    dates where either leg is finite). Show n/window with any statistic computed from a and b."""

    dates: np.ndarray
    a: np.ndarray
    b: np.ndarray
    n: int
    window: int


def _series(s, unit):
    if isinstance(s, dict):
        d, v = list(s.keys()), list(s.values())
    elif hasattr(s, "index") and hasattr(s, "values") and not isinstance(s, (tuple, list)):
        d, v = list(s.index), list(np.asarray(s.values))
    else:
        d, v = s
    d = np.array(
        [np.datetime64(x, unit) if not isinstance(x, np.datetime64) else x.astype(f"datetime64[{unit}]") for x in d],
        dtype=f"datetime64[{unit}]",
    )
    v = np.array([np.nan if x is None else x for x in v], dtype=float)
    if d.shape != v.shape:
        raise ValueError("dates and values differ in length")
    if np.unique(d).size != d.size:
        raise ValueError(f"duplicate dates at {unit} resolution")
    ok = np.isfinite(v)
    return d[ok], v[ok]


def paired(a, b, start=None, end=None, unit: str = "D") -> Paired:
    """Align two dated series on the dates where BOTH have a finite value.

    a, b  : {date: value} mappings, (dates, values) pairs, or objects with .index and .values (a pandas Series);
            dates anything numpy.datetime64 accepts (str, datetime, datetime64); None/NaN = missing
    start, end : optional inclusive window
    unit  : date resolution ("D" days by default; "h" for hourly series)
    Never difference or ratio two series over a window without this: each leg stops on its own day."""
    da, va = _series(a, unit)
    db, vb = _series(b, unit)
    lo = None if start is None else np.datetime64(start, unit)
    hi = None if end is None else np.datetime64(end, unit)

    def inwin(d):
        m = np.ones(d.shape, bool)
        if lo is not None:
            m &= d >= lo
        if hi is not None:
            m &= d <= hi
        return m

    ma, mb = inwin(da), inwin(db)
    da, va, db, vb = da[ma], va[ma], db[mb], vb[mb]
    common, ia, ib = np.intersect1d(da, db, return_indices=True)
    if lo is not None and hi is not None:
        window = int((hi - lo) / np.timedelta64(1, unit)) + 1
    else:
        window = int(np.union1d(da, db).size)
    return Paired(common, va[ia], vb[ib], int(common.size), window)
