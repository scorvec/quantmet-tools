"""Degree days: the midpoint and integral conventions, degree hours by time of day, degree days from daily extremes and
a diurnal shape, expected degree days of a Gaussian temperature, weighted aggregation in the right order, local-day
assembly from UTC data, a 366-day calendar and a two-hinge balance-point fit.

Every degree-day function returns (hdd, cdd) in the units of its temperatures. The default base is 65 (degF, the
North American convention); pass base=18 (or 15.5, 18.3...) for degC.

Numerical notes that matter in practice:
- MIDPOINT AND INTEGRAL ARE DIFFERENT QUANTITIES. The midpoint degree day max(b - (hi + lo)/2, 0) is zero whenever
  the mean of the extremes clears the base, yet a cool dawn still sits below it for hours; the integral (the time
  mean of max(b - T(t), 0), i.e. degree hours / 24) sees that. They agree only when the whole day is on one side of
  the base. A day that swings 58-78 around a 65 base (mean 68, amplitude 10) has a midpoint HDD of 0 and an integral
  HDD of 1.83 -- shoulder seasons are where the conventions part. Totals over a winter barely move.
  Neither is "right": the midpoint is the traded and published convention, the integral is a second, complementary
  measure. Degree hours split into time-of-day blocks keep where in the day the heating or cooling falls.
- KINK FIRST, WEIGHT AFTER. A degree day is a convex (hinged) function of temperature, so by Jensen's inequality
  max(b - sum w T, 0) <= sum w max(b - T, 0): weighting temperatures across locations (or ensemble members) first
  and taking the degree day of the weighted mean UNDERSTATES both HDD and CDD, without exception, and the error is
  largest exactly when it matters -- a warm aggregate annihilates the heating of every cold location in it (a
  national mean above the base reads zero HDD while half the population heats). ``weighted_dd`` applies the kink per
  location (per member) and weights afterwards. The same holds for any time aggregation.
- EXPECTED DEGREE DAYS ARE NOT THE DEGREE DAYS OF THE EXPECTED TEMPERATURE. For T ~ N(mu, sigma^2),
  E[max(b - T, 0)] = (b - mu) Phi(z) + sigma phi(z), z = (b - mu)/sigma (Thom 1954), which exceeds max(b - mu, 0) by
  up to sigma/sqrt(2 pi) (0.4 sigma) at mu = b. Use it to turn an ensemble or seasonal mean and spread into degree
  days; HDD - CDD = b - mu exactly, as for the midpoint.
- LOCAL DAYS, NOT UTC DAYS. A calendar day is local, and a local day has 23 or 25 hours across a DST change.
  ``local_days`` assembles them from UTC timestamps with zoneinfo and, by default, drops days with fewer samples than
  that day really has -- a partial first or last day otherwise shows up as a spuriously mild (or hot) day.
- A NORMALISED SHAPE IS NOT A TYPICAL DAY. dd_from_extremes builds T(h) = lo + (hi - lo) w(h). If w is the mean of
  normalised daily curves it is flatter than any real day (days peak at different hours), which biases the integral
  degree days; take w from a model of the typical day, and treat the reconstruction as an approximation (a day's
  curve cannot be recovered from its two extremes).
- 366 SLOTS. A day-of-year climatology indexed 0..365 with 29 Feb in its own slot (59) keeps 1 March at 60 in every
  year; indexing by tm_yday shifts every day after February by one in leap years.
- BALANCE POINTS ARE FITTED, NOT ASSUMED. Energy demand against temperature is a "smile": a heating hinge, a
  dead band, a cooling hinge. fit_balance_points finds both hinges by least squares over a grid (the five-parameter
  change-point model of Kissock, Haberl & Claridge 2003), optionally with extra regressors (day of week, trend).

References: Thom (1954, Mon. Wea. Rev. 82, 1-6) on degree days and the normal distribution; Kissock, Haberl &
Claridge (2003, ASHRAE Research Project 1050: Inverse Modeling Toolkit) on change-point models; PEP 615 (zoneinfo).

Requires: numpy, scipy (normal distribution); zoneinfo (standard library, Python >= 3.9) for local days.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass

import numpy as np

__all__ = [
    "midpoint_dd",
    "integral_dd",
    "degree_hours",
    "dd_from_extremes",
    "expected_dd",
    "weighted_dd",
    "local_days",
    "LocalDays",
    "day_length_hours",
    "doy366",
    "fit_balance_points",
    "BalancePoints",
]

BASE = 65.0


def _as(x):
    return np.asarray(x, dtype=float)


def midpoint_dd(tmax, tmin, base: float = BASE):
    """Midpoint degree days from daily extremes: m = (tmax + tmin)/2, (max(base - m, 0), max(m - base, 0))."""
    m = 0.5 * (_as(tmax) + _as(tmin))
    return np.maximum(base - m, 0.0), np.maximum(m - base, 0.0)


def integral_dd(temps, base: float = BASE, axis: int = -1, method: str = "mean"):
    """Integral degree days from sub-daily temperatures along `axis` (one day per slice).

    method "mean"   : the mean of max(base - T, 0) over the samples, each standing for an equal share of the day
                      (hourly data: degree hours / 24);
    method "linear" : the exact time mean of the hinge of the piecewise-linear curve through the samples (trapezoid
                      with the base crossings solved inside each interval) -- better for 3- or 6-hourly samples.
                      Spans first to last sample; pass the next day's first sample appended to close the day.
    NaN in a slice -> NaN (strict)."""
    t = np.moveaxis(_as(temps), axis, -1)
    if method == "mean":
        return np.maximum(base - t, 0.0).mean(-1), np.maximum(t - base, 0.0).mean(-1)
    if method != "linear":
        raise ValueError("method is 'mean' or 'linear'")
    if t.shape[-1] < 2:
        raise ValueError("'linear' needs at least two samples")
    a, b = t[..., :-1], t[..., 1:]

    def seg(u, v):  # mean over [0, 1] of max(u + (v - u) s, 0) for a linear segment u -> v
        both = (u >= 0) & (v >= 0)
        none = (u <= 0) & (v <= 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            cross = np.where(u > 0, u * u, v * v) / (2.0 * np.abs(v - u))
        return np.where(both, 0.5 * (u + v), np.where(none, 0.0, cross))

    h = seg(base - a, base - b).mean(-1)
    c = seg(a - base, b - base).mean(-1)
    return h, c


def degree_hours(
    temps, hours, base: float = BASE, blocks=((0, 6), (6, 12), (12, 18), (18, 24)), step: float = 1.0, axis: int = -1
):
    """Heating and cooling degree hours by time-of-day block.

    temps : (..., n) samples along `axis`; hours : (n,) local hour of each sample (0 <= h < 24)
    blocks: [start, end) local-hour blocks; step: hours each sample represents (1 for hourly data).
    Returns (hdh, cdh), each (..., len(blocks)). Over blocks that tile the day, sum(hdh)/24 = integral_dd."""
    t = np.moveaxis(_as(temps), axis, -1)
    hr = _as(hours)
    if hr.shape != (t.shape[-1],):
        raise ValueError("one local hour per sample")
    hh = np.maximum(base - t, 0.0) * step
    cc = np.maximum(t - base, 0.0) * step
    H, C = [], []
    for lo, hi in blocks:
        m = (hr >= lo) & (hr < hi)
        H.append(hh[..., m].sum(-1))
        C.append(cc[..., m].sum(-1))
    return np.stack(H, -1), np.stack(C, -1)


def dd_from_extremes(tmax, tmin, shape, base: float = BASE):
    """Degree days from a day's extremes and a diurnal shape: T(h) = tmin + (tmax - tmin) w(h), with w = the shape
    rescaled to [0, 1] along its last axis (so any curve -- hourly climatology, a model's typical day -- can be passed).
    tmax/tmin broadcast against shape's leading axes. Returns the integral (hdd, cdd) of the reconstructed curve."""
    w = _as(shape)
    lo = w.min(-1, keepdims=True)
    rng = w.max(-1, keepdims=True) - lo
    if (rng <= 0).any():
        raise ValueError("the shape must vary over the day")
    w = (w - lo) / rng
    tmax = _as(tmax)[..., None]
    tmin = _as(tmin)[..., None]
    return integral_dd(tmin + (tmax - tmin) * w, base)


def expected_dd(mu, sigma, base: float = BASE):
    """Expected degree days of a Gaussian temperature N(mu, sigma^2) (Thom 1954):
        E[max(b - T, 0)] = (b - mu) Phi(z) + sigma phi(z),   E[max(T - b, 0)] = (mu - b) Phi(-z) + sigma phi(z),
    z = (b - mu)/sigma; sigma = 0 gives the degree days of mu. Returns (hdd, cdd)."""
    from scipy.stats import norm

    mu, sigma = np.broadcast_arrays(_as(mu), _as(sigma))
    if (sigma < 0).any():
        raise ValueError("sigma must be non-negative")
    s = np.where(sigma > 0, sigma, 1.0)
    z = (base - mu) / s
    h = (base - mu) * norm.cdf(z) + s * norm.pdf(z)
    c = (mu - base) * norm.cdf(-z) + s * norm.pdf(z)
    return (np.where(sigma > 0, h, np.maximum(base - mu, 0.0)), np.where(sigma > 0, c, np.maximum(mu - base, 0.0)))


def weighted_dd(temps, weights, base: float = BASE, axis: int = -1, min_coverage: float = 1.0):
    """Weighted (e.g. population-weighted) degree days with the kink applied FIRST, per location, then weighted:
        hdd = sum_i w_i max(base - T_i, 0) / sum_i w_i     (and cdd likewise)
    temps : (..., nloc) daily temperatures (or midpoints) along `axis`; weights : (nloc,), any non-negative scale.
    NaN locations drop out with their weight renormalised only while the weight present is >= min_coverage of the
    total (default 1.0: any missing location makes the aggregate NaN). Weighting temperatures first and kinking the
    mean is biased low by Jensen's inequality (module docstring)."""
    t = np.moveaxis(_as(temps), axis, -1)
    w = _as(weights)
    if w.shape != (t.shape[-1],) or (w < 0).any():
        raise ValueError("weights: one non-negative weight per location")
    ok = np.isfinite(t)
    wv = np.where(ok, w, 0.0)
    have = wv.sum(-1)
    tot = w.sum()
    out = []
    for f in (np.maximum(base - t, 0.0), np.maximum(t - base, 0.0)):
        with np.errstate(invalid="ignore", divide="ignore"):
            v = (np.where(ok, f, 0.0) * wv).sum(-1) / have
        out.append(np.where(have >= min_coverage * tot - 1e-12 * tot, v, np.nan))
    return tuple(out)


# --------------------------------------------------------------------------
# Local days
# --------------------------------------------------------------------------
def _zone(tz):
    from zoneinfo import ZoneInfo

    return ZoneInfo(tz) if isinstance(tz, str) else tz


def _utc_seconds(times):
    t = np.asarray(times)
    if np.issubdtype(t.dtype, np.datetime64):
        return t.astype("datetime64[s]").astype(np.int64)
    out = []
    for x in t.ravel():
        if isinstance(x, _dt.datetime):
            naive = x.astimezone(_dt.timezone.utc).replace(tzinfo=None) if x.tzinfo is not None else x
            out.append(int((np.datetime64(naive, "s") - np.datetime64(0, "s")) / np.timedelta64(1, "s")))
        else:
            out.append(int(np.datetime64(x, "s").astype(np.int64)))
    return np.array(out, np.int64)


def day_length_hours(dates, tz) -> np.ndarray:
    """Length in hours of each local calendar day in time zone tz (IANA name or tzinfo): 24, or 23 / 25 across a DST
    change (other values for historical offset changes)."""
    z = _zone(tz)
    out = []
    for d in np.atleast_1d(np.asarray(dates, dtype="datetime64[D]")):
        day = d.astype(_dt.date)
        a = _dt.datetime(day.year, day.month, day.day, tzinfo=z)
        nx = day + _dt.timedelta(days=1)
        b = _dt.datetime(nx.year, nx.month, nx.day, tzinfo=z)
        out.append((b.astimezone(_dt.timezone.utc) - a.astimezone(_dt.timezone.utc)).total_seconds() / 3600.0)
    return np.array(out)


@dataclass
class LocalDays:
    """Local calendar days assembled from UTC samples: dates (datetime64[D]), index (one array of sample indices per
    day), n (samples present), expected (the fewest samples a gap-free day of that length holds at the sampling step,
    floor(hours / step)), hours (the day's length, 23/24/25) and local_hour (per sample, the local wall-clock hour,
    fractional; it repeats or skips an hour on DST days)."""

    dates: np.ndarray
    index: list
    n: np.ndarray
    expected: np.ndarray
    hours: np.ndarray
    local_hour: np.ndarray

    def reduce(self, values, func=np.mean, axis: int = -1):
        """Apply func (e.g. np.max, np.min, np.mean) per local day along `axis` (the sample axis): (..., nday)."""
        v = np.moveaxis(_as(values), axis, -1)
        if not self.index:
            return np.empty((*v.shape[:-1], 0))
        return np.stack([func(v[..., i], axis=-1) for i in self.index], -1)


def local_days(times_utc, tz, step_hours=None, complete: bool = True) -> LocalDays:
    """Group UTC timestamps into local calendar days of time zone tz (IANA name, e.g. "America/Chicago"), DST-safe.

    times_utc  : datetime64 (taken as UTC), naive datetimes (UTC) or aware datetimes
    step_hours : sampling interval; None = the median spacing
    complete   : keep only days with every sample the day really has (n >= floor(hours / step): 23, 24 or 25 for hourly
                 data), so a partial first or last day, or a gap, never yields a spuriously mild or hot day; False
                 keeps every day with any sample.
    Use LocalDays.reduce to take the daily max, min or mean of a series sampled at these times."""
    z = _zone(tz)
    sec = _utc_seconds(times_utc)
    if step_hours is None:
        step_hours = float(np.median(np.diff(np.sort(sec)))) / 3600.0 if sec.size > 1 else 1.0
    loc_date = np.empty(sec.size, "datetime64[D]")
    loc_hour = np.empty(sec.size)
    cache = {}
    for i, s in enumerate(sec):
        hour = int(s // 3600)
        if hour not in cache:
            u = _dt.datetime.fromtimestamp(hour * 3600, _dt.timezone.utc).astimezone(z)
            cache[hour] = (np.datetime64(u.date(), "D"), u.hour + u.minute / 60.0)
        d, h0 = cache[hour]
        loc_date[i] = d
        loc_hour[i] = h0 + (s - hour * 3600) / 3600.0
    days, inv = np.unique(loc_date, return_inverse=True)
    hours = day_length_hours(days, z) if days.size else np.zeros(0)
    expected = np.floor(hours / step_hours + 1e-9).astype(int)
    idx = [np.flatnonzero(inv == k) for k in range(days.size)]
    n = np.array([i.size for i in idx], int)
    keep = n >= expected if complete else n > 0
    return LocalDays(days[keep], [i for i, k in zip(idx, keep) if k], n[keep], expected[keep], hours[keep], loc_hour)


def doy366(dates) -> np.ndarray:
    """Day-of-year index 0..365 on a 366-slot calendar: 29 Feb is slot 59 and 1 March is 60 in EVERY year, so a
    climatology lines up across leap and common years (31 Dec is 365 always). Slot 59 holds only the leap years: fill
    it from them or from its neighbours, and smooth cyclically over the 366 slots (harmonics in 2 pi slot / 366)."""
    d = np.asarray(dates, dtype="datetime64[D]")
    y = d.astype("datetime64[Y]")
    doy = (d - y.astype("datetime64[D]")).astype(int)  # 0-based
    yr = y.astype(int) + 1970
    leap = (yr % 4 == 0) & ((yr % 100 != 0) | (yr % 400 == 0))
    return np.where(~leap & (doy >= 59), doy + 1, doy)


# --------------------------------------------------------------------------
# Balance points
# --------------------------------------------------------------------------
@dataclass
class BalancePoints:
    """heat_base, cool_base (the hinges), intercept (load in the dead band, at zero extra regressors), heat_slope
    (load per degree below heat_base, >0 when heating adds load), cool_slope (per degree above cool_base), middle_slope
    (0 unless linear=True), extra (coefficients of X), sse, r2, n."""

    heat_base: float
    cool_base: float
    intercept: float
    heat_slope: float
    cool_slope: float
    middle_slope: float
    extra: np.ndarray
    sse: float
    r2: float
    n: int

    def predict(self, temp, X=None):
        t = _as(temp)
        y = (
            self.intercept
            + self.heat_slope * np.maximum(self.heat_base - t, 0.0)
            + self.cool_slope * np.maximum(t - self.cool_base, 0.0)
            + self.middle_slope * t
        )
        if X is not None and self.extra.size:
            y = y + np.asarray(X, float) @ self.extra
        return y


def fit_balance_points(
    temp, load, X=None, grid=None, step: float = 1.0, linear: bool = False, min_side: int = 10
) -> BalancePoints:
    """Two-hinge (five-parameter change-point) fit of load against temperature,
        load = b0 + a_h max(Th - T, 0) + a_c max(T - Tc, 0) [+ b1 T] [+ X beta],
    with (Th <= Tc) chosen by least squares over a grid of hinge pairs, every pair solved exactly by OLS.

    temp, load : (n,), non-finite pairs dropped; X : optional (n, k) extra regressors (day-of-week dummies, a trend)
    grid       : candidate hinge temperatures; None = from the 5th to the 95th percentile of temp in `step` units
    linear     : add a slope in the dead band (b1 T)
    min_side   : at least this many points must lie beyond each hinge."""
    t = _as(temp).ravel()
    y = _as(load).ravel()
    Xe = np.zeros((t.size, 0)) if X is None else np.asarray(X, float).reshape(t.size, -1)
    ok = np.isfinite(t) & np.isfinite(y) & np.isfinite(Xe).all(1)
    t, y, Xe = t[ok], y[ok], Xe[ok]
    n = t.size
    if grid is None:
        lo, hi = np.percentile(t, [5, 95])
        grid = np.arange(np.floor(lo / step) * step, np.ceil(hi / step) * step + step / 2, step)
    grid = np.unique(_as(grid))
    B = np.column_stack([np.ones(n)] + ([t] if linear else []) + [Xe])
    kb = B.shape[1]
    Hm = np.maximum(grid[None, :] - t[:, None], 0.0)  # heating hinge column for every candidate
    Cm = np.maximum(t[:, None] - grid[None, :], 0.0)  # cooling hinge column
    i, j = np.nonzero(np.triu(np.ones((grid.size, grid.size), bool), 1 if linear else 0))
    keep = ((Hm > 0).sum(0)[i] >= min_side) & ((Cm > 0).sum(0)[j] >= min_side)
    i, j = i[keep], j[keep]
    if not i.size:
        raise ValueError("no hinge pair leaves min_side points beyond both hinges")
    # every pair solved through its normal equations (batched); the winner is refitted by lstsq below
    BtH, BtC = B.T @ Hm, B.T @ Cm
    M = np.empty((i.size, kb + 2, kb + 2))
    M[:, :kb, :kb] = B.T @ B
    M[:, :kb, kb] = M[:, kb, :kb] = BtH[:, i].T
    M[:, :kb, kb + 1] = M[:, kb + 1, :kb] = BtC[:, j].T
    M[:, kb, kb] = (Hm * Hm).sum(0)[i]
    M[:, kb + 1, kb + 1] = (Cm * Cm).sum(0)[j]
    M[:, kb, kb + 1] = M[:, kb + 1, kb] = (Hm.T @ Cm)[i, j]
    yc = y - y.mean()  # the intercept absorbs it; better conditioned
    v = np.concatenate([np.broadcast_to(B.T @ yc, (i.size, kb)), (Hm.T @ yc)[i, None], (Cm.T @ yc)[j, None]], 1)
    try:
        beta = np.linalg.solve(M, v[..., None])[..., 0]
    except np.linalg.LinAlgError:  # collinear extra regressors
        beta = (np.linalg.pinv(M) @ v[..., None])[..., 0]
    k = int(np.argmin(yc @ yc - np.sum(beta * v, 1)))
    th, tc = grid[i[k]], grid[j[k]]
    A = np.column_stack([B[:, : 1 + linear], Hm[:, i[k]], Cm[:, j[k]], Xe])
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    sse = float(np.sum((y - A @ coef) ** 2))
    sst = float(yc @ yc)
    j = 2 if linear else 1
    return BalancePoints(
        float(th),
        float(tc),
        float(coef[0]),
        float(coef[j]),
        float(coef[j + 1]),
        float(coef[1]) if linear else 0.0,
        coef[j + 2 :].copy(),
        sse,
        float(1 - sse / sst) if sst > 0 else float("nan"),
        int(n),
    )
