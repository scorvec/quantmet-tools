"""Takaya-Nakamura (2001) wave-activity flux: the phase-independent flux of
quasi-stationary Rossby wave activity on a pressure level, per ensemble
member, with its divergence and the members' agreement on its sign.

Horizontal components on the sphere (TN01 eq. 38), lambda = longitude,
phi = latitude, psi' the streamfunction anomaly, (U, V) the basic state,
p = pressure / 1000 hPa:

    W = p cos(phi) / (2 |U|) * (
        U/(a^2 cos^2 phi) (psi_l^2 - psi psi_ll) + V/(a^2 cos phi) (psi_l psi_p - psi psi_lp),
        U/(a^2 cos phi)   (psi_l psi_p - psi psi_lp) + V/a^2       (psi_p^2 - psi psi_pp) )

(subscripts l, p = partial derivatives in lambda, phi). W points along the
group velocity of the packets -- where wave energy is HEADING, ducted along
the jet waveguides -- and its convergence marks where the downstream flow
will amplify: downstream development, blocking onset, the arcs that carry
tropical forcing into the PNA.

Numerical notes that matter in practice:
- PER-MEMBER, THEN AVERAGE. W is quadratic in psi', so the flux of the
  ENSEMBLE-MEAN psi' fades with lead as the members' phases decorrelate --
  which reads as "waves dying" when they are merely uncertain. The mean of
  the members' fluxes does not fade (ensemble_tn01_flux). The same lesson
  holds for every quadratic eddy diagnostic (EP flux, [v'T'], eddy kinetic
  energy): form the quadratic per member, then average.
- MIXED DERIVATIVE WITHOUT A METRIC TERM. The cross term uses the plain
  angle derivative psi_{lambda phi}, as TN01 write it. Differentiating the
  metric-scaled psi_x = psi_lambda / (a cos phi) in y instead adds
  psi_lambda tan(phi) / a^2 -- a term that oscillates at twice the wave's
  wavenumber and breaks the phase independence the flux is built for (a
  few % to ~10 % of the cross term for synoptic waves at 45 deg).
- PHASE INDEPENDENCE SURVIVES THE DISCRETIZATION when second derivatives
  are the centred difference of the centred first derivative: for
  psi' = A cos(m lambda + n phi) every bracket is then exactly constant,
  with m, n replaced by their discrete values sin(m dl)/dl, sin(n dp)/dp.
- PERIODIC LONGITUDE. lambda-derivatives wrap around the globe; a
  one-sided difference at the seam (np.gradient's default) would put a
  first-order error and a spurious divergence along the date line of the
  grid. A grid that is not a full uniform circle is rejected unless
  periodic=False.
- QUASI-STATIONARY THEORY. TN01 assume slowly varying, quasi-stationary
  waves: low-pass psi' in time before the flux (lead_smooth; the site uses
  a 5-day running mean along the lead) and linearise about a basic state
  consistent with psi' -- the site uses the day-of-year climatology plus the
  trailing 30-day mean analysis anomaly (lowpass_anomaly), so packets are
  steered by the waveguide that is actually there in a year with a
  displaced jet. Masked where |U| < umin or |lat| < lat_min (the theory
  needs a westerly waveguide).

Streamfunction: quantmet.helmholtz.streamfunction (spherical-harmonic
Poisson inversion on a pole-free Gauss-Legendre grid), wrapped by
streamfunction_anomaly; the flux itself needs numpy only.

Powers the 250 hPa wave-activity flux loop on
https://scorvec.com/circulation.html (AIFS-ENS, 25 members, day 0-15).

Requires: numpy (flux); pyshtools + xarray for streamfunction_anomaly and
the spectral divergence filter (pip install quantmet[waf]).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

A_EARTH = 6.371e6
UMIN = 3.0               # m/s: basic-state wind below which the flux is masked
LAT_MIN = 20.0           # deg: tropical mask
TSMOOTH = 5              # samples (days): running mean of psi' along the lead
WINDOW_DAYS = 30         # trailing window of the low-passed basic state
MIN_COUNT = 20           # analyses required inside that window

__all__ = ["WAF", "tn01_flux", "tn01_flux_cartesian", "ensemble_tn01_flux",
           "EnsembleWAF", "lead_smooth", "lowpass_anomaly",
           "streamfunction_anomaly", "sh_truncate",
           "UMIN", "LAT_MIN", "TSMOOTH", "WINDOW_DAYS", "MIN_COUNT"]


@dataclass
class WAF:
    """wx, wy (m^2/s^2), div (m/s^2), mask (True = masked) on the input grid."""
    wx: np.ndarray
    wy: np.ndarray
    div: np.ndarray
    mask: np.ndarray


def _dlon(f, lonr, periodic):
    if not periodic:
        return np.gradient(f, lonr, axis=-1)
    return (np.roll(f, -1, axis=-1) - np.roll(f, 1, axis=-1)) / (2.0 * (lonr[1] - lonr[0]))


def _check_lon(lon, periodic):
    """Radians, and whether the last column repeats the first (lon[-1] =
    lon[0] + 360, as on the extended Driscoll-Healy grids of pyshtools)."""
    lon = np.asarray(lon, float)
    closed = lon.size > 2 and np.isclose(lon[-1] - lon[0], 360.0)
    if periodic:
        d = np.diff(lon[:-1] if closed else lon)
        n = lon.size - closed
        if not (np.allclose(d, d[0]) and np.isclose(n * abs(d[0]), 360.0)):
            raise ValueError("periodic=True needs a uniform longitude grid covering 360 deg; "
                             "pass periodic=False for a regional grid")
    return np.deg2rad(lon), bool(closed and periodic)


def tn01_flux(psi_a, U, V, lat, lon, p_hpa: float = 250.0, umin: float = UMIN,
              lat_min: float = LAT_MIN, periodic: bool = True,
              div_lmax: int | None = None) -> WAF:
    """Horizontal TN01 flux from psi' (m^2/s) and the basic state (U, V)
    (m/s), all on one (lat, lon) grid (leading dimensions allowed on psi_a).

    p_hpa    : the level; the flux carries the factor p/1000 hPa.
    umin, lat_min : mask |U, V| < umin and |lat| < lat_min (NaN in wx, wy,
               div; div is computed with masked points set to zero).
    periodic : longitude wraps (full uniform circle required).
    div_lmax : spectrally truncate the divergence at degree div_lmax (the
               site uses 15, planetary scale: second derivatives of the flux
               carry grid-scale ripple). Needs pyshtools and a Driscoll-Healy
               grid (see sh_truncate)."""
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    lonr, closed = _check_lon(lon, periodic)
    latr = np.deg2rad(lat)
    psi = np.asarray(psi_a, float)
    U = np.asarray(U, float)
    V = np.asarray(V, float)
    if closed:                         # compute on the open circle, re-close after
        op = lambda f: f[..., :-1] if f.ndim and f.shape[-1] == lon.size else f   # noqa: E731
        r = tn01_flux(psi[..., :-1], op(U), op(V), lat, lon[:-1], p_hpa=p_hpa,
                      umin=umin, lat_min=lat_min, periodic=True)
        cl = lambda f: np.concatenate([f, f[..., :1]], axis=-1)   # noqa: E731
        div = cl(r.div)
        if div_lmax is not None:
            div = np.where(cl(r.mask), np.nan, sh_truncate(np.nan_to_num(div), div_lmax))
        return WAF(cl(r.wx), cl(r.wy), div, cl(r.mask))
    cos = np.clip(np.cos(latr), 1e-3, None)[:, None]
    a2 = A_EARTH ** 2
    pl = _dlon(psi, lonr, periodic)
    pp = np.gradient(psi, latr, axis=-2)
    pll = _dlon(pl, lonr, periodic)
    plp = np.gradient(pl, latr, axis=-2)
    ppp = np.gradient(pp, latr, axis=-2)
    spd = np.hypot(U, V)
    pref = (p_hpa / 1000.0) * cos / (2.0 * np.maximum(spd, 1e-6))
    xx = pl * pl - psi * pll
    xy = pl * pp - psi * plp
    yy = pp * pp - psi * ppp
    wx = pref * (U * xx / (a2 * cos ** 2) + V * xy / (a2 * cos))
    wy = pref * (U * xy / (a2 * cos) + V * yy / a2)
    bad = np.broadcast_to((spd < umin) | (np.abs(lat)[:, None] < lat_min), wx.shape)
    wx = np.where(bad, np.nan, wx)
    wy = np.where(bad, np.nan, wy)
    wx0 = np.nan_to_num(wx)
    wy0 = np.nan_to_num(wy)
    div = (_dlon(wx0, lonr, periodic) + np.gradient(wy0 * cos, latr, axis=-2)) / (A_EARTH * cos)
    if div_lmax is not None:
        div = sh_truncate(div, div_lmax)
    div = np.where(bad, np.nan, div)
    return WAF(wx, wy, div, np.array(bad))


def tn01_flux_cartesian(psi_a, U, V, dx: float, dy: float, phat: float = 1.0,
                        periodic_y: bool = False):
    """TN01 flux on a Cartesian (beta-plane or channel) grid (..., ny, nx),
    periodic in x: W = phat/(2|U|) (U(psi_x^2 - psi psi_xx) + V(psi_x psi_y -
    psi psi_xy), U(psi_x psi_y - psi psi_xy) + V(psi_y^2 - psi psi_yy)).
    Returns (wx, wy, div). No masking."""
    psi = np.asarray(psi_a, float)
    U = np.asarray(U, float)
    V = np.asarray(V, float)

    def ddx(f):
        return (np.roll(f, -1, axis=-1) - np.roll(f, 1, axis=-1)) / (2 * dx)

    def ddy(f):
        if periodic_y:
            return (np.roll(f, -1, axis=-2) - np.roll(f, 1, axis=-2)) / (2 * dy)
        return np.gradient(f, dy, axis=-2)

    px, py = ddx(psi), ddy(psi)
    pxx, pxy, pyy = ddx(px), ddy(px), ddy(py)
    spd = np.maximum(np.hypot(U, V), 1e-6)
    wx = phat / (2 * spd) * (U * (px * px - psi * pxx) + V * (px * py - psi * pxy))
    wy = phat / (2 * spd) * (U * (px * py - psi * pxy) + V * (py * py - psi * pyy))
    return wx, wy, ddx(wx) + ddy(wy)


@dataclass
class EnsembleWAF:
    """Mean of the members' fluxes and divergences, and the share of members
    agreeing with the majority sign of the divergence (0.5-1; NaN where
    masked). The site hatches agree < 0.6."""
    wx: np.ndarray
    wy: np.ndarray
    div: np.ndarray
    agree: np.ndarray
    n: int


def ensemble_tn01_flux(psi_members, U, V, lat, lon, **kw) -> EnsembleWAF:
    """TN01 flux PER MEMBER, then averaged. psi_members iterates (lat, lon)
    arrays (or is a (member, lat, lon) array); U, V are the shared basic
    state; kw go to tn01_flux. Memory stays O(one member) plus the sums."""
    sx = sy = sd = None
    npos = nneg = nfin = None
    n = 0
    for psi in psi_members:
        r = tn01_flux(psi, U, V, lat, lon, **kw)
        fin = np.isfinite(r.div)
        if n == 0:
            sx, sy, sd = np.zeros_like(r.wx), np.zeros_like(r.wy), np.zeros_like(r.div)
            npos, nneg, nfin = (np.zeros(r.div.shape) for _ in range(3))
        sx += np.nan_to_num(r.wx)
        sy += np.nan_to_num(r.wy)
        sd += np.nan_to_num(r.div)
        npos += fin & (r.div > 0)
        nneg += fin & (r.div < 0)
        nfin += fin
        n += 1
    if n == 0:
        raise ValueError("no members supplied")
    with np.errstate(invalid="ignore", divide="ignore"):
        nf = np.where(nfin > 0, nfin, np.nan)
        agree = np.maximum(npos, nneg) / nf
    return EnsembleWAF(sx / nf, sy / nf, sd / nf, agree, n)


def lead_smooth(psi, window: int = TSMOOTH, axis: int = 0) -> np.ndarray:
    """Centred running mean of psi' along the lead axis (window samples),
    truncated at the ends (each end point averages what exists). Quasi-
    stationarity for TN01: fast synoptic packets enter the phase-independent
    form with error."""
    x = np.moveaxis(np.asarray(psi, float), axis, 0)
    n = x.shape[0]
    half = window // 2
    out = np.empty_like(x)
    for i in range(n):
        out[i] = x[max(0, i - half):min(n, i + half + 1)].mean(axis=0)
    return np.moveaxis(out, 0, axis)


def lowpass_anomaly(fields, times, clim, end, days: int = WINDOW_DAYS,
                    min_count: int = MIN_COUNT):
    """Trailing mean of (analysis - climatology) for a low-passed basic state.

    fields : (time, ...) analyses (e.g. u, v or psi on one grid)
    times  : datetime64-like, one per row
    clim   : callable doy -> climatology field of fields.shape[1:]
    end    : last time included; the window is (end - days, end]
    Returns (anomaly, count); anomaly is None when fewer than min_count
    analyses fall in the window (use the climatology alone then).
    Basic state = clim(doy) + anomaly; take psi' against the same flow."""
    t = np.asarray(times, dtype="datetime64[s]")
    end = np.datetime64(end, "s")
    sel = (t > end - np.timedelta64(int(days * 86400), "s")) & (t <= end)
    count = int(sel.sum())
    if count < min_count:
        return None, count
    f = np.asarray(fields, float)[sel]
    doy = (t[sel].astype("datetime64[D]") - t[sel].astype("datetime64[Y]").astype("datetime64[D]")).astype(int) + 1
    acc = np.zeros(f.shape[1:])
    for fi, d in zip(f, doy):
        acc += fi - np.asarray(clim(float(d)), float)
    return acc / count, count


def sh_truncate(field, lmax: int) -> np.ndarray:
    """Spherical-harmonic truncation of a global field at degree lmax.
    field: (lat, lon) on a Driscoll-Healy grid (lat from +90 down; nlon =
    nlat or 2 nlat), plain or extended with the -90 row and the 360 column
    as returned by quantmet.helmholtz. Needs pyshtools."""
    import pyshtools as pysh
    f = np.asarray(field, float)
    ext = f.shape[0] % 2 == 1                  # extended grid: -90 row and 360 column included
    nlat, nlon = f.shape[0] - ext, f.shape[1] - ext
    clm = pysh.SHGrid.from_array(f[:nlat, :nlon], grid="DH").expand()
    clm.coeffs[:, lmax + 1:, :] = 0.0
    out = clm.expand(grid="DH2" if nlon == 2 * nlat else "DH", lmax=clm.lmax, extend=ext).data
    return out


def streamfunction_anomaly(u2d, v2d, psi_clim=None, lmax: int = 63):
    """psi' = psi(u, v) - psi_clim on the Driscoll-Healy output grid of
    quantmet.helmholtz.streamfunction (xarray u, v with latitude/longitude
    coords; lmax ~63 keeps synoptic and planetary scales). Returns
    (psi_a, lat, lon)."""
    from .helmholtz import streamfunction
    psi, lat, lon = streamfunction(u2d, v2d, lmax=lmax)
    if psi_clim is not None:
        psi = psi - np.asarray(psi_clim, float)
    return psi, lat, lon
