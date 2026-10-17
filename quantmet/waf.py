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
  needs a westerly waveguide). W is homogeneous of degree zero in (U, V)
  (the 1/|U| cancels the U, V in the brackets): its MAGNITUDE does not grow
  as |U| -> 0, its DIRECTION becomes ill-conditioned (a wind known to
  +-1.5 m/s turns it by ~27 deg at 3 m/s, ~3 deg in a 25 m/s jet).
- THE MASK EDGE AND THE TRUNCATED DIVERGENCE. The divergence is truncated
  at a low degree (div_lmax), a global filter, so whatever is done at the
  mask edge spreads into the valid cells. Zero-filling the masked flux
  leaves a step whose centred difference lands half in the first masked
  cell and half in the first valid one. div_method="zero" (the default)
  sets the masked cells' divergence to zero before truncating, which drops
  the outer half; "as_computed" keeps both halves; "taper" ramps the flux to
  zero over taper_deg (mask_taper) and takes its divergence spectrally
  (sh_divergence), which moves the whole edge term W.grad(w) into the
  valid ramp. In the synthetic tests of tests/test_waf_divergence.py,
  "zero" had the smallest error for a flux of known divergence and about
  half the spurious divergence of the other two for a divergence-free flux
  crossing the mask edge; "taper" had the largest of both, and the least
  sensitivity to a noisy wind direction in the weak-wind rim (small for all
  three on a real basic state). 0.4.0 used "zero" on a closed (360-column)
  grid and "as_computed" on an open one; since 0.4.1 every method is the
  same on both.

Streamfunction: quantmet.helmholtz.streamfunction (spherical-harmonic
Poisson inversion on a pole-free Gauss-Legendre grid), wrapped by
streamfunction_anomaly; the flux itself needs numpy only.

Powers the 250 hPa wave-activity flux loop on
https://scorvec.com/circulation.html (AIFS-ENS, 25 members, day 0-15).

Requires: numpy (flux); scipy for mask_taper; pyshtools + xarray for
streamfunction_anomaly and the truncated divergence (sh_truncate,
sh_divergence) (pip install quantmet[waf]).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import A_EARTH
from .degdays import doy366

UMIN = 3.0  # m/s: basic-state wind below which the flux is masked
LAT_MIN = 20.0  # deg: tropical mask
TSMOOTH = 5  # samples (days): running mean of psi' along the lead
WINDOW_DAYS = 30  # trailing window of the low-passed basic state
MIN_COUNT = 20  # analyses required inside that window
TAPER_DEG = 5.0  # deg: cos^2 ramp of the flux into the mask (div_method="taper")
DIV_METHODS = ("zero", "as_computed", "taper")  # the first is the default

__all__ = [
    "WAF",
    "tn01_flux",
    "tn01_flux_cartesian",
    "ensemble_tn01_flux",
    "EnsembleWAF",
    "lead_smooth",
    "lowpass_anomaly",
    "streamfunction_anomaly",
    "sh_truncate",
    "sh_divergence",
    "mask_taper",
    "masked_divergence",
    "UMIN",
    "LAT_MIN",
    "TSMOOTH",
    "WINDOW_DAYS",
    "MIN_COUNT",
    "TAPER_DEG",
    "DIV_METHODS",
]


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
            raise ValueError(
                "periodic=True needs a uniform longitude grid covering 360 deg; pass periodic=False for a regional grid"
            )
    return np.deg2rad(lon), bool(closed and periodic)


def _flux_open(psi, U, V, lat, lonr, periodic, p_hpa, umin, lat_min):
    """wx, wy (NaN where masked), the mask broadcast to the flux and the
    mask in its own (basic-state) shape, on an open longitude circle."""
    latr = np.deg2rad(lat)
    cos = np.clip(np.cos(latr), 1e-3, None)[:, None]
    a2 = A_EARTH**2
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
    wx = pref * (U * xx / (a2 * cos**2) + V * xy / (a2 * cos))
    wy = pref * (U * xy / (a2 * cos) + V * yy / a2)
    base = (spd < umin) | (np.abs(lat)[:, None] < lat_min)
    bad = np.broadcast_to(base, wx.shape)
    wx = np.where(bad, np.nan, wx)
    wy = np.where(bad, np.nan, wy)
    return wx, wy, np.array(bad), base


def _fd_div(wx0, wy0, lat, lonr, periodic):
    """Centred-difference divergence of a NaN-free flux (periodic lambda)."""
    latr = np.deg2rad(lat)
    cos = np.clip(np.cos(latr), 1e-3, None)[:, None]
    return (_dlon(wx0, lonr, periodic) + np.gradient(wy0 * cos, latr, axis=-2)) / (A_EARTH * cos)


def tn01_flux(
    psi_a,
    U,
    V,
    lat,
    lon,
    p_hpa: float = 250.0,
    umin: float = UMIN,
    lat_min: float = LAT_MIN,
    periodic: bool = True,
    div_lmax: int | None = None,
    div_method: str = "zero",
    taper_deg: float = TAPER_DEG,
) -> WAF:
    """Horizontal TN01 flux from psi' (m^2/s) and the basic state (U, V)
    (m/s), all on one (lat, lon) grid (leading dimensions allowed on psi_a).

    p_hpa    : the level; the flux carries the factor p/1000 hPa.
    umin, lat_min : mask |U, V| < umin and |lat| < lat_min (NaN in wx, wy
               and div; the mask is returned).
    periodic : longitude wraps (full uniform circle required).
    div_lmax : spectrally truncate the divergence at degree div_lmax (the
               site uses 15, planetary scale: second derivatives of the flux
               carry grid-scale ripple). Needs pyshtools and a global
               Driscoll-Healy grid (see sh_truncate). Without it the
               divergence is the centred difference of the flux with the
               masked cells set to zero, whatever div_method says (no filter
               spreads the mask edge: it stays in the cells next to it).
    div_method : how the truncated divergence treats the mask edge (only
               with div_lmax; wx, wy and the mask are the same for all):
        "zero"        (default) centred-difference divergence of the
                      zero-filled flux, the masked cells' divergence set to
                      zero, truncated, masked. Drops the half of the edge
                      step's derivative that falls in the masked cells.
        "as_computed" the same divergence truncated as computed: the masked
                      cells next to the edge keep the outer half of the
                      step's derivative into the filter.
        "taper"       wx, wy multiplied by a cos^2 ramp, 0 on the masked
                      cells and 1 from taper_deg of great-circle distance
                      on (mask_taper), divergence of the tapered flux taken
                      spectrally up to div_lmax (sh_divergence, the weak
                      form: the flux is not differenced), masked after.
               Synthetic tests (tests/test_waf_divergence.py, CHANGELOG
               0.4.1) chose the default: "zero" had the smallest error for a
               flux of known divergence and about half the spurious
               divergence of the others for a divergence-free flux; "taper"
               had the largest of both and the least leakage of a noisy
               wind direction in the weak-wind rim. 0.4.0 applied "zero" on
               a closed (360-column) grid and "as_computed" on an open one,
               so the two layouts disagreed; every method is now the same
               on both, and "zero" reproduces 0.4.0 on a closed grid, as
               "as_computed" does on an open one.
    taper_deg : ramp width (deg) for div_method="taper"; 0 is a hard edge
               (the zero-filled flux, divergence taken spectrally)."""
    if div_method not in DIV_METHODS:
        raise ValueError(f"div_method must be one of {DIV_METHODS}, got {div_method!r}")
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    lonr, closed = _check_lon(lon, periodic)
    psi = np.asarray(psi_a, float)
    U = np.asarray(U, float)
    V = np.asarray(V, float)
    if closed:  # compute on the open circle, re-close after

        def op(f):
            return f[..., :-1] if f.ndim and f.shape[-1] == lon.size else f

        psi, U, V, lono, lonr = op(psi), op(U), op(V), lon[:-1], lonr[:-1]
    else:
        lono = lon
    wx, wy, bad, base = _flux_open(psi, U, V, lat, lonr, periodic, p_hpa, umin, lat_min)
    div = _masked_div_open(wx, wy, bad, base, lat, lono, lonr, periodic, closed, div_lmax, div_method, taper_deg)
    cl = (lambda f: np.concatenate([f, f[..., :1]], axis=-1)) if closed else (lambda f: f)
    return WAF(cl(wx), cl(wy), div, cl(bad))


def _masked_div_open(wx, wy, bad, base, lat, lono, lonr, periodic, closed, lmax, method, taper_deg):
    """The divergence stage on the open circle; returned re-closed if closed."""
    cl = (lambda f: np.concatenate([f, f[..., :1]], axis=-1)) if closed else (lambda f: f)
    wx0 = np.nan_to_num(np.where(bad, 0.0, wx))
    wy0 = np.nan_to_num(np.where(bad, 0.0, wy))
    if lmax is None:
        div = cl(_fd_div(wx0, wy0, lat, lonr, periodic))
    elif method == "taper":
        wt = _taper_like(base, lat, lono, taper_deg, wx0.shape)
        div = cl(sh_divergence(wt * wx0, wt * wy0, lat, lono, lmax=lmax))
    else:
        div = _fd_div(wx0, wy0, lat, lonr, periodic)
        if method == "zero":
            div = np.where(bad, 0.0, div)
        div = sh_truncate(cl(div), lmax)
    return np.where(cl(bad), np.nan, div)


def masked_divergence(
    wx,
    wy,
    mask,
    lat,
    lon,
    lmax: int | None = 15,
    method: str = "zero",
    taper_deg: float = TAPER_DEG,
    periodic: bool = True,
) -> np.ndarray:
    """Divergence of a flux that is only valid outside a mask, as tn01_flux
    takes it (wx, wy: (..., lat, lon); mask: True = invalid, broadcastable to
    them; values under the mask are ignored). method and taper_deg as
    div_method / taper_deg of tn01_flux; lmax None = centred difference of
    the zero-filled flux, untruncated. NaN on the mask."""
    if method not in DIV_METHODS:
        raise ValueError(f"method must be one of {DIV_METHODS}, got {method!r}")
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    lonr, closed = _check_lon(lon, periodic)
    wx = np.asarray(wx, float)
    wy = np.asarray(wy, float)
    base = np.asarray(mask, bool)
    base = np.broadcast_to(base, np.broadcast_shapes(base.shape, (lat.size, lon.size)))
    if closed:
        wx, wy, base, lono, lonr = wx[..., :-1], wy[..., :-1], base[..., :-1], lon[:-1], lonr[:-1]
    else:
        lono = lon
    bad = np.broadcast_to(base, wx.shape)
    return _masked_div_open(wx, wy, bad, base, lat, lono, lonr, periodic, closed, lmax, method, taper_deg)


def _taper_like(base, lat, lon, width_deg, shape):
    """mask_taper of each 2-D mask in base, broadcast to the flux shape."""
    base = np.broadcast_to(base, np.broadcast_shapes(base.shape, (lat.size, lon.size)))
    if base.ndim == 2:
        return np.broadcast_to(mask_taper(base, lat, lon, width_deg), shape)
    out = np.empty(base.shape)
    for idx in np.ndindex(base.shape[:-2]):
        out[idx] = mask_taper(base[idx], lat, lon, width_deg)
    return np.broadcast_to(out, shape)


_TAPER_CACHE: dict = {}


def mask_taper(mask, lat, lon, width_deg: float = TAPER_DEG) -> np.ndarray:
    """cos^2 edge weight for a (lat, lon) mask (True = masked, lat/lon in
    degrees): 0 on masked cells, sin^2(pi/2 * d / width_deg) at great-circle
    distance d from the nearest masked cell centre, 1 from width_deg on.
    width_deg = 0 gives the plain 0/1 indicator. The last mask is cached
    (an ensemble shares one basic state, hence one mask)."""
    mask = np.asarray(mask, bool)
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    if mask.shape != (lat.size, lon.size):
        raise ValueError(f"mask shape {mask.shape} != (lat, lon) = {(lat.size, lon.size)}")
    if width_deg <= 0 or not mask.any() or mask.all():
        return (~mask).astype(float)
    key = (mask.tobytes(), lat.tobytes(), lon.tobytes(), float(width_deg))
    hit = _TAPER_CACHE.get("last")
    if hit is not None and hit[0] == key:
        return hit[1].copy()
    from scipy.spatial import cKDTree

    la, lo = np.meshgrid(np.deg2rad(lat), np.deg2rad(lon), indexing="ij")
    xyz = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1).reshape(-1, 3)
    flat = mask.ravel()
    chord = 2.0 * np.sin(np.deg2rad(width_deg) / 2.0)
    d, _ = cKDTree(xyz[flat]).query(xyz[~flat], k=1, distance_upper_bound=chord * (1 + 1e-9))
    ang = np.rad2deg(2.0 * np.arcsin(np.clip(np.where(np.isfinite(d), d, chord) / 2.0, 0.0, 1.0)))
    w = np.zeros(flat.size)
    w[~flat] = np.sin(0.5 * np.pi * np.clip(ang / width_deg, 0.0, 1.0)) ** 2
    w = w.reshape(mask.shape)
    _TAPER_CACHE["last"] = (key, w)
    return w.copy()


def tn01_flux_cartesian(psi_a, U, V, dx: float, dy: float, phat: float = 1.0, periodic_y: bool = False):
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
        out[i] = x[max(0, i - half) : min(n, i + half + 1)].mean(axis=0)
    return np.moveaxis(out, 0, axis)


def lowpass_anomaly(fields, times, clim, end, days: int = WINDOW_DAYS, min_count: int = MIN_COUNT):
    """Trailing mean of (analysis - climatology) for a low-passed basic state.

    fields : (time, ...) analyses (e.g. u, v or psi on one grid)
    times  : datetime64-like, one per row
    clim   : callable doy -> climatology field of fields.shape[1:], called
             with the 1-based day on the 366-slot calendar
             (quantmet.degdays.doy366 + 1): 29 Feb is 60 and 1 March is 61
             in every year, so a 366-row daily table is read as
             table[doy - 1] and a harmonic fit takes the phase
             2 pi (doy - 1) / 366. (quantmet <= 0.4.1 passed the plain day
             of the year, which reads every common-year day after February
             from the slot of the day before.)
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
    doy = doy366(t[sel]) + 1
    acc = np.zeros(f.shape[1:])
    for fi, d in zip(f, doy):
        acc += fi - np.asarray(clim(float(d)), float)
    return acc / count, count


def _dh_layout(nrow: int, ncol: int):
    """(n, south, closed) of a Driscoll-Healy grid: n rows from +90 at 180/n
    spacing (n even), plus the -90 row if nrow is odd; n or 2n columns from
    0 E, plus the 360 column if ncol is odd."""
    south, closed = nrow % 2 == 1, ncol % 2 == 1
    n, nlon = nrow - south, ncol - closed
    if n < 4 or n % 2 or nlon not in (n, 2 * n):
        raise ValueError(
            f"({nrow}, {ncol}) is not a Driscoll-Healy grid (n or n+1 rows, n, 2n, n+1 or 2n+1 columns, n even)"
        )
    return n, south, closed


def sh_truncate(field, lmax: int) -> np.ndarray:
    """Spherical-harmonic truncation of a global field at degree lmax.
    field: (..., lat, lon) on a Driscoll-Healy grid (lat from +90 down; nlon
    = nlat or 2 nlat), plain or extended with the -90 row and/or the 360
    column (as returned by quantmet.helmholtz, or an extended grid with the
    longitude re-opened). Needs pyshtools."""
    import pyshtools as pysh

    f = np.asarray(field, float)
    if f.ndim > 2:
        return np.stack([sh_truncate(g, lmax) for g in f.reshape(-1, *f.shape[-2:])]).reshape(f.shape)
    n, south, closed = _dh_layout(*f.shape)
    nlon = f.shape[1] - closed
    clm = pysh.SHGrid.from_array(f[:n, :nlon], grid="DH").expand()
    clm.coeffs[:, lmax + 1 :, :] = 0.0
    grid = "DH2" if nlon == 2 * n else "DH"
    if south == closed:  # plain or fully extended: as in 0.4.0
        return clm.expand(grid=grid, lmax=clm.lmax, extend=south).data
    return clm.expand(grid=grid, lmax=clm.lmax, extend=True).data[: n + south, : nlon + closed]


_DH_TABLES: dict = {}


def _dh_tables(n: int, lmax: int):
    """Quadrature weights and 4pi-normalised Legendre tables on the n
    Driscoll-Healy colatitudes theta_j = pi j / n (plus the south pole)."""
    key = (n, lmax)
    if key not in _DH_TABLES:
        import pyshtools as pysh

        th = np.pi * np.arange(n) / n
        k = np.arange(n // 2)
        # Driscoll & Healy (1994) weights, scaled to integrate sin(theta) dtheta over [0, pi]
        w = np.sin(th) * (np.sin(np.outer(th, 2 * k + 1)) / (2 * k + 1)).sum(1)
        w *= 2.0 / w.sum()
        nlm = (lmax + 1) * (lmax + 2) // 2
        P = np.empty((n + 1, nlm))
        dP = np.zeros((n, nlm))  # d/dz, z = cos(theta) = sin(lat); unused at the pole
        P[0] = pysh.legendre.PlmBar(lmax, 1.0)
        for j in range(1, n):
            P[j], dP[j] = pysh.legendre.PlmBar_d1(lmax, np.cos(th[j]))
        P[n] = pysh.legendre.PlmBar(lmax, -1.0)
        ls = np.concatenate([np.full(ll + 1, ll) for ll in range(lmax + 1)])
        ms = np.concatenate([np.arange(ll + 1) for ll in range(lmax + 1)])
        _DH_TABLES[key] = (w, np.sin(th), P, dP, ls, ms)
    return _DH_TABLES[key]


def sh_divergence(wx, wy, lat, lon, lmax: int | None = None, radius: float = A_EARTH) -> np.ndarray:
    """Divergence of a horizontal vector field (wx eastward, wy northward;
    (..., lat, lon) on a global Driscoll-Healy grid as for sh_truncate),
    spectrally and truncated at degree lmax (default: the grid's full
    n/2 - 1). The coefficients come from the weak form
        D_lm = -(1/4 pi) * integral of W . grad(Y_lm) dOmega
    (integration by parts on the closed sphere), evaluated with the Driscoll-
    Healy quadrature: the flux itself is never differenced, so a sharp or
    noisy field does not feed grid-scale derivatives into the low degrees.
    Exact (to round-off) for band-limited fields the quadrature resolves.
    NaN is rejected: fill or taper masked cells first. Needs pyshtools."""
    wx = np.asarray(wx, float)
    wy = np.asarray(wy, float)
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    if wx.shape != wy.shape:
        raise ValueError("wx and wy differ in shape")
    n, south, closed = _dh_layout(*wx.shape[-2:])
    nlon = wx.shape[-1] - closed
    if lat.size != wx.shape[-2] or lon.size != wx.shape[-1]:
        raise ValueError("lat / lon do not match the field")
    if not (
        np.allclose(lat[:n], 90.0 - 180.0 * np.arange(n) / n)
        and (not south or np.isclose(lat[-1], -90.0))
        and np.allclose(lon[:nlon], 360.0 * np.arange(nlon) / nlon)
    ):
        raise ValueError("sh_divergence needs a Driscoll-Healy grid: lat from +90 at 180/n, lon from 0 at 360/nlon")
    lmx = n // 2 - 1
    L = lmx if lmax is None else int(lmax)
    if not 0 <= L <= lmx:
        raise ValueError(f"lmax must be in [0, {lmx}] on this grid")
    x = wx[..., :n, :nlon]
    y = wy[..., :n, :nlon]
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("NaN in the flux: fill or taper the masked cells first")
    w, cosl, P, dP, _ls, ms = _dh_tables(n, L)
    fx = np.fft.rfft(x, axis=-1)[..., ms]  # (..., n, nlm): sum_k f e^{-i m lambda_k}
    fy = np.fft.rfft(y, axis=-1)[..., ms]
    cx, sx, cy, sy = fx.real, -fx.imag, fy.real, -fy.imag
    j = slice(1, n)  # the north-pole row has zero weight
    # integrand per unit solid angle: Wx dY/dlambda / cos(lat) + Wy dY/dphi, dY/dphi = cos(lat) dP/dz
    a_ = (w[j] / cosl[j])[:, None] * ms * P[j]  # m P / cos(lat), weighted
    b_ = (w[j] * cosl[j])[:, None] * dP[j]  # cos(lat) dP/dz, weighted
    s = -1.0 / (2.0 * radius * nlon)  # -(2 pi / nlon) / (4 pi a)
    dc = s * (np.einsum("...jk,jk->...k", sx[..., j, :], -a_) + np.einsum("...jk,jk->...k", cy[..., j, :], b_))
    ds = s * (np.einsum("...jk,jk->...k", cx[..., j, :], a_) + np.einsum("...jk,jk->...k", sy[..., j, :], b_))
    rows = P[: n + 1] if south else P[:n]
    onehot = np.zeros((ms.size, nlon // 2 + 1))
    onehot[np.arange(ms.size), ms] = 1.0
    cm = np.einsum("...k,jk,km->...jm", dc, rows, onehot)
    sm = np.einsum("...k,jk,km->...jm", ds, rows, onehot)
    spec = (nlon / 2.0) * (cm - 1j * sm)
    spec[..., 0] = nlon * cm[..., 0]
    out = np.fft.irfft(spec, n=nlon, axis=-1)
    return np.concatenate([out, out[..., :1]], axis=-1) if closed else out


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
