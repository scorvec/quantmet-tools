"""Transformed-Eulerian-mean (TEM) diagnostics on pressure levels: eddy
covariances by zonal wavenumber, the residual circulation, the full
primitive-equation Eliassen-Palm flux and its divergence, the residual
streamfunction, downward control, and the tropical upwelling it implies.

Primitive-equation TEM in pressure coordinates (Andrews, Holton & Leovy
1987, eqs. 3.5.1-3.5.5), with theta the potential temperature, subscripts
partial derivatives, [.] the zonal mean and primes deviations from it:

    fhat   = f - (a cos phi)^-1 d(u cos phi)/d phi
    v*     = v - d/dp( [v'th'] / th_p )
    omega* = omega + (a cos phi)^-1 d/d phi( cos phi [v'th'] / th_p )
    F_phi  = a cos phi ( u_p [v'th'] / th_p - [u'v'] )
    F_p    = a cos phi ( fhat [v'th'] / th_p - [u'omega'] )
    du/dt  = fhat v* - omega* u_p + (a cos phi)^-1 div F + X

and the residual mass streamfunction (kg/s, positive = clockwise with north
to the right, i.e. rising near the equator and poleward aloft in the NH):

    psi*   = (2 pi a cos phi / g) ( int_top^p [v] dp' - [v'th'] / th_p )
    v*     = g / (2 pi a cos phi) d psi*/dp
    omega* = -g / (2 pi a^2 cos phi) d psi*/d phi

Numerical notes that matter in practice:
- NYQUIST HALVING. Eddy covariances are summed from the cross-spectrum:
  the k-th zonal wavenumber contributes 2 Re(A_k conj B_k) / n^2 for
  0 < k < n/2. On an EVEN longitude grid the k = n/2 (Nyquist) coefficient
  has no conjugate partner, so its contribution is Re(A_k conj B_k) / n^2,
  NOT twice that. Doubling it (the naive loop) is invisible on a 0.25-deg
  grid where k = 720 carries nothing, and wrong by exactly the Nyquist
  variance on a coarse grid -- e.g. [v'T'] summed to k = 72 on a 2.5-deg
  (144-point) grid. With the halving, the wavenumber sum equals the direct
  zonal mean of the product exactly (Parseval).
- ONE MEAN STATE FOR EVERY WAVE GROUP. The EP-flux divergence by wavenumber
  group uses the same fhat, u_p and th_p as the total, so the groups add up
  to the total (the flux is linear in the covariances).
- DOWNWARD CONTROL (Haynes et al. 1991) builds psi* from the zonal
  momentum forcing, -fhat v* = G in steady state:
      psi*(phi, p) = -(2 pi a cos phi / g) int_top^p G / fhat dp'
  It is undefined where fhat -> 0 (the deep tropics): those points are NaN
  (|fhat| < fhat_min) and fill_tropics interpolates across them in
  sin(lat), the profile of uniform tropical upwelling. From once-daily
  analyses downward control is far better constrained than the Eulerian
  continuity route (residual_streamfunction): the analysed tropical mean
  meridional circulation (~0.3 mm/s) sits below the noise of a single
  zonal mean, while the momentum forcing does not.
- INTEGRATION FROM THE TOP. Both streamfunctions are integrated from the
  highest level supplied, which is taken as psi* = 0 (plus the eddy term,
  for the Eulerian route). Supply levels high enough (0.1 hPa for the
  stratosphere) that the mass above is negligible.
- UPWELLING INDEX. The tropical upward mass flux across a level is
  psi*(lat_n) - psi*(lat_s) between the turnaround latitudes; the mean
  log-pressure w* over that band is g H F / (2 pi a^2 p (sin lat_n -
  sin lat_s)). Fix the turnaround latitudes (e.g. from a climatology) for
  daily work: the max-minus-min of each day's own noisy profile is biased
  high (on MERRA-2 at 70 hPa, 9.4 instead of 6.4 x 1e9 kg/s annual mean),
  and a fixed band keeps the index linear, so a mean of daily values is the
  index of the mean circulation.
- STATIC STABILITY. th_p is the LOCAL zonal-mean d theta/dp, floored at
  -1e-7 K/Pa. That is safe in the stratosphere; tropospheric work near a
  neutral boundary layer wants the QG global-mean profile instead
  (quantmet.epflux).

Powers the zonal-momentum budget, the Brewer-Dobson downward-control
section and the 100 hPa eddy heat flux on
https://scorvec.com/stratosphere.html.

Requires: numpy only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

A_EARTH = 6.371e6
G = 9.80665
OMEGA = 7.292e-5
KAPPA = 0.2857            # R/c_p, dry air
P_REF = 1000.0            # hPa, for potential temperature
H_SCALE = 7000.0          # m, log-pressure scale height
DEFAULT_GROUPS = {"k1": (1, 1), "k2": (2, 2), "k3p": (3, None)}

__all__ = ["cospectrum", "eddy_covariances", "eddy_heat_flux", "HeatFlux",
           "tem_terms", "residual_streamfunction", "residual_velocities",
           "downward_control", "fill_tropics", "upwelling_w", "Upwelling",
           "DEFAULT_GROUPS"]


# --------------------------------------------------------------------------
# Eddy covariances
# --------------------------------------------------------------------------
def cospectrum(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Contribution of each zonal wavenumber k = 1 .. n//2 to [a'b'] along
    the last (longitude) axis: (..., n//2). Sums exactly to the zonal mean
    of (a - [a])(b - [b]); the Nyquist term of an even grid is not doubled."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n = a.shape[-1]
    if b.shape[-1] != n:
        raise ValueError("a and b need the same longitude axis (last)")
    c = 2.0 * np.real(np.fft.rfft(a, axis=-1) * np.conj(np.fft.rfft(b, axis=-1)))[..., 1:] / n ** 2
    if n % 2 == 0:
        c[..., -1] /= 2.0
    return c


def _group_sum(c: np.ndarray, k0: int, k1: int | None) -> np.ndarray:
    kmax = c.shape[-1]
    hi = kmax if k1 is None else min(k1, kmax)
    if k0 > hi:
        return np.zeros(c.shape[:-1])
    return c[..., k0 - 1:hi].sum(-1)


def eddy_covariances(u, v, t, omega=None, groups=DEFAULT_GROUPS) -> dict:
    """Zonal means and eddy covariances of fields shaped (..., lat, lon).

    Returns a dict with ubar, vbar, tbar (and wbar = [omega]), the total
    covariances vT = [v'T'], uv = [u'v'] (and uw = [u'omega']) over every
    wavenumber, and the same by wavenumber group as "<name>_<group>"
    (default groups: k1, k2, k3p = 3 and above). T in K, omega in Pa/s.
    Fields must be finite: fill gaps before calling."""
    f = {"u": np.asarray(u, float), "v": np.asarray(v, float), "t": np.asarray(t, float)}
    if omega is not None:
        f["omega"] = np.asarray(omega, float)
    for k, x in f.items():
        if not np.isfinite(x).all():
            raise ValueError(f"{k} has non-finite values")
    out = {"ubar": f["u"].mean(-1), "vbar": f["v"].mean(-1), "tbar": f["t"].mean(-1)}
    pairs = [("vT", "v", "t"), ("uv", "u", "v")]
    if omega is not None:
        out["wbar"] = f["omega"].mean(-1)
        pairs.append(("uw", "u", "omega"))
    for name, a, b in pairs:
        c = cospectrum(f[a], f[b])
        out[name] = c.sum(-1)
        for g, (k0, k1) in (groups or {}).items():
            out[f"{name}_{g}"] = _group_sum(c, k0, k1)
    return out


@dataclass
class HeatFlux:
    """Eddy heat flux [v'T'] (K m/s).

    total : summed over k = 1 .. kmax, shape (...) with a lat_band, else (..., lat)
    by_k  : per wavenumber, (..., kmax) or (..., lat, kmax)
    k     : the wavenumbers 1 .. kmax
    """
    total: np.ndarray
    by_k: np.ndarray
    k: np.ndarray


def eddy_heat_flux(v, t, lat, kmax: int | None = None, lat_band=None,
                   poleward: bool = True) -> HeatFlux:
    """[v'T'] by zonal wavenumber from v, T shaped (..., lat, lon).

    kmax     : highest wavenumber kept (default n//2, all). Match it to the
               coarsest data being compared: a 0.25-deg forecast against a
               2.5-deg climatology wants kmax = 72 on both.
    lat_band : (lo, hi) degrees -> cos(lat)-weighted band mean; None keeps
               every latitude.
    poleward : multiply by sign(lat) so positive means poleward in both
               hemispheres (the upward EP flux at 100 hPa).
    The Nyquist term is halved (see the module notes), so kmax = n/2 on an
    even grid gives exactly the direct zonal covariance."""
    lat = np.asarray(lat, float)
    c = cospectrum(v, t)                                     # (..., lat, n//2)
    kmax = c.shape[-1] if kmax is None else min(int(kmax), c.shape[-1])
    c = c[..., :kmax]
    if poleward:
        c = c * np.sign(lat)[:, None]
    if lat_band is not None:
        lo, hi = min(lat_band), max(lat_band)
        m = (lat >= lo) & (lat <= hi)
        if not m.any():
            raise ValueError("no latitude inside lat_band")
        w = np.cos(np.deg2rad(lat[m]))
        c = np.tensordot(np.moveaxis(c[..., m, :], -2, -1), w / w.sum(), axes=([-1], [0]))
    return HeatFlux(total=c.sum(-1), by_k=c, k=np.arange(1, kmax + 1))


# --------------------------------------------------------------------------
# TEM terms
# --------------------------------------------------------------------------
def _p_axis(plev):
    p = np.asarray(plev, float)
    d = np.diff(p)
    if not (np.all(d > 0) or np.all(d < 0)):
        raise ValueError("plev must be strictly monotonic")
    return p


def _geometry(lat):
    phi = np.deg2rad(np.asarray(lat, float))
    c = np.cos(phi)
    cc = np.where(np.abs(c) < 1e-3, 1e-3, c)
    return phi, c, cc


def tem_terms(lat, plev, r: dict, groups=None) -> dict:
    """TEM terms on (..., plev, lat) from the zonal means and covariances of
    eddy_covariances (which must include omega). plev in hPa, any monotonic
    order; lat in degrees.

    Returns (SI units; tendencies in m/s^2):
      fhat, vstar (m/s), omstar (Pa/s), psi_e = [v'th']/th_p,
      Fphi, Fp        the primitive-equation EP flux,
      epd             (a cos phi)^-1 div F, the resolved-wave forcing,
      epd_<group>     the same by wavenumber group (from r's "<name>_<group>"),
      cor = fhat v*,  vad = -omega* u_p   (the residual-circulation terms).
    """
    phi, c, cc = _geometry(lat)
    p = _p_axis(plev) * 100.0
    pk = (P_REF / _p_axis(plev))[:, None] ** KAPPA
    th = r["tbar"] * pk
    th_p = np.gradient(th, p, axis=-2)
    th_p = np.where(np.abs(th_p) < 1e-7, -1e-7, th_p)
    u = r["ubar"]
    u_p = np.gradient(u, p, axis=-2)
    f = 2 * OMEGA * np.sin(phi)
    fhat = f - np.gradient(u * c, phi, axis=-1) / (A_EARTH * cc)
    psi_e = r["vT"] * pk / th_p
    vstar = r["vbar"] - np.gradient(psi_e, p, axis=-2)
    omstar = r["wbar"] + np.gradient(c * psi_e, phi, axis=-1) / (A_EARTH * cc)
    out = {"fhat": fhat, "vstar": vstar, "omstar": omstar, "psi_e": psi_e,
           "cor": fhat * vstar, "vad": -omstar * u_p}

    def epd(sfx):
        pe = r[f"vT{sfx}"] * pk / th_p
        Fphi = A_EARTH * c * (u_p * pe - r[f"uv{sfx}"])
        Fp = A_EARTH * c * (fhat * pe - r[f"uw{sfx}"])
        div = np.gradient(Fphi * c, phi, axis=-1) / (A_EARTH * cc) + np.gradient(Fp, p, axis=-2)
        return div / (A_EARTH * cc), Fphi, Fp

    out["epd"], out["Fphi"], out["Fp"] = epd("")
    if groups is None:
        groups = [k[3:] for k in r if k.startswith("vT_")]
    for g in groups:
        out[f"epd_{g}"] = epd("_" + g)[0]
    return out


# --------------------------------------------------------------------------
# Residual circulation
# --------------------------------------------------------------------------
def _cumtrapz_from_top(q, p):
    """Cumulative trapezoid of q (..., plev, lat) in p from the smallest p,
    returned in the input level order."""
    o = np.argsort(p)
    qs = np.take(q, o, axis=-2)
    ps = p[o]
    inc = 0.5 * (qs[..., 1:, :] + qs[..., :-1, :]) * np.diff(ps)[:, None]
    top = np.where(np.isnan(qs[..., :1, :]), np.nan, 0.0)          # NaN at the top stays NaN
    cum = np.concatenate([top, np.cumsum(inc, axis=-2)], axis=-2)
    return np.take(cum, np.argsort(o), axis=-2)


def residual_streamfunction(lat, plev, vbar, tbar, vT) -> np.ndarray:
    """Residual mass streamfunction psi* (kg/s) on (..., plev, lat) by the
    integration-by-parts identity (the eddy flux is never differentiated in
    the vertical): psi* = (2 pi a cos phi / g)(int_top^p [v] dp - [v'th']/th_p).
    NaN in [v] propagates."""
    _, c, _ = _geometry(lat)
    ph = _p_axis(plev)
    p = ph * 100.0
    pk = (P_REF / ph)[:, None] ** KAPPA
    th_p = np.gradient(np.asarray(tbar, float) * pk, p, axis=-2)
    th_p = np.where(np.abs(th_p) < 1e-7, -1e-7, th_p)
    iv = _cumtrapz_from_top(np.asarray(vbar, float), p)
    return 2 * np.pi * A_EARTH * c / G * (iv - np.asarray(vT, float) * pk / th_p)


def residual_velocities(psi, lat, plev):
    """(v* m/s, omega* Pa/s) implied by a residual streamfunction psi* (kg/s)
    on (..., plev, lat); they satisfy continuity by construction."""
    phi, _, cc = _geometry(lat)
    p = _p_axis(plev) * 100.0
    vstar = G / (2 * np.pi * A_EARTH * cc) * np.gradient(psi, p, axis=-2)
    omstar = -G / (2 * np.pi * A_EARTH ** 2 * cc) * np.gradient(psi, phi, axis=-1)
    return vstar, omstar


def downward_control(lat, plev, fhat, forcing, fhat_min: float = 1e-6) -> np.ndarray:
    """Steady downward control: psi* (kg/s) on (..., plev, lat) from a zonal
    forcing G (m/s^2) -- the EP-flux divergence, a parameterized drag, an
    analysis increment, minus du/dt, or their sum (the map is linear, so the
    parts add):  psi* = -(2 pi a cos phi / g) int_top^p G / fhat dp.
    NaN where |fhat| < fhat_min at or above p; see fill_tropics."""
    _, c, _ = _geometry(lat)
    p = _p_axis(plev) * 100.0
    fh = np.where(np.abs(fhat) < fhat_min, np.nan, fhat)
    return -(2 * np.pi * A_EARTH * c / G) * _cumtrapz_from_top(np.asarray(forcing, float) / fh, p)


def fill_tropics(psi, lat, edge: float = 15.0) -> np.ndarray:
    """Replace |lat| < edge (downward control undefined as fhat -> 0) by
    linear interpolation in sin(lat) between the grid latitudes nearest to
    -edge and +edge: the profile of uniform tropical upwelling. Any latitude
    order; lat is the last axis."""
    lat = np.asarray(lat, float)
    out = np.array(psi, float, copy=True)
    s = np.sin(np.deg2rad(lat))
    i0 = int(np.argmin(np.abs(lat + edge)))
    i1 = int(np.argmin(np.abs(lat - edge)))
    inside = (lat > lat[i0]) & (lat < lat[i1])
    w = (s[inside] - s[i0]) / (s[i1] - s[i0])
    out[..., inside] = out[..., [i0]] + (out[..., [i1]] - out[..., [i0]]) * w
    return out


@dataclass
class Upwelling:
    """Tropical upwelling between turnaround latitudes, per level.

    w     : mean log-pressure w* over the band (m/s)
    flux  : upward mass flux psi*(lat_n) - psi*(lat_s) (kg/s)
    lat_s, lat_n : the turnaround latitudes used (degrees)
    """
    w: np.ndarray
    flux: np.ndarray
    lat_s: np.ndarray
    lat_n: np.ndarray


def upwelling_w(psi, lat, plev, lat_s=None, lat_n=None, search=(15.0, 40.0),
                H: float = H_SCALE) -> Upwelling:
    """Tropical upward mass flux and mean w* = g H F / (2 pi a^2 p Dsin) from
    psi* (plev, lat). lat_s / lat_n: scalars or one per level; if None, each
    level's own extremum in `search` (NH maximum, SH minimum) -- biased high
    on noisy single days, see the module notes."""
    lat = np.asarray(lat, float)
    ph = _p_axis(plev)
    psi = np.asarray(psi, float).reshape(ph.size, lat.size)
    o = np.argsort(lat)
    la, ps = lat[o], psi[:, o]
    nb = (la >= search[0]) & (la <= search[1])
    sb = (la <= -search[0]) & (la >= -search[1])
    if lat_n is None:
        lat_n = la[nb][np.argmax(ps[:, nb], axis=1)]
    if lat_s is None:
        lat_s = la[sb][np.argmin(ps[:, sb], axis=1)]
    lat_n = np.broadcast_to(np.asarray(lat_n, float), ph.shape)
    lat_s = np.broadcast_to(np.asarray(lat_s, float), ph.shape)
    F = np.array([np.interp(n_, la, row) - np.interp(s_, la, row)
                  for row, s_, n_ in zip(ps, lat_s, lat_n)])
    ds = np.sin(np.deg2rad(lat_n)) - np.sin(np.deg2rad(lat_s))
    w = F * G * H / (ph * 100.0 * 2 * np.pi * A_EARTH ** 2 * ds)
    return Upwelling(w=w, flux=F, lat_s=np.array(lat_s), lat_n=np.array(lat_n))
