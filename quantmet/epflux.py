"""Quasi-geostrophic Eliassen-Palm flux and its divergence on pressure levels.

Given u, v, T snapshots on (plev, lat, lon), compute the QG E-P flux of a
chosen planetary-wavenumber band and the flux divergence expressed as a zonal
force per unit mass (m/s/day) -- the wave driving of the zonal-mean flow:

    F_phi = -a cos(phi) [u'v']
    F_p   =  a cos(phi) f [v'theta'] / (d theta_bar / dp)
    force = (div F) / (a cos(phi))

with primes the deviation from the zonal mean band-passed to zonal
wavenumbers 1..kmax and [.] the zonal mean (Edmon, Hoskins & McIntyre 1980).

Implementation notes that matter in practice:
- Static stability (``theta_p``): the default since v0.2 is the ZONAL-MEAN
  profile d[theta]/dp (phi, p). The GLOBAL-mean profile (``theta_p="global"``,
  the v0.1 behaviour and the textbook QG choice) carries tropospheric
  stability above the polar tropopause, 3-4x too small there, which inflates
  F_p and its p-derivative at 250-300 hPa poleward of ~70 deg; against an
  analysis primitive-equation E-P budget the zonal profile raised the
  stratospheric pattern correlation (SH 0.83 -> 0.87, NH 0.69 -> 0.81). Either
  way it is floored at -5e-5 K/Pa (never neutral or unstable): a POINTWISE
  d(theta)/dp would cross zero in the troposphere and blow the term up.
- Vertical derivatives are taken in ln p on the ACTUAL levels,
  d/dp = (1/p) d/d(ln p): accurate on log-spaced stratospheric levels
  (1, 2, 3, 5, 7, 10 ... hPa) where a centred difference in p is lopsided.
- Below ground (``psfc``): models extrapolate fields under the surface. Where
  more than ``bg_frac`` of a latitude circle's longitudes lie below ground at a
  level, the eddy fluxes are set to NaN BEFORE any derivative (so the level
  just above, whose centred difference would reach into extrapolated data,
  goes too): Antarctica, Greenland, Tibet, the Andes.
- cos(phi) is floored (85 deg) in the divergence *before* any smoothing.
  Smoothing first lets the polar 1/cos(phi) blow-up bleed equatorward and
  bury the real signal; the polar rows are masked in the returned force.
- Ensembles: the flux is quadratic in the eddies, so ensemble products must
  average PER-MEMBER fluxes (`ensemble_ep_flux`), never take the flux of the
  ensemble-mean fields -- ensemble averaging damps the eddies with lead time
  and the flux fades with them.

Powers the live E-P flux & wave-driving forecast loop on
https://scorvec.com/stratosphere.html (GEFS, 31 levels 1-1000 hPa, day 0-15).

Requires: numpy (scipy for smooth_deg).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

A_EARTH = 6.371e6
OMEGA = 7.292e-5
KAPPA = 0.2854
P_REF = 1000.0            # hPa, for potential temperature
__all__ = ["EPFlux", "zonal_bandpass", "ep_flux", "ensemble_ep_flux", "below_ground_share"]


@dataclass
class EPFlux:
    """E-P flux and derived wave driving on the (plev, lat) plane.

    fphi   : meridional component, -a cos(phi) [u'v']            (m^3 s^-2)
    fp     : vertical (pressure) component, a cos f [v'th']/th_p  (m^3 s^-2 Pa/m)
    force  : (div F)/(a cos(phi)) as m/s per DAY; polar rows (|lat| > 82) NaN
    ubar   : zonal-mean zonal wind (m/s)
    lat    : degrees; plev: hPa (as passed in)
    """
    fphi: np.ndarray
    fp: np.ndarray
    force: np.ndarray
    ubar: np.ndarray
    lat: np.ndarray
    plev: np.ndarray
    below: np.ndarray | None = None


def zonal_bandpass(f: np.ndarray, kmax: int = 3, kmin: int = 1) -> np.ndarray:
    """Keep zonal wavenumbers kmin..kmax of f(..., lon) (last axis), zero the
    mean and everything outside the band. Exactly real on output."""
    F = np.fft.rfft(f, axis=-1)
    F[..., :kmin] = 0.0
    F[..., kmax + 1:] = 0.0
    return np.fft.irfft(F, n=f.shape[-1], axis=-1)


def below_ground_share(psfc_pa: np.ndarray, plev: np.ndarray) -> np.ndarray:
    """(plev, lat) share of longitudes where the level lies below the surface (p_level > p_s); psfc_pa (lat, lon) in Pa,
    plev in hPa."""
    p = np.asarray(plev, float)[:, None, None] * 100.0
    return (p > np.asarray(psfc_pa, float)[None]).mean(axis=-1)


def _nan_smooth_lat(f, sigma_pts):
    """Gaussian along latitude (axis 1) ignoring NaN (normalised convolution); NaN stays NaN."""
    from scipy.ndimage import gaussian_filter1d
    ok = np.isfinite(f)
    num = gaussian_filter1d(np.where(ok, f, 0.0), sigma_pts, axis=1, mode="nearest")
    den = gaussian_filter1d(ok.astype(float), sigma_pts, axis=1, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[~ok] = np.nan
    return out


def _quadratics(u, v, t, plev, kmax):
    """Per-snapshot zonal-mean quadratics: [u'v'], [v'th'], theta_bar, ubar."""
    fac = (P_REF / np.asarray(plev, float))[:, None, None] ** KAPPA
    th = t * fac
    up = zonal_bandpass(u, kmax)
    vp = zonal_bandpass(v, kmax)
    thp = zonal_bandpass(th, kmax)
    return ((up * vp).mean(axis=-1), (vp * thp).mean(axis=-1),
            th.mean(axis=-1), u.mean(axis=-1))


def _assemble(uv, vth, thbar, ubar, lat, plev, polar_mask_deg, theta_p="zonal", below=None,
              bg_frac=0.10, smooth_deg=0.0):
    lat = np.asarray(lat, float)
    plev = np.asarray(plev, float)
    if theta_p not in ("zonal", "global"):
        raise ValueError("theta_p must be 'zonal' or 'global'")
    latr = np.deg2rad(lat)
    cosp = np.cos(latr)[None, :]
    f_cor = 2 * OMEGA * np.sin(latr)[None, :]
    p = plev * 100.0
    lnp = np.log(p)

    w = np.cos(latr)
    if theta_p == "zonal":
        dthdp = np.gradient(thbar, lnp, axis=0) / p[:, None]
    else:
        th_prof = (thbar * w[None, :]).sum(axis=1) / w.sum()
        dthdp = (np.gradient(th_prof, lnp) / p)[:, None]
    dthdp = np.where(dthdp > -5e-5, -5e-5, dthdp)

    if below is not None:
        bg = np.asarray(below) > bg_frac
        uv = np.where(bg, np.nan, uv)
        vth = np.where(bg, np.nan, vth)

    fphi = -A_EARTH * cosp * uv
    fp = A_EARTH * cosp * f_cor * vth / dthdp

    cosp_safe = np.clip(cosp, np.cos(np.deg2rad(85.0)), None)
    div = (np.gradient(fphi * cosp, latr, axis=1) / (A_EARTH * cosp_safe)
           + np.gradient(fp, lnp, axis=0) / p[:, None])
    force = div / (A_EARTH * cosp_safe) * 86400.0
    force[:, np.abs(lat) > polar_mask_deg] = np.nan        # before smoothing: nothing leaks in from the pole
    if smooth_deg and smooth_deg > 0:
        force = _nan_smooth_lat(force, smooth_deg / float(np.abs(np.diff(lat)).mean()))
    return EPFlux(fphi, fp, force, ubar, lat, plev, below)


def ep_flux(u: np.ndarray, v: np.ndarray, t: np.ndarray,
            lat: np.ndarray, plev: np.ndarray,
            kmax: int = 3, polar_mask_deg: float = 82.0, theta_p: str = "zonal",
            psfc: np.ndarray | None = None, bg_frac: float = 0.10,
            smooth_deg: float = 0.0) -> EPFlux:
    """QG E-P flux of zonal wavenumbers 1..kmax from one (plev, lat, lon)
    snapshot of u, v, T. plev in hPa (monotonic), lat in degrees (any order);
    returns fields on the input (plev, lat) grid.

    theta_p   : "zonal" (default) or "global" static stability (see module notes)
    psfc      : surface pressure (lat, lon) in Pa -> below-ground masking
    smooth_deg: NaN-aware Gaussian smoothing of the force along latitude only
                (degrees; 0 = none, the v0.1 behaviour). The polar rows are
                blanked first, so nothing leaks in from the pole."""
    uv, vth, thbar, ubar = _quadratics(np.asarray(u, float), np.asarray(v, float),
                                       np.asarray(t, float), plev, kmax)
    below = below_ground_share(psfc, plev) if psfc is not None else None
    return _assemble(uv, vth, thbar, ubar, lat, plev, polar_mask_deg, theta_p, below, bg_frac, smooth_deg)


def ensemble_ep_flux(members, lat: np.ndarray, plev: np.ndarray,
                     kmax: int = 3, polar_mask_deg: float = 82.0, theta_p: str = "zonal",
                     bg_frac: float = 0.10, smooth_deg: float = 0.0) -> EPFlux:
    """Ensemble E-P flux: average the per-member QUADRATICS (equivalently the
    per-member fluxes -- both are linear in [u'v'], [v'theta']), then form the
    flux and divergence once. `members` iterates (u, v, t) triples of
    (plev, lat, lon) arrays, or (u, v, t, psfc) quadruples for below-ground
    masking (the member-mean below-ground share is compared with bg_frac);
    memory stays O(one member)."""
    n = 0
    uv_s = vth_s = th_s = ub_s = bg_s = None
    for mem in members:
        u, v, t = mem[:3]
        psfc = mem[3] if len(mem) > 3 else None
        uv, vth, thbar, ubar = _quadratics(np.asarray(u, float),
                                           np.asarray(v, float),
                                           np.asarray(t, float), plev, kmax)
        bg = below_ground_share(psfc, plev) if psfc is not None else None
        if n == 0:
            uv_s, vth_s, th_s, ub_s, bg_s = uv, vth, thbar, ubar, bg
        else:
            uv_s += uv; vth_s += vth; th_s += thbar; ub_s += ubar
            if bg is not None and bg_s is not None:
                bg_s = bg_s + bg
        n += 1
    if n == 0:
        raise ValueError("no members supplied")
    return _assemble(uv_s / n, vth_s / n, th_s / n, ub_s / n,
                     lat, plev, polar_mask_deg, theta_p,
                     None if bg_s is None else bg_s / n, bg_frac, smooth_deg)
