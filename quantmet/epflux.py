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
- The static stability d(theta_bar)/dp uses the GLOBAL-mean (area-weighted)
  theta profile, a function of pressure only. A local d(theta)/dp crosses
  zero in the troposphere and blows the heat-flux term up by many orders of
  magnitude; the global profile is the standard QG choice and is floored at
  -5e-5 K/Pa besides.
- cos(phi) is floored (85 deg) in the divergence *before* any smoothing.
  Smoothing first lets the polar 1/cos(phi) blow-up bleed equatorward and
  bury the real signal; the polar rows are masked in the returned force.
- Ensembles: the flux is quadratic in the eddies, so ensemble products must
  average PER-MEMBER fluxes (`ensemble_ep_flux`), never take the flux of the
  ensemble-mean fields -- ensemble averaging damps the eddies with lead time
  and the flux fades with them.

Powers the live E-P flux & wave-driving forecast loop on
https://scorvec.com/enso-atmosphere.html (AIFS-ENS, day 0-15).

Requires: numpy only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

A_EARTH = 6.371e6
OMEGA = 7.292e-5
KAPPA = 0.2854
P_REF = 1000.0            # hPa, for potential temperature
__all__ = ["EPFlux", "zonal_bandpass", "ep_flux", "ensemble_ep_flux"]


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


def zonal_bandpass(f: np.ndarray, kmax: int = 3, kmin: int = 1) -> np.ndarray:
    """Keep zonal wavenumbers kmin..kmax of f(..., lon) (last axis), zero the
    mean and everything outside the band. Exactly real on output."""
    F = np.fft.rfft(f, axis=-1)
    F[..., :kmin] = 0.0
    F[..., kmax + 1:] = 0.0
    return np.fft.irfft(F, n=f.shape[-1], axis=-1)


def _quadratics(u, v, t, plev, kmax):
    """Per-snapshot zonal-mean quadratics: [u'v'], [v'th'], theta_bar, ubar."""
    fac = (P_REF / np.asarray(plev, float))[:, None, None] ** KAPPA
    th = t * fac
    up = zonal_bandpass(u, kmax)
    vp = zonal_bandpass(v, kmax)
    thp = zonal_bandpass(th, kmax)
    return ((up * vp).mean(axis=-1), (vp * thp).mean(axis=-1),
            th.mean(axis=-1), u.mean(axis=-1))


def _assemble(uv, vth, thbar, ubar, lat, plev, polar_mask_deg):
    lat = np.asarray(lat, float)
    plev = np.asarray(plev, float)
    latr = np.deg2rad(lat)
    cosp = np.cos(latr)[None, :]
    f_cor = 2 * OMEGA * np.sin(latr)[None, :]
    p_pa = (plev * 100.0)[:, None]

    w = np.cos(latr)
    th_prof = (thbar * w[None, :]).sum(axis=1) / w.sum()
    dthdp = np.gradient(th_prof, p_pa[:, 0])[:, None]
    dthdp = np.where(dthdp > -5e-5, -5e-5, dthdp)

    fphi = -A_EARTH * cosp * uv
    fp = A_EARTH * cosp * f_cor * vth / dthdp

    cosp_safe = np.clip(cosp, np.cos(np.deg2rad(85.0)), None)
    div = (np.gradient(fphi * cosp, latr, axis=1) / (A_EARTH * cosp_safe)
           + np.gradient(fp, axis=0) / np.gradient(p_pa, axis=0))
    force = div / (A_EARTH * cosp_safe) * 86400.0
    force[:, np.abs(lat) > polar_mask_deg] = np.nan
    return EPFlux(fphi, fp, force, ubar, lat, plev)


def ep_flux(u: np.ndarray, v: np.ndarray, t: np.ndarray,
            lat: np.ndarray, plev: np.ndarray,
            kmax: int = 3, polar_mask_deg: float = 82.0) -> EPFlux:
    """QG E-P flux of zonal wavenumbers 1..kmax from one (plev, lat, lon)
    snapshot of u, v, T. plev in hPa, lat in degrees (any order); returns
    fields on the input (plev, lat) grid. No smoothing is applied -- display
    smoothing is the caller's business (the force field benefits from a light
    lat-smooth before contouring)."""
    uv, vth, thbar, ubar = _quadratics(np.asarray(u, float), np.asarray(v, float),
                                       np.asarray(t, float), plev, kmax)
    return _assemble(uv, vth, thbar, ubar, lat, plev, polar_mask_deg)


def ensemble_ep_flux(members, lat: np.ndarray, plev: np.ndarray,
                     kmax: int = 3, polar_mask_deg: float = 82.0) -> EPFlux:
    """Ensemble E-P flux: average the per-member QUADRATICS (equivalently the
    per-member fluxes -- both are linear in [u'v'], [v'theta']), then form the
    flux and divergence once. `members` iterates (u, v, t) triples of
    (plev, lat, lon) arrays; memory stays O(one member)."""
    n = 0
    uv_s = vth_s = th_s = ub_s = None
    for u, v, t in members:
        uv, vth, thbar, ubar = _quadratics(np.asarray(u, float),
                                           np.asarray(v, float),
                                           np.asarray(t, float), plev, kmax)
        if n == 0:
            uv_s, vth_s, th_s, ub_s = uv, vth, thbar, ubar
        else:
            uv_s += uv; vth_s += vth; th_s += thbar; ub_s += ubar
        n += 1
    if n == 0:
        raise ValueError("no members supplied")
    return _assemble(uv_s / n, vth_s / n, th_s / n, ub_s / n,
                     lat, plev, polar_mask_deg)
