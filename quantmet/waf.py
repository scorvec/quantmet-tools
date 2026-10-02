"""Takaya-Nakamura (2001) horizontal wave-activity flux for quasi-stationary Rossby waves.

    W = p cos(phi) / (2 |U|) *
        [ U/(a^2 cos^2 phi) (psi_l^2 - psi psi_ll) + V/(a^2 cos phi) (psi_l psi_p - psi psi_lp) ,
          U/(a^2 cos phi)   (psi_l psi_p - psi psi_lp) + V/a^2 (psi_p^2 - psi psi_pp) ]

(TN01 eq. 38; l = longitude, p = latitude, in radians; psi the streamfunction ANOMALY from the basic state (U, V);
p as the fraction p/1000 hPa). The flux is parallel to the local group velocity of a stationary Rossby wave packet and
independent of the wave phase, so its convergence marks where packets deposit activity (downstream amplification).

Implementation notes that matter in practice:
- The cross term uses the plain mixed derivative psi_lp, exactly as eq. 38. Writing it as d/dy of
  psi_x = psi_l/(a cos phi) adds psi_l tan(phi)/a^2 -- a term oscillating at twice the wavenumber that breaks the
  phase independence (5-10 % of the cross term for synoptic waves at 45 deg).
- Longitude derivatives are periodic. On a closed grid (last column = 360 deg = the first) the difference is taken on
  the open circle and the seam column re-appended, so no one-sided error appears at Greenwich.
- W is quadratic in psi': for an ENSEMBLE average the per-member fluxes (``ensemble_flux``), never the flux of the
  ensemble-mean psi', which fades with lead time as members decorrelate.
- The flux is only defined in a westerly basic state: points with |U| below ``umin`` or equatorward of ``latmin``
  are masked.

Powers the wave-activity-flux loops at https://scorvec.com/circulation.html (AIFS-ENS 250 hPa, member-mean flux).

Reference: Takaya & Nakamura (2001, JAS 58, 608-627).

Requires: numpy only.
"""
from __future__ import annotations

import numpy as np

A_EARTH = 6.371e6
__all__ = ["dlon", "tn01_flux", "flux_divergence", "ensemble_flux"]


def dlon(f: np.ndarray, lonr: np.ndarray) -> np.ndarray:
    """Centred, periodic derivative along the last axis (longitude, radians); handles closed and open grids."""
    closed = np.isclose(np.rad2deg(lonr[-1] - lonr[0]), 360.0)
    g = f[..., :-1] if closed else f
    d = (np.roll(g, -1, axis=-1) - np.roll(g, 1, axis=-1)) / (2.0 * (lonr[1] - lonr[0]))
    return np.concatenate([d, d[..., :1]], axis=-1) if closed else d


def tn01_flux(psi_a: np.ndarray, U: np.ndarray, V: np.ndarray, lat: np.ndarray, lon: np.ndarray,
              p_hpa: float = 250.0, umin: float = 3.0, latmin: float = 20.0):
    """(Wx, Wy) in m^2 s^-2 from the streamfunction anomaly psi_a (m^2/s) and the basic-state wind (U, V), all
    (lat, lon) on one regular grid (lat in degrees, either order; lon in degrees, global). Masked (NaN) where
    |U| < umin or |lat| < latmin."""
    lat = np.asarray(lat, float)
    latr = np.deg2rad(lat); lonr = np.deg2rad(np.asarray(lon, float))
    cosp = np.clip(np.cos(latr), 1e-3, None)[:, None]
    a2 = A_EARTH * A_EARTH
    psi_a = np.asarray(psi_a, float)
    pl = dlon(psi_a, lonr)
    pp = np.gradient(psi_a, latr, axis=0)
    pll = dlon(pl, lonr)
    plp = np.gradient(pl, latr, axis=0)
    ppp = np.gradient(pp, latr, axis=0)
    spd = np.hypot(U, V)
    pref = (p_hpa / 1000.0) * cosp / (2.0 * np.maximum(spd, 1e-6))
    xx = pl * pl - psi_a * pll
    xy = pl * pp - psi_a * plp
    yy = pp * pp - psi_a * ppp
    wx = pref * (U * xx / (a2 * cosp ** 2) + V * xy / (a2 * cosp))
    wy = pref * (U * xy / (a2 * cosp) + V * yy / a2)
    bad = (spd < umin) | (np.abs(lat)[:, None] < latmin)
    wx = np.where(bad, np.nan, wx); wy = np.where(bad, np.nan, wy)
    return wx, wy


def flux_divergence(wx: np.ndarray, wy: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Spherical divergence of W (m s^-2); NaN inputs treated as zero, NaN where either input was NaN. Negative =
    convergence (wave activity piling up). Second derivatives of psi carry grid-scale ripple: smooth or spectrally
    truncate before display."""
    latr = np.deg2rad(np.asarray(lat, float)); lonr = np.deg2rad(np.asarray(lon, float))
    cosp = np.clip(np.cos(latr), 1e-3, None)[:, None]
    bad = ~(np.isfinite(wx) & np.isfinite(wy))
    wx0, wy0 = np.nan_to_num(wx), np.nan_to_num(wy)
    d = (dlon(wx0, lonr) + np.gradient(wy0 * cosp, latr, axis=0)) / (A_EARTH * cosp)
    return np.where(bad, np.nan, d)


def ensemble_flux(psi_members, U, V, lat, lon, **kw):
    """Mean of the per-member fluxes (psi_members iterates (lat, lon) streamfunction anomalies). Returns (Wx, Wy,
    agree) where agree is the fraction of members whose divergence has the sign of the mean divergence."""
    sx = sy = None
    divs = []
    n = 0
    for psi in psi_members:
        wx, wy = tn01_flux(psi, U, V, lat, lon, **kw)
        sx = wx if sx is None else sx + wx
        sy = wy if sy is None else sy + wy
        divs.append(flux_divergence(wx, wy, lat, lon))
        n += 1
    if n == 0:
        raise ValueError("no members supplied")
    wx, wy = sx / n, sy / n
    dm = flux_divergence(wx, wy, lat, lon)
    agree = np.mean([np.sign(d) == np.sign(dm) for d in divs], axis=0)
    return wx, wy, agree
