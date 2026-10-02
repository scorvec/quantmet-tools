"""Ertel potential vorticity on pressure levels, the dynamic tropopause, and isentropic PV.

    PV = -g [ (zeta + f) d(theta)/dp - (dv/dp)(d(theta)/dx) + (du/dp)(d(theta)/dy) ]      (PVU = 1e-6 K m^2 kg^-1 s^-1)

computed on the pressure grid from T, u, v snapshots (lev, lat, lon) on a regular, global-in-longitude
latitude-longitude grid. The dynamic tropopause (DT) is the 2-PVU surface; theta, p and the wind on it are found by
searching UPWARD from the lowest level and interpolating linearly in PV between the bracketing levels (p in ln p).
Isentropic PV interpolates linearly in theta.

Implementation notes that matter in practice:
- Pressure derivatives use np.gradient on the ACTUAL pressure levels (second order on a non-uniform grid);
  horizontal derivatives are centred, periodic in longitude.
- PV does not survive ensemble averaging: a mean of fifty tropopause folds is a smear, so compute it per member
  (or on one member) and never on the ensemble-mean fields.
- A light 3-point smoothing in each horizontal direction (``smooth=True``, the default) tames the grid-scale noise of
  zeta on a 0.25 deg grid; turn it off for coarse grids or analysis work.
- Where a column never reaches the threshold (the deep tropics, when the data stop at 100 hPa) the tropopause is at or
  above the top level. ``dynamic_tropopause`` then returns the TOP-level values (a lower bound on theta, flagged in
  ``never``) instead of a hole -- set ``fill_top=False`` to get NaN there.
- The upward search takes the FIRST crossing, so a stratospheric intrusion folded under tropospheric air (a
  tropopause fold) is found at its lower edge; that is the conventional DT choice.

Powers the dynamic-tropopause loops (theta on 2 PVU, PV on 330/350 K) at https://scorvec.com/circulation.html.

References: Hoskins, McIntyre & Robertson (1985, QJRMS); Morgan & Nielsen-Gammon (1998, MWR) on DT maps.

Requires: numpy only.
"""
from __future__ import annotations

import numpy as np

A_EARTH = 6.371e6
OMEGA = 7.2921e-5
G0 = 9.80665
KAPPA = 0.2857
P0 = 1.0e5
__all__ = ["potential_temperature", "ertel_pv", "dynamic_tropopause", "on_isentrope"]


def potential_temperature(t: np.ndarray, p_pa: np.ndarray) -> np.ndarray:
    """theta (K) from T (K) on levels p_pa (Pa) along axis 0."""
    p = np.asarray(p_pa, float).reshape((-1,) + (1,) * (np.ndim(t) - 1))
    return np.asarray(t, float) * (P0 / p) ** KAPPA


def ertel_pv(t: np.ndarray, u: np.ndarray, v: np.ndarray, p_pa: np.ndarray,
             lat: np.ndarray, lon: np.ndarray, smooth: bool = True):
    """Ertel PV (PVU) and theta (K) on the pressure grid for one time.

    t, u, v : (lev, lat, lon); p_pa : (lev,) in Pa (any order, monotonic); lat, lon in degrees, regular, lon global
    (periodic). Returns (pv, theta), both (lev, lat, lon). Southern-hemisphere PV is negative (f < 0)."""
    t = np.asarray(t, float); u = np.asarray(u, float); v = np.asarray(v, float)
    p_pa = np.asarray(p_pa, float)
    th = potential_temperature(t, p_pa)
    phi = np.deg2rad(np.asarray(lat, float)); lam = np.deg2rad(np.asarray(lon, float))
    cosphi = np.clip(np.cos(phi), 1e-6, None)[None, :, None]
    dlam = lam[1] - lam[0]

    def ddx(a):
        return (np.roll(a, -1, -1) - np.roll(a, 1, -1)) / (2 * dlam) / (A_EARTH * cosphi)

    def ddy(a):
        return np.gradient(a, phi, axis=1) / A_EARTH

    zeta = ddx(v) - ddy(u * cosphi) / cosphi
    f = (2 * OMEGA * np.sin(phi))[None, :, None]
    th_p = np.gradient(th, p_pa, axis=0)
    u_p = np.gradient(u, p_pa, axis=0)
    v_p = np.gradient(v, p_pa, axis=0)
    pv = -G0 * ((zeta + f) * th_p - v_p * ddx(th) + u_p * ddy(th)) * 1e6
    if smooth:
        pv = (pv + np.roll(pv, 1, -1) + np.roll(pv, -1, -1)) / 3.0
        pv[:, 1:-1] = (pv[:, :-2] + pv[:, 1:-1] + pv[:, 2:]) / 3.0
    return pv, th


def _bottom_up(p_pa):
    """Index order putting the highest pressure (lowest level) first."""
    return np.argsort(-np.asarray(p_pa, float))


def dynamic_tropopause(pv: np.ndarray, theta: np.ndarray, u: np.ndarray, v: np.ndarray,
                       p_pa: np.ndarray, thr: float = 2.0, fill_top: bool = True) -> dict:
    """theta, p (hPa), u, v on the ``thr``-PVU surface, searching upward from the lowest level.

    Use |PV| (``np.abs(pv)``) for a both-hemisphere search. Returns a dict with 'theta', 'p', 'u', 'v' (lat, lon) and
    'never' (bool: the column never reached thr; values there are the top level's if fill_top, else NaN)."""
    o = _bottom_up(p_pa)
    pv, th, u, v = (np.asarray(x, float)[o] for x in (pv, theta, u, v))
    p = np.asarray(p_pa, float)[o]
    above = pv >= thr
    nl = pv.shape[0]
    idx = np.argmax(above, axis=0)
    never = ~above.any(axis=0)
    k1 = np.clip(idx, 1, nl - 1); k0 = k1 - 1
    jj, ii = np.meshgrid(np.arange(pv.shape[1]), np.arange(pv.shape[2]), indexing="ij")
    pv0, pv1 = pv[k0, jj, ii], pv[k1, jj, ii]
    w = np.clip((thr - pv0) / np.where(np.abs(pv1 - pv0) < 1e-6, 1e-6, pv1 - pv0), 0, 1)
    w = np.where(idx == 0, 0.0, w)                       # already above thr at the lowest level: that level itself
    k0 = np.where(idx == 0, 0, k0); k1 = np.where(idx == 0, 0, k1)

    def interp(a):
        return a[k0, jj, ii] * (1 - w) + a[k1, jj, ii] * w

    lnp = np.log(p)[:, None, None] * np.ones_like(pv)
    out = {"theta": interp(th), "p": np.exp(interp(lnp)) / 100.0, "u": interp(u), "v": interp(v)}
    top = {"theta": th[-1], "p": np.full(th.shape[1:], p[-1] / 100.0), "u": u[-1], "v": v[-1]}
    for k in out:
        out[k] = np.where(never, top[k] if fill_top else np.nan, out[k])
    out["never"] = never
    return out


def on_isentrope(pv: np.ndarray, theta: np.ndarray, u: np.ndarray, v: np.ndarray, theta0: float) -> dict:
    """PV and wind interpolated linearly in theta onto the theta0 isentrope; NaN where theta0 is outside the column.
    Assumes theta increases monotonically with height (statically stable); levels in any order."""
    th = np.asarray(theta, float)
    o = np.argsort(th.mean(axis=(1, 2)))                 # theta ascending = upward
    pv, th, u, v = (np.asarray(x, float)[o] for x in (pv, th, u, v))
    nl = pv.shape[0]
    idx = np.clip(np.sum(th <= theta0, axis=0) - 1, 0, nl - 2)
    jj, ii = np.meshgrid(np.arange(pv.shape[1]), np.arange(pv.shape[2]), indexing="ij")
    t0, t1 = th[idx, jj, ii], th[idx + 1, jj, ii]
    w = np.clip((theta0 - t0) / np.where(np.abs(t1 - t0) < 1e-3, 1e-3, t1 - t0), 0, 1)
    valid = (th.min(axis=0) <= theta0) & (th.max(axis=0) >= theta0)

    def interp(a):
        return np.where(valid, a[idx, jj, ii] * (1 - w) + a[idx + 1, jj, ii] * w, np.nan)

    return {"pv": interp(pv), "u": interp(u), "v": interp(v)}
