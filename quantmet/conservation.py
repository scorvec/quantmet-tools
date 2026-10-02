"""Global conservation diagnostics for gridded model output: dry-air mass, the column water budget, column total
energy, and atmospheric angular momentum (AAM) with its mountain and friction torques.

Used to test whether data-driven forecast models conserve what the equations do (dry mass, water, energy, angular
momentum), against a physics model and a reanalysis: https://scorvec.com/aifs-aam-budget.html.

Grids: regular latitude-longitude, global; latitude in either order; integration weights are cos(phi) dlam dphi
(midpoint rule), which on a regular grid that includes the poles gives the pole rows zero weight, as it should.

Vertical integration on pressure levels: layer edges halfway between levels, the top edge at 0 hPa and the bottom
edge at 1100 hPa, then CLIPPED to the surface pressure p_s, so each column integrates exactly 0..p_s:
    dp_k = clip(min(edge_k+1, p_s) - min(edge_k, p_s), 0, inf).
Levels below the ground therefore get zero (or partial) weight -- the values a model extrapolates there never enter.
Stopping the top at the highest level instead (a common shortcut) drops the mass above it: 50 hPa of a 1000 hPa
column is 5 % of the mass.

Formulas (g = 9.80665, a = 6.371e6, Omega = 7.292115e-5):
  dry-air pressure      p_d = p_s - g * TCW                     (TCW: total column water, vapour + condensate, kg/m2)
  dry-air mass          M_d = (a^2/g) sum p_d cos(phi) dlam dphi          (kg; ~5.135e18)
  column water budget   dW/dt = E - P (no horizontal term globally) -> implied E = P + dW/dt
  column energy         sum_k (c_p T + Phi + L_v q [+ (u^2+v^2)/2]) dp_k / g       (J/m2; KE is ~0.05 % of the total)
  relative AAM          M_r = (a^3/g) sum u cos^2(phi) dp dlam dphi       (kg m2/s; 1 "Hadley" unit = 1e18)
  mass (Omega) AAM      M_O = (Omega a^4/g) sum p_s cos^3(phi) dlam dphi
  mountain torque       T_m = a^2 sum h (dp_s/dlam) cos(phi) dlam dphi    (= -sum p_s dh/dlam dA; N m, + = adds
                                                                            westerly AAM to the atmosphere)
  friction torque       T_f = -a^3 sum tau_x cos^2(phi) dlam dphi         (tau_x: surface stress ON the surface, i.e.
                                                                            the same sign as the wind)
Hemispheric splits give the equator row half to each hemisphere, so NH + SH = global exactly.

References: Trenberth & Smith (2005, J. Climate) on dry mass; Peixoto & Oort (1992); Egger, Weickmann &
Hoinka (2007, Rev. Geophys.) on AAM and torques.

Requires: numpy only.
"""
from __future__ import annotations

import numpy as np

A_EARTH = 6.371e6
G = 9.80665
OMEGA = 7.292115e-5
CP = 1004.7
LV = 2.501e6
__all__ = ["layer_thickness", "area_weights", "global_mean", "global_integral", "dry_air_pressure", "dry_air_mass",
           "implied_evaporation", "column_energy", "relative_aam", "mass_aam", "mountain_torque", "friction_torque"]


def layer_thickness(p_pa, ps, top_pa: float = 0.0, bottom_pa: float = 110000.0) -> np.ndarray:
    """Layer thickness dp (Pa) per level, (lev, ...ps.shape), with edges midway between levels, the outer edges at
    top_pa / bottom_pa, all clipped to the surface pressure ps (Pa). Levels in any order (output in the input order)."""
    p = np.asarray(p_pa, float)
    o = np.argsort(p); ps = np.asarray(ps, float)
    q = p[o]
    edges = np.empty(len(q) + 1)
    edges[1:-1] = 0.5 * (q[1:] + q[:-1])
    edges[0] = top_pa
    edges[-1] = bottom_pa
    sh = (-1,) + (1,) * ps.ndim
    lo = np.minimum(edges[:-1].reshape(sh), ps[None])
    hi = np.minimum(edges[1:].reshape(sh), ps[None])
    dp = np.clip(hi - lo, 0.0, None)
    out = np.empty_like(dp)
    out[o] = dp
    return out


def area_weights(lat, lon) -> np.ndarray:
    """cos(phi) dlam dphi (sr) per cell of a regular global grid, (lat, lon); sums to ~4 pi."""
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    dlam = np.deg2rad(abs(lon[1] - lon[0])); dphi = np.deg2rad(abs(lat[1] - lat[0]))
    return np.cos(np.deg2rad(lat))[:, None] * dlam * dphi * np.ones(len(lon))[None, :]


def _hemi(field_w, lat):
    lat = np.asarray(lat, float)
    row = field_w.sum(axis=-1)
    eq = np.isclose(lat, 0.0)
    wN = (lat > 0) + 0.5 * eq; wS = (lat < 0) + 0.5 * eq
    return {"global": float(row.sum()), "nh": float((row * wN).sum()), "sh": float((row * wS).sum())}


def global_integral(f, lat, lon, hemispheres: bool = False):
    """sum f cos(phi) dlam dphi (per steradian -> multiply by a^2 for m^2)."""
    fw = np.asarray(f, float) * area_weights(lat, lon)
    return _hemi(fw, lat) if hemispheres else float(fw.sum())


def global_mean(f, lat, lon) -> float:
    w = area_weights(lat, lon)
    return float((np.asarray(f, float) * w).sum() / w.sum())


def dry_air_pressure(ps, tcw):
    """p_s - g * TCW (Pa); TCW in kg/m2 (vapour + condensate: use tcw, not tcwv, where both exist)."""
    return np.asarray(ps, float) - G * np.asarray(tcw, float)


def dry_air_mass(ps, tcw, lat, lon, hemispheres: bool = False):
    """Dry-air mass (kg): (a^2/g) * integral of p_d."""
    r = global_integral(dry_air_pressure(ps, tcw), lat, lon, hemispheres)
    k = A_EARTH ** 2 / G
    return {h: v * k for h, v in r.items()} if hemispheres else r * k


def implied_evaporation(precip_mm, tcw_start, tcw_end, days: float = 1.0):
    """Global water budget E = P + dW/dt (no horizontal convergence globally), mm/day; precip_mm = precipitation over
    the interval (mm = kg/m2), tcw in kg/m2, all global means (or any consistent area means)."""
    return (np.asarray(precip_mm, float) + (np.asarray(tcw_end, float) - np.asarray(tcw_start, float))) / days


def column_energy(t, phi, q, p_pa, ps, u=None, v=None, top_pa: float = 0.0) -> dict:
    """Column-integrated energy components (J/m2), each (lat, lon): 'cpT', 'Phi' (geopotential, m2/s2 input), 'Lq',
    'KE' (if u and v are given) and 'total'. Inputs (lev, lat, lon) on p_pa (Pa); ps (lat, lon) in Pa."""
    dp = layer_thickness(p_pa, ps, top_pa=top_pa) / G                   # kg/m2 per layer
    out = {"cpT": (CP * np.asarray(t, float) * dp).sum(0), "Phi": (np.asarray(phi, float) * dp).sum(0),
           "Lq": (LV * np.asarray(q, float) * dp).sum(0)}
    if u is not None and v is not None:
        out["KE"] = (0.5 * (np.asarray(u, float) ** 2 + np.asarray(v, float) ** 2) * dp).sum(0)
    out["total"] = sum(out.values())
    return out


def relative_aam(u, p_pa, ps, lat, lon, hemispheres: bool = False):
    """M_r (kg m2/s) from zonal wind u (lev, lat, lon) on p_pa (Pa), surface pressure ps (Pa)."""
    lat = np.asarray(lat, float)
    uint = (np.asarray(u, float) * layer_thickness(p_pa, ps)).sum(0)    # (lat, lon), Pa m/s
    f = (A_EARTH ** 3 / G) * uint * np.cos(np.deg2rad(lat))[:, None]
    return global_integral(f, lat, lon, hemispheres)


def mass_aam(ps, lat, lon, hemispheres: bool = False):
    """M_O (kg m2/s): the angular momentum of the atmosphere's solid-body rotation with the Earth."""
    lat = np.asarray(lat, float)
    f = (OMEGA * A_EARTH ** 4 / G) * np.asarray(ps, float) * np.cos(np.deg2rad(lat))[:, None] ** 2
    return global_integral(f, lat, lon, hemispheres)


def mountain_torque(ps, h, lat, lon, hemispheres: bool = False):
    """T_m (N m) = a^2 sum h dp_s/dlam cos(phi) dlam dphi; h surface height (m); periodic centred dlam that follows
    the array's longitude direction."""
    lon = np.asarray(lon, float); ps = np.asarray(ps, float)
    dlam = np.deg2rad(abs(lon[1] - lon[0]))
    dpd = (np.roll(ps, -1, axis=-1) - np.roll(ps, 1, axis=-1)) / (2 * dlam) * np.sign(lon[1] - lon[0])
    return global_integral(A_EARTH ** 2 * np.asarray(h, float) * dpd, lat, lon, hemispheres)


def friction_torque(tau_x, lat, lon, hemispheres: bool = False):
    """T_f (N m) = -a^3 sum tau_x cos^2(phi) dlam dphi; tau_x eastward surface stress (N/m2), same sign as the wind."""
    lat = np.asarray(lat, float)
    f = -(A_EARTH ** 3) * np.asarray(tau_x, float) * np.cos(np.deg2rad(lat))[:, None]
    return global_integral(f, lat, lon, hemispheres)
