"""Global Helmholtz decomposition by spherical-harmonic Poisson inversion.

Given u, v on the sphere, compute divergence and vorticity, invert the
Laplacian spectrally (chi_lm = -a^2 delta_lm / l(l+1); psi_lm likewise from
vorticity), and return velocity potential / streamfunction plus the
irrotational (divergent) and nondivergent wind components.

Implementation notes that matter in practice:
- Wind is sampled on a Gauss-Legendre (GLQ) grid before differentiating, so
  there are no pole points and no 1/cos(phi) blowups in the divergence.
- The l=0 mode is excluded (undefined for the inverse Laplacian).
- cos(phi) is clipped when evaluating grad(chi) near the poles of the
  output Driscoll-Healy grid.

Powers the Walker-circulation monitor at https://scorvec.com/circulation.html
(velocity potential of the 200 hPa divergent wind, and the zonal mass
streamfunction built from the divergent zonal wind), and supplies the
streamfunction for the wave-activity flux (quantmet.waf).

Accuracy: vorticity and divergence are second-order finite differences on
the GLQ grid, so the inversion error falls roughly as lmax^-1.7 (solid-body
rotation: 2.5 % at lmax = 31, 0.8 % at 63, 0.24 % at 127).

The output grid is pyshtools' extended Driscoll-Healy grid: it includes
both poles and repeats longitude 0 at 360.

Requires: pyshtools, xarray.
"""
from __future__ import annotations

import numpy as np
import xarray as xr
import pyshtools as pysh

A_EARTH = 6.371e6
__all__ = ["velocity_potential", "streamfunction", "irrotational_wind",
           "nondivergent_wind"]


def _to_0360(da: xr.DataArray) -> xr.DataArray:
    lon = da["longitude"].values
    if lon.min() < 0:
        da = da.assign_coords(longitude=np.where(lon < 0, lon + 360, lon)).sortby("longitude")
    return da.sortby("latitude")


def _laplace_invert(field: np.ndarray, lmax: int) -> "pysh.SHGrid":
    clm = pysh.SHGrid.from_array(field, grid="GLQ").expand()
    l = np.arange(clm.lmax + 1, dtype=float)
    fac = np.zeros_like(l)
    fac[1:] = -(A_EARTH ** 2) / (l[1:] * (l[1:] + 1))
    out = clm.copy()
    out.coeffs *= fac[None, :, None]
    return out.expand(grid="DH2")


def _grid_uv(u2d: xr.DataArray, v2d: xr.DataArray, lmax: int):
    u2d, v2d = _to_0360(u2d), _to_0360(v2d)
    glat, glon = pysh.expand.GLQGridCoord(lmax)
    ug = u2d.interp(latitude=glat, longitude=glon).transpose("latitude", "longitude").values
    vg = v2d.interp(latitude=glat, longitude=glon).transpose("latitude", "longitude").values
    return ug, vg, np.deg2rad(glat), np.deg2rad(glon)


def velocity_potential(u2d: xr.DataArray, v2d: xr.DataArray, lmax: int = 120):
    """chi (m^2/s) with its output-grid lats/lons: inverts del^2 chi = div."""
    ug, vg, latr, lonr = _grid_uv(u2d, v2d, lmax)
    cosp = np.cos(latr)[:, None]
    div = (np.gradient(ug, lonr, axis=1) + np.gradient(vg * cosp, latr, axis=0)) / (A_EARTH * cosp)
    g = _laplace_invert(div, lmax)
    return g.data, np.array(g.lats()), np.array(g.lons())


def streamfunction(u2d: xr.DataArray, v2d: xr.DataArray, lmax: int = 120):
    """psi (m^2/s): inverts del^2 psi = zeta (relative vorticity)."""
    ug, vg, latr, lonr = _grid_uv(u2d, v2d, lmax)
    cosp = np.cos(latr)[:, None]
    zeta = (np.gradient(vg, lonr, axis=1) - np.gradient(ug * cosp, latr, axis=0)) / (A_EARTH * cosp)
    g = _laplace_invert(zeta, lmax)
    return g.data, np.array(g.lats()), np.array(g.lons())


def irrotational_wind(chi: np.ndarray, dlat: np.ndarray, dlon: np.ndarray):
    """(u_chi, v_chi) = grad(chi) on the chi grid."""
    latr, lonr = np.deg2rad(dlat), np.deg2rad(dlon)
    cosp = np.clip(np.cos(latr), 1e-3, None)[:, None]
    return (np.gradient(chi, lonr, axis=1) / (A_EARTH * cosp),
            np.gradient(chi, latr, axis=0) / A_EARTH)


def nondivergent_wind(psi: np.ndarray, dlat: np.ndarray, dlon: np.ndarray):
    """(u_psi, v_psi) = k x grad(psi) on the psi grid."""
    latr, lonr = np.deg2rad(dlat), np.deg2rad(dlon)
    cosp = np.clip(np.cos(latr), 1e-3, None)[:, None]
    return (-np.gradient(psi, latr, axis=0) / A_EARTH,
            np.gradient(psi, lonr, axis=1) / (A_EARTH * cosp))
