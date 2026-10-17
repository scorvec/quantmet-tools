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
- The input is wrapped in longitude (a column at lon + 360 and one at
  lon - 360) and, if it lacks them, given pole rows before it is interpolated
  to the GLQ grid, so any regular global grid works at any lmax. (Before
  0.3.1 a 1.5 or 2.5 deg grid at lmax = 120 left the GLQ columns past its
  last longitude NaN, and chi came out entirely NaN.)
- Longitude derivatives are periodic centred differences on every global
  grid (GLQ and the extended DH2 output, whose 360 column repeats 0); before
  0.3.1 np.gradient's one-sided differences were used at the seam columns.

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

from .constants import A_EARTH

__all__ = ["velocity_potential", "streamfunction", "irrotational_wind", "nondivergent_wind"]


def _to_0360(da: xr.DataArray) -> xr.DataArray:
    """Longitudes mapped to [0, 360) and sorted (a repeated seam column dropped); latitudes ascending."""
    lon = np.mod(da["longitude"].values.astype(float), 360.0)
    da = da.assign_coords(longitude=lon).sortby("longitude")
    keep = np.r_[True, np.diff(da["longitude"].values) > 1e-9]
    return da.isel(longitude=np.flatnonzero(keep)).sortby("latitude")


def _pole_row(row: np.ndarray) -> np.ndarray:
    """A wind component at a pole: only zonal wavenumber 1 of the nearest row survives (a cross-polar flow seen in
    local east/north components), so that is what the pole row gets."""
    c = np.fft.rfft(row)
    keep = np.zeros_like(c)
    keep[1:2] = c[1:2]
    return np.fft.irfft(keep, n=row.size)


def _padded(da: xr.DataArray) -> xr.DataArray:
    """Wrap one column each side in longitude, and add pole rows if the grid stops short of the poles, so that
    interpolation to any GLQ grid stays inside the data."""
    lon = da["longitude"].values
    da = xr.concat(
        [
            da.isel(longitude=[-1]).assign_coords(longitude=[lon[-1] - 360.0]),
            da,
            da.isel(longitude=[0]).assign_coords(longitude=[lon[0] + 360.0]),
        ],
        dim="longitude",
    )
    da = da.transpose("latitude", "longitude")
    lat = da["latitude"].values
    rows, lats = [], []
    if lat[0] > -90.0 + 1e-9:
        rows.append(_pole_row(da.values[0])[None])
        lats.append([-90.0])
    rows.append(da.values)
    lats.append(lat)
    if lat[-1] < 90.0 - 1e-9:
        rows.append(_pole_row(da.values[-1])[None])
        lats.append([90.0])
    return xr.DataArray(
        np.concatenate(rows, 0),
        dims=("latitude", "longitude"),
        coords={"latitude": np.concatenate(lats), "longitude": da["longitude"].values},
    )


def _ddlon(f: np.ndarray, lon_deg) -> np.ndarray:
    """d f / d lambda (per radian) along the last axis: periodic centred differences on a global uniform grid (with
    or without a repeated 360 column), np.gradient on a regional one."""
    lon = np.asarray(lon_deg, float)
    d = np.diff(lon)
    dl = d.mean()
    if np.allclose(d, dl, rtol=1e-6, atol=1e-9):
        dlr = np.deg2rad(dl)
        span = lon[-1] - lon[0]
        if np.isclose(span, 360.0, atol=1e-6):  # repeated seam column (DH2 extended grid)
            core = f[..., :-1]
            g = (np.roll(core, -1, axis=-1) - np.roll(core, 1, axis=-1)) / (2 * dlr)
            return np.concatenate([g, g[..., :1]], axis=-1)
        if np.isclose(span + dl, 360.0, atol=1e-6):  # full circle, no repeat (GLQ)
            return (np.roll(f, -1, axis=-1) - np.roll(f, 1, axis=-1)) / (2 * dlr)
    return np.gradient(f, np.deg2rad(lon), axis=-1)


def _laplace_invert(field: np.ndarray, lmax: int) -> pysh.SHGrid:
    clm = pysh.SHGrid.from_array(field, grid="GLQ").expand()
    l = np.arange(clm.lmax + 1, dtype=float)
    fac = np.zeros_like(l)
    fac[1:] = -(A_EARTH**2) / (l[1:] * (l[1:] + 1))
    out = clm.copy()
    out.coeffs *= fac[None, :, None]
    return out.expand(grid="DH2")


def _grid_uv(u2d: xr.DataArray, v2d: xr.DataArray, lmax: int):
    u2d, v2d = _padded(_to_0360(u2d)), _padded(_to_0360(v2d))
    glat, glon = pysh.expand.GLQGridCoord(lmax)
    ug = u2d.interp(latitude=glat, longitude=glon).values
    vg = v2d.interp(latitude=glat, longitude=glon).values
    return ug, vg, np.deg2rad(glat), np.deg2rad(glon)


def velocity_potential(u2d: xr.DataArray, v2d: xr.DataArray, lmax: int = 120):
    """chi (m^2/s) with its output-grid lats/lons: inverts del^2 chi = div."""
    ug, vg, latr, lonr = _grid_uv(u2d, v2d, lmax)
    cosp = np.cos(latr)[:, None]
    div = (_ddlon(ug, np.rad2deg(lonr)) + np.gradient(vg * cosp, latr, axis=0)) / (A_EARTH * cosp)
    g = _laplace_invert(div, lmax)
    return g.data, np.array(g.lats()), np.array(g.lons())


def streamfunction(u2d: xr.DataArray, v2d: xr.DataArray, lmax: int = 120):
    """psi (m^2/s): inverts del^2 psi = zeta (relative vorticity)."""
    ug, vg, latr, lonr = _grid_uv(u2d, v2d, lmax)
    cosp = np.cos(latr)[:, None]
    zeta = (_ddlon(vg, np.rad2deg(lonr)) - np.gradient(ug * cosp, latr, axis=0)) / (A_EARTH * cosp)
    g = _laplace_invert(zeta, lmax)
    return g.data, np.array(g.lats()), np.array(g.lons())


def irrotational_wind(chi: np.ndarray, dlat: np.ndarray, dlon: np.ndarray):
    """(u_chi, v_chi) = grad(chi) on the chi grid (periodic in longitude on a global grid)."""
    latr = np.deg2rad(dlat)
    cosp = np.clip(np.cos(latr), 1e-3, None)[:, None]
    return (_ddlon(chi, dlon) / (A_EARTH * cosp), np.gradient(chi, latr, axis=0) / A_EARTH)


def nondivergent_wind(psi: np.ndarray, dlat: np.ndarray, dlon: np.ndarray):
    """(u_psi, v_psi) = k x grad(psi) on the psi grid (periodic in longitude on a global grid)."""
    latr = np.deg2rad(dlat)
    cosp = np.clip(np.cos(latr), 1e-3, None)[:, None]
    return (-np.gradient(psi, latr, axis=0) / A_EARTH, _ddlon(psi, dlon) / (A_EARTH * cosp))
