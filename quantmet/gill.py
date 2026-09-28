"""Gill (1980) steady tropical response to heating: a linear, damped
shallow-water model on the equatorial beta-plane, solved directly.

    eps u - y v + p_x = 0
    eps v + y u + p_y = 0            (nondimensional; Gill 1980, QJRMS)
    eps p + u_x + v_y = -Q

x, y are scaled by the equatorial radius of deformation L = sqrt(c / beta),
time by T = 1 / sqrt(c beta), velocity by c and p (a geopotential) by c^2.
(u, v, p) are the LOWER-tropospheric fields of the first baroclinic mode: a
positive heating Q gives low-level convergence and low pressure; the upper-
tropospheric wind is the negative of u, v. Rayleigh friction and Newtonian
cooling share one rate eps (Gill's choice; eps = 0.1 with c = 30 m/s is a
~4.4-day damping).

Numerical notes that matter in practice:
- DIRECT SOLVE, not time-stepping. The steady problem is one sparse linear
  system (3 N unknowns, centred differences on a collocated lat-lon grid,
  periodic in longitude, v = 0 at the two latitude walls), handed to
  scipy.sparse.linalg.spsolve. An explicitly time-stepped version was
  unstable: the Coriolis term y*u grows with |y|, and at the poleward rows
  of a +-30 deg domain it exceeds the stable step of any reasonable dt. The
  direct solve has no step, no spin-up and no convergence test; it is the
  exact steady state of the discrete equations.
- SPONGE. Damping rises linearly from eps to eps + sponge_damp between
  sponge_lat and sponge_lat + sponge_width degrees, so Rossby waves are
  absorbed before the walls instead of reflecting off them.
- CHECKERBOARD CONTROL. Centred differences on a collocated grid admit a
  2-dx mode that the pressure gradient cannot see; a weak del^4 term in x
  (coefficient `smooth`, in GRID units, eigenvalue 16*smooth on the 2-dx
  mode) removes it. At smooth = 0.02 a wave resolved by 20 points is damped
  at a rate of ~1e-4, far below eps. Because it is in grid units, halving
  the grid spacing weakens it 16x at a fixed physical scale.
- PERIODIC WRAP AT SMALL eps. Longitude is periodic, so the forced Kelvin
  wave travels round the equator and re-enters from the west, on top of
  the Rossby response there. Its e-folding length is 1/eps deformation
  radii (about 11,500 km at eps = 0.1, c = 30 m/s) against a circumference
  of ~35 radii (2 pi a / L), so the wrapped Kelvin wave arrives at
  exp(-35 eps) of its amplitude -- 3 % at eps = 0.1 -- where the Rossby
  signal has itself decayed: the measured westward decay rate 1-4 radii
  west of the heating is 20 % slow at eps = 0.1, 4-5 % at eps = 0.2 and
  ~2 % at eps = 0.3. It is the wrap, not the walls: moving the walls from
  +-30 to +-50 deg changes nothing, while shrinking L (c = 10 or 5 m/s,
  circumference 60 or 85 radii) cuts the eps = 0.1 error to 6 and 4 %.
  Quantitative decay-rate work wants eps >= 0.2 (or a smaller c); the
  pattern of the response is robust either way.
- TOTAL-SST HEATING (heating_from_ssta). Heating is proportional to the SST
  anomaly only where the TOTAL SST (climatology + anomaly) is warm enough to
  support deep convection, tapered over 26-28 C. Thresholding on the
  climatology instead leaves a warm eastern-Pacific cold tongue unheated;
  on the site's Walker product this one choice took the Gill El Nino
  response from about half the regressed amplitude to about 70 %.

Analytic checks (tests/test_gill.py), for Gill's heating exp(-y^2/4) F(x),
which projects only on the Kelvin wave and the n = 1 Rossby wave: east of
the heating the equatorial p decays as exp(-eps x) (the damped Kelvin wave,
exactly);
west of it the n = 1 Rossby wave decays as exp(kappa x) with
kappa = (sqrt(1/eps^2 + 4 (3 + eps^2)) - 1/eps) / 2 -- the damped
dispersion relation WITHOUT the long-wave approximation, because the model
keeps eps v. (The long-wave value 3 eps is 0.6 at eps = 0.2; the exact root
is 0.548.) For heating symmetric about the equator u and p are symmetric
and v antisymmetric.

Powers the idealized leg of the Walker-circulation-by-basin product at
https://scorvec.com/circulation.html.

Requires: numpy, scipy.
"""
from __future__ import annotations

import numpy as np

A_EARTH = 6.371e6
C_WAVE = 30.0            # m/s: first baroclinic mode, moist-reduced
BETA = 2.28e-11          # 1/(m s): 2 Omega / a
EPS = 0.1                # nondimensional damping (Gill 1980)
L_V = 2.5e6              # J/kg, latent heat of condensation
C_P = 1004.0             # J/(kg K)
G = 9.80665
W_PER_MM_DAY = L_V / 86400.0          # 1 mm/day of rain = 28.9 W/m^2 of column heating

__all__ = ["gill_scales", "gill_response", "rossby_decay_rate",
           "heating_from_ssta", "column_heating", "heating_from_precip",
           "to_dimensional", "band_mean", "C_WAVE", "BETA", "EPS",
           "W_PER_MM_DAY"]


def gill_scales(c: float = C_WAVE, beta: float = BETA):
    """(L, T): equatorial deformation radius (m) and time scale (s)."""
    return np.sqrt(c / beta), 1.0 / np.sqrt(c * beta)


def rossby_decay_rate(eps: float) -> float:
    """Westward e-folding rate (per deformation radius) of the steady damped
    n = 1 Rossby response, exact (no long-wave approximation). With damping
    eps on every equation the dispersion relation w^2 - k^2 - k/w = 3 at
    w = i eps and k = i kappa gives kappa^2 - kappa/eps - (3 + eps^2) = 0;
    the westward-decaying root is returned as a positive rate."""
    return 0.5 * (np.sqrt(1.0 / eps ** 2 + 4.0 * (3.0 + eps ** 2)) - 1.0 / eps)


def _periodic_laplacian(n: int):
    import scipy.sparse as sp
    L = sp.diags([np.ones(n - 1), -2 * np.ones(n), np.ones(n - 1)], [-1, 0, 1],
                 shape=(n, n)).tolil()
    L[0, n - 1] = 1
    L[n - 1, 0] = 1
    return L.tocsr()


def gill_response(Q: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                  eps: float = EPS, c: float = C_WAVE, beta: float = BETA,
                  sponge_lat: float = 24.0, sponge_width: float = 6.0,
                  sponge_damp: float = 0.6, smooth: float = 0.02):
    """Steady (u, v, p) for heating Q(lat, lon), all nondimensional.

    lat : ascending, uniformly spaced degrees (the walls are its first and
          last rows); lon : uniformly spaced degrees covering the full circle
          (periodic). Q is in the model's units (see heating_from_precip for
          a physical scaling; heating_from_ssta gives a shape only).
    eps : damping; c, beta : set the length scale L = sqrt(c/beta) (the
          nondimensional answer depends on them only through the grid).
    sponge_lat, sponge_width, sponge_damp : extra damping ramped linearly
          from 0 at |lat| = sponge_lat to sponge_damp at sponge_lat +
          sponge_width (and beyond).
    smooth : del^4 checkerboard control in x, in grid units (0 disables).

    Returns u, v, p with Q's shape. Multiply by to_dimensional for m/s and
    m^2/s^2."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    Q = np.nan_to_num(np.asarray(Q, float))
    ny, nx = Q.shape
    if (ny, nx) != (lat.size, lon.size):
        raise ValueError("Q must be (lat, lon)")
    if np.any(np.diff(lat) <= 0):
        raise ValueError("lat must be ascending")
    dlon = np.diff(lon)
    if not np.allclose(dlon, dlon[0]) or not np.isclose(nx * dlon[0], 360.0):
        raise ValueError("lon must be uniform and cover 360 degrees (periodic)")
    Lg, _ = gill_scales(c, beta)
    y = np.deg2rad(lat) * A_EARTH / Lg
    x = np.deg2rad(lon) * A_EARTH / Lg
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    N = ny * nx
    damp = eps + sponge_damp * np.clip((np.abs(lat) - sponge_lat) / sponge_width, 0, 1)

    Dx1 = sp.diags([np.ones(nx - 1), -np.ones(nx - 1)], [1, -1], shape=(nx, nx)).tolil()
    Dx1[0, nx - 1] = -1
    Dx1[nx - 1, 0] = 1
    Dx = sp.kron(sp.identity(ny), Dx1.tocsr() / (2 * dx), format="csr")
    Dy1 = (sp.diags([np.ones(ny - 1), -np.ones(ny - 1)], [1, -1], shape=(ny, ny)).tolil()
           / (2 * dy))
    Dy1[0, 0] = -1 / dy
    Dy1[0, 1] = 1 / dy
    Dy1[ny - 1, ny - 1] = 1 / dy
    Dy1[ny - 1, ny - 2] = -1 / dy
    Dy = sp.kron(Dy1.tocsr(), sp.identity(nx), format="csr")
    Yd = sp.diags(np.repeat(y, nx))
    Ed = sp.diags(np.repeat(damp, nx))
    Lx = sp.kron(sp.identity(ny), _periodic_laplacian(nx), format="csr")
    S = smooth * (Lx @ Lx)
    A = sp.bmat([[Ed + S, -Yd, Dx],
                 [Yd, Ed + S, Dy],
                 [Dx, Dy, Ed + S]], format="lil")
    b = np.concatenate([np.zeros(N), np.zeros(N), -Q.ravel()])
    for j in (0, ny - 1):                      # walls: v = 0
        for i in range(nx):
            r = N + j * nx + i
            A.rows[r] = [r]
            A.data[r] = [1.0]
            b[r] = 0.0
    sol = spla.spsolve(A.tocsc(), b)
    return (sol[:N].reshape(ny, nx), sol[N:2 * N].reshape(ny, nx),
            sol[2 * N:].reshape(ny, nx))


def to_dimensional(u, v, p, c: float = C_WAVE):
    """Nondimensional (u, v, p) -> (m/s, m/s, m^2/s^2): velocity scale c,
    geopotential scale c^2."""
    return np.asarray(u) * c, np.asarray(v) * c, np.asarray(p) * c * c


def heating_from_ssta(ssta: np.ndarray, sst_clim: np.ndarray, lat: np.ndarray,
                      t_lo: float = 26.0, t_hi: float = 28.0,
                      lat_edge: float = 22.0, lat_taper: float = 4.0,
                      smooth: bool = True) -> np.ndarray:
    """Heating SHAPE (units of the SST anomaly, K) on (lat, lon): the SST
    anomaly where the TOTAL SST (sst_clim + ssta, deg C) supports deep
    convection, ramped from 0 at t_lo to 1 at t_hi so there is no cliff;
    confined equatorward of lat_edge with a lat_taper-degree ramp; and, with
    `smooth`, a 1-2-1 filter in longitude (periodic). Using the total rather
    than the climatological SST lets a warm cold tongue convect -- the
    nonlinearity that makes a strong El Nino's response more than a scaled
    weak one. The amplitude is arbitrary: calibrate it (the site matches the
    ENSO regression) or use heating_from_precip for a physical scale."""
    ssta = np.nan_to_num(np.asarray(ssta, float))
    total = np.nan_to_num(np.asarray(sst_clim, float), nan=0.0) + ssta
    taper = np.clip((total - t_lo) / (t_hi - t_lo), 0, 1)
    latw = np.clip((lat_edge - np.abs(np.asarray(lat, float))) / lat_taper, 0, 1)[:, None]
    Q = ssta * taper * latw
    if smooth:
        Q = (Q + 0.5 * (np.roll(Q, 1, 1) + np.roll(Q, -1, 1))) / 2.0
    return Q


def column_heating(p_mm_day):
    """Column-integrated latent heating (W/m^2) of a rain rate (mm/day):
    L_v * P, i.e. 28.9 W/m^2 per mm/day. Over a 1000-100 hPa column that is
    a mass-weighted mean heating of 0.27 K/day per mm/day."""
    return np.asarray(p_mm_day, float) * W_PER_MM_DAY


def heating_from_precip(p_anom_mm_day, c: float = C_WAVE, beta: float = BETA,
                        depth: float = 15e3, dp: float = 9.0e4,
                        t_ref: float = 300.0, mode_factor: float = 1.0):
    """Gill heating Q (nondimensional, for gill_response) from a rain-rate
    anomaly (mm/day), by the latent-heat conversion and a first-baroclinic
    projection. Linear: Q = K * mode_factor * P with K ~ 0.033 per mm/day at
    the defaults.

    Derivation (documented so the scale can be checked or replaced):
      1. Column heating H = L_v P = 28.9 W/m^2 per mm/day (column_heating).
      2. Mass-weighted mean heating rate over the heated column of pressure
         depth dp: Tdot_mean = g H / (c_p dp)  (K/s; 0.27 K/day per mm/day
         for dp = 900 hPa).
      3. First baroclinic mode of a troposphere of height `depth`: heating
         sin(pi z/depth) with peak (pi/2) Tdot_mean, geopotential
         cos(pi z/depth). Projecting the thermodynamic equation
         d/dt(dphi/dz) + N^2 w = (g/t_ref) Tdot gives the shallow-water mass
         equation dphi/dt + c^2 div(u) = -Q_d with c = N depth / pi and
         Q_d = g depth Tdot_mean / (2 t_ref)   (m^2/s^3).
      4. Nondimensionalize with velocity c, length L = sqrt(c/beta), time
         T = 1/sqrt(c beta): Q = Q_d T / c^2.
    `mode_factor` is the fraction of the heating that projects on this mode:
    1 for the deep, sine-shaped profile of convective rain; less for
    stratiform or bottom-heavy heating (and part of that projection feeds
    higher modes the model does not carry). The same c sets both the length
    scale and the conversion, so choose it once for both calls. This is an
    order-of-magnitude physical scale, not a calibration; the site's Walker
    product calibrates amplitude against observed regressions."""
    _, T = gill_scales(c, beta)
    tdot_mean = G * column_heating(p_anom_mm_day) / (C_P * dp)
    Q_d = G * depth * tdot_mean / (2.0 * t_ref)
    return mode_factor * Q_d * T / (c * c)


def band_mean(f: np.ndarray, lat: np.ndarray, half: float = 5.0) -> np.ndarray:
    """cos(lat)-weighted mean of f(lat, lon) over |lat| <= half: the
    equatorial-band profile used for Walker-cell indices."""
    lat = np.asarray(lat, float)
    m = np.abs(lat) <= half
    w = np.cos(np.deg2rad(lat[m]))
    return (np.asarray(f)[m] * w[:, None]).sum(0) / w.sum()
