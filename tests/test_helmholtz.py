import numpy as np
import pytest

pytest.importorskip("pyshtools")
xr = pytest.importorskip("xarray")

from quantmet.helmholtz import velocity_potential


def test_velocity_potential_of_known_field():
    A = 6.371e6
    C = 5e6
    lat = np.arange(-90, 90.01, 1.0)
    lon = np.arange(0, 360, 1.0)
    LA, LO = np.meshgrid(np.deg2rad(lat), np.deg2rad(lon), indexing="ij")
    u = (C * np.cos(LA) * -np.sin(LO) + C * np.cos(LA) ** 2 * np.sin(LA) * 2 * np.cos(2 * LO)) / (A * np.cos(LA))
    v = (C * -np.sin(LA) * np.cos(LO) + C * (np.cos(LA) ** 3 - 2 * np.cos(LA) * np.sin(LA) ** 2) * np.sin(2 * LO)) / A
    u[np.abs(lat) == 90] = 0

    def da(f):
        return xr.DataArray(f, dims=("latitude", "longitude"), coords=dict(latitude=lat, longitude=lon))

    chi, dl, dn = velocity_potential(da(u), da(v), lmax=63)
    DL, DN = np.meshgrid(np.deg2rad(dl), np.deg2rad(dn), indexing="ij")
    ref = C * np.cos(DL) * np.cos(DN) + C * np.cos(DL) ** 2 * np.sin(DL) * np.sin(2 * DN)
    assert np.sqrt(np.mean((chi - ref) ** 2)) < 5e-3 * np.sqrt(np.mean(ref**2))


A = 6.371e6
C = 5e6


def _analytic(lat, lon):
    """chi = C cos(phi) cos(lam) + C cos^2(phi) sin(phi) sin(2 lam) and its gradient (u = dchi/dlam / (a cos phi),
    v = dchi/dphi / a), written with cos(phi) cancelled so the poles are finite."""
    LA, LO = np.meshgrid(np.deg2rad(lat), np.deg2rad(lon), indexing="ij")
    chi = C * np.cos(LA) * np.cos(LO) + C * np.cos(LA) ** 2 * np.sin(LA) * np.sin(2 * LO)
    u = (-C * np.sin(LO) + 2 * C * np.cos(LA) * np.sin(LA) * np.cos(2 * LO)) / A
    v = (-C * np.sin(LA) * np.cos(LO) + C * (np.cos(LA) ** 3 - 2 * np.cos(LA) * np.sin(LA) ** 2) * np.sin(2 * LO)) / A
    return chi, u, v


def _da(f, lat, lon):
    return xr.DataArray(f, dims=("latitude", "longitude"), coords=dict(latitude=lat, longitude=lon))


@pytest.mark.parametrize("res,lmax", [(1.5, 120), (2.5, 120), (2.5, 63), (1.5, 63)])
def test_coarse_grids_at_high_lmax(res, lmax):
    """0.3.0 interpolated to the GLQ grid without a longitude wrap: on 1.5 and 2.5 deg grids at lmax = 120 the GLQ
    columns past the last input longitude were NaN and chi came out 100 % NaN."""
    lat = np.arange(90, -90 - 1e-9, -res)
    lon = np.arange(0, 360, res)  # north-to-south, like GRIB
    _, u, v = _analytic(lat, lon)
    chi, dl, dn = velocity_potential(_da(u, lat, lon), _da(v, lat, lon), lmax=lmax)
    assert np.isfinite(chi).all()
    ref, _, _ = _analytic(dl, dn)
    assert np.sqrt(np.mean((chi - ref) ** 2)) < 1e-2 * np.sqrt(np.mean(ref**2))


def test_offset_grid_without_poles_or_zero_meridian():
    """A 2 deg cell-centred grid (lat +-89, lon 1..359) at lmax 150: the GLQ grid reaches 89.1 deg and starts at
    lon 0, both outside the data unless it is wrapped and given pole rows."""
    lat = np.arange(-89, 89.1, 2.0)
    lon = np.arange(1, 360, 2.0)
    _, u, v = _analytic(lat, lon)
    chi, dl, dn = velocity_potential(_da(u, lat, lon), _da(v, lat, lon), lmax=150)
    assert np.isfinite(chi).all()
    ref, _, _ = _analytic(dl, dn)
    assert np.sqrt(np.mean((chi - ref) ** 2)) < 1e-2 * np.sqrt(np.mean(ref**2))


def test_irrotational_wind_is_periodic_at_the_seam():
    from quantmet.helmholtz import irrotational_wind

    lat = np.arange(-90, 90.01, 1.0)
    lon = np.arange(0, 360, 1.0)
    _, u, v = _analytic(lat, lon)
    chi, dl, dn = velocity_potential(_da(u, lat, lon), _da(v, lat, lon), lmax=63)
    assert np.isclose(dn[-1], 360.0) and np.allclose(chi[:, 0], chi[:, -1])  # extended DH2 grid
    uc, _vc = irrotational_wind(chi, dl, dn)
    _, ua, _va = _analytic(dl, dn)
    rows = np.abs(dl) < 80
    err = np.abs(uc - ua)[rows]
    scale = np.abs(ua[rows]).max()
    assert err[:, [0, -1]].max() < 2 * err[:, 1:-1].max() + 1e-4 * scale  # seam no worse than the interior
    assert err.max() < 1e-2 * scale
    assert np.allclose(uc[:, 0], uc[:, -1])


def test_streamfunction_on_a_coarse_grid():
    """Solid-body rotation u = U cos(phi): psi = -U a sin(phi) (+ const); 2.5 deg input at lmax 120."""
    from quantmet.helmholtz import streamfunction

    res = 2.5
    lat = np.arange(90, -90 - 1e-9, -res)
    lon = np.arange(0, 360, res)
    U = 20.0
    u = U * np.cos(np.deg2rad(lat))[:, None] * np.ones(len(lon))
    v = np.zeros_like(u)
    psi, dl, dn = streamfunction(_da(u, lat, lon), _da(v, lat, lon), lmax=120)
    assert np.isfinite(psi).all()
    ref = -U * A * np.sin(np.deg2rad(dl))[:, None] * np.ones(len(dn))
    assert np.sqrt(np.mean((psi - ref) ** 2)) < 2e-2 * np.sqrt(np.mean(ref**2))


def test_tc_irrotational_on_grid_at_2p5_deg():
    from quantmet.tc import irrotational_on_grid

    res = 2.5
    lat = np.arange(90, -90 - 1e-9, -res)
    lon = np.arange(0, 360, res)
    _, u, v = _analytic(lat, lon)
    tl = np.arange(-60, 60.1, 5.0)
    tn = np.arange(0, 360, 5.0)
    uc, vc = irrotational_on_grid(_da(u, lat, lon), _da(v, lat, lon), tl, tn, lmax=106)
    assert np.isfinite(uc).all() and np.isfinite(vc).all()
    _, ua, va = _analytic(tl, tn)
    assert np.abs(uc - ua).max() < 2e-2 * np.abs(ua).max()
    assert np.abs(vc - va).max() < 2e-2 * np.abs(va).max()
