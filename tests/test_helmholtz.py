import numpy as np
import pytest

pytest.importorskip("pyshtools")
xr = pytest.importorskip("xarray")

from quantmet.helmholtz import velocity_potential  # noqa: E402


def test_velocity_potential_of_known_field():
    A = 6.371e6
    C = 5e6
    lat = np.arange(-90, 90.01, 1.0)
    lon = np.arange(0, 360, 1.0)
    LA, LO = np.meshgrid(np.deg2rad(lat), np.deg2rad(lon), indexing="ij")
    u = (C * np.cos(LA) * -np.sin(LO) + C * np.cos(LA) ** 2 * np.sin(LA) * 2 * np.cos(2 * LO)) / (A * np.cos(LA))
    v = (C * -np.sin(LA) * np.cos(LO) + C * (np.cos(LA) ** 3 - 2 * np.cos(LA) * np.sin(LA) ** 2) * np.sin(2 * LO)) / A
    u[np.abs(lat) == 90] = 0
    da = lambda f: xr.DataArray(f, dims=("latitude", "longitude"), coords=dict(latitude=lat, longitude=lon))  # noqa: E731
    chi, dl, dn = velocity_potential(da(u), da(v), lmax=63)
    DL, DN = np.meshgrid(np.deg2rad(dl), np.deg2rad(dn), indexing="ij")
    ref = C * np.cos(DL) * np.cos(DN) + C * np.cos(DL) ** 2 * np.sin(DL) * np.sin(2 * DN)
    assert np.sqrt(np.mean((chi - ref) ** 2)) < 5e-3 * np.sqrt(np.mean(ref ** 2))
