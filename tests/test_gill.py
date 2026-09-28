import numpy as np
import pytest

from quantmet import gill


def _gill_case(eps, dl=2.0, latmax=40.0):
    L, _ = gill.gill_scales()
    lat = np.arange(-latmax, latmax + 0.01, dl)
    lon = np.arange(0.0, 360.0, dl)
    y = np.deg2rad(lat) * gill.A_EARTH / L
    x = np.deg2rad(lon) * gill.A_EARTH / L
    half = 2.0
    x0 = x[lon.size // 2]
    F = np.where(np.abs(x - x0) < half, np.cos(np.pi / 2 * (x - x0) / half), 0.0)
    Q = np.exp(-y[:, None] ** 2 / 4) * F[None, :]      # Gill's heating: Kelvin + n=1 Rossby only
    u, v, p = gill.gill_response(Q, lat, lon, eps=eps, sponge_lat=latmax - 6)
    return lat, x, x0, half, u, v, p


@pytest.fixture(scope="module")
def eps02():
    return _gill_case(0.2)


def _rate(x, x0, pe, lo, hi):
    m = (x - x0 > lo) & (x - x0 < hi) if lo > 0 else (x - x0 < lo) & (x - x0 > hi)
    return np.polyfit(x[m] - x0, np.log(np.abs(pe[m])), 1)[0]


def test_kelvin_eastward_decay_equals_eps(eps02):
    lat, x, x0, half, u, v, p = eps02
    pe = p[np.argmin(np.abs(lat))]
    k = -_rate(x, x0, pe, half + 0.5, half + 6)
    assert abs(k - 0.2) < 0.002


def test_symmetry_of_the_response(eps02):
    lat, x, x0, half, u, v, p = eps02
    s = np.abs(u).max()
    np.testing.assert_allclose(u, u[::-1], atol=1e-9 * s)
    np.testing.assert_allclose(p, p[::-1], atol=1e-9 * np.abs(p).max())
    np.testing.assert_allclose(v, -v[::-1], atol=1e-9 * s)
    assert np.abs(v).max() > 1e-3 * s                       # v is not trivially zero


@pytest.mark.parametrize("eps,tol", [(0.2, 0.05), (0.3, 0.03)])
def test_rossby_westward_decay(eps, tol, eps02):
    lat, x, x0, half, u, v, p = eps02 if eps == 0.2 else _gill_case(eps)
    pe = p[np.argmin(np.abs(lat))]
    k = _rate(x, x0, pe, -(half + 0.5), -(half + 3.0))
    exact = gill.rossby_decay_rate(eps)
    assert abs(k / exact - 1) < tol
    assert abs(exact - 3 * eps) > 0.05 * exact or eps < 0.05     # not the long-wave value


def test_heating_signs():
    lat, x, x0, half, u, v, p = _gill_case(0.3, dl=2.0)
    eq = np.argmin(np.abs(lat))
    j = np.argmin(np.abs(x - x0))
    assert p[eq, j] < 0                                        # heating: low pressure
    assert u[eq, j + 5] < 0 and u[eq, j - 5] > 0               # Kelvin easterlies east, Rossby westerlies west


def test_rossby_decay_rate_limits():
    assert abs(gill.rossby_decay_rate(0.01) - 0.03) < 1e-3     # long-wave limit 3 eps
    assert abs(gill.rossby_decay_rate(0.2) - 0.548) < 1e-3


def test_heating_from_ssta_uses_total_sst():
    lat = np.arange(-30.0, 30.1, 2.0)
    ssta = np.full((lat.size, 4), 2.0)
    cold = gill.heating_from_ssta(ssta, np.full_like(ssta, 23.0), lat)       # total 25 C
    warm = gill.heating_from_ssta(ssta, np.full_like(ssta, 25.0), lat)       # total 27 C: half on
    hot = gill.heating_from_ssta(ssta, np.full_like(ssta, 26.5), lat)        # total 28.5 C
    eq = np.argmin(np.abs(lat))
    assert cold.max() == 0
    np.testing.assert_allclose(warm[eq], 1.0)
    np.testing.assert_allclose(hot[eq], 2.0)
    assert (hot[np.abs(lat) >= 22] == 0).all()


def test_heating_from_precip_scale():
    np.testing.assert_allclose(gill.column_heating(1.0), 28.935, rtol=1e-3)
    q1 = gill.heating_from_precip(1.0)
    assert 0.02 < q1 < 0.05
    np.testing.assert_allclose(gill.heating_from_precip(np.array([2.0, -3.0])), [2 * q1, -3 * q1])
    np.testing.assert_allclose(gill.heating_from_precip(1.0, mode_factor=0.5), 0.5 * q1)
    # the dimensional response of a 10 mm/day anomaly is O(1-10) m/s
    lat, x, x0, half, u, v, p = _gill_case(0.3, dl=2.0)
    ud, _, _ = gill.to_dimensional(u * gill.heating_from_precip(10.0), v, p)
    assert 0.5 < np.abs(ud).max() < 30


def test_bad_grids_rejected():
    lat = np.arange(-30.0, 30.1, 2.0)
    with pytest.raises(ValueError):
        gill.gill_response(np.zeros((lat.size, 10)), lat, np.arange(0.0, 20.0, 2.0))
    with pytest.raises(ValueError):
        gill.gill_response(np.zeros((lat.size, 180)), lat[::-1], np.arange(0.0, 360.0, 2.0))
