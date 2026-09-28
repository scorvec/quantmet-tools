import numpy as np
import pytest

from quantmet.qbo import zero_line

pytest.importorskip("scipy")
LAT = np.arange(-50.0, 50.01, 1.0)


def _profile(dip=None, easterly_edge=-2.0):
    """NH-winter westerlies peaking at 35N, easing to 0 at 25N ... then the
    QBO: westerly (weak) down to `easterly_edge`, strong easterlies south of it."""
    u = np.where(LAT > 25, 25 * np.exp(-((LAT - 35) / 10) ** 2), 0.0)
    u = u + np.where(LAT <= 25, 4.0, 0.0) * np.clip((25 - LAT) / 3, 0, 1) * (LAT > easterly_edge)
    u = np.where(LAT <= easterly_edge, -12.0, u)
    u = np.where((LAT > 25) & (u < 4), np.maximum(u, 4.0), u)
    if dip is not None:
        lo, hi, val = dip
        u = np.where((LAT >= lo) & (LAT <= hi), val, u)
    return u


def test_no_westerlies_is_undefined():
    z, f = zero_line(-5.0 * np.ones_like(LAT), LAT, "nh")
    assert f == 1 and np.isnan(z)


def test_finds_the_easterly_edge():
    z, f = zero_line(_profile(), LAT, "nh")
    assert f == 0 and -3.5 < z < 0.0


def test_shallow_dip_is_stepped_over():
    u = _profile(dip=(12, 20, -0.3))
    z, f = zero_line(u, LAT, "nh")
    assert f == 0 and -3.5 < z < 0.0                        # the -0.3 m/s dip is not a critical line
    z2, f2 = zero_line(u, LAT, "nh", umin=0.2)               # ... unless the threshold says it is
    assert f2 == 0 and 16 < z2 < 22
    z3, f3 = zero_line(_profile(dip=(12, 20, -3.0)), LAT, "nh")   # a real easterly pocket counts
    assert f3 == 0 and 16 < z3 < 22


def test_censored_when_westerlies_reach_the_bound():
    u = np.where(LAT > 25, 25 * np.exp(-((LAT - 35) / 10) ** 2), 0) + 5.0
    z, f = zero_line(u, LAT, "nh")
    assert f == 2 and z == -45.0


def test_southern_hemisphere_mirrors_northern():
    rng = np.random.default_rng(0)
    rows = np.stack([_profile(easterly_edge=e) + 0.2 * rng.normal(size=LAT.size) for e in (-8.0, -2.0, 5.0)])
    zn, fn = zero_line(rows, LAT, "nh")
    zs, fs = zero_line(rows[:, ::-1], LAT, "sh")             # u_sh(lat) = u_nh(-lat)
    np.testing.assert_array_equal(fn, fs)
    np.testing.assert_allclose(zs, -zn)
    zd, fd = zero_line(rows[:, ::-1], LAT[::-1], "nh")        # latitude order does not matter
    np.testing.assert_allclose(zd, zn)


def test_leading_shape_is_kept():
    u = np.broadcast_to(_profile(), (2, 3, LAT.size))
    z, f = zero_line(u, LAT)
    assert z.shape == (2, 3) and f.shape == (2, 3)
