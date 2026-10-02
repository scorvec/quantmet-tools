import numpy as np

from quantmet import conservation as C

LAT = np.arange(-90, 90.1, 1.0); LON = np.arange(0, 360, 1.0)


def test_area_weights_sum_to_sphere():
    assert abs(C.area_weights(LAT, LON).sum() - 4 * np.pi) / (4 * np.pi) < 1e-4


def test_uniform_surface_pressure_dry_mass():
    ps = np.full((len(LAT), len(LON)), 98_000.0); tcw = np.full_like(ps, 25.0)
    m = C.dry_air_mass(ps, tcw, LAT, LON)
    exact = 4 * np.pi * C.A_EARTH ** 2 * (98_000.0 - C.G * 25.0) / C.G
    assert abs(m / exact - 1) < 1e-4
    h = C.dry_air_mass(ps, tcw, LAT, LON, hemispheres=True)
    assert abs(h["nh"] + h["sh"] - h["global"]) < 1e-6 * h["global"]


def test_layer_thickness_spans_exactly_zero_to_ps():
    p = np.array([50, 100, 200, 300, 500, 700, 850, 925, 1000], float) * 100
    ps = np.array([[101_000.0, 60_000.0]])
    dp = C.layer_thickness(p, ps)
    assert np.allclose(dp.sum(0), ps)
    assert dp[-1, 0, 1] == 0.0                                 # 1000 hPa is underground at ps = 600 hPa


def test_solid_body_rotation_aam():
    """Solid-body rotation u = U0 cos(phi), uniform ps: M_r = (a^3/g) ps U0 2 pi int cos^3 dphi, int = 4/3."""
    u0, ps0 = 10.0, 100_000.0
    p = np.array([100, 300, 500, 700, 900], float) * 100
    ps = np.full((len(LAT), len(LON)), ps0)
    u = np.broadcast_to((u0 * np.cos(np.deg2rad(LAT)))[None, :, None], (len(p), len(LAT), len(LON)))
    mr = C.relative_aam(u, p, ps, LAT, LON)
    exact = (C.A_EARTH ** 3 / C.G) * ps0 * u0 * 2 * np.pi * (4 / 3)
    assert abs(mr / exact - 1) < 1e-3
    mo = C.mass_aam(ps, LAT, LON)
    exact_o = (C.OMEGA * C.A_EARTH ** 4 / C.G) * ps0 * 2 * np.pi * (4 / 3)      # int cos^3 dphi = 4/3
    assert abs(mo / exact_o - 1) < 1e-3


def test_torques_vanish_for_flat_or_still():
    ps = np.full((len(LAT), len(LON)), 1e5); h = np.random.default_rng(0).uniform(0, 3000, ps.shape)
    assert abs(C.mountain_torque(ps, h, LAT, LON)) < 1e-6
    assert C.friction_torque(np.zeros_like(ps), LAT, LON) == 0.0


def test_column_energy_and_water():
    p = np.array([200, 500, 850], float) * 100; ps = np.full((2, 2), 1e5)
    t = np.full((3, 2, 2), 250.0)
    e = C.column_energy(t, np.zeros_like(t), np.zeros_like(t), p, ps)
    assert np.allclose(e["cpT"], C.CP * 250.0 * 1e5 / C.G)
    assert np.isclose(C.implied_evaporation(3.0, 25.0, 25.5), 3.5)
