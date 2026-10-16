import numpy as np
import pytest

from quantmet import thermo as T
from quantmet.constants import CPD, EPSILON, G, LV0, RD, T0

C = T0  # degC -> K offset


# --------------------------------------------------------------------------
# vapour pressure and humidity
# --------------------------------------------------------------------------
def test_reference_vapour_pressure():
    # Murphy & Koop (2005): 611.657 Pa at the triple point over both phases; 2339 Pa at 20 degC (IAPWS 2339.2)
    for ph in ("liquid", "ice"):
        assert abs(T.saturation_vapour_pressure(273.16, ph, "murphy-koop") - 611.657) < 0.01
    assert abs(T.saturation_vapour_pressure(293.15, "liquid", "murphy-koop") - 2339.2) < 0.5


def test_magnus_accuracy_claims():
    def ref(t, ph="liquid"):
        return T.saturation_vapour_pressure(t, ph, "murphy-koop")

    t = np.linspace(-40, 50, 400) + C
    assert np.abs(T.saturation_vapour_pressure(t) / ref(t) - 1).max() < 0.004  # Alduchov-Eskridge
    tb = np.linspace(-30, 35, 400) + C
    assert np.abs(T.saturation_vapour_pressure(tb, formula="bolton") / ref(tb) - 1).max() < 0.003
    ti = np.linspace(-80, 0, 400) + C
    assert np.abs(T.saturation_vapour_pressure(ti, "ice") / ref(ti, "ice") - 1).max() < 0.003
    auto = T.saturation_vapour_pressure(np.array([263.15, 283.15]), "auto")
    np.testing.assert_allclose(
        auto, [T.saturation_vapour_pressure(263.15, "ice"), T.saturation_vapour_pressure(283.15)]
    )
    with pytest.raises(ValueError):
        T.saturation_vapour_pressure(250.0, "ice", "bolton")


def test_dewpoint_roundtrips():
    t = np.linspace(-40, 45, 50) + C
    for f in ("magnus", "bolton", "murphy-koop"):
        np.testing.assert_allclose(T.dewpoint(T.vapour_pressure(t, f), formula=f), t, atol=1e-8)
    np.testing.assert_allclose(T.dewpoint(T.saturation_vapour_pressure(t, "ice"), "ice"), t, atol=1e-8)
    np.testing.assert_allclose(T.relative_humidity(t, t), 100.0)
    td = T.dewpoint_from_rh(303.15, 40.0)
    assert abs(T.relative_humidity(303.15, td) - 40.0) < 1e-9
    assert T.relative_humidity(263.15, 263.15, phase="ice") > 100.0  # water-saturated = ice-supersaturated


def test_humidity_conversions_pressure_aware():
    p = np.array([100000.0, 85000.0, 50000.0])
    td = np.array([25.0, 10.0, -20.0]) + C
    e = T.vapour_pressure(td)
    q = T.specific_humidity_from_dewpoint(td, p)
    r = T.mixing_ratio(e, p)
    np.testing.assert_allclose(T.specific_humidity_from_mixing_ratio(r), q, rtol=1e-12)
    np.testing.assert_allclose(T.mixing_ratio_from_specific_humidity(q), r, rtol=1e-12)
    np.testing.assert_allclose(T.vapour_pressure_from_specific_humidity(q, p), e, rtol=1e-12)
    np.testing.assert_allclose(T.vapour_pressure_from_mixing_ratio(r, p), e, rtol=1e-12)
    np.testing.assert_allclose(T.dewpoint_from_specific_humidity(q, p), td, atol=1e-8)
    t = td + 8.0
    rh = T.relative_humidity_from_specific_humidity(t, q, p)
    np.testing.assert_allclose(rh, T.relative_humidity(t, td), rtol=1e-12)
    np.testing.assert_allclose(T.specific_humidity_from_relative_humidity(t, rh, p), q, rtol=1e-12)
    # the eps e / p shortcut: 1.6 % low for q, 4.2 % low for r at a 30 degC dewpoint and 1000 hPa
    e30 = T.vapour_pressure(303.15)
    np.testing.assert_allclose(1 - EPSILON * e30 / 1e5 / T.specific_humidity(e30, 1e5), (1 - EPSILON) * e30 / 1e5)
    np.testing.assert_allclose(1 - EPSILON * e30 / 1e5 / T.mixing_ratio(e30, 1e5), e30 / 1e5)
    assert 0.04 < e30 / 1e5 < 0.045
    assert np.isnan(T.saturation_mixing_ratio(330.0, 10000.0))  # e_s > p: undefined, not negative
    assert T.saturation_mixing_ratio(263.15, 1e5, "ice") < T.saturation_mixing_ratio(263.15, 1e5)
    np.testing.assert_allclose(T.saturation_specific_humidity(t, p), T.specific_humidity_from_dewpoint(t, p))
    assert T.dewpoint_depression(300.0, 290.0) == 10.0
    np.testing.assert_allclose(
        T.vapour_pressure_deficit(300.0, rh=40.0),
        T.vapour_pressure_deficit(300.0, td=T.dewpoint_from_rh(300.0, 40.0)),
        rtol=1e-10,
    )


# --------------------------------------------------------------------------
# potential temperatures, density
# --------------------------------------------------------------------------
def test_potential_and_virtual_temperatures():
    assert T.potential_temperature(288.0, 1e5) == 288.0
    th = T.potential_temperature(250.0, 50000.0)
    np.testing.assert_allclose(th, 250.0 * 2 ** (RD / CPD))
    np.testing.assert_allclose(T.temperature_from_potential_temperature(th, 50000.0), 250.0)
    q = 0.015
    thm = T.potential_temperature(300.0, 80000.0, q=q)
    np.testing.assert_allclose(T.temperature_from_potential_temperature(thm, 80000.0, q=q), 300.0)
    assert thm < T.potential_temperature(300.0, 80000.0)  # moist kappa is smaller
    np.testing.assert_allclose(T.virtual_temperature(300.0, q), 300.0 * (1 + 0.6078 * q), rtol=1e-4)
    np.testing.assert_allclose(T.density_temperature(300.0, q), T.virtual_temperature(300.0, q))
    assert T.density_temperature(300.0, q, ql=0.002, qi=0.001) < T.virtual_temperature(300.0, q)
    np.testing.assert_allclose(
        T.virtual_potential_temperature(300.0, 80000.0, q),
        T.potential_temperature(T.virtual_temperature(300.0, q), 80000.0),
    )
    # ISA sea level: 1.225 kg/m3
    assert abs(T.air_density(101325.0, 288.15) - 1.225) < 5e-4
    assert T.air_density(1e5, 300.0, q=0.02) < T.air_density(1e5, 300.0)  # moist air is lighter
    np.testing.assert_allclose(T.equivalent_temperature(300.0, 0.01), 300.0 + LV0 * (0.01 / 0.99) / CPD)
    np.testing.assert_allclose(T.exner(50000.0), 0.5 ** (RD / CPD))


def test_bolton_theta_e_matches_a_numerical_pseudo_adiabat():
    """theta_E is the potential temperature a parcel reaches when lifted pseudo-adiabatically until dry: lift to the
    LCL, integrate the pseudo-adiabat to 20 hPa, where r_s ~ 1e-16."""
    for tc, tdc, p in [(30, 20, 1e5), (25, 10, 85000.0), (10, 5, 70000.0), (20, 15, 1e5)]:
        t, td = tc + C, tdc + C
        tl, pl = T.lcl_bolton(t, td, p)
        ttop = T.moist_adiabat(tl, pl, [2000.0])[0]
        assert T.saturation_mixing_ratio(ttop, 2000.0) < 1e-12
        assert abs(T.equivalent_potential_temperature(t, td, p) - T.potential_temperature(ttop, 2000.0)) < 0.15


def test_reversible_theta_e():
    # conserved along an unsaturated adiabat (exact moist Poisson ascent at fixed q)
    t1, td1, p1 = 303.15, 288.15, 1e5
    e = T.vapour_pressure(td1)
    q = T.specific_humidity(e, p1)
    ref = T.equivalent_potential_temperature(t1, td1, p1, "reversible")
    for p2 in (95000.0, 90000.0, 85000.0):
        t2 = T.temperature_from_potential_temperature(T.potential_temperature(t1, p1, q=q), p2, q=q)
        assert abs(T.equivalent_potential_temperature(t2, T.dewpoint(e * p2 / p1), p2, "reversible") - ref) < 0.01
    # a few K below the pseudo-adiabatic value for moist air; both -> theta for dry air
    assert 1.0 < T.equivalent_potential_temperature(t1, td1, p1) - ref < 6.0
    dry = T.equivalent_potential_temperature(280.0, 180.0, 70000.0, "reversible")
    assert abs(dry - T.potential_temperature(280.0, 70000.0)) < 0.01
    assert (
        abs(T.equivalent_potential_temperature(280.0, 180.0, 70000.0) - T.potential_temperature(280.0, 70000.0)) < 0.2
    )
    np.testing.assert_allclose(
        T.saturation_equivalent_potential_temperature(290.0, 9e4), T.equivalent_potential_temperature(290.0, 290.0, 9e4)
    )


def test_wet_bulb_potential_temperature_davies_jones():
    # at 1000 hPa a saturated parcel's theta_w is its temperature
    t = np.array([263.15, 283.15, 298.15, 303.15])
    np.testing.assert_allclose(T.wet_bulb_potential_temperature(t, t, 1e5), t, atol=0.01)
    # elsewhere: the 1000 hPa temperature of the pseudo-adiabat through the LCL
    for tc, tdc, p in [(30, 20, 1e5), (25, 10, 85000.0), (10, 5, 70000.0), (35, 25, 95000.0)]:
        tl, pl = T.lcl_bolton(tc + C, tdc + C, p)
        assert abs(T.wet_bulb_potential_temperature(tc + C, tdc + C, p) - T.moist_adiabat(tl, pl, [1e5])[0]) < 0.05


def test_liquid_and_ice_liquid_theta():
    th = T.potential_temperature(285.0, 85000.0)
    assert T.liquid_water_potential_temperature(285.0, 85000.0, 0.0) == th
    assert T.ice_liquid_potential_temperature(285.0, 85000.0, 0.0, 0.0) == th
    tl = T.liquid_water_potential_temperature(285.0, 85000.0, 1e-3)
    np.testing.assert_allclose(th - tl, th / 285.0 * LV0 / CPD * 1e-3)
    til = T.ice_liquid_potential_temperature(285.0, 85000.0, 1e-3, 0.0)
    assert abs(til - tl) < 0.01 * (th - tl)  # same to first order
    assert T.ice_liquid_potential_temperature(250.0, 5e4, 0.0, 1e-3) < T.ice_liquid_potential_temperature(
        250.0, 5e4, 1e-3, 0.0
    )  # ice carries L_s > L_v


# --------------------------------------------------------------------------
# wet bulb
# --------------------------------------------------------------------------
def test_stull_reference_point_and_mask():
    assert abs(T.wet_bulb_stull(20 + C, 50.0) - (13.7 + C)) < 0.01  # Stull (2011) worked example
    assert np.isnan(T.wet_bulb_stull(55 + C, 50.0)) and np.isnan(T.wet_bulb_stull(20 + C, 3.0))
    assert np.isfinite(T.wet_bulb_stull(55 + C, 50.0, strict=False))
    np.testing.assert_array_equal(T.stull_valid(np.array([-25, -20, 50, 51]) + C, 50), [False, True, True, False])


def test_psychrometric_wet_bulb():
    t = np.array([0.0, 15.0, 30.0, 40.0]) + C
    np.testing.assert_allclose(T.wet_bulb_temperature(t, 1e5, td=t), t, atol=1e-5)  # saturated: Tw = T
    td = t - 12.0
    tw = T.wet_bulb_temperature(t, 1e5, td=td)
    assert (tw > td).all() and (tw < t).all()
    assert (T.wet_bulb_temperature(t, 70000.0, td=td) < tw).all()  # lower pressure, cooler
    # standard psychrometric value: 20 degC and 50 % at sea level -> 13.8 degC
    assert abs(T.wet_bulb_temperature(20 + C, 101325.0, rh=50.0) - (13.8 + C)) < 0.05
    # it is the adiabatic-saturation balance it claims to solve
    r = T.mixing_ratio(T.vapour_pressure(td), 1e5)
    lhs = (CPD + r * 1846.0) * (t - tw)
    rhs = (T.saturation_mixing_ratio(tw, 1e5) - r) * T.latent_heat(tw)
    np.testing.assert_allclose(lhs, rhs, rtol=1e-5)


def test_stull_against_psychrometric_and_pressure():
    tc, rh = np.meshgrid(np.arange(0, 50.1, 1.0), np.arange(5, 99.1, 1.0))
    err = T.wet_bulb_stull(tc + C, rh) - T.wet_bulb_temperature(tc + C, 101325.0, rh=rh)
    assert err.min() > -1.25 and err.max() < 1.2 and np.abs(err).mean() < 0.4
    # dry heat on a plateau: 40 degC, 15 %, 840 hPa
    st = T.wet_bulb_stull(40 + C, 15.0)
    sea = T.wet_bulb_temperature(40 + C, 101325.0, rh=15.0)
    high = T.wet_bulb_temperature(40 + C, 84000.0, rh=15.0)
    assert 1.0 < sea - high < 1.3  # the pressure effect alone
    assert st - high > 1.4  # Stull, blind to pressure, is 1.5 degC too warm


# --------------------------------------------------------------------------
# LCL
# --------------------------------------------------------------------------
def test_romps_lcl_published_value_and_consistency():
    tl, pl, zl = T.lcl_romps(300.0, 1e5, 50.0)
    assert abs(zl / 1433.844139279 - 1) < 1e-9  # Romps (2017) code's own test value
    # self-consistency: a dry adiabat with Romps's constants saturates (in his vapour pressure) at T_LCL
    c = T.ROMPS2017
    cpa, cpv = c["cva"] + c["rgasa"], c["cvv"] + c["rgasv"]

    def pstar(t):
        return (
            c["ptrip"]
            * (t / c["ttrip"]) ** ((cpv - c["cvl"]) / c["rgasv"])
            * np.exp((c["e0v"] - (c["cvv"] - c["cvl"]) * c["ttrip"]) / c["rgasv"] * (1 / c["ttrip"] - 1 / t))
        )

    pv0 = 0.5 * pstar(300.0)
    qv = c["rgasa"] * pv0 / (c["rgasv"] * 1e5 + (c["rgasa"] - c["rgasv"]) * pv0)
    rm, cpm = (1 - qv) * c["rgasa"] + qv * c["rgasv"], (1 - qv) * cpa + qv * cpv
    p_at = 1e5 * (tl / 300.0) ** (cpm / rm)
    np.testing.assert_allclose(p_at, pl, rtol=1e-12)
    np.testing.assert_allclose(pv0 * pl / 1e5, pstar(tl), rtol=1e-9)  # vapour pressure scales with p
    np.testing.assert_allclose(zl, cpm * (300.0 - tl) / c["g"])
    t, z = T.lcl_romps(290.0, 9e4, 100.0)[0], T.lcl_romps(290.0, 9e4, 100.0)[2]
    assert abs(t - 290.0) < 1e-6 and abs(z) < 1e-3
    assert abs(T.lcl_romps(250.0, 7e4, 100.0, "ice")[0] - 250.0) < 1e-6  # ice-saturated: the parcel is at its LDL
    auto = T.lcl_romps(np.array([250.0, 300.0]), 8e4, 60.0, "auto")[0]
    np.testing.assert_allclose(auto, [T.lcl_romps(250.0, 8e4, 60.0, "ice")[0], T.lcl_romps(300.0, 8e4, 60.0)[0]])


def test_bolton_lcl_close_to_romps():
    for tc, rh in [(30, 50), (20, 80), (10, 30), (35, 20)]:
        t = tc + C
        tr = T.lcl_romps(t, 1e5, rh)[0]
        tb = T.lcl_bolton(t, T.dewpoint_from_rh(t, rh), 1e5)[0]
        assert abs(tr - tb) < 0.1


# --------------------------------------------------------------------------
# energies, lapse rate
# --------------------------------------------------------------------------
def test_static_energy_and_enthalpy():
    assert T.dry_static_energy(300.0, 1000.0) == CPD * 300.0 + G * 1000.0
    assert T.moist_static_energy(300.0, 1000.0, 0.01) == CPD * 300.0 + G * 1000.0 + LV0 * 0.01
    assert T.enthalpy(T0) == 0.0
    np.testing.assert_allclose(T.enthalpy(T0, q=0.01), 0.01 * LV0)
    # evaporating dq at T0 costs exactly L_v0 dq; freezing releases L_f0
    np.testing.assert_allclose(T.enthalpy(T0, 0.011, 0.001) - T.enthalpy(T0, 0.010, 0.002), LV0 * 0.001)
    np.testing.assert_allclose(T.enthalpy(T0, 0.0, 0.0, 0.001) - T.enthalpy(T0, 0.0, 0.001, 0.0), -3.337e5 * 0.001)
    # differences between states with the same total water do not depend on the reference state
    a = (305.0, 0.015, 0.001, 0.0)
    b = (280.0, 0.008, 0.005, 0.003)
    for tr in (0.0, 250.0, T0):
        np.testing.assert_allclose(
            T.enthalpy(*a, t_ref=tr) - T.enthalpy(*b, t_ref=tr), T.enthalpy(*a) - T.enthalpy(*b), rtol=1e-10
        )
    np.testing.assert_allclose(T.latent_heat(T0 + 10), LV0 - 2372.0 * 10)


def test_moist_lapse_rate():
    assert abs(T.moist_lapse_rate(200.0, 1e5) / (G / CPD) - 1) < 1e-3  # no vapour: the dry adiabatic rate
    g300 = T.moist_lapse_rate(300.0, 1e5) * 1e3
    assert 3.4 < g300 < 4.0
    assert T.moist_lapse_rate(300.0, 7e4) < g300  # more vapour at lower pressure
    # the adiabat integrates the lapse rate: dT/dp hydrostatic consistency
    p = np.array([90000.0, 89000.0])
    tt = T.moist_adiabat(295.0, 90000.0, p)
    tv = T.virtual_temperature(tt.mean(), T.saturation_specific_humidity(tt.mean(), p.mean()))
    dtdz = -(tt[1] - tt[0]) / (np.log(p[0] / p[1]) * RD * tv / G)
    assert abs(dtdz / T.moist_lapse_rate(tt.mean(), p.mean()) - 1) < 0.005
    out = T.moist_adiabat(np.array([290.0, 300.0]), 1e5, [1e5, 5e4, 85000.0])
    assert out.shape == (3, 2) and np.allclose(out[0], [290.0, 300.0])


# --------------------------------------------------------------------------
# columns
# --------------------------------------------------------------------------
P = np.array([1000, 925, 850, 700, 500, 300, 200, 100]) * 100.0


def test_column_integral_zero_to_ps():
    ones = np.ones((P.size, 3))
    ps = np.array([101000.0, 90000.0, 60000.0])
    np.testing.assert_allclose(T.column_integral(ones, P, ps) * G, ps)  # integrates exactly 0..p_s
    np.testing.assert_allclose(T.column_integral(np.ones(P.size), P) * G, 1e5)  # no p_s: lowest level is the surface
    q = np.broadcast_to(np.array([16, 12, 9, 5, 2, 0.4, 0.1, 0.01])[:, None] * 1e-3, (P.size, 3)).copy()
    ref = T.precipitable_water(q, P, ps)
    q2 = q.copy()
    q2[:2, 2] = np.nan  # below ground at 600 hPa: ignored
    np.testing.assert_allclose(T.precipitable_water(q2, P, ps), ref)
    np.testing.assert_allclose(T.precipitable_water(q[::-1], P[::-1], ps), ref)  # any level order
    np.testing.assert_allclose(T.precipitable_water(q, np.broadcast_to(P[:, None], q.shape), ps), ref)  # model levels
    np.testing.assert_allclose(T.precipitable_water(q.T, P, ps, axis=1), ref)
    assert 30 < ref[0] < 60
    cw = T.column_water(P, ps, q=q, ql=q * 0.1)
    np.testing.assert_allclose(cw["total"], cw["vapour"] + cw["liquid"])
    np.testing.assert_allclose(cw["liquid"], 0.1 * cw["vapour"])
    t = np.full_like(q, 250.0)
    z = np.zeros_like(q)
    np.testing.assert_allclose(T.column_moist_static_energy(t, z, q, P, ps), (CPD * 250.0 * ps / G) + LV0 * ref)
    np.testing.assert_allclose(T.column_enthalpy(t, P, ps), CPD * (250.0 - T0) * ps / G)


def test_thickness_hypsometric():
    np.testing.assert_allclose(T.thickness(np.full(P.size, 250.0), P, 1e5, 5e4), RD * 250.0 / G * np.log(2))
    # constant lapse rate: T = T0 - L z, p = p0 (T/T0)^(g/(Rd L)) -> z(p) analytic
    t0, lapse = 288.15, 0.0065
    p = np.arange(1000, 299, -25) * 100.0
    tt = t0 * (p / 1e5) ** (RD * lapse / G)
    z500 = (t0 - t0 * 0.5 ** (RD * lapse / G)) / lapse
    assert abs(T.thickness(tt, p, 1e5, 5e4) - z500) < 0.05  # within 5 cm of 5.6 km
    assert abs(T.layer_mean_temperature(tt, p, 85000.0, 70000.0) - 0.5 * (tt[6] + tt[12])) < 0.1
    q = np.full(P.size, 0.01)
    np.testing.assert_allclose(
        T.thickness(np.full(P.size, 280.0), P, 1e5, 5e4, q=q) / T.thickness(np.full(P.size, 280.0), P, 1e5, 5e4),
        1 + 0.6078 * 0.01,
        rtol=1e-4,
    )
    with pytest.raises(ValueError):
        T.thickness(np.full(P.size, 250.0), P, 105000.0, 5e4)  # no extrapolation


# --------------------------------------------------------------------------
# heat index and wind chill against the NWS charts
# --------------------------------------------------------------------------
HI_T = np.arange(80, 112, 2)
HI_CHART = {
    40: [80, 81, 83, 85, 88, 91, 94, 97, 101, 105, 109, 114, 119, 124, 130, 136],
    50: [81, 83, 85, 88, 91, 95, 99, 103, 108, 113, 118, 124, 131, 137],
    60: [82, 84, 88, 91, 95, 100, 105, 110, 116, 123, 129, 137],
    80: [84, 89, 94, 100, 106, 113, 121, 129],
}


def test_heat_index_nws_chart():
    for rh, row in HI_CHART.items():
        np.testing.assert_array_equal(np.round(T.heat_index(HI_T[: len(row)], rh)), row)
    # RH 100 %: the chart (88, 90 degF: 121, 132) has no adjustment; at 80-86 degF the NWS adds the high-RH one
    np.testing.assert_array_equal(np.round(T.heat_index(np.array([88, 90]), 100)), [121, 132])
    t = np.array([80, 82, 84, 86.0])
    adj = (100 - 85) / 10 * (87 - t) / 5
    np.testing.assert_array_equal(np.round(T.heat_index(t, 100) - adj), [87, 95, 103, 112])


def test_heat_index_branches():
    # below 80 degF the simple formula, not Rothfusz
    assert T.heat_index(70.0, 50.0) == pytest.approx(0.5 * (70 + 61 + 1.2 * 2 + 0.094 * 50))

    # low-RH adjustment
    def roth(t, r):
        return (
            -42.379
            + 2.04901523 * t
            + 10.14333127 * r
            - 0.22475541 * t * r
            - 6.83783e-3 * t * t
            - 5.481717e-2 * r * r
            + 1.22874e-3 * t * t * r
            + 8.5282e-4 * t * r * r
            - 1.99e-6 * t * t * r * r
        )

    assert T.heat_index(100.0, 10.0) == pytest.approx(roth(100.0, 10.0) - 0.75 * np.sqrt(12 / 17))
    assert T.heat_index(84.0, 95.0) == pytest.approx(roth(84.0, 95.0) + 1.0 * 0.6)
    assert T.heat_index(95.0, 50.0) == pytest.approx(roth(95.0, 50.0))
    assert T.heat_index(np.array([[70.0, 95.0]]), 50.0).shape == (1, 2)


WC_T = np.arange(40, -46, -5)
WC_CHART = {
    5: [36, 31, 25, 19, 13, 7, 1, -5, -11, -16, -22, -28, -34, -40, -46, -52, -57, -63],
    10: [34, 27, 21, 15, 9, 3, -4, -10, -16, -22, -28, -35, -41, -47, -53, -59, -66, -72],
    15: [32, 25, 19, 13, 6, 0, -7, -13, -19, -26, -32, -39, -45, -51, -58, -64, -71, -77],
    20: [30, 24, 17, 11, 4, -2, -9, -15, -22, -29, -35, -42, -48, -55, -61, -68, -74, -81],
    30: [28, 22, 15, 8, 1, -5, -12, -19, -26, -33, -39, -46, -53, -60, -67, -73, -80, -87],
    60: [25, 17, 10, 3, -4, -11, -19, -26, -33, -40, -48, -55, -62, -69, -76, -84, -91, -98],
}


def test_wind_chill_nws_chart():
    for v, row in WC_CHART.items():
        np.testing.assert_array_equal(np.round(T.wind_chill(WC_T, v)) + 0.0, row)
    assert T.wind_chill(55.0, 20.0) == 55.0  # too warm: no wind chill
    assert T.wind_chill(20.0, 2.0) == 20.0  # too calm


def test_units():
    np.testing.assert_allclose(T.c_to_f([100.0, -40.0, 0.0]), [212.0, -40.0, 32.0])
    np.testing.assert_allclose(T.f_to_c(T.c_to_f(23.4)), 23.4)
    np.testing.assert_allclose(T.k_to_f(T.f_to_k(71.0)), 71.0)
    np.testing.assert_allclose(T.k_to_c(T.c_to_k(-3.0)), -3.0)
    np.testing.assert_allclose(T.ms_to_kt(10.0), 19.43844, rtol=1e-6)
    np.testing.assert_allclose(T.kt_to_mph(1.0), 1.150779, rtol=1e-6)
    np.testing.assert_allclose(T.mph_to_ms(T.ms_to_mph(7.0)), 7.0)
    np.testing.assert_allclose(T.mph_to_kt(T.kt_to_mph(9.0)), 9.0)
    np.testing.assert_allclose(T.kt_to_ms(T.ms_to_kt(3.0)), 3.0)
