import numpy as np

from quantmet.pv import ertel_pv, dynamic_tropopause, on_isentrope, potential_temperature

LEV = np.array([1000, 850, 700, 500, 400, 300, 250, 200, 150, 100], float) * 100.0
LAT = np.arange(-80, 80.1, 2.0)
LON = np.arange(0, 360, 2.0)


def _resting(tz):
    """Resting atmosphere with T a function of pressure only."""
    shape = (len(LEV), len(LAT), len(LON))
    t = np.broadcast_to(tz[:, None, None], shape).copy()
    z = np.zeros(shape)
    return t, z, z.copy()


def test_resting_atmosphere_pv_is_f_times_stability():
    tz = np.maximum(288.0 - 6.5e-3 * 7000 * np.log(1e5 / LEV), 216.65)   # troposphere + isothermal stratosphere
    t, u, v = _resting(tz)
    pv, th = ertel_pv(t, u, v, LEV, LAT, LON, smooth=False)
    f = 2 * 7.2921e-5 * np.sin(np.deg2rad(LAT))
    th_p = np.gradient(potential_temperature(tz, LEV), LEV)
    expect = -9.80665 * f[None, :] * th_p[:, None] * 1e6
    assert np.allclose(pv[:, :, 0], expect, rtol=1e-6, atol=1e-9)
    assert (pv[:, LAT > 5] > 0).all() and (pv[:, LAT < -5] < 0).all()    # stable: PV has the sign of f


def test_dynamic_tropopause_interpolates_in_pv():
    pv = np.broadcast_to(np.linspace(0.2, 5.0, len(LEV))[:, None, None], (len(LEV), 3, 4)).copy()
    th = np.broadcast_to(np.linspace(290, 380, len(LEV))[:, None, None], pv.shape).copy()
    z = np.zeros_like(pv)
    dt = dynamic_tropopause(pv, th, z, z, LEV, thr=2.0)
    frac = (2.0 - pv[3, 0, 0]) / (pv[4, 0, 0] - pv[3, 0, 0])
    assert np.allclose(dt["theta"], th[3, 0, 0] + frac * (th[4, 0, 0] - th[3, 0, 0]))
    assert not dt["never"].any()


def test_dynamic_tropopause_never_reached_uses_top_level():
    pv = np.full((len(LEV), 2, 2), 0.5); th = np.broadcast_to(np.linspace(300, 370, len(LEV))[:, None, None], pv.shape)
    z = np.zeros_like(pv)
    dt = dynamic_tropopause(pv, th, z, z, LEV)
    assert dt["never"].all() and np.allclose(dt["p"], 100.0) and np.allclose(dt["theta"], 370.0)
    assert np.isnan(dynamic_tropopause(pv, th, z, z, LEV, fill_top=False)["theta"]).all()


def test_isentrope_linear_interpolation_and_out_of_range():
    th = np.broadcast_to(np.linspace(300, 390, len(LEV))[:, None, None], (len(LEV), 2, 2)).copy()
    pv = th / 100.0
    z = np.zeros_like(th)
    r = on_isentrope(pv, th, z, z, 345.0)
    assert np.allclose(r["pv"], 3.45)
    assert np.isnan(on_isentrope(pv, th, z, z, 400.0)["pv"]).all()
