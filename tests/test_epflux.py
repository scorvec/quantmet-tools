import numpy as np
import pytest

from quantmet import epflux

pytest.importorskip("scipy")


def _theta_profile(plev, N2=4e-4, H=7000.0, g=9.80665, th0=300.0):
    z = -H * np.log(plev / 1000.0)
    return th0 * np.exp(N2 * z / g)


def test_charney_drazin_uc_at_60N_is_about_28():
    plev = np.geomspace(1, 300, 40)
    lat = np.arange(-90.0, 90.1, 1.0)
    uc = epflux.charney_drazin_uc(lat, plev, _theta_profile(plev), k=1)
    j = int(np.argmin(np.abs(lat - 60)))
    i = int(np.argmin(np.abs(plev - 30)))
    assert 27.0 < uc[i, j] < 30.0
    np.testing.assert_allclose(uc[1:-1, j], uc[i, j], rtol=1e-6)   # constant N^2: no height dependence
    assert uc[i, int(np.argmin(np.abs(lat - 60)))] > epflux.charney_drazin_uc(lat, plev, _theta_profile(plev), k=2)[i, j]


def _jet(lat, plev, amp):
    return amp * np.exp(-((lat[None, :] - 60) / 12.0) ** 2) * np.log(1000 / plev)[:, None] / np.log(1000 / 10)


def test_member_axis_is_never_filtered():
    plev = np.geomspace(1, 500, 30)
    lat = np.arange(-89.5, 90, 0.5)
    th = _theta_profile(plev)
    members = np.stack([_jet(lat, plev, a) for a in (10.0, 40.0, 80.0)])
    ens = epflux.charney_drazin_excess(members, lat, plev, th)
    assert ens.shape == members.shape
    for m in range(3):
        one = epflux.charney_drazin_excess(members[m], lat, plev, th)
        np.testing.assert_array_equal(np.isnan(ens[m]), np.isnan(one))
        np.testing.assert_allclose(ens[m], one, rtol=0, atol=1e-12)
    # masks: tropics, poles, troposphere
    assert np.isnan(ens[:, :, np.abs(lat) < 20]).all()
    assert np.isnan(ens[:, plev > 400]).all()
    assert np.isfinite(ens[:, plev <= 400][:, :, (lat > 30) & (lat < 80)]).all()


def test_zonal_bandpass_filters_only_longitude():
    nlon = 64
    lam = np.arange(nlon) * 2 * np.pi / nlon
    rng = np.random.default_rng(3)
    offs = rng.normal(size=(4, 7, 1)) * 10                 # member x latitude zonal means
    amp = rng.normal(size=(4, 7, 1))
    f = offs + amp * np.cos(2 * lam) + 0.5 * np.cos(9 * lam)
    out = epflux.zonal_bandpass(f, kmax=3)
    np.testing.assert_allclose(out, np.broadcast_to(amp * np.cos(2 * lam), f.shape), atol=1e-12)


def test_ensemble_of_one_equals_single_and_symmetric_state_is_quiet():
    nlev, nlat, nlon = 10, 37, 48
    plev = np.geomspace(1, 300, nlev)
    lat = np.linspace(-90, 90, nlat)
    lam = np.arange(nlon) * 2 * np.pi / nlon
    rng = np.random.default_rng(5)
    t = 220 + 5 * np.log(plev)[:, None, None] + rng.normal(size=(nlev, nlat, nlon))
    u = 20 + rng.normal(size=t.shape)
    v = rng.normal(size=t.shape) + np.cos(lam)
    a = epflux.ep_flux(u, v, t, lat, plev)
    b = epflux.ensemble_ep_flux([(u, v, t)], lat, plev)
    np.testing.assert_allclose(a.fp, b.fp)
    sym = np.broadcast_to(t.mean(-1, keepdims=True), t.shape)
    q = epflux.ep_flux(np.broadcast_to(u.mean(-1, keepdims=True), u.shape),
                       np.broadcast_to(v.mean(-1, keepdims=True), v.shape), sym, lat, plev)
    assert np.abs(q.fphi).max() < 1e-6 and np.abs(q.fp).max() < 1e-3


# ---- v0.3: zonal/global static stability, ln-p derivatives, below-ground masking ----
from quantmet.epflux import ep_flux, ensemble_ep_flux, below_ground_share  # noqa: E402

PLEV_V3 = np.array([1, 2, 5, 10, 20, 50, 100, 200, 300, 500, 700, 850, 1000], float)
LAT_V3 = np.arange(-90, 90.1, 2.5); LON_V3 = np.arange(0, 360, 2.5)


def _fields_v3(wave=0.0):
    shape = (len(PLEV_V3), len(LAT_V3), len(LON_V3))
    lam = np.deg2rad(LON_V3)[None, None, :]
    u = np.broadcast_to((30 * np.cos(np.deg2rad(LAT_V3)))[None, :, None], shape) + wave * np.cos(lam)
    v = np.zeros(shape) + wave * np.sin(lam)
    t = np.broadcast_to((220 + 0.07 * PLEV_V3)[:, None, None], shape) + wave * 0.5 * np.cos(lam + 0.4)
    return u, v, t


def test_zonal_mean_flow_has_no_flux():
    u, v, t = _fields_v3(0.0)
    for tp in ("zonal", "global"):
        r = ep_flux(u, v, t, LAT_V3, PLEV_V3, theta_p=tp)
        assert np.allclose(r.fphi, 0) and np.allclose(r.fp, 0)
        assert np.allclose(np.nan_to_num(r.force), 0)


def test_ensemble_equals_single_for_identical_members():
    u, v, t = _fields_v3(5.0)
    a = ep_flux(u, v, t, LAT_V3, PLEV_V3)
    b = ensemble_ep_flux([(u, v, t)] * 3, LAT_V3, PLEV_V3)
    assert np.allclose(a.fp, b.fp, equal_nan=True) and np.allclose(a.force, b.force, equal_nan=True)


def test_below_ground_masks_fluxes():
    u, v, t = _fields_v3(5.0)
    ps = np.full((len(LAT_V3), len(LON_V3)), 101_000.0)
    ps[LAT_V3 < -70] = 65_000.0                                    # an Antarctic plateau
    share = below_ground_share(ps, PLEV_V3)
    assert share[PLEV_V3 == 850][0][LAT_V3 < -70].min() == 1.0
    r = ep_flux(u, v, t, LAT_V3, PLEV_V3, psfc=ps)
    assert np.isnan(r.fp[PLEV_V3 == 850][0][LAT_V3 < -72]).all()
    assert np.isfinite(r.fp[PLEV_V3 == 850][0][np.abs(LAT_V3) < 60]).all()
