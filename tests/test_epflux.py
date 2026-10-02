import numpy as np

from quantmet.epflux import ep_flux, ensemble_ep_flux, below_ground_share

PLEV = np.array([1, 2, 5, 10, 20, 50, 100, 200, 300, 500, 700, 850, 1000], float)
LAT = np.arange(-90, 90.1, 2.5); LON = np.arange(0, 360, 2.5)


def _fields(wave=0.0):
    shape = (len(PLEV), len(LAT), len(LON))
    lam = np.deg2rad(LON)[None, None, :]
    u = np.broadcast_to((30 * np.cos(np.deg2rad(LAT)))[None, :, None], shape) + wave * np.cos(lam)
    v = np.zeros(shape) + wave * np.sin(lam)
    t = np.broadcast_to((220 + 0.07 * PLEV)[:, None, None], shape) + wave * 0.5 * np.cos(lam + 0.4)
    return u, v, t


def test_zonal_mean_flow_has_no_flux():
    u, v, t = _fields(0.0)
    for tp in ("zonal", "global"):
        r = ep_flux(u, v, t, LAT, PLEV, theta_p=tp)
        assert np.allclose(r.fphi, 0) and np.allclose(r.fp, 0)
        assert np.allclose(np.nan_to_num(r.force), 0)


def test_ensemble_equals_single_for_identical_members():
    u, v, t = _fields(5.0)
    a = ep_flux(u, v, t, LAT, PLEV)
    b = ensemble_ep_flux([(u, v, t)] * 3, LAT, PLEV)
    assert np.allclose(a.fp, b.fp, equal_nan=True) and np.allclose(a.force, b.force, equal_nan=True)


def test_below_ground_masks_fluxes():
    u, v, t = _fields(5.0)
    ps = np.full((len(LAT), len(LON)), 101_000.0)
    ps[LAT < -70] = 65_000.0                                    # an Antarctic plateau
    share = below_ground_share(ps, PLEV)
    assert share[PLEV == 850][0][LAT < -70].min() == 1.0
    r = ep_flux(u, v, t, LAT, PLEV, psfc=ps)
    assert np.isnan(r.fp[PLEV == 850][0][LAT < -72]).all()
    assert np.isfinite(r.fp[PLEV == 850][0][np.abs(LAT) < 60]).all()
