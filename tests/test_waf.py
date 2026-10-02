import numpy as np

from quantmet.waf import tn01_flux, flux_divergence, ensemble_flux

LAT = np.arange(-90, 90.1, 2.5); LON = np.arange(0, 360, 2.5)


def test_zonally_uniform_anomaly_has_no_zonal_flux_and_no_divergence_from_waves():
    psi = np.broadcast_to(np.sin(np.deg2rad(LAT))[:, None] * 1e6, (len(LAT), len(LON))).copy()
    U = np.full_like(psi, 20.0); V = np.zeros_like(psi)
    wx, wy = tn01_flux(psi, U, V, LAT, LON)
    ok = np.isfinite(wx)
    assert np.allclose(wx[ok], 0.0, atol=1e-12)               # psi_l = psi_ll = 0
    d = flux_divergence(wx, wy, LAT, LON)
    rows = np.isfinite(d).all(axis=1)
    assert rows.any() and np.allclose(d[rows].std(axis=1), 0.0, atol=1e-12)  # no longitude structure


def test_phase_independence_of_plane_wave():
    """TN01 is phase-independent: shifting a zonal plane wave must not change Wx (the reason for eq. 38's form)."""
    lat = np.arange(20, 70.1, 1.0)
    k = 6
    lam = np.deg2rad(LON)
    amp = (np.cos(np.deg2rad(lat - 45) * 3) * 5e6)[:, None]
    U = np.full((len(lat), len(LON)), 25.0); V = np.zeros_like(U)
    w1 = tn01_flux(amp * np.cos(k * lam)[None], U, V, lat, LON)[0]
    w2 = tn01_flux(amp * np.cos(k * lam + 1.0)[None], U, V, lat, LON)[0]
    inner = slice(3, -3)
    assert np.allclose(w1[inner], w2[inner], rtol=1e-3, atol=1e-6)
    assert np.nanstd(w1[inner], axis=1).max() < 1e-3 * np.nanmax(np.abs(w1[inner]))


def test_ensemble_flux_is_mean_of_member_fluxes():
    rng = np.random.default_rng(0)
    U = np.full((len(LAT), len(LON)), 20.0); V = np.zeros_like(U)
    mem = [rng.normal(size=U.shape) * 1e6 for _ in range(3)]
    wx, wy, agree = ensemble_flux(mem, U, V, LAT, LON)
    ref = np.mean([tn01_flux(m, U, V, LAT, LON)[0] for m in mem], axis=0)
    assert np.allclose(wx, ref, equal_nan=True)
    assert np.nanmax(agree) <= 1.0
