import numpy as np
import pytest

from quantmet import waf

A = waf.A_EARTH


def test_plane_rossby_wave_has_constant_flux_and_no_divergence():
    # stationary Rossby wave in uniform westerlies on a doubly periodic beta-plane box
    beta, U = 1.6e-11, 15.0
    nx, ny = 128, 96
    Lx, Ly = 2.4e7, 1.2e7
    dx, dy = Lx / nx, Ly / ny
    k = 2 * np.pi * 3 / Lx
    l = 2 * np.pi * 2 / Ly
    assert beta / (k * k + l * l) > 0          # a real stationary wavenumber exists
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    X, Y = np.meshgrid(x, y)
    Amp = 1e7
    kd, ld = np.sin(k * dx) / dx, np.sin(l * dy) / dy      # centred-difference wavenumbers
    ref = None
    for phase in (0.0, 0.7, 2.1):
        psi = Amp * np.cos(k * X + l * Y + phase)
        wx, wy, div = waf.tn01_flux_cartesian(psi, np.full_like(psi, U), np.zeros_like(psi), dx, dy,
                                              phat=0.25, periodic_y=True)
        np.testing.assert_allclose(wx, 0.25 / 2 * Amp ** 2 * kd ** 2, rtol=1e-10)
        np.testing.assert_allclose(wy, 0.25 / 2 * Amp ** 2 * kd * ld, rtol=1e-10)
        assert np.abs(div).max() < 1e-12 * np.abs(wx).max() / dx
        if ref is not None:
            np.testing.assert_allclose(wx, ref, rtol=1e-10)
        ref = wx
    assert wx.mean() > 0 and wy.mean() > 0          # energy heads east and (for l > 0) north


def _sphere(dl=2.0):
    lat = np.arange(-88.0, 88.1, dl)
    lon = np.arange(0.0, 360.0, dl)
    return lat, lon


def test_sphere_flux_is_phase_independent_and_matches_tn01():
    lat, lon = _sphere()
    LA, LO = np.meshgrid(np.deg2rad(lat), np.deg2rad(lon), indexing="ij")
    m, n, Amp, U0 = 5, 4.0, 5e6, 20.0
    dl, dp = np.deg2rad(2.0), np.deg2rad(2.0)
    md, nd = np.sin(m * dl) / dl, np.sin(n * dp) / dp
    U = np.full(LA.shape, U0)
    V = np.zeros_like(U)
    for phase in (0.0, 1.0):
        r = waf.tn01_flux(Amp * np.cos(m * LO + n * LA + phase), U, V, lat, lon, p_hpa=250)
        inner = (np.abs(lat) > 25) & (np.abs(lat) < 80)
        cos = np.cos(np.deg2rad(lat[inner]))[:, None]
        wx_ref = 0.25 * cos / 2 * Amp ** 2 * md ** 2 / (A ** 2 * cos ** 2)
        wy_ref = 0.25 * cos / 2 * Amp ** 2 * md * nd / (A ** 2 * cos)
        np.testing.assert_allclose(r.wx[inner], np.broadcast_to(wx_ref, r.wx[inner].shape), rtol=1e-10)
        np.testing.assert_allclose(r.wy[inner], np.broadcast_to(wy_ref, r.wy[inner].shape), rtol=1e-10)
    assert np.isnan(r.wx[np.abs(lat) < waf.LAT_MIN]).all()


def test_zonal_wave_on_sphere_has_zero_divergence_across_the_seam():
    lat, lon = _sphere()
    LA, LO = np.meshgrid(np.deg2rad(lat), np.deg2rad(lon), indexing="ij")
    psi = 5e6 * np.cos(4 * LO) * np.cos(LA) ** 2
    U = 20 * np.ones_like(psi)
    r = waf.tn01_flux(psi, U, 0 * U, lat, lon)
    rows = np.isfinite(r.wx).all(1)
    # the flux is zonally uniform in every row, the seam columns included ...
    np.testing.assert_allclose(r.wx[rows], r.wx[rows][:, :1] * np.ones((1, lon.size)), rtol=1e-10)
    assert np.abs(r.wy[rows]).max() < 1e-10 * np.abs(r.wx[rows]).max()
    # ... so it has no divergence anywhere
    assert np.nanmax(np.abs(r.div)) < 1e-10 * np.nanmax(np.abs(r.wx)) / A
    # a one-sided difference at the seam (periodic=False) would not keep that
    q = waf.tn01_flux(psi, U, 0 * U, lat, lon, periodic=False)
    assert np.nanmax(np.abs(q.div)) > 1e3 * np.nanmax(np.abs(r.div))


def test_grid_checks_and_closed_grid():
    lat = np.arange(-88.0, 88.1, 4.0)
    lon = np.arange(0.0, 360.0, 4.0)
    LA, LO = np.meshgrid(np.deg2rad(lat), np.deg2rad(lon), indexing="ij")
    psi = 5e6 * np.cos(3 * LO + 2 * LA)
    U = 15 + 10 * np.cos(LA)
    V = np.sin(2 * LO) * np.cos(LA)
    r = waf.tn01_flux(psi, U, V, lat, lon)
    lonc = np.r_[lon, 360.0]
    close = lambda f: np.concatenate([f, f[:, :1]], axis=1)   # noqa: E731
    rc = waf.tn01_flux(close(psi), close(U), close(V), lat, lonc)
    np.testing.assert_allclose(rc.wx, close(r.wx), equal_nan=True)
    np.testing.assert_allclose(rc.div, close(r.div), equal_nan=True)
    with pytest.raises(ValueError):
        waf.tn01_flux(psi[:, :40], U[:, :40], V[:, :40], lat, lon[:40])
    waf.tn01_flux(psi[:, :40], U[:, :40], V[:, :40], lat, lon[:40], periodic=False)


def test_per_member_flux_does_not_fade_with_phase_spread():
    lat, lon = _sphere(4.0)
    LA, LO = np.meshgrid(np.deg2rad(lat), np.deg2rad(lon), indexing="ij")
    U = 20 * np.ones(LA.shape)
    members = [5e6 * np.cos(4 * LO + 2 * LA + ph) for ph in (0.0, np.pi)]   # opposite phases
    ens = waf.ensemble_tn01_flux(members, U, 0 * U, lat, lon)
    one = waf.tn01_flux(members[0], U, 0 * U, lat, lon)
    of_mean = waf.tn01_flux(np.mean(members, 0), U, 0 * U, lat, lon)
    np.testing.assert_allclose(ens.wx, one.wx, rtol=1e-10, equal_nan=True)
    assert np.nanmax(np.abs(of_mean.wx)) < 1e-12 * np.nanmax(np.abs(one.wx))
    assert ens.n == 2 and np.nanmin(ens.agree) >= 0.5


def test_lead_smooth_and_lowpass_anomaly():
    x = np.arange(10.0)[:, None] * np.ones((1, 3))
    s = waf.lead_smooth(x, 5, axis=0)
    np.testing.assert_allclose(s[5], 5.0)
    np.testing.assert_allclose(s[0], 1.0)            # truncated window at the end
    times = np.datetime64("2026-01-01") + np.arange(40) * np.timedelta64(12, "h")
    f = np.ones((40, 2, 2)) * 3.0
    anom, n = waf.lowpass_anomaly(f, times, lambda doy: np.full((2, 2), 1.0), times[-1], days=10)
    assert n == 20
    np.testing.assert_allclose(anom, 2.0)
    none, n2 = waf.lowpass_anomaly(f, times, lambda doy: 0.0, times[-1], days=5)
    assert none is None and n2 == 10


def test_streamfunction_path_with_pyshtools():
    pytest.importorskip("pyshtools")
    xr = pytest.importorskip("xarray")
    lat = np.arange(-90.0, 90.1, 1.0)
    lon = np.arange(0.0, 360.0, 1.0)
    LA, LO = np.meshgrid(np.deg2rad(lat), np.deg2rad(lon), indexing="ij")
    u = 20 * np.cos(LA)                                # solid-body rotation: psi = -20 a sin(lat)
    da = lambda f: xr.DataArray(f, dims=("latitude", "longitude"), coords=dict(latitude=lat, longitude=lon))  # noqa: E731
    psi, la, lo = waf.streamfunction_anomaly(da(u), da(0 * u), lmax=63)
    ref = -20 * A * np.sin(np.deg2rad(la))[:, None]
    assert np.abs(psi - ref).max() < 1.5e-2 * np.abs(ref).max()
    UU = 20 * np.cos(np.deg2rad(la))[:, None] * np.ones_like(psi)
    r = waf.tn01_flux(1e-3 * psi * np.cos(3 * np.deg2rad(lo))[None, :], UU, 0 * UU, la, lo, div_lmax=15)
    assert r.div.shape == psi.shape and np.isfinite(r.div[(np.abs(la) > 30) & (np.abs(la) < 75)]).all()
