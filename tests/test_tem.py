import numpy as np
import pytest

from quantmet import tem

A, G, OM, KAP = tem.A_EARTH, tem.G, tem.OMEGA, tem.KAPPA
trapezoid = getattr(np, "trapezoid", None) or np.trapz


def _lon(n):
    return np.arange(n) * 2 * np.pi / n


def test_zonally_symmetric_state_has_no_eddy_flux():
    lat = np.linspace(-80, 80, 81)
    plev = np.geomspace(1, 300, 30)
    P, L = np.meshgrid(plev, np.deg2rad(lat), indexing="ij")
    base = [30 * np.cos(L) * np.log(P), 2 * np.sin(2 * L), 220 + 30 * np.cos(L) ** 2, 1e-3 * np.sin(L)]
    u, v, t, w = (np.repeat(b[..., None], 48, axis=-1) for b in base)
    r = tem.eddy_covariances(u, v, t, w)
    for k in ("vT", "uv", "uw", "vT_k1", "uv_k2", "uw_k3p"):
        assert np.abs(r[k]).max() < 1e-12
    tt = tem.tem_terms(lat, plev, r)
    for k in ("Fphi", "Fp", "epd", "epd_k1", "epd_k2", "epd_k3p"):
        assert np.abs(tt[k]).max() < 1e-12 * max(1.0, np.abs(tt["cor"]).max())
    np.testing.assert_allclose(tt["vstar"], r["vbar"], atol=1e-10)
    np.testing.assert_allclose(tt["omstar"], r["wbar"], atol=1e-12)


@pytest.mark.parametrize("n", [36, 35])
@pytest.mark.parametrize("k", [1, 3, 7])
def test_single_tilted_wave_heat_flux(n, k):
    lam = _lon(n)
    V, T, a, b = 7.0, 3.0, 0.4, 1.3
    v = 5.0 + V * np.cos(k * lam + a)
    t = 250.0 + T * np.cos(k * lam + b)
    c = tem.cospectrum(v, t)
    assert c.shape == (n // 2,)
    np.testing.assert_allclose(c.sum(), 0.5 * V * T * np.cos(a - b), rtol=1e-12)
    np.testing.assert_allclose(c[k - 1], 0.5 * V * T * np.cos(a - b), rtol=1e-12)
    assert np.abs(np.delete(c, k - 1)).max() < 1e-12


def test_nyquist_term_is_not_doubled():
    n = 36
    k = n // 2
    lam = _lon(n)
    V, T, a, b = 7.0, 3.0, 0.4, 1.3
    v = V * np.cos(k * lam + a)                 # = V cos(a) (-1)^j: the sine part is unresolved
    t = T * np.cos(k * lam + b)
    direct = np.mean((v - v.mean()) * (t - t.mean()))
    np.testing.assert_allclose(direct, V * T * np.cos(a) * np.cos(b), rtol=1e-12)
    c = tem.cospectrum(v, t)
    np.testing.assert_allclose(c[-1], direct, rtol=1e-12)
    assert not np.isclose(c[-1], 2 * direct)
    lat = np.array([-60.0, 60.0])
    hf = tem.eddy_heat_flux(np.stack([v, v]), np.stack([t, t]), lat, kmax=k, poleward=False)
    np.testing.assert_allclose(hf.total, direct, rtol=1e-12)


@pytest.mark.parametrize("n", [64, 63])
def test_wavenumber_sum_equals_direct_covariance(n):
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(3, 5, n)), rng.normal(size=(3, 5, n))
    direct = ((a - a.mean(-1, keepdims=True)) * (b - b.mean(-1, keepdims=True))).mean(-1)
    np.testing.assert_allclose(tem.cospectrum(a, b).sum(-1), direct, rtol=1e-10, atol=1e-14)


def test_heat_flux_band_and_poleward_sign():
    n, lam = 72, _lon(72)
    lat = np.arange(-80.0, 80.1, 2.5)
    amp = np.cos(np.deg2rad(lat))[:, None]
    v = amp * np.cos(2 * lam)
    t = amp * np.cos(2 * lam - 0.5)[None, :] * np.sign(lat)[:, None]   # poleward in both hemispheres
    hf_n = tem.eddy_heat_flux(v, t, lat, kmax=10, lat_band=(45, 75))
    hf_s = tem.eddy_heat_flux(v, t, lat, kmax=10, lat_band=(-75, -45))
    assert hf_n.total > 0 and hf_s.total > 0
    np.testing.assert_allclose(hf_n.total, hf_s.total, rtol=1e-12)
    assert hf_n.by_k.shape == (10,) and np.argmax(hf_n.by_k) == 1
    raw = tem.eddy_heat_flux(v, t, lat, kmax=10, lat_band=(-75, -45), poleward=False)
    np.testing.assert_allclose(raw.total, -hf_s.total, rtol=1e-12)


def _state(nlat=161, nlev=240, eddy_top_zero=True):
    """A continuity-consistent Eulerian circulation plus an eddy heat flux."""
    lat = np.linspace(-80, 80, nlat)
    plev = np.geomspace(0.5, 150.0, nlev)
    p = plev * 100.0
    p0, p1 = p[0], p[-1]
    P, PHI = np.meshgrid(p, np.deg2rad(lat), indexing="ij")
    C = 5e9
    s = np.sin(np.pi * (P - p0) / (p1 - p0))
    ds = np.pi / (p1 - p0) * np.cos(np.pi * (P - p0) / (p1 - p0))
    shape = np.cos(PHI) ** 2 * np.sin(PHI)
    dshape = -2 * np.cos(PHI) * np.sin(PHI) ** 2 + np.cos(PHI) ** 3
    vbar = G / (2 * np.pi * A * np.cos(PHI)) * C * ds * shape
    wbar = -G / (2 * np.pi * A ** 2 * np.cos(PHI)) * C * s * dshape
    tbar = 210.0 + 25.0 * np.cos(PHI) ** 2 + 10 * np.log(P / p1)
    prof = np.sin(np.pi * (P - p0) / (p1 - p0)) if eddy_top_zero else np.log(P / p0) + 1
    vT = 8.0 * np.cos(PHI) ** 2 * np.sin(2 * PHI) * prof
    ubar = 40 * np.cos(PHI) * np.sin(2 * PHI) ** 2 * np.log(P / 10.0)
    r = dict(ubar=ubar, vbar=vbar, tbar=tbar, wbar=wbar, vT=vT, uv=np.zeros_like(vT), uw=np.zeros_like(vT))
    return lat, plev, r


def _interior(x, m=3):
    return x[m:-m, m:-m]


def test_residual_streamfunction_closes_continuity():
    lat, plev, r = _state(eddy_top_zero=False)
    tt = tem.tem_terms(lat, plev, r, groups=[])
    psi = tem.residual_streamfunction(lat, plev, r["vbar"], r["tbar"], r["vT"])
    vs, ws = tem.residual_velocities(psi, lat, plev)
    # psi*'s own v*, omega* satisfy continuity to rounding (the discrete derivatives commute)
    phi, c = np.deg2rad(lat), np.cos(np.deg2rad(lat))
    p = plev * 100
    div = np.gradient(vs * c, phi, axis=1) / (A * c) + np.gradient(ws, p, axis=0)
    scale = np.abs(np.gradient(ws, p, axis=0)).max()
    assert np.abs(div).max() < 1e-10 * scale
    # ... and they are the TEM residual velocities computed directly from the eddy flux
    for mine, direct in ((vs, tt["vstar"]), (ws, tt["omstar"])):
        err = np.abs(_interior(mine - direct)).max() / np.abs(_interior(direct)).max()
        assert err < 2e-3


def test_downward_control_equals_direct_integral():
    lat, plev, r = _state()
    tt = tem.tem_terms(lat, plev, r, groups=[])
    G_total = -tt["fhat"] * tt["vstar"]              # steady state: -fhat v* = G
    psi_dc = tem.downward_control(lat, plev, tt["fhat"], G_total)
    psi_direct = tem.residual_streamfunction(lat, plev, r["vbar"], r["tbar"], r["vT"])
    ex = np.abs(lat) >= 15
    rel = np.nanmax(np.abs(_interior(psi_dc[:, ex] - psi_direct[:, ex]))) / np.abs(psi_direct).max()
    assert rel < 2e-3
    assert np.isnan(psi_dc[:, np.abs(lat) < 0.5]).all()   # fhat -> 0 on the equator
    # linear: the parts add up to the whole
    half = tem.downward_control(lat, plev, tt["fhat"], 0.3 * G_total)
    rest = tem.downward_control(lat, plev, tt["fhat"], 0.7 * G_total)
    np.testing.assert_allclose(half + rest, psi_dc, rtol=1e-10)
    # plev order does not matter
    flip = tem.downward_control(lat, plev[::-1], tt["fhat"][::-1], G_total[::-1])[::-1]
    np.testing.assert_allclose(flip, psi_dc, rtol=1e-10)


def test_fill_tropics_is_linear_in_sin_lat():
    lat = np.arange(-40.0, 40.1, 1.0)
    psi = np.tile(np.cbrt(lat), (3, 1))
    out = tem.fill_tropics(psi, lat, edge=15)
    s = np.sin(np.deg2rad(lat))
    m = np.abs(lat) < 15
    np.testing.assert_allclose(np.diff(out[0, m]) / np.diff(s[m]), (out[0, 55] - out[0, 25]) / (s[55] - s[25]))
    np.testing.assert_array_equal(out[:, ~m], psi[:, ~m])
    np.testing.assert_allclose(tem.fill_tropics(psi[:, ::-1], lat[::-1], 15), out[:, ::-1])


def test_upwelling_w_matches_area_mean_omega():
    lat, plev, r = _state()
    psi = tem.residual_streamfunction(lat, plev, r["vbar"], r["tbar"], r["vT"])
    up = tem.upwelling_w(psi, lat, plev)
    _, ws = tem.residual_velocities(psi, lat, plev)
    for i in (60, 120, 180):
        m = (lat >= up.lat_s[i]) & (lat <= up.lat_n[i])
        c = np.cos(np.deg2rad(lat[m]))
        w_direct = -trapezoid(ws[i, m] * c, lat[m]) / trapezoid(c, lat[m]) * tem.H_SCALE / (plev[i] * 100)
        np.testing.assert_allclose(up.w[i], w_direct, rtol=1e-3)
    assert (up.flux[5:-5] > 0).all()
    fixed = tem.upwelling_w(psi, lat, plev, lat_s=-30.0, lat_n=30.0)
    np.testing.assert_allclose(fixed.lat_n, 30.0)


def test_wave_groups_add_up_to_total():
    rng = np.random.default_rng(1)
    nlev, nlat, nlon = 12, 37, 48
    lat = np.linspace(-80, 80, nlat)
    plev = np.geomspace(1, 200, nlev)
    base_t = 220 + 20 * np.cos(np.deg2rad(lat))[None, :, None] + 5 * np.log(plev)[:, None, None]
    u, v, w = (rng.normal(size=(nlev, nlat, nlon)) for _ in range(3))
    t = base_t + rng.normal(size=(nlev, nlat, nlon))
    r = tem.eddy_covariances(10 + u, v, t, 1e-3 * w)
    tt = tem.tem_terms(lat, plev, r)
    np.testing.assert_allclose(tt["epd_k1"] + tt["epd_k2"] + tt["epd_k3p"], tt["epd"], rtol=1e-9, atol=1e-18)
    # leading (member) axis is carried through
    rm = tem.eddy_covariances(np.stack([10 + u, 10 + u]), np.stack([v, v]), np.stack([t, t]), np.stack([1e-3 * w] * 2))
    tm = tem.tem_terms(lat, plev, rm)
    np.testing.assert_allclose(tm["epd"][1], tt["epd"], rtol=1e-12)


def test_nan_input_rejected():
    u = np.ones((2, 3, 8))
    u[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        tem.eddy_covariances(u, u, u)
