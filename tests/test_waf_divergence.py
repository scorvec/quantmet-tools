"""Synthetic tests of the truncated wave-activity-flux divergence near a mask.

The TN01 flux has a 1/|U| factor, so it is masked where the basic-state wind
is weak (and in the tropics). Its divergence is then truncated at a low
spherical-harmonic degree (T15), a global filter, so whatever the method does
at the mask edge spreads into the valid cells. Three methods
(quantmet.waf.DIV_METHODS):

  zero         centred-difference divergence of the zero-filled flux, the
               masked cells' divergence set to zero, truncated (0.4.0 on a
               closed 360-column grid; the default since 0.4.1 on every grid)
  as_computed  the same divergence truncated as computed (0.4.0 on an open
               grid)
  taper        the flux tapered to zero over taper_deg with a cos^2 ramp,
               divergence taken spectrally (weak form), truncated

Outcome (python tests/test_waf_divergence.py 20; medians): "zero" has the
smallest error in (a) and half the spurious divergence of the others in (b);
"taper" has the largest of both (and grows with taper_deg) but the least rim
leakage in (c), where all three stay a few % of the signal. Hence the default.

Cases, on the extended DH2 grid of a T63 streamfunction (1.41 deg):
  (a) KNOWN DIVERGENCE: W = grad(chi) + k x grad(psi) with chi, psi random
      band-limited (degree <= 12) fields, so div W = lap(chi) exactly and T15
      leaves it unchanged; tropical band plus random weak-wind caps masked.
      Error of each method in the valid cells.
  (b) DIVERGENCE-FREE: W = k x grad(psi), same masks; any divergence in the
      valid cells is spurious (normalised by the divergence of a gradient flux
      of the same spectrum).
  (c) NEAR-SINGULAR WEAK-WIND BAND: the real TN01 flux of a random psi' in a
      westerly that drops to zero across a band (and a cap), so the flux next
      to the mask edge carries a large 1/|U|. Leakage = the change in the
      divergence in cells whose own flux does not change when the rim's wind is
      raised to U_FLOOR.

python tests/test_waf_divergence.py prints the full table (more realisations).
"""

from __future__ import annotations

import numpy as np
import pytest

from quantmet import waf

pysh = pytest.importorskip("pyshtools")

A = waf.A_EARTH
N = 128  # DH rows: the T63 streamfunction grid
LMAX = 15
U_FLOOR = 10.0  # m/s: rim wind for the leakage reference in (c)
FAR = 10.0  # deg: "far from the mask"
METHODS = ("zero", "as_computed", "taper")


def grid(n=N):
    lat = 90.0 - 180.0 * np.arange(n + 1) / n
    lon = 360.0 * np.arange(2 * n + 1) / (2 * n)
    return lat, lon


def _coeffs(rng, lmin, lmax, slope=1.0, n=N):
    c = np.zeros((2, n // 2, n // 2))
    for ll in range(lmin, lmax + 1):
        c[:, ll, : ll + 1] = rng.standard_normal((2, ll + 1)) * ll**-slope
    c[1, :, 0] = 0.0
    return c


def _grad(c):
    """(east, north) components of grad f for coefficients c on the grid."""
    th, ph = pysh.expand.MakeGradientDH(c, sampling=2, extend=True, radius=A)
    return ph, -th


def _lap(c):
    ll = np.arange(c.shape[1])
    return pysh.expand.MakeGridDH(c * (-ll * (ll + 1) / A**2)[None, :, None], sampling=2, extend=True)


def _dist_deg(lat, lon, la0, lo0):
    LA, LO = np.meshgrid(np.deg2rad(lat), np.deg2rad(lon), indexing="ij")
    a0, o0 = np.deg2rad(la0), np.deg2rad(lo0)
    c = np.sin(LA) * np.sin(a0) + np.cos(LA) * np.cos(a0) * np.cos(LO - o0)
    return np.rad2deg(np.arccos(np.clip(c, -1.0, 1.0)))


def random_mask(rng, lat, lon, ncap=3):
    """Tropical band plus ncap weak-wind caps (radius 8-20 deg, centres 30-70 deg)."""
    m = np.broadcast_to(np.abs(lat)[:, None] < waf.LAT_MIN, (lat.size, lon.size)).copy()
    for _ in range(ncap):
        la0 = rng.uniform(30, 70) * rng.choice([-1, 1])
        m |= _dist_deg(lat, lon, la0, rng.uniform(0, 360)) < rng.uniform(8, 20)
    return m


def _wmean(f, sel, lat):
    w = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], f.shape)[sel]
    return np.sum(w * f[sel]) / np.sum(w)


def _rms(f, sel, lat):
    return float(np.sqrt(_wmean(f * f, sel, lat)))


def _far(mask, lat, lon):
    return waf.mask_taper(mask, lat, lon, FAR) >= 1.0


def case_a(rng, method, taper_deg=waf.TAPER_DEG):
    lat, lon = grid()
    chi, psi = _coeffs(rng, 1, 12), _coeffs(rng, 1, 12)
    gx, gy = _grad(chi)
    rx, ry = _grad(psi)
    wx, wy = gx - ry, gy + rx  # grad(chi) + k x grad(psi)
    truth = _lap(chi)
    mask = random_mask(rng, lat, lon)
    d = waf.masked_divergence(wx, wy, mask, lat, lon, LMAX, method, taper_deg)
    ok, far = ~mask, _far(mask, lat, lon)
    e = d - truth
    return {
        "rms": _rms(e, ok, lat) / _rms(truth, ok, lat),
        "far": _rms(e, far, lat) / _rms(truth, far, lat),
        "max": float(np.abs(e[ok]).max() / np.abs(truth[ok]).max()),
    }


def case_b(rng, method, taper_deg=waf.TAPER_DEG):
    lat, lon = grid()
    psi = _coeffs(rng, 1, 12)
    rx, ry = _grad(psi)
    wx, wy = -ry, rx  # k x grad(psi): no divergence
    ref = _lap(psi)  # what a gradient flux of the same spectrum diverges by
    mask = random_mask(rng, lat, lon)
    d = waf.masked_divergence(wx, wy, mask, lat, lon, LMAX, method, taper_deg)
    ok, far = ~mask, _far(mask, lat, lon)
    return {
        "rms": _rms(d, ok, lat) / _rms(ref, ok, lat),
        "far": _rms(d, far, lat) / _rms(ref, far, lat),
        "max": float(np.abs(d[ok]).max() / np.abs(ref[ok]).max()),
    }


def weak_wind(rng, lat, lon, kind):
    """Basic state (U, V): 25 m/s westerlies that fall linearly to zero across a
    10-deg ramp onto a wavy zonal band (kind 'band') or a 6-deg cap ('cap')."""
    U0, h = 25.0, 10.0
    LA, LO = np.meshgrid(lat, np.deg2rad(lon), indexing="ij")
    hemi = rng.choice([-1, 1])
    if kind == "band":
        phc = hemi * rng.uniform(40, 60) + 4.0 * np.sin(2 * LO + rng.uniform(0, 2 * np.pi))
        r = np.abs(LA - phc) - 1.0
    else:
        r = _dist_deg(lat, lon, hemi * rng.uniform(40, 65), rng.uniform(0, 360)) - 6.0
    U = U0 * np.clip(r / h, 0.0, 1.0)
    V = 2.0 * np.sin(3 * LO) * np.cos(np.deg2rad(LA)) * np.clip(r / h, 0.0, 1.0)
    return U, V


def case_a_src(rng, method, taper_deg=waf.TAPER_DEG):
    """(a) with the divergence concentrated INSIDE the masked caps: chi is a
    sum of bumps centred in them (T12), so strong flux leaves the mask."""
    lat, lon = grid()
    mask = np.broadcast_to(np.abs(lat)[:, None] < waf.LAT_MIN, (lat.size, lon.size)).copy()
    bump = np.zeros(mask.shape)
    for _ in range(3):
        la0, lo0, rad = rng.uniform(30, 70) * rng.choice([-1, 1]), rng.uniform(0, 360), rng.uniform(10, 20)
        dist = _dist_deg(lat, lon, la0, lo0)
        mask |= dist < rad
        bump += rng.choice([-1, 1]) * np.exp(-((dist / (0.6 * rad)) ** 2))
    chi = pysh.SHGrid.from_array(bump[:-1, :-1], grid="DH").expand().coeffs
    chi[:, 13:, :] = 0.0
    c = np.zeros((2, N // 2, N // 2))
    c[:, : chi.shape[1], : chi.shape[2]] = chi
    wx, wy = _grad(c)
    truth = _lap(c)
    d = waf.masked_divergence(wx, wy, mask, lat, lon, LMAX, method, taper_deg)
    ok, far = ~mask, _far(mask, lat, lon)
    e = d - truth
    return {
        "rms": _rms(e, ok, lat) / _rms(truth, ok, lat),
        "far": _rms(e, far, lat) / _rms(truth, far, lat),
        "max": float(np.abs(e[ok]).max() / np.abs(truth[ok]).max()),
    }


def _rotate(U, V, ang):
    c, s = np.cos(ang), np.sin(ang)
    return c * U - s * V, s * U + c * V


def case_c(rng, method, taper_deg=waf.TAPER_DEG, kind="band", noise=1.5):
    """The TN01 flux is homogeneous of degree 0 in (U, V) (the 1/|U| cancels the
    U, V in the brackets), so its MAGNITUDE does not blow up near the mask: what
    becomes singular as |U| -> 0 is its DIRECTION. A basic state known to
    +-noise m/s has a direction error ~ noise/|U|: 27 deg at the 3 m/s mask
    edge, 3 deg in a 25 m/s jet. Rotate the rim's wind (UMIN <= |U| < U_FLOOR)
    by arctan(delta/|U|), delta a smooth random field of rms noise m/s (speed
    and mask unchanged): leakage = the change in the divergence in the valid
    cells outside the rim, whose own flux is unchanged."""
    lat, lon = grid()
    psi = pysh.expand.MakeGridDH(1e7 * _coeffs(rng, 4, 20, slope=0.5), sampling=2, extend=True)
    U, V = weak_wind(rng, lat, lon, kind)
    spd = np.hypot(U, V)
    rim = (spd >= waf.UMIN) & (spd < U_FLOOR)
    delta = pysh.expand.MakeGridDH(_coeffs(rng, 1, 10, slope=0.0), sampling=2, extend=True)
    delta *= noise / np.sqrt(np.mean(delta**2))
    Up, Vp = _rotate(U, V, np.where(rim, np.arctan2(delta, np.maximum(spd, 1e-9)), 0.0))
    r = waf.tn01_flux(psi, Up, Vp, lat, lon, div_lmax=LMAX, div_method=method, taper_deg=taper_deg)
    q = waf.tn01_flux(psi, U, V, lat, lon, div_lmax=LMAX, div_method=method, taper_deg=taper_deg)
    assert np.array_equal(r.mask, q.mask)
    same = ~r.mask & ~rim  # cells whose own flux is identical in both
    far = same & _far(r.mask, lat, lon)
    e = r.div - q.div
    return {
        "rms": _rms(e, same, lat) / _rms(q.div, same, lat),
        "far": _rms(e, far, lat) / _rms(q.div, far, lat),
        "max": float(np.nanmax(np.abs(e[same])) / np.nanmax(np.abs(q.div[same]))),
    }


CASES = {
    "a": case_a,
    "a_src": case_a_src,
    "b": case_b,
    "c_band": lambda rng, m, t=waf.TAPER_DEG: case_c(rng, m, t, "band"),
    "c_cap": lambda rng, m, t=waf.TAPER_DEG: case_c(rng, m, t, "cap"),
}


def run(case, method, nreal, seed=0, taper_deg=waf.TAPER_DEG):
    """Metrics of one case and method over nreal realisations (the same seeds
    for every method, so the comparison is paired)."""
    out = [CASES[case](np.random.default_rng(seed + k), method, taper_deg) for k in range(nreal)]
    return {key: np.array([o[key] for o in out]) for key in out[0]}


# ----------------------------------------------------------------------------- tests


def test_spectral_divergence_is_exact_for_band_limited_fields():
    rng = np.random.default_rng(1)
    lat, lon = grid()
    chi, psi = _coeffs(rng, 1, 12), _coeffs(rng, 1, 12)
    gx, gy = _grad(chi)
    rx, ry = _grad(psi)
    truth = _lap(chi)
    d = waf.sh_divergence(gx - ry, gy + rx, lat, lon, lmax=LMAX)
    assert np.abs(d - truth).max() < 1e-12 * np.abs(truth).max()
    assert np.abs(waf.sh_divergence(-ry, rx, lat, lon, lmax=LMAX)).max() < 1e-12 * np.abs(truth).max()
    # the same on the plain (open) grid, and with the full degree range
    dp = waf.sh_divergence((gx - ry)[:-1, :-1], (gy + rx)[:-1, :-1], lat[:-1], lon[:-1])
    assert np.abs(dp - truth[:-1, :-1]).max() < 1e-10 * np.abs(truth).max()
    with pytest.raises(ValueError):
        waf.sh_divergence(np.where(lat[:, None] > 80, np.nan, gx), gy, lat, lon, lmax=LMAX)


def test_taper_weights():
    lat, lon = grid(64)
    mask = np.broadcast_to(np.abs(lat)[:, None] < 20, (lat.size, lon.size))
    w = waf.mask_taper(mask, lat, lon, 5.0)
    assert (w[mask] == 0).all() and (w[np.abs(lat) > 25.1] == 1).all()
    row = np.argmin(np.abs(lat - 22.5))  # 2.5 deg from the nearest masked row (lat 19.69)
    d = lat[row] - lat[np.abs(lat) < 20].max()
    np.testing.assert_allclose(w[row], np.sin(np.pi / 2 * d / 5.0) ** 2, rtol=1e-6)
    np.testing.assert_array_equal(waf.mask_taper(mask, lat, lon, 0.0), (~mask).astype(float))


def test_methods_agree_on_open_and_closed_grids():
    rng = np.random.default_rng(2)
    lat, lon = grid(64)
    psi = pysh.expand.MakeGridDH(1e7 * _coeffs(rng, 4, 20, slope=0.5, n=64), sampling=2, extend=True)
    U, V = weak_wind(rng, lat, lon, "band")
    for m in waf.DIV_METHODS:
        rc = waf.tn01_flux(psi, U, V, lat, lon, div_lmax=LMAX, div_method=m)
        ro = waf.tn01_flux(psi[:, :-1], U[:, :-1], V[:, :-1], lat, lon[:-1], div_lmax=LMAX, div_method=m)
        np.testing.assert_allclose(rc.div[:, :-1], ro.div, rtol=1e-12, atol=1e-25, equal_nan=True)
        np.testing.assert_array_equal(rc.mask[:, :-1], ro.mask)


def test_legacy_methods_reproduce_040():
    """zero = the 0.4.0 closed-grid algorithm, as_computed = the 0.4.0 open-grid one."""
    rng = np.random.default_rng(4)
    lat, lon = grid(64)
    psi = pysh.expand.MakeGridDH(1e7 * _coeffs(rng, 4, 20, slope=0.5, n=64), sampling=2, extend=True)
    U, V = weak_wind(rng, lat, lon, "cap")
    r = waf.tn01_flux(psi, U, V, lat, lon)  # untruncated: identical for all methods
    wx0, wy0 = np.nan_to_num(r.wx), np.nan_to_num(r.wy)
    cos = np.clip(np.cos(np.deg2rad(lat)), 1e-3, None)[:, None]
    dl = np.deg2rad(lon[1] - lon[0])

    def ddl(f):  # periodic centred difference, 360 column re-appended
        d = (np.roll(f[:, :-1], -1, 1) - np.roll(f[:, :-1], 1, 1)) / (2 * dl)
        return np.concatenate([d, d[:, :1]], axis=1)

    fd = (ddl(wx0) + np.gradient(wy0 * cos, np.deg2rad(lat), axis=0)) / (A * cos)
    np.testing.assert_allclose(np.where(r.mask, np.nan, fd), r.div, rtol=1e-12, equal_nan=True)
    zero = np.where(r.mask, np.nan, waf.sh_truncate(np.where(r.mask, 0.0, fd), LMAX))
    asis = np.where(r.mask, np.nan, waf.sh_truncate(fd, LMAX))
    for m, ref in (("zero", zero), ("as_computed", asis)):
        got = waf.tn01_flux(psi, U, V, lat, lon, div_lmax=LMAX, div_method=m).div
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-25, equal_nan=True)
    with pytest.raises(ValueError):
        waf.tn01_flux(psi, U, V, lat, lon, div_lmax=LMAX, div_method="nope")
    assert waf.DIV_METHODS[0] == "zero"
    np.testing.assert_array_equal(
        waf.tn01_flux(psi, U, V, lat, lon, div_lmax=LMAX).div,
        waf.tn01_flux(psi, U, V, lat, lon, div_lmax=LMAX, div_method="zero").div,
    )


def test_flux_is_homogeneous_of_degree_zero_in_the_basic_state():
    # why (c) perturbs the rim wind's direction, not its speed
    rng = np.random.default_rng(5)
    lat, lon = grid(64)
    psi = pysh.expand.MakeGridDH(1e7 * _coeffs(rng, 4, 20, slope=0.5, n=64), sampling=2, extend=True)
    U, V = weak_wind(rng, lat, lon, "band")
    r = waf.tn01_flux(psi, U, V, lat, lon, umin=0.0)
    q = waf.tn01_flux(psi, 7.0 * U, 7.0 * V, lat, lon, umin=0.0)
    np.testing.assert_allclose(q.wx, r.wx, rtol=1e-10, atol=1e-12 * np.nanmax(np.abs(r.wx)), equal_nan=True)


def test_values_under_the_mask_never_enter():
    rng = np.random.default_rng(6)
    lat, lon = grid(64)
    gx, gy = _grad(_coeffs(rng, 1, 12, n=64))
    mask = random_mask(rng, lat, lon)
    junk = np.where(mask, 1e6 * rng.standard_normal(mask.shape), 0.0)
    for m in waf.DIV_METHODS:
        a = waf.masked_divergence(gx, gy, mask, lat, lon, LMAX, m)
        b = waf.masked_divergence(gx + junk, np.where(mask, np.nan, gy), mask, lat, lon, LMAX, m)
        np.testing.assert_array_equal(a, b)
        assert np.isnan(a[mask]).all() and np.isfinite(a[~mask]).all()


def _paired(case, nreal=4):
    return {m: run(case, m, nreal, seed=50) for m in METHODS}


def test_a_known_divergence_zero_has_the_smallest_error():
    r = _paired("a")
    assert (r["zero"]["rms"] < r["as_computed"]["rms"]).all()
    assert (r["as_computed"]["rms"] < r["taper"]["rms"]).all()
    assert np.median(r["zero"]["rms"]) < 0.3


def test_b_divergence_free_flux_zero_halves_the_spurious_divergence():
    r = _paired("b")
    assert (r["zero"]["rms"] < 0.7 * r["as_computed"]["rms"]).all()
    assert (r["as_computed"]["rms"] < r["taper"]["rms"]).all()
    assert np.median(r["zero"]["rms"]) < 0.12


def test_c_rim_leakage_is_small_and_least_with_the_taper():
    r = _paired("c_band")
    assert (r["taper"]["rms"] < r["as_computed"]["rms"]).all()
    assert (r["as_computed"]["rms"] < r["zero"]["rms"]).all()
    assert max(np.median(v["rms"]) for v in r.values()) < 0.1


if __name__ == "__main__":
    import sys

    nreal = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    tapers = [float(t) for t in sys.argv[2:]] or [waf.TAPER_DEG]
    print(
        f"{nreal} realisations per case; median [10th-90th percentile]; rms/far relative to the true "
        f"(a) or reference (b, c) divergence in the valid cells, far = >= {FAR:g} deg from the mask"
    )
    for case in CASES:
        res = {m: run(case, m, nreal) for m in ("zero", "as_computed")}
        for t in tapers:
            res[f"taper {t:g}"] = run(case, "taper", nreal, taper_deg=t)
        print(f"\ncase {case}")
        for m, r in res.items():
            cells = "  ".join(
                f"{k} {np.median(v):7.3f} [{np.percentile(v, 10):.3f}-{np.percentile(v, 90):.3f}]" for k, v in r.items()
            )
            print(f"  {m:12s} {cells}")
        for other in ("zero", f"taper {tapers[0]:g}"):
            ratio = res["as_computed"]["rms"] / res[other]["rms"]
            print(
                f"  as_computed / {other}: rms ratio median {np.median(ratio):.2f}, "
                f"{other} lower in {np.mean(ratio > 1) * 100:.0f} % of realisations"
            )
