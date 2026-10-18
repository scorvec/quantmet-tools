import shutil

import numpy as np
import pytest

if shutil.which("cc") is None:
    pytest.skip("no C compiler", allow_module_level=True)


@pytest.fixture()
def tc(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTMET_CACHE_DIR", str(tmp_path))
    from quantmet import tc_detect

    tc_detect._lib.cache_clear()
    yield tc_detect
    tc_detect._lib.cache_clear()


def _field(centres, dl=0.5):
    lat = np.arange(60.0, -60.01, -dl)
    lon = np.arange(0.0, 360.0, dl)
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    p = 1012.0 + 0.002 * LA**2
    for la, lo, depth in centres:
        dlon = (LO - lo + 180) % 360 - 180  # wraps the date line / prime meridian
        p -= depth * np.exp(-((LA - la) ** 2 + dlon**2) / (2 * 1.2**2))
    return p, lat, lon


def test_finds_lows_including_across_the_seam(tc):
    p, lat, lon = _field([(15.0, 0.0, 20.0), (-20.0, 200.0, 12.0), (30.0, 100.0, 1.0)])
    clat, clon, _cp = tc.detect_candidates(p, lat, lon, ring_deg=2.5, depth_hpa=2.0)
    got = sorted(zip(np.round(clat, 1), np.round(clon, 1)))
    assert got == [(-20.0, 200.0), (15.0, 0.0)]  # the 1 hPa low is too shallow
    assert tc.library_path().exists() and tc.library_path().parent == tc._cache_dir()
    full = tc.detect_candidates(p, lat, lon, full=True, wind=np.hypot(*np.gradient(p)), p_max=1000.0)
    assert list(np.round(full["lat"], 1)) == [15.0] and full["depth"][0] > 10
    none = tc.detect_candidates(p, lat, lon, lat_band=(-10, 10))
    assert none[0].size == 0


def _annulus_depth(f, iy, ix, ry, rx):
    """Python reference: mean over the outer box (half-widths ry, rx) minus the inner box (ry//2, rx//2), both centred
    on (iy, ix) with longitude wrap and rows clipped at the edges, minus the centre value."""
    ny, nx = f.shape

    def box(hy, hx):
        rows = np.arange(max(0, iy - hy), min(ny, iy + hy + 1))
        cols = np.arange(ix - hx, ix + hx + 1) % nx
        return f[np.ix_(rows, cols)]

    o, i = box(ry, rx), box(ry // 2, rx // 2)
    return (o.sum() - i.sum()) / (o.size - i.size) - f[iy, ix]


@pytest.mark.parametrize("dlat,dlon", [(0.25, 0.25), (0.5, 0.25), (0.25, 0.5)])
def test_ring_depth_matches_a_centred_annulus_on_a_gradient(tc, dlat, dlon):
    """0.3.0 centred the ring one column WEST of the candidate; on a background zonal gradient that biases the depth
    by one column's worth of the gradient. The kernel must equal a centred Python annulus to float32 round-off --
    also when dlat != dlon, the ring being square in degrees."""
    lat = np.arange(40.0, -40.001, -dlat)
    lon = np.arange(0.0, 360.0, dlon)
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    rng = np.random.default_rng(1)
    f = 1010.0 + 0.08 * (LO - 150.0) + 0.3 * rng.standard_normal(LA.shape)  # strong east-west gradient + noise
    for la, lo in ((20.0, 150.0), (-15.0, 359.0), (5.0, 0.5)):  # incl. both sides of the seam
        d = np.hypot((LO - lo + 180) % 360 - 180, LA - la)
        f -= 15.0 * np.exp(-((d / 1.5) ** 2))
    f = f.astype(np.float32).astype(float)
    out = tc.detect_candidates(f, lat, lon, ring_deg=2.5, depth_hpa=2.0, full=True, min_deg=1.0)
    ry, rx = round(2.5 / dlat), round(2.5 / dlon)
    assert out["lat"].size >= 3
    for la, lo, dep in zip(out["lat"], out["lon"], out["depth"]):
        iy = int(np.argmin(np.abs(lat - la)))
        ix = int(np.argmin(np.abs(lon - lo)))
        assert abs(dep - _annulus_depth(f, iy, ix, ry, rx)) < 1e-3
    k = int(np.argmin(np.hypot(out["lon"] - 150.0, out["lat"] - 20.0)))
    iy = int(np.argmin(np.abs(lat - out["lat"][k])))
    ix = int(np.argmin(np.abs(lon - out["lon"][k])))
    west = _annulus_depth(f, iy, ix - 1, ry, rx) + f[iy, ix - 1] - f[iy, ix]  # ring one column west
    assert abs(out["depth"][k] - west) > 0.5 * 0.08 * dlon  # ... is what 0.3.0 reported


def test_ties_are_kept(tc):
    """A flat-bottomed low (two equal minimum points side by side) gives both as candidates: 'no lower neighbour'."""
    p, lat, lon = _field([(15.0, 100.0, 20.0)])
    iy = int(np.argmin(np.abs(lat - 15.0)))
    ix = int(np.argmin(np.abs(lon - 100.0)))
    p[iy, ix + 1] = p[iy, ix]
    _clat, clon, _ = tc.detect_candidates(p, lat, lon)
    assert sorted(np.round(clon, 2)) == [100.0, 100.5]
