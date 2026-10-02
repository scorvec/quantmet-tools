import shutil

import numpy as np
import pytest

if shutil.which("cc") is None:
    pytest.skip("no C compiler", allow_module_level=True)


@pytest.fixture()
def tc(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTMET_CACHE_DIR", str(tmp_path))
    from quantmet import tc_detect
    tc_detect._LIB = None
    yield tc_detect
    tc_detect._LIB = None


def _field(centres, dl=0.5):
    lat = np.arange(60.0, -60.01, -dl)
    lon = np.arange(0.0, 360.0, dl)
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    p = 1012.0 + 0.002 * LA ** 2
    for (la, lo, depth) in centres:
        dlon = (LO - lo + 180) % 360 - 180                    # wraps the date line / prime meridian
        p -= depth * np.exp(-((LA - la) ** 2 + dlon ** 2) / (2 * 1.2 ** 2))
    return p, lat, lon


def test_finds_lows_including_across_the_seam(tc):
    p, lat, lon = _field([(15.0, 0.0, 20.0), (-20.0, 200.0, 12.0), (30.0, 100.0, 1.0)])
    clat, clon, cp = tc.detect_candidates(p, lat, lon, ring_deg=2.5, depth_hpa=2.0)
    got = sorted(zip(np.round(clat, 1), np.round(clon, 1)))
    assert got == [(-20.0, 200.0), (15.0, 0.0)]                # the 1 hPa low is too shallow
    assert tc.library_path().exists() and tc.library_path().parent == tc._cache_dir()
    full = tc.detect_candidates(p, lat, lon, full=True, wind=np.hypot(*np.gradient(p)), p_max=1000.0)
    assert list(np.round(full["lat"], 1)) == [15.0] and full["depth"][0] > 10
    none = tc.detect_candidates(p, lat, lon, lat_band=(-10, 10))
    assert none[0].size == 0
