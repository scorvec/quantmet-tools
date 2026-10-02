import numpy as np

from quantmet.tc import recurvature, hart_parameters, et_times, outflow_index, within, pv_advection


def test_recurvature_westward_then_eastward():
    steps = np.arange(0, 241, 12)
    lon = np.r_[np.linspace(140, 130, 9), np.linspace(130.5, 150, 12)]
    lat = np.linspace(15, 40, len(steps))
    r = recurvature(dict(steps=steps, lat=lat, lon=lon))
    assert r is not None and 84 <= r <= 108
    assert recurvature(dict(steps=steps, lat=lat, lon=np.linspace(140, 160, len(steps)))) is None


def _vortex(levs, warm):
    lat = np.arange(10, 40.1, 0.25); lon = np.arange(120, 150.1, 0.25)
    r = np.hypot(*np.meshgrid((lon - 135) * np.cos(np.deg2rad(25)), lat - 25)) * 111.0   # km, (lat, lon)
    base = {925: 750, 850: 1450, 700: 3000, 600: 4300, 500: 5800, 400: 7500, 300: 9500}
    z = []
    for p in levs:
        # warm core: the low is deepest at the bottom and weakens upward; cold core: deepest aloft
        depth = (60 + 0.15 * (p - 300)) if warm else (60 + 0.15 * (1000 - p))
        z.append(base[p] - depth * np.exp(-(r / 250.0) ** 2))
    return np.array(z), lat, lon


def test_symmetric_warm_core_vortex():
    levs = [925, 850, 700, 600, 500, 400, 300]
    z, lat, lon = _vortex(levs, warm=True)
    r = hart_parameters(z, levs, lat, lon, 25.0, 135.0, motion_deg=0.0, low=(925, 850, 700, 600), thick=(600, 925))
    assert abs(r["B"]) < 1.0 and r["VTL"] > 0 and r["VTU"] > 0


def test_cold_core_reads_negative():
    levs = [925, 850, 700, 600, 500, 400, 300]
    z, lat, lon = _vortex(levs, warm=False)
    r = hart_parameters(z, levs, lat, lon, 25.0, 135.0, motion_deg=0.0, low=(925, 850, 700, 600), thick=(600, 925))
    assert r["VTL"] < 0 and r["VTU"] < 0


def test_et_times():
    s = dict(h=[0, 12, 24, 36, 48], B=[0, 5, 12, 20, 30], VTL=[50, 40, 10, -5, -20])
    assert et_times(s) == (24, 36)


def test_outflow_index_and_advection_sign():
    lat = np.arange(0, 60.1, 0.5); lon = np.arange(100, 160, 0.5)
    pv = np.broadcast_to((lat / 10.0)[:, None], (len(lat), len(lon))).copy()     # PV rising poleward
    vchi = np.full_like(pv, 5.0); uchi = np.zeros_like(pv)                         # poleward outflow
    adv = pv_advection(uchi, vchi, pv, lat, lon, sigma_deg=0)
    assert np.nanmedian(adv[5:-5]) < 0                          # low PV pushed poleward: negative advection
    assert outflow_index(adv, lat, lon, 30.0, 130.0) > 0
    assert within(lat, lon, 30.0, 130.0, 500.0).sum() > 0
