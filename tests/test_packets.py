import numpy as np

from quantmet.packets import packet_envelope, band_mean


def test_gaussian_packet_envelope():
    lon = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    env = 20.0 * np.exp(-((lon - np.pi) / 0.6) ** 2)
    v = env * np.cos(8 * lon)                                 # carrier k=8 inside the 4-15 band
    e = packet_envelope(v, 4, 15)
    core = np.abs(lon - np.pi) < 1.2
    assert np.max(np.abs(e[core] - env[core])) < 0.05 * env.max()


def test_band_mean_weights():
    lat = np.array([0.0, 60.0]); f = np.array([[1.0], [3.0]])
    w = np.cos(np.deg2rad(lat))
    assert np.allclose(band_mean(f, lat, -1, 61, lat_axis=0), (f[:, 0] * w).sum() / w.sum())
