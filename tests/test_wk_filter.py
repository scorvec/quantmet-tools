"""The site's self-tests (OLR Hovmoller and Kelvin tracker), ported, plus the 0.2 additions."""
import numpy as np
import pytest

from quantmet.wk_filter import WAVES, wave_mask, wk_filter

KELVIN_20 = dict(k=(1, 14), p=(2.5, 20.0), h=(8, 90), n=None)     # the velocity-potential tracker's band


def _wave(s, period, nt=1024, nx=144, amp=1.0):
    t = np.arange(nt)[:, None]
    lam = (np.arange(nx) / nx)[None, :]
    return amp * np.cos(2 * np.pi * (s * lam - t / period))          # eastward if s, period > 0


@pytest.mark.parametrize("s,period,keep,reject", [
    (4, 8.0, "Kelvin", "ER"),        # eastward s=+4, 8 d: Kelvin, not ER
    (-4, 20.0, "ER", "Kelvin"),      # westward s=-4, 20 d: ER, not Kelvin
    (2, 50.0, "MJO", "Kelvin"),      # eastward s=+2, 50 d: MJO, not Kelvin
])
def test_olr_hovmoller_selftest(s, period, keep, reject):
    x = _wave(s, period)
    assert np.std(wk_filter(x, keep)) > 5 * np.std(wk_filter(x, reject))


@pytest.mark.parametrize("s,period,kept", [(5, 8.0, True), (-5, 8.0, False), (2, 45.0, False)])
def test_kelvin_tracker_selftest(s, period, kept):
    x = _wave(s, period, nt=150)[:, None, :]                          # (time, lat, lon)
    y = wk_filter(x, KELVIN_20, pad_end=60, ramp=10)[:, 0]
    r = np.std(y[20:-20]) / np.std(x[20:-20, 0])
    assert (r > 0.7) if kept else (r < 0.1)


def test_custom_band_differs_from_canonical():
    x = _wave(1, 25.0, nt=730)                                       # k=1, 25 d: 18 m/s, he ~ 35 m
    assert np.std(wk_filter(x, "Kelvin")) > 0.5                      # inside 2.5-30 d
    assert np.std(wk_filter(x, KELVIN_20)) < 0.05                    # outside 2.5-20 d


def test_middle_dimensions_are_filtered_independently():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(256, 3, 2, 72))
    y = wk_filter(x, "Kelvin")
    assert y.shape == x.shape and y.dtype == float
    for i in range(3):
        for j in range(2):
            np.testing.assert_allclose(y[:, i, j], wk_filter(x[:, i, j], "Kelvin"), atol=1e-12)


def test_mask_is_hermitian_so_output_is_real():
    for nt, nx in ((256, 144), (255, 143)):
        for w in (*WAVES, KELVIN_20):
            m = wave_mask(nt, nx, w)
            conj = np.roll(np.roll(m[::-1, ::-1], 1, axis=0), 1, axis=1)   # cell (-m, -n)
            np.testing.assert_array_equal(m, conj)
    x = np.random.default_rng(1).normal(size=(256, 144))
    F = np.fft.fft2(x) * wave_mask(256, 144, "ER")
    assert np.abs(np.fft.ifft2(F).imag).max() < 1e-12


def test_soft_landing_keeps_the_last_days():
    x = _wave(4, 8.0, nt=200)
    plain = wk_filter(x, "Kelvin")
    soft = wk_filter(x, "Kelvin", pad_end=48)
    tail = slice(-5, None)
    assert np.std(soft[tail]) > 1.5 * np.std(plain[tail])
    assert np.std(soft[tail]) > 0.6 * np.std(x[tail])


def test_er_bounds_for_any_wavenumber_and_bad_input():
    band = dict(k=(-25, -11), p=(9.7, 48), h=(8, 90), n=1)
    assert wave_mask(512, 144, band).any()
    with pytest.raises(ValueError):
        wk_filter(np.zeros((10, 8)), "Yanai")
    with pytest.raises(ValueError):
        wk_filter(np.zeros((10, 8)), dict(k=(1, 3)))
    with pytest.raises(ValueError):
        wk_filter(np.zeros(10), "Kelvin")


def test_sampling_interval():
    # 6-hourly samples of an 8-day wave: same band decision as daily
    x = _wave(4, 8.0 * 4, nt=1024)                                    # period in samples
    assert np.std(wk_filter(x, "Kelvin", dt=0.25)) > 0.5
    assert np.std(wk_filter(x, "Kelvin", dt=1.0)) < 0.05              # read as 32-day: out of band
