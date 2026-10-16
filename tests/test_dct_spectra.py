"""quantmet.dct_spectra against exact single DCT modes (each wavelength must land in its own bin) and a power law."""

import numpy as np
import pytest

from quantmet.dct_spectra import ke_spectrum, spectral_slope, fidelity_resolution


def _mode(n, m):
    """cos(pi m (i + 1/2) / n): an exact DCT-II basis function, m/2 wavelengths across n points."""
    return np.cos(np.pi * m * (np.arange(n) + 0.5) / n)


@pytest.mark.parametrize("k", [4, 8, 16, 32])
@pytest.mark.parametrize("axis", [0, 1])
def test_exact_mode_lands_on_its_wavelength_square(k, axis):
    n, res = 256, 3.0
    f = _mode(n, 2 * k)
    u = (f[:, None] * np.ones((1, n))) if axis == 0 else (np.ones((n, 1)) * f[None, :])
    wl, p = ke_spectrum(u, np.zeros_like(u), res)
    j = int(np.argmax(p))
    assert np.isclose(wl[j], n * res / k)  # 192, 96, 48, 24 km (0.3.0: 256, 109.7, 51.2, 24.8)
    assert p[j] > 0.999 * p.sum()  # all of the energy in that one bin
    assert np.isclose(p.sum(), 0.25)  # 0.5 * mean(u^2) of a unit cosine


def test_exact_mode_on_a_rectangular_domain():
    """Wavelengths are res * (short side) / k; a mode along the long side lands there too."""
    ny, nx, res = 128, 256, 3.0
    for u, true in (
        (np.ones((ny, 1)) * _mode(nx, 16)[None, :], 2 * nx * res / 16),  # 8 waves in 768 km
        (_mode(ny, 8)[:, None] * np.ones((1, nx)), 2 * ny * res / 8),
    ):  # 4 waves in 384 km
        wl, p = ke_spectrum(u, np.zeros_like(u), res)
        assert np.isclose(wl[int(np.argmax(p))], true)  # 96 km both


def test_wavelength_axis_and_slope_of_a_power_law():
    n, res = 256, 3.0
    wl, p = ke_spectrum(np.zeros((n, n)), np.zeros((n, n)), res)
    assert np.isclose(wl[0], n * res) and np.allclose(wl, n * res / np.arange(1, len(wl) + 1))
    # k^-3 by construction: 2-D DCT amplitude ~ K^-2 with K = m/2 the full wavenumber -> ring sum ~ k^-3
    rng = np.random.default_rng(0)
    m = np.hypot(*np.meshgrid(np.arange(n), np.arange(n), indexing="ij")) / 2
    amp = np.zeros_like(m)
    amp[m > 0] = m[m > 0] ** -2.0
    import scipy.fft

    u = scipy.fft.idctn(amp * rng.standard_normal((n, n)), norm="ortho")
    v = scipy.fft.idctn(amp * rng.standard_normal((n, n)), norm="ortho")
    wl, p = ke_spectrum(u, v, res)
    assert abs(spectral_slope(wl, p, 15, 100) + 3.0) < 0.15  # pairing of half-wavenumbers steepens it slightly


def test_fidelity_resolution_interpolates_the_crossing():
    wl = np.array([1000.0, 500.0, 250.0, 125.0])
    assert np.isnan(fidelity_resolution(wl, np.ones(4), np.ones(4)))
    r = fidelity_resolution(wl, np.array([1.0, 0.8, 0.4, 0.1]), np.ones(4))
    assert 250.0 < r < 500.0
