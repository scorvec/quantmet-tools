"""Kinetic-energy power spectra on limited-area grids via the 2-D Discrete
Cosine Transform (Denis, Cote & Laprise 2002, MWR).

The DCT sidesteps the periodicity assumption that corrupts FFT spectra on
regional domains: no detrending, no windowing, no spurious low-wavenumber
leakage from the domain-scale trend. Variance is binned into elliptic rings
of constant normalized wavenumber, giving a 1-D spectrum directly
comparable across models with different grid spacings.

Used in production to compare HRRR vs RRFS effective resolution
(https://scorvec.com): the spectral rolloff shows where each model's
numerics stop resolving the flow it advertises.
"""
from __future__ import annotations

import numpy as np
import scipy.fft
from scipy import stats

__all__ = ["ke_spectrum", "spectral_slope", "effective_resolution",
           "fidelity_resolution"]


def ke_spectrum(u: np.ndarray, v: np.ndarray, res_km: float):
    """1-D kinetic-energy spectrum from 2-D wind components.

    Parameters
    ----------
    u, v : 2-D arrays (same shape), wind components on a uniform grid
    res_km : grid spacing in km

    Returns
    -------
    wavelengths : km, descending
    power : KE spectral density per wavenumber bin
    """
    du = np.abs(scipy.fft.dctn(u, norm="ortho") / np.sqrt(u.size)) ** 2
    dv = np.abs(scipy.fft.dctn(v, norm="ortho") / np.sqrt(v.size)) ** 2
    p2 = 0.5 * (du + dv)
    s0, s1 = p2.shape
    nbwv, nbwx = min(s0, s1), max(s0, s1)
    fr = np.arange(s0)[:, None] / s0
    fc = np.arange(s1)[None, :] / s1
    knrm = (np.sqrt(fr ** 2 + fc ** 2) * nbwv).flatten()
    eps = 0.1 * np.sqrt(1 / nbwx)
    kbins = np.arange(-1.0 + eps, nbwv + eps, 1.0)
    abins, _, _ = stats.binned_statistic(knrm, p2.flatten(),
                                         statistic=np.nansum, bins=kbins)
    p1 = abins[1:]
    p1 = p1[0::2][:len(p1[1::2])] + p1[1::2]           # adjacent-bin pairing
    wavenumbers = np.arange(1, len(p1))
    wavelengths = res_km * nbwv / wavenumbers
    return wavelengths, p1[1:]


def spectral_slope(wavelengths, power, lo_km: float, hi_km: float) -> float:
    """Log-log slope over [lo_km, hi_km] (k^-3 synoptic, k^-5/3 mesoscale)."""
    m = (wavelengths >= lo_km) & (wavelengths <= hi_km) & (power > 0)
    return float(np.polyfit(np.log(1 / wavelengths[m]), np.log(power[m]), 1)[0])


def effective_resolution(wavelengths, power, ref_slope: float = -5.0 / 3.0,
                         fit_km=(300.0, 600.0), drop_db: float = 3.0) -> float:
    """Effective resolution: wavelength where the spectrum falls `drop_db`
    below the mesoscale power law extrapolated from `fit_km` (Skamarock 2004
    in spirit). Returns km."""
    m = (wavelengths >= fit_km[0]) & (wavelengths <= fit_km[1]) & (power > 0)
    b = np.polyfit(np.log(1 / wavelengths[m]), np.log(power[m]), 1)
    model = np.exp(np.polyval(b, np.log(1 / wavelengths)))
    ratio = 10 * np.log10(np.maximum(power, 1e-300) / model)
    below = np.where((wavelengths < fit_km[0]) & (ratio < -drop_db))[0]
    return float(wavelengths[below[0]]) if len(below) else float("nan")

def fidelity_resolution(wl, p, p_ref, frac=0.5):
    """Finest wavelength at which the spectrum still carries at least `frac`
    of a reference spectrum's energy density — e.g. a forecast against its own
    analysis. Unlike dissipation-range criteria (which can be undefined when a
    spectrum has no clean roll-off), this is defined whenever the ratio drops
    below `frac` anywhere, and it directly measures scale-dependent energy
    loss rather than numerical damping. Returns NaN only if the spectrum never
    falls below the threshold (fidelity maintained at all resolved scales)."""
    import numpy as np
    wl = np.asarray(wl, float)
    r = np.asarray(p, float) / np.asarray(p_ref, float)
    below = r < frac
    if not below.any():
        return float("nan")
    i = np.argmax(below)                      # first scale (largest wl) below
    if i == 0:
        return float(wl[0])
    # log-interpolate the crossing between i-1 and i
    w1, w2 = np.log(wl[i - 1]), np.log(wl[i])
    r1, r2 = r[i - 1], r[i]
    t = (frac - r1) / (r2 - r1) if r2 != r1 else 0.0
    return float(np.exp(w1 + t * (w2 - w1)))
