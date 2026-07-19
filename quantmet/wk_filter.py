"""Wheeler-Kiladis space-time filtering for convectively coupled equatorial
waves (Wheeler & Kiladis 1999, JAS).

Given a (time x longitude) anomaly field, isolate Kelvin, n=1 equatorial
Rossby, and MJO bands by masking the 2-D FFT with each wave's dispersion
relation on the equatorial beta-plane. The ER frequency bounds come from
solving the full cubic dispersion relation per zonal wavenumber and taking
the Rossby (smallest-|nu|) root; Kelvin uses the implied equivalent depth.

FFT convention care: every spectral cell is folded to its positive-frequency
physical representative before testing band membership, keeping the mask
Hermitian so the inverse transform is exactly real.

Runs daily on a GMGSI longwave-IR OLR proxy at https://scorvec.com
(wave-overlay Hovmoller).
"""
from __future__ import annotations

import numpy as np

G = 9.81
A_EARTH = 6.371e6
BETA = 2.0 * 7.292e-5 / A_EARTH
DAY = 86400.0

WAVES = {
    "Kelvin": dict(k=(1, 14),   p=(2.5, 30),  h=(8, 90), n=None),
    "ER":     dict(k=(-10, -1), p=(9.7, 48),  h=(8, 90), n=1),
    "MJO":    dict(k=(1, 5),    p=(30, 96),   h=None,    n=None),
}

__all__ = ["wk_filter", "kelvin_he", "er_freq", "lanczos_lowpass", "WAVES"]


def kelvin_he(s: float, f: float) -> float:
    """Equivalent depth implied by a Kelvin wave at (s>0, f>0): ω = (s/a)·√(g·he)."""
    if s <= 0 or f <= 0:
        return np.nan
    omega = 2 * np.pi * f / DAY
    c = omega / (s / A_EARTH)
    return c * c / G


def er_freq(s: int, he: float, n: int = 1) -> float:
    """|frequency| (cyc/day) of the n-mode equatorial-Rossby wave at wavenumber s, depth he.
    Solves the nondim cubic ν³−(K²+2n+1)ν−K=0 and takes the Rossby (smallest-|ν|) root."""
    c = np.sqrt(G * he)
    k_dim = s / A_EARTH
    K = k_dim * np.sqrt(c / BETA)
    roots = np.roots([1.0, 0.0, -(K * K + 2 * n + 1), -K])
    real = roots[np.abs(roots.imag) < 1e-9].real
    nu = real[np.argmin(np.abs(real))]               # Rossby branch
    omega = nu * np.sqrt(BETA * c)
    return abs(omega) / (2 * np.pi) * DAY


_ER_BOUNDS = {}
for _s in range(-20, 0):
    _f = sorted(er_freq(_s, h) for h in (8.0, 90.0))
    _ER_BOUNDS[_s] = (_f[0], _f[1])


def _taper(a: np.ndarray, frac: float = 0.05) -> np.ndarray:
    """Split-cosine-bell taper over the first/last `frac` of the time axis (axis 0)."""
    nt = a.shape[0]
    w = np.ones(nt)
    m = max(1, int(frac * nt))
    ramp = 0.5 * (1 - np.cos(np.pi * (np.arange(m) + 1) / (m + 1)))
    w[:m] = ramp; w[-m:] = ramp[::-1]
    return a * w[:, None]


def wk_filter(anom: np.ndarray, wave: str) -> np.ndarray:
    """Bandpass `anom` (ntime × nlon, detrended) to one wave band via 2-D FFT.
    numpy convention: cell (m,n) reconstructs exp(2πi(f_m t + s_n x/nx)); the physical
    wave exp(i(sλ − 2π f t)) therefore sits at s_phys = wavenum[n], f_phys = −freq_t[m].
    A cell is kept iff its positive-frequency representative is in the band (the conjugate
    cell maps to the same representative, so the mask is Hermitian → real inverse)."""
    w = WAVES[wave]
    nt, nx = anom.shape
    F = np.fft.fft2(_taper(anom))
    ft = np.fft.fftfreq(nt, d=1.0)                   # cycles/day
    sx = np.fft.fftfreq(nx, d=1.0) * nx              # planetary wavenumber (integers)
    S = np.broadcast_to(sx[None, :], (nt, nx)).astype(float).copy()
    Fr = np.broadcast_to(-ft[:, None], (nt, nx)).astype(float).copy()
    neg = Fr < 0                                     # fold every cell to positive frequency
    S[neg] *= -1; Fr[neg] *= -1
    with np.errstate(divide="ignore", invalid="ignore"):
        period = np.where(Fr > 0, 1.0 / Fr, np.inf)
    kmin, kmax = w["k"]; pmin, pmax = w["p"]
    # exclude the zonal mean (k=0): it's a longitude-uniform offset, not a spatial pattern — so LF
    # shows the wavenumber-1..3 standing convective pattern (the warm-pool shift), not a flat stripe.
    m = (S >= kmin) & (S <= kmax) & (np.abs(S) >= 1) & (period >= pmin) & (period <= pmax) & (Fr > 0)
    if w["h"] is not None:
        hmin, hmax = w["h"]
        if w["n"] is None:                           # Kelvin: depth from (s,f)
            with np.errstate(divide="ignore", invalid="ignore"):
                c = (2 * np.pi * Fr / DAY) / (S / A_EARTH)
                he = c * c / G
            m &= (he >= hmin) & (he <= hmax)
        else:                                        # ER: per-wavenumber frequency bounds
            flo = np.full((nt, nx), np.nan); fhi = np.full((nt, nx), np.nan)
            si = np.round(S).astype(int)
            for s, (lo, hi) in _ER_BOUNDS.items():
                cell = si == s
                flo[cell] = lo; fhi[cell] = hi
            m &= (Fr >= flo) & (Fr <= fhi)
    return np.fft.ifft2(F * m.astype(float)).real


def lanczos_lowpass(anom: np.ndarray, cutoff_days: float = 120.0, half: int = 60) -> np.ndarray:
    """Low-frequency envelope: a Lanczos low-pass in time (periods > cutoff_days), then remove the
    zonal (longitude) mean at each time so it's the *spatial* standing pattern, not a uniform
    tropical-mean stripe. Keeps all spatial scales — so the persistent convection stays where it
    actually is (no wavenumber-1..3 truncation, which spread it into spurious lobes)."""
    from scipy.ndimage import convolve1d
    fc = 1.0 / cutoff_days
    k = np.arange(-half, half + 1)
    w = np.sinc(2 * fc * k) * 2 * fc * np.sinc(k / half)        # Lanczos-windowed ideal low-pass
    w /= w.sum()
    lp = convolve1d(anom, w, axis=0, mode="nearest")            # time low-pass per longitude
    return lp - lp.mean(axis=1, keepdims=True)                  # remove zonal mean (k=0)
