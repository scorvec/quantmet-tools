"""Wheeler-Kiladis space-time filtering for convectively coupled equatorial
waves (Wheeler & Kiladis 1999, JAS).

Given a (time, ..., longitude) anomaly field, isolate Kelvin, n=1 equatorial
Rossby, and MJO bands by masking the 2-D (time, longitude) FFT with each
wave's dispersion relation on the equatorial beta-plane. The ER frequency
bounds come from solving the full cubic dispersion relation per zonal
wavenumber and taking the Rossby (smallest-|nu|) root; Kelvin uses the
implied equivalent depth.

Numerical notes that matter in practice:
- FOLD TO POSITIVE FREQUENCY. Every spectral cell is folded to its
  positive-frequency physical representative before testing band
  membership. A cell and its conjugate map to the same representative, so
  the mask is Hermitian and the inverse transform is exactly real -- no
  imaginary residue to silently discard. The time-Nyquist row (period of
  two samples) has no positive-frequency partner and is never kept.
- ANY NUMBER OF MIDDLE DIMENSIONS. Time is axis 0 and longitude the last
  axis; latitude, level or member axes in between are filtered
  independently with the same mask (it broadcasts), never mixed.
- SOFT LANDING AT THE END (pad_end). The split-cosine taper (5 % of the
  record at each end) otherwise falls on the most recent -- operationally
  most important -- days and drags them toward zero. With pad_end = P the
  record is extended by a `ramp`-sample slide from the last row to zero,
  then zeros up to P samples; the taper and the circular wrap then act on
  the padding, which is dropped after filtering. The end is still
  provisional (the filter cannot see the future), but it is no longer
  damped by construction.
- BANDS ARE DATA. The canonical bands are in WAVES, but products differ:
  the site's OLR Hovmoller uses Kelvin periods 2.5-30 days and its
  velocity-potential Kelvin tracker 2.5-20 days. Pass a dict with the same
  keys (k, p, h, n) to filter any band; ER frequency bounds are solved for
  whatever wavenumbers the band covers.

Runs daily on a GMGSI longwave-IR OLR proxy and on 200 hPa velocity
potential at https://scorvec.com/enso.html (equatorial-wave Hovmoller and
Kelvin-wave tracker).
"""
from __future__ import annotations

from functools import lru_cache

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

__all__ = ["wk_filter", "wave_mask", "kelvin_he", "er_freq", "lanczos_lowpass", "WAVES"]


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


@lru_cache(maxsize=64)
def _er_bounds(s: int, h: tuple, n: int):
    f = sorted(er_freq(s, float(he), n) for he in h)
    return f[0], f[1]


def _band(wave):
    if isinstance(wave, str):
        try:
            return WAVES[wave]
        except KeyError:
            raise ValueError(f"unknown wave {wave!r}; use one of {list(WAVES)} or a band dict") from None
    w = dict(wave)
    missing = {"k", "p"} - set(w)
    if missing:
        raise ValueError(f"band dict needs keys k and p (missing {sorted(missing)})")
    w.setdefault("h", None)
    w.setdefault("n", None)
    return w


def _taper(a: np.ndarray, frac: float = 0.05) -> np.ndarray:
    """Split-cosine-bell taper over the first/last `frac` of the time axis (axis 0)."""
    nt = a.shape[0]
    w = np.ones(nt)
    m = max(1, int(frac * nt))
    ramp = 0.5 * (1 - np.cos(np.pi * (np.arange(m) + 1) / (m + 1)))
    w[:m] = ramp
    w[-m:] = ramp[::-1]
    return a * w.reshape((nt,) + (1,) * (a.ndim - 1))


def wave_mask(nt: int, nx: int, wave, dt: float = 1.0) -> np.ndarray:
    """Boolean (nt, nx) mask in numpy's fft2 layout for one band.

    numpy convention: cell (m, n) reconstructs exp(2πi(f_m t + s_n x/nx)); the
    physical wave exp(i(sλ − 2π f t)) therefore sits at s_phys = wavenum[n],
    f_phys = −freq_t[m]. A cell is kept iff its positive-frequency
    representative is in the band. dt is the sampling interval in days."""
    w = _band(wave)
    ft = np.fft.fftfreq(nt, d=dt)                    # cycles/day
    sx = np.fft.fftfreq(nx, d=1.0) * nx              # planetary wavenumber (integers)
    S = np.broadcast_to(sx[None, :], (nt, nx)).astype(float).copy()
    Fr = np.broadcast_to(-ft[:, None], (nt, nx)).astype(float).copy()
    neg = Fr < 0                                     # fold every cell to positive frequency
    S[neg] *= -1
    Fr[neg] *= -1
    with np.errstate(divide="ignore", invalid="ignore"):
        period = np.where(Fr > 0, 1.0 / Fr, np.inf)
    kmin, kmax = w["k"]
    pmin, pmax = w["p"]
    # the zonal mean (k = 0) is a longitude-uniform offset, not a wave: always excluded
    m = (S >= kmin) & (S <= kmax) & (np.abs(S) >= 1) & (period >= pmin) & (period <= pmax) & (Fr > 0)
    if nt % 2 == 0:
        m[nt // 2] = False                           # time Nyquist: no positive-frequency partner
    if w["h"] is not None:
        hmin, hmax = w["h"]
        if w["n"] is None:                           # Kelvin: depth from (s,f)
            with np.errstate(divide="ignore", invalid="ignore"):
                c = (2 * np.pi * Fr / DAY) / (S / A_EARTH)
                he = c * c / G
            m &= (he >= hmin) & (he <= hmax)
        else:                                        # ER: per-wavenumber frequency bounds
            flo = np.full((nt, nx), np.nan)
            fhi = np.full((nt, nx), np.nan)
            si = np.round(S).astype(int)
            for s in range(int(np.floor(kmin)), int(np.ceil(kmax)) + 1):
                if s >= 0:
                    continue
                lo, hi = _er_bounds(s, (float(hmin), float(hmax)), int(w["n"]))
                cell = si == s
                flo[cell] = lo
                fhi[cell] = hi
            with np.errstate(invalid="ignore"):
                m &= (Fr >= flo) & (Fr <= fhi)
    return m


def wk_filter(anom: np.ndarray, wave="Kelvin", dt: float = 1.0, pad_end: int = 0,
              ramp: int = 10, taper: float = 0.05) -> np.ndarray:
    """Bandpass `anom` (time, ..., lon; detrended anomalies, full longitude
    circle) to one wave band via the 2-D FFT over (time, lon).

    wave    : a key of WAVES, or a band dict {k: (kmin, kmax), p: (pmin,
              pmax) days, h: (hmin, hmax) m or None, n: None (Kelvin/MJO
              form) or 1 (ER)}; k > 0 eastward.
    dt      : sampling interval in days.
    pad_end : soft landing -- extend the record by a `ramp`-sample slide from
              the last row to zero, then zeros to pad_end samples; the
              padding is dropped after filtering. 0 = off (the 0.1 behaviour).
    taper   : split-cosine-bell fraction at each end of the (padded) record.
    Returns a real array of anom's shape."""
    x = np.asarray(anom, float)
    if x.ndim < 2:
        raise ValueError("anom must be (time, ..., lon)")
    nt = x.shape[0]
    if pad_end:
        r = min(ramp, pad_end)
        slide = x[-1][None] * (1 - np.arange(1, r + 1) / (r + 1)).reshape((r,) + (1,) * (x.ndim - 1))
        x = np.concatenate([x, slide, np.zeros((pad_end - r,) + x.shape[1:])], axis=0)
    ntp, nx = x.shape[0], x.shape[-1]
    m = wave_mask(ntp, nx, wave, dt).reshape((ntp,) + (1,) * (x.ndim - 2) + (nx,))
    F = np.fft.fft2(_taper(x, taper), axes=(0, -1))
    out = np.fft.ifft2(F * m, axes=(0, -1)).real
    return out[:nt]


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
    return lp - lp.mean(axis=-1, keepdims=True)                 # remove zonal mean (k=0)
