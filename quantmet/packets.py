"""Rossby wave-packet envelope (Zimin, Szunyogh, Patil, Hunt & Ott 2003, MWR 131).

The envelope of the meridional wind along each latitude circle: keep zonal wavenumbers kmin..kmax, build the analytic
signal (positive wavenumbers doubled, negative zeroed, so the real part is the band-passed field) and take its modulus.
A train of troughs and ridges then reads as ONE positive hump instead of alternating signs, so a longitude-time
(Hovmoller) plot of the envelope shows a wave packet as a single bright streak whose slope is the group velocity.

Implementation notes that matter in practice:
- The band matters: kmin ~4 removes the planetary stationary waves (whose envelope is just "everywhere"), kmax ~15
  removes grid-scale noise. Zimin et al. use 4-17 for 300 hPa v.
- The transform is along longitude only, latitude by latitude; average the ENVELOPE over a latitude band (not v),
  since v changes sign across a packet's meridional structure.
- For the analytic signal to be meaningful the band must be well inside (0, n/2); the function checks it.

Powers the wave-packet panel of the tropical-cyclone / jet-stream card at https://scorvec.com/circulation.html.

Requires: numpy only.
"""
from __future__ import annotations

import numpy as np

__all__ = ["packet_envelope", "band_mean"]


def packet_envelope(v: np.ndarray, kmin: int = 4, kmax: int = 15) -> np.ndarray:
    """Envelope (same units as v) of v(..., lon) for zonal wavenumbers kmin..kmax; the LAST axis is longitude and must
    span the full circle with uniform spacing. NaNs are treated as zero."""
    v = np.nan_to_num(np.asarray(v, float))
    n = v.shape[-1]
    if not (1 <= kmin <= kmax < n // 2):
        raise ValueError(f"need 1 <= kmin <= kmax < nlon/2 (got {kmin}, {kmax}, nlon {n})")
    F = np.fft.fft(v, axis=-1)
    Z = np.zeros_like(F)
    Z[..., kmin:kmax + 1] = 2.0 * F[..., kmin:kmax + 1]
    return np.abs(np.fft.ifft(Z, axis=-1))


def band_mean(field: np.ndarray, lat: np.ndarray, lat_min: float, lat_max: float, lat_axis: int = -2) -> np.ndarray:
    """cos(lat)-weighted mean of field over lat_min..lat_max along ``lat_axis`` (removed from the output)."""
    field = np.asarray(field, float)
    lat = np.asarray(lat, float)
    m = (lat >= lat_min) & (lat <= lat_max)
    if not m.any():
        raise ValueError("no latitudes in the band")
    w = np.cos(np.deg2rad(lat[m]))
    sub = np.compress(m, field, axis=lat_axis)
    shape = [1] * sub.ndim
    shape[lat_axis] = -1
    return (sub * w.reshape(shape)).sum(axis=lat_axis) / w.sum()
