"""Fast tropical-cyclone candidate detection on MSLP fields.

A C kernel (tc_detect.c, auto-compiled on import) finds closed-circulation
candidates: local minima that are at least `depth` hPa deeper than the mean
pressure on a surrounding ring, using summed-area tables so the ring test
is O(1) per gridpoint regardless of ring radius. Longitude-wrapped ghost
columns handle the dateline. ~100x faster than the scipy equivalent;
tracks 101 ECMWF ensemble members to day 15 in seconds at
https://scorvec.com/tc.html.
"""
from __future__ import annotations

import ctypes
import subprocess
import tempfile
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        so = _HERE / "tc_detect.dylib"
        if not so.exists():
            subprocess.run(["cc", "-O3", "-shared", "-fPIC",
                            str(_HERE / "tc_detect.c"), "-o", str(so)], check=True)
        _LIB = ctypes.CDLL(str(so))
    return _LIB


def detect_candidates(mslp_hpa: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                      ring_deg: float = 2.5, depth_hpa: float = 2.0,
                      max_cands: int = 4096):
    """Return (lat, lon, mslp) arrays of closed-low candidates.

    mslp_hpa : 2-D (lat, lon) sea-level pressure, hPa
    ring_deg : ring radius for the closed-circulation test
    depth_hpa: required depth below the ring-mean
    """
    lib = _lib()
    f = np.ascontiguousarray(mslp_hpa, dtype=np.float32)
    ny, nx = f.shape
    dy = abs(float(lat[1] - lat[0]))
    r = max(1, int(round(ring_deg / dy)))
    out = np.zeros((max_cands, 2), dtype=np.int32)
    n = lib.detect(f.ctypes.data_as(ctypes.c_void_p), ny, nx, r,
                   ctypes.c_float(depth_hpa),
                   out.ctypes.data_as(ctypes.c_void_p), max_cands)
    iy, ix = out[:n, 0], out[:n, 1]
    return lat[iy], lon[ix], mslp_hpa[iy, ix]
