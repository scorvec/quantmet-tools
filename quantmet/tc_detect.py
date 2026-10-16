"""Fast tropical-cyclone candidate detection on MSLP fields.

A C kernel (tc_detect.c, compiled on first use) finds closed-circulation
candidates: local minima of sea-level pressure (no lower neighbour; a tie
with a neighbour still counts, so a flat-bottomed low can return adjacent
candidates) that are at least `depth` hPa deeper than the mean pressure on a
surrounding annulus, square in degrees and centred on the candidate,
using a summed-area table so the ring test is O(1) per gridpoint regardless
of ring radius. Longitude-wrapped ghost columns handle the date line. ~10x
the throughput of the chained scipy filters it replaces; it tracked the
101-member ECMWF ensemble to day 15 in seconds for the site's former TC
tracker.

Build notes: the shared library is compiled with `cc` into a per-user cache
directory ($QUANTMET_CACHE_DIR, else $XDG_CACHE_HOME/quantmet, else
~/.cache/quantmet) -- never into the installed package, which may be
read-only -- with the platform's shared-library suffix, and named by a hash
of the C source so an upgraded kernel is rebuilt rather than silently
reused. The build writes to a temporary file and renames it into place, so
two processes compiling at once cannot load a half-written library.
"""

from __future__ import annotations

import ctypes
import functools
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
_SRC = _HERE / "tc_detect.c"

__all__ = ["detect_candidates", "library_path"]


def _cache_dir() -> Path:
    env = os.environ.get("QUANTMET_CACHE_DIR")
    if env:
        return Path(env)
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "quantmet"


def _suffix() -> str:
    """Shared-library file suffix; the loader (ctypes) does not depend on it outside Windows."""
    return ".dll" if sys.platform == "win32" else ".so"


def library_path() -> Path:
    """Where the compiled kernel lives (it may not exist yet)."""
    h = hashlib.sha256(_SRC.read_bytes()).hexdigest()[:12]
    return _cache_dir() / f"tc_detect_{h}{_suffix()}"


@functools.cache
def _lib():
    """The compiled kernel, built on first use and loaded once per process."""
    so = library_path()
    if not so.exists():
        so.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(suffix=_suffix(), dir=so.parent)
        os.close(fd)
        try:
            subprocess.run([os.environ.get("CC", "cc"), "-O3", "-shared", "-fPIC", str(_SRC), "-o", tmp], check=True)
            os.replace(tmp, so)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    lib = ctypes.CDLL(str(so))
    f = lib.detect_step
    P = ctypes.c_void_p
    I = ctypes.c_int
    f.argtypes = [P, P, I, I, I, I, ctypes.c_float, I, I, I, I, ctypes.c_float, I, I, P, P, P, P, P, I]
    f.restype = ctypes.c_int
    return lib


def detect_candidates(
    mslp_hpa: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    ring_deg: float = 2.5,
    depth_hpa: float = 2.0,
    max_cands: int = 4096,
    min_deg: float = 1.0,
    lat_band=None,
    p_max: float | None = None,
    wind: np.ndarray | None = None,
    wind_deg: float = 2.0,
    full: bool = False,
):
    """Closed-low candidates on a regular global (lat, lon) grid.

    mslp_hpa : 2-D (lat, lon) sea-level pressure, hPa; longitude must span
               the full circle (it wraps). The latitude and longitude spacings
               may differ: every half-width below is converted to rows with
               dlat and to columns with dlon.
    ring_deg : outer half-width of the annulus (the inner half-width is half
               of it), in degrees of latitude and of longitude; the candidate
               must be `depth_hpa` below the annulus mean
    min_deg  : half-width of the local-minimum neighbourhood (no neighbour
               lower; ties allowed)
    lat_band : (lo, hi) degrees to search (default: every row)
    p_max    : only centres below this pressure (hPa)
    wind     : optional (lat, lon) wind speed; the maximum within wind_deg of
               each centre is reported when full=True
    Returns (lat, lon, mslp) arrays, or with full=True a dict adding
    depth (hPa below the ring mean) and wind_max."""
    lib = _lib()
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    f = np.ascontiguousarray(mslp_hpa, dtype=np.float32)
    ny, nx = f.shape
    w = np.zeros_like(f) if wind is None else np.ascontiguousarray(wind, dtype=np.float32)
    dy = abs(float(lat[1] - lat[0]))
    dx = abs(float(lon[1] - lon[0]))
    ry, rx = (max(1, round(ring_deg / d)) for d in (dy, dx))
    my, mx = (max(1, round(min_deg / d)) for d in (dy, dx))
    wy, wx = (max(0, round(wind_deg / d)) for d in (dy, dx))
    if lat_band is None:
        iy0, iy1 = 0, ny
    else:
        rows = np.flatnonzero((lat >= min(lat_band)) & (lat <= max(lat_band)))
        iy0, iy1 = (int(rows.min()), int(rows.max()) + 1) if rows.size else (0, 0)
    oy = np.zeros(max_cands, np.int32)
    ox = np.zeros(max_cands, np.int32)
    op = np.zeros(max_cands, np.float32)
    od = np.zeros(max_cands, np.float32)
    ow = np.zeros(max_cands, np.float32)

    def ptr(a):
        return a.ctypes.data_as(ctypes.c_void_p)

    n = lib.detect_step(
        ptr(f),
        ptr(w),
        ny,
        nx,
        iy0,
        iy1,
        ctypes.c_float(np.inf if p_max is None else p_max),
        my,
        mx,
        ry,
        rx,
        ctypes.c_float(depth_hpa),
        wy,
        wx,
        ptr(oy),
        ptr(ox),
        ptr(op),
        ptr(od),
        ptr(ow),
        max_cands,
    )
    if n < 0:
        raise MemoryError("tc_detect kernel could not allocate its summed-area table")
    iy, ix = oy[:n], ox[:n]
    if full:
        return {
            "lat": lat[iy],
            "lon": lon[ix],
            "mslp": np.asarray(mslp_hpa)[iy, ix],
            "depth": od[:n].astype(float),
            "wind_max": ow[:n].astype(float) if wind is not None else None,
        }
    return lat[iy], lon[ix], np.asarray(mslp_hpa)[iy, ix]
