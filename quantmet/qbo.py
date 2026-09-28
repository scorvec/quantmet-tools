"""The QBO zero-wind line: where the zonal-mean zonal wind turns easterly on
the equatorward side of the winter westerlies.

Holton and Tan (1980): planetary waves propagating up and equatorward from
the winter midlatitudes meet a critical line where [u] = 0. With easterly
QBO winds in the lower stratosphere that line sits in the winter
subtropics, the extratropical waveguide narrows and more wave activity is
steered toward the polar vortex. zero_line tracks WHERE the line is, one
latitude per profile, so it can be followed day by day, level by level and
member by member.

The rule (defaults are the site's, chosen on robustness before looking at
any vortex relation):
  1. Smooth [u] in latitude with a Gaussian of `sigma` degrees.
  2. Start at the strongest westerlies between start = (25, 45) degrees in
     the hemisphere asked for. If even that maximum is <= 0 there are no
     winter westerlies to bound: undefined, flag 1 (most of May-September in
     the NH).
  3. Scan toward and across the equator. The index is the first zero
     crossing (linear interpolation) whose easterly region beyond it reaches
     at least `umin` (1 m/s).
  4. No qualifying crossing before `bound` degrees in the other hemisphere:
     censored at -bound, flag 2.

Numerical notes that matter in practice:
- THE DIP RULE. A zero crossing is only a critical line if the easterlies
  beyond it are real. Shallow dips (min |u| < umin, inside analysis noise)
  are stepped over and the scan continues past them: without that, a
  -0.3 m/s wobble at 18N in a westerly-QBO autumn reads as a critical line
  and the index jumps 20 degrees for a day. The smoothing alone does not fix
  it (a dip wider than sigma survives), and raising sigma blurs the real
  line; the amplitude test is what separates them.
- CENSORING IS INFORMATION. With westerly QBO winds at a level there may be
  no easterlies anywhere within reach; that is flag 2 at -bound, not a
  missing value, and statistics must treat it as a censored observation.
- SOUTHERN HEMISPHERE by mirroring the grid: the SH index is the NH rule on
  -lat, returned negative (positive = north throughout).

Powers the QBO zero-wind-line tracker on
https://scorvec.com/stratosphere.html (MERRA-2 reference, GEOS FP analyses
and per-member AIFS-ENS / IFS-ENS forecasts).

Requires: numpy, scipy (Gaussian smoothing).
"""
from __future__ import annotations

import numpy as np

SIGMA, UMIN, START, BOUND = 2.0, 1.0, (25.0, 45.0), 45.0

__all__ = ["zero_line", "SIGMA", "UMIN", "START", "BOUND"]


def zero_line(u, lat, hemi: str = "nh", sigma: float = SIGMA, umin: float = UMIN,
              start=START, bound: float = BOUND):
    """Zero-wind-line latitude of [u] profiles.

    u     : (..., lat) zonal-mean zonal wind, m/s; lat in any order, uniformly
            spaced (degrees).
    hemi  : "nh" or "sh", the winter hemisphere whose westerlies bound it.
    Returns (index, flag), both with u's leading shape: index in degrees
    (positive north; NaN where flag == 1, -bound / +bound where censored),
    flag 0 = found, 1 = undefined (no westerlies to bound), 2 = censored."""
    from scipy.ndimage import gaussian_filter1d
    if hemi not in ("nh", "sh"):
        raise ValueError("hemi must be 'nh' or 'sh'")
    lat = np.asarray(lat, float)
    u = np.asarray(u, float)
    o = np.argsort(lat)
    lat, u = lat[o], u[..., o]
    if hemi == "sh":
        lat, u = -lat[::-1], u[..., ::-1]
    if sigma > 0:
        u = gaussian_filter1d(u, sigma / abs(lat[1] - lat[0]), axis=-1, mode="nearest")
    shp = u.shape[:-1]
    U = u.reshape(-1, u.shape[-1])
    keep = lat >= -bound
    la, U = lat[keep], U[:, keep]
    st = np.flatnonzero((la >= start[0]) & (la <= start[1]))
    if st.size == 0:
        raise ValueError("no grid latitude inside `start`")
    z = np.full(U.shape[0], np.nan)
    fl = np.zeros(U.shape[0], int)
    for k in range(U.shape[0]):
        row = U[k]
        if not np.isfinite(row).all():
            fl[k] = 1
            continue
        j = st[np.argmax(row[st])]
        if row[j] <= 0:
            fl[k] = 1
            continue
        found = False
        while j > 0:
            if row[j - 1] <= 0 < row[j]:
                jj = j - 1
                while jj >= 0 and row[jj] <= 0:
                    jj -= 1
                if row[jj + 1:j].min() <= -umin:            # deep enough: a critical line
                    z[k] = la[j - 1] + (la[j] - la[j - 1]) * (0 - row[j - 1]) / (row[j] - row[j - 1])
                    found = True
                    break
                if jj < 0:
                    break
                j = jj + 1                                  # shallow dip: step over it
                continue
            j -= 1
        if not found:
            z[k], fl[k] = -bound, 2
    z = z.reshape(shp)
    return (-z if hemi == "sh" else z), fl.reshape(shp)
