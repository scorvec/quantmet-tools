"""Ensemble diagnostics: Benjamini-Hochberg FDR masking, ensemble sensitivity, member-group composites, and a
trailing-pair estimator of systematic cycle offsets (the "windshield wiper").

Ensemble sensitivity (Torn & Hakim 2008; Ancell & Hakim 2007): across ensemble members, regress a later field J
(e.g. day-7 z500) on an earlier scalar x (e.g. a tropical cyclone's day-3 latitude):
    dJ/dx = cov(J, x) / var(x),      r = corr(J, x),     t = r sqrt((n-2)/(1-r^2)),  two-sided p from Student t.
It is an ASSOCIATION across members, not a causal response.

Field significance: thousands of grid points tested at 5 % give hundreds of false positives. ``bh`` controls the
false-discovery rate (Benjamini & Hochberg 1995): the expected share of false positives among the shaded points is
<= q. Apply it over everything shown together (all panels of a figure), not panel by panel.

Cycle offsets: some models' runs from one initialisation hour sit systematically off the runs from another at the
SAME valid time (seen in AIFS-ENS 00Z vs 12Z tropical-wind indices, growing roughly linearly with lead). Animating
alternating cycles then rocks back and forth. ``cycle_offset`` estimates the offset as a function of lead from the
trailing same-valid-time differences between the two families (no lookahead: only pairs initialised before the target
run), to be SUBTRACTED from the target run. Archive RAW values; archiving corrected ones collapses future estimates.

Requires: numpy, scipy (quantmet.stats for the FDR).
"""
from __future__ import annotations

import numpy as np

__all__ = ["bh", "sensitivity", "group_composite", "cycle_offset"]


def bh(p: np.ndarray, q: float = 0.10) -> np.ndarray:
    """Benjamini-Hochberg discoveries at FDR q: boolean mask, same shape as p; NaN p-values are never significant.
    Thin alias of quantmet.stats.benjamini_hochberg."""
    from .stats import benjamini_hochberg
    return benjamini_hochberg(p, q)


def sensitivity(x: np.ndarray, J: np.ndarray, q: float = 0.10, min_sd: float = 0.0) -> dict | None:
    """Regression of J (n, ...) on the scalar predictor x (n,) across n members.

    Returns {slope (J units per x unit), r, p, sig (BH at q), n} or None if fewer than 4 members or x's spread is
    below min_sd (members agree on x: nothing to regress on)."""
    from scipy import stats as sst
    x = np.asarray(x, float); J = np.asarray(J, float)
    n = len(x)
    if n < 4 or x.std() <= min_sd:
        return None
    xa = x - x.mean()
    Ja = J - J.mean(0)
    sxy = np.tensordot(xa, Ja, axes=(0, 0))
    slope = sxy / np.sum(xa ** 2)
    r = sxy / (np.sqrt(np.sum(xa ** 2)) * np.sqrt(np.sum(Ja ** 2, 0)) + 1e-30)
    t = r * np.sqrt((n - 2) / np.clip(1 - r ** 2, 1e-12, None))
    p = 2 * sst.t.sf(np.abs(t), n - 2)
    return dict(slope=slope, r=r, p=p, sig=bh(p, q), n=n)


def group_composite(A: np.ndarray, B: np.ndarray, q: float = 0.10, min_group: int = 8) -> dict | None:
    """Mean(A) - mean(B) of two member groups (n_a, ...), (n_b, ...) with a Welch t-test per point and BH masking;
    None if either group has fewer than min_group members."""
    from scipy import stats as sst
    A = np.asarray(A, float); B = np.asarray(B, float)
    if len(A) < min_group or len(B) < min_group:
        return None
    t, p = sst.ttest_ind(A, B, axis=0, equal_var=False)
    return dict(diff=A.mean(0) - B.mean(0), p=p, sig=bh(p, q), n=(len(A), len(B)))


def cycle_offset(records, target_init, leads, ref_hour: int = 0, alt_hour: int = 12, n_pairs: int = 7,
                 min_pairs: int = 3):
    """Offset of the ``alt_hour`` family relative to the ``ref_hour`` family, as a function of lead.

    records : iterable of (init, leads_days, values) with init a numpy/pandas datetime, leads_days (L,), values (L, k)
              -- RAW ensemble means of k quantities.
    target_init : the init of the run to correct; leads : its leads (days).
    Pairs are (alt run, the ref run initialised (alt_hour - ref_hour) h earlier); only pairs with alt init strictly
    before target_init are used (no lookahead), newest n_pairs first. Differences alt - ref at common VALID times are
    averaged per alt-run lead and interpolated to ``leads``. Returns (L_target, k) to subtract, zeros when the
    target is a ref run or fewer than min_pairs pairs exist."""
    leads = np.asarray(leads, float)
    recs = [(np.datetime64(i, "s"), np.asarray(l, float), np.asarray(v, float)) for i, l, v in records]
    k = recs[0][2].shape[1] if recs else 1
    zero = np.zeros((len(leads), k))
    tgt = np.datetime64(target_init, "s")
    hour = lambda t: int((t - t.astype("datetime64[D]")) / np.timedelta64(1, "h"))
    if hour(tgt) != alt_hour:
        return zero
    by_init = {r[0]: r for r in recs}
    gap = np.timedelta64(alt_hour - ref_hour, "h")
    diffs: dict[float, list] = {}
    used = 0
    for init, lb, vb in sorted(recs, key=lambda r: r[0], reverse=True):
        if used >= n_pairs:
            break
        if hour(init) != alt_hour or init >= tgt:
            continue
        mate = by_init.get(init - gap)
        if mate is None:
            continue
        _, la, va = mate
        va_by_valid = {mate[0] + np.timedelta64(int(round(x * 86400)), "s"): j for j, x in enumerate(la)}
        hit = False
        for i, ld in enumerate(lb):
            j = va_by_valid.get(init + np.timedelta64(int(round(ld * 86400)), "s"))
            if j is not None:
                diffs.setdefault(round(float(ld), 3), []).append(vb[i] - va[j]); hit = True
        used += hit
    if used < min_pairs:
        return zero
    grid = np.array(sorted(diffs))
    mean = np.array([np.mean(diffs[g], axis=0) for g in grid])
    return np.stack([np.interp(leads, grid, mean[:, c]) for c in range(k)], axis=1)
