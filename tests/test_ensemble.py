import numpy as np

from quantmet.ensemble import bh, sensitivity, group_composite, cycle_offset


def test_bh_uniform_pvalues_find_nothing():
    p = np.random.default_rng(1).uniform(size=5000)
    assert bh(p, 0.10).sum() <= 5                              # FDR control: essentially no discoveries under the null


def test_bh_finds_strong_signals_and_ignores_nan():
    p = np.r_[np.full(50, 1e-6), np.random.default_rng(2).uniform(size=950), np.nan]
    s = bh(p, 0.10)
    assert s[:50].all() and not s[-1]


def test_sensitivity_recovers_slope():
    rng = np.random.default_rng(3)
    x = rng.normal(size=40)
    J = np.stack([3.0 * x + rng.normal(scale=0.1, size=40), rng.normal(size=40)], axis=1)
    r = sensitivity(x, J)
    assert abs(r["slope"][0] - 3.0) < 0.05 and r["sig"][0] and not r["sig"][1]
    assert sensitivity(np.ones(10), J[:10]) is None


def test_group_composite_needs_members():
    a = np.zeros((5, 3)); b = 1.0 + np.random.default_rng(5).normal(scale=0.01, size=(9, 3))
    assert group_composite(a, b) is None
    r = group_composite(np.zeros((9, 3)) + np.random.default_rng(4).normal(scale=0.01, size=(9, 3)), b + 0.0)
    assert np.allclose(r["diff"], -1.0, atol=0.02) and r["sig"].all()


def test_cycle_offset_recovers_linear_drift_without_lookahead():
    import datetime as dt
    recs = []
    leads00 = np.arange(0, 15, 1.0); leads12 = np.r_[0.0, np.arange(0.5, 14.6, 1.0)]
    for d in range(10):
        day = dt.datetime(2026, 9, 1) + dt.timedelta(days=d)
        truth = lambda vt: np.sin(vt / 3.0)                    # a common signal at valid time vt (days)
        t0 = d; t12 = d + 0.5
        recs.append((np.datetime64(day), leads00, truth(t0 + leads00)[:, None]))
        recs.append((np.datetime64(day + dt.timedelta(hours=12)), leads12,
                     (truth(t12 + leads12) + 0.1 * leads12)[:, None]))   # 12Z family drifts by 0.1 per day of lead
    tgt = np.datetime64(dt.datetime(2026, 9, 10, 12))
    off = cycle_offset(recs, tgt, leads12)
    assert np.allclose(off[1:-1, 0], 0.1 * leads12[1:-1], atol=1e-9)    # leads with a same-valid-time 00Z mate
    assert not cycle_offset(recs, np.datetime64(dt.datetime(2026, 9, 10)), leads00).any()   # 00Z: reference family
