import numpy as np
import pytest

from quantmet import stats as S


def _pvals(seed=0):
    rng = np.random.default_rng(seed)
    p = np.concatenate([rng.uniform(size=180), rng.uniform(0, 0.004, size=20)])
    p = p.reshape(10, 20)
    p[3, 4] = np.nan
    p[7, 1] = np.nan
    return p


def test_bh_matches_scipy():
    sst = pytest.importorskip("scipy.stats")
    if not hasattr(sst, "false_discovery_control"):
        pytest.skip("scipy < 1.11 has no false_discovery_control")
    for seed in range(5):
        p = _pvals(seed)
        ok = np.isfinite(p)
        ref = sst.false_discovery_control(p[ok], method="bh")
        adj = S.fdr_adjust(p)
        np.testing.assert_allclose(adj[ok], ref, rtol=1e-12)
        assert np.isnan(adj[~ok]).all()
        for alpha in (0.05, 0.10):
            np.testing.assert_array_equal(S.benjamini_hochberg(p, alpha)[ok], ref <= alpha)
            assert not S.benjamini_hochberg(p, alpha)[~ok].any()


def test_bh_edge_cases():
    assert not S.benjamini_hochberg(np.array([np.nan, np.nan])).any()
    assert not S.benjamini_hochberg(np.array([0.5, 0.9])).any()
    np.testing.assert_array_equal(S.benjamini_hochberg(np.array([0.01, 0.02, 0.03]), 0.1), [True] * 3)


def _ar1(n, rho, rng, sd=1.0):
    e = rng.normal(scale=sd, size=n)
    x = np.empty(n)
    x[0] = e[0] / np.sqrt(1 - rho ** 2)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + e[i]
    return x


def test_prais_winsten_recovers_trend_and_rho():
    rng = np.random.default_rng(42)
    n, rho, slope = 3000, 0.7, 0.002
    t = np.arange(n, dtype=float)
    y = 1.0 + slope * t + _ar1(n, rho, rng)
    X = np.column_stack([np.ones(n), t])
    r = S.prais_winsten(y, X)
    assert abs(r.rho[0] - rho) < 0.04
    assert abs(r.beta[1] - slope) < 3 * r.se[1]
    ols_se = np.sqrt(np.var(y - X @ np.linalg.lstsq(X, y, rcond=None)[0]) / np.sum((t - t.mean()) ** 2))
    assert r.se[1] > 1.8 * ols_se                        # OLS understates it ~ sqrt((1+rho)/(1-rho))


def test_prais_winsten_segments_and_many_series():
    rng = np.random.default_rng(7)
    n = 2400
    t = np.arange(n, dtype=float)
    Y = np.column_stack([0.001 * t + _ar1(n, 0.6, rng), -0.003 * t + _ar1(n, 0.3, rng)])
    keep = np.ones(n, bool)
    keep[800:900] = False                                # an excluded window
    X = np.column_stack([np.ones(n), t])
    r = S.prais_winsten(Y[keep], X[keep], S.segment_starts(keep))
    assert r.beta.shape == (2, 2) and r.dof == keep.sum() - 2
    np.testing.assert_allclose(r.rho, [0.6, 0.3], atol=0.06)
    assert abs(r.beta[1, 0] - 0.001) < 3 * r.se[1, 0] and abs(r.beta[1, 1] + 0.003) < 3 * r.se[1, 1]
    st = S.segment_starts(keep)
    assert st.sum() == 2 and st[800]


def test_acf_inflation_bartlett():
    rng = np.random.default_rng(3)
    white = rng.normal(size=(20000, 2))
    np.testing.assert_allclose(S.acf_inflation(white), 1.0, atol=0.1)
    x = _ar1(40000, 0.5, rng)
    k = np.arange(1, 11)
    expect = 1 + 2 * np.sum((1 - k / 11) * 0.5 ** k)
    np.testing.assert_allclose(S.acf_inflation(x), expect, rtol=0.05)
    # an oscillating series: Bartlett stays near 1 where AR(1) from rho_1 would not
    tt = np.arange(40000)
    osc = np.cos(2 * np.pi * tt / 6.0) + 0.8 * rng.normal(size=tt.size)
    infl = S.acf_inflation(osc)
    r1 = np.corrcoef(osc[1:], osc[:-1])[0, 1]
    assert infl < (1 + r1) / (1 - r1) or r1 < 0
    # gaps: rows on a daily grid
    times = np.datetime64("2026-01-01") + np.r_[np.arange(100), np.arange(110, 300)]
    val = S.acf_inflation(rng.normal(size=times.size), times=times)
    assert 0.5 < float(val) < 1.5


def test_event_bootstrap():
    rng = np.random.default_rng(11)
    ne, L = 60, 5
    ev_means = rng.normal(size=ne)
    X = np.repeat(ev_means, L)[:, None] + 0.0 * rng.normal(size=(ne * L, 1))
    events = np.repeat(np.arange(ne), L)
    mean, se = S.event_bootstrap(X, events, nboot=4000, rng=1)
    np.testing.assert_allclose(mean[0], ev_means.mean(), rtol=1e-12)
    np.testing.assert_allclose(se[0], ev_means.std() / np.sqrt(ne), rtol=0.1)
    # a day bootstrap would understate it by ~sqrt(L)
    day_se = np.std([X[rng.integers(0, ne * L, ne * L)].mean() for _ in range(2000)])
    assert se[0] > 1.8 * day_se
    m1, s1 = S.event_bootstrap(np.ones((4, 2)), np.zeros(4), nboot=10)
    assert np.isnan(s1).all() and (m1 == 1).all()


def test_label_events():
    key = np.array([1, 1, 2, 2, 2, 3, 3, 1])
    act = np.array([1, 1, 1, 0, 1, 1, 1, 1], bool)
    np.testing.assert_array_equal(S.label_events(key, act), [0, 0, 1, -1, 2, 3, 3, 4])


def test_partial_corr_test():
    rng = np.random.default_rng(5)
    n = 60
    z = rng.normal(size=n)
    x = z + 0.5 * rng.normal(size=n)
    y_null = 2 * z + rng.normal(size=n)                  # related to x only through z
    y_alt = 2 * z + 0.8 * (x - z) + 0.5 * rng.normal(size=n)
    null = S.partial_corr_test(x, y_null, [z], nperm=1999, nboot=499, rng=0)
    alt = S.partial_corr_test(x, y_alt, [z], nperm=999, nboot=499, rng=0)
    assert alt["p_perm"] < 0.01 and alt["ci"][0] > 0 and alt["dR2"] > 0
    assert abs(null["p_perm"] - null["p_F"]) < 0.03       # permutation agrees with the exact F test
    assert abs(null["r"]) < abs(alt["r"])


def test_partial_corr_test_size():
    rng = np.random.default_rng(8)
    rej = 0
    nsim = 200
    for _ in range(nsim):
        z = rng.normal(size=30)
        x = z + 0.7 * rng.normal(size=30)
        y = 1.5 * z + rng.normal(size=30)
        rej += S.partial_corr_test(x, y, [z], nperm=199, nboot=10, rng=rng)["p_perm"] <= 0.05
    assert 0.015 <= rej / nsim <= 0.095


def test_compare_r_and_auc():
    rng = np.random.default_rng(9)
    y = rng.normal(size=200)
    res = S.compare_r(y + 0.5 * rng.normal(size=200), rng.normal(size=200), y, nboot=999, rng=0)
    assert res["diff_abs_r"] > 0.5 and res["p_not_better"] < 0.01 and res["ci"][0] > 0
    lab = np.r_[np.zeros(50), np.ones(50)]
    assert S.auc(np.arange(100), lab) == 1.0
    assert S.auc(-np.arange(100), lab) == 0.0
    sc = rng.normal(size=100) + lab
    brute = np.mean([(a > b) + 0.5 * (a == b) for a in sc[lab == 1] for b in sc[lab == 0]])
    np.testing.assert_allclose(S.auc(sc, lab), brute)
    np.testing.assert_allclose(S.auc(np.r_[1, 1, 2, 2], np.r_[0, 1, 0, 1]), 0.5)
