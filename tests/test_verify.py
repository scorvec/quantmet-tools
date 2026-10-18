import numpy as np
import pytest

from quantmet import verify as V

SQPI = np.sqrt(np.pi)


def _abs_normal(m, s):
    """E|Z| for Z ~ N(m, s^2)."""
    from scipy.stats import norm

    return s * np.sqrt(2 / np.pi) * np.exp(-m * m / (2 * s * s)) + m * (1 - 2 * norm.cdf(-m / s))


# --------------------------------------------------------------------------
# CRPS
# --------------------------------------------------------------------------
def test_crps_gaussian_against_the_integral():
    from scipy.integrate import quad
    from scipy.stats import norm

    for mu, s, y in [(0.0, 1.0, 0.3), (2.0, 0.5, -1.0), (-1.0, 3.0, 4.0)]:

        def f(x, mu=mu, s=s, y=y):
            return (norm.cdf(x, mu, s) - (x >= y)) ** 2

        ref = quad(f, -np.inf, y)[0] + quad(f, y, np.inf)[0]
        assert abs(V.crps_gaussian(mu, s, y) - ref) < 1e-8
    np.testing.assert_allclose(V.crps_gaussian(1.0, 0.0, 3.5), 2.5)
    np.testing.assert_allclose(V.crps_gaussian(0.0, 1.0, 0.0), 2 * 0.3989422804014327 - 1 / SQPI)
    assert V.crps_gaussian(np.zeros((2, 3)), 1.0, 0.0).shape == (2, 3)


def test_crps_ensemble_of_quantiles_converges_to_gaussian():
    from scipy.stats import norm

    m = 4000
    q = norm.ppf((np.arange(m) + 0.5) / m)
    for mu, s, y in [(0.0, 1.0, 0.3), (5.0, 2.0, 1.0)]:
        assert abs(V.crps_ensemble(mu + s * q, y) - V.crps_gaussian(mu, s, y)) < 1e-3 * s


def test_crps_ensemble_exact_weights_nan_axis():
    rng = np.random.default_rng(0)
    x = rng.normal(size=7)
    w = rng.uniform(size=7)
    y = 0.3
    wn = w / w.sum()
    brute = (wn * abs(x - y)).sum() - 0.5 * (wn[:, None] * wn[None] * abs(x[:, None] - x[None])).sum()
    np.testing.assert_allclose(V.crps_ensemble(x, y, weights=w), brute, rtol=1e-12)
    xe = np.r_[x, np.nan]
    np.testing.assert_allclose(V.crps_ensemble(xe, y), V.crps_ensemble(x, y), rtol=1e-12)  # NaN member dropped
    assert np.isnan(V.crps_ensemble(np.full(4, np.nan), 0.0)) and np.isnan(V.crps_ensemble(x, np.nan))
    np.testing.assert_allclose(V.crps_ensemble([2.0], 0.5), 1.5)  # one member: |error|
    ens = rng.normal(size=(4, 11, 3))
    obs = rng.normal(size=(4, 3))
    np.testing.assert_allclose(V.crps_ensemble(ens, obs, axis=1), V.crps_ensemble(np.moveaxis(ens, 1, -1), obs))
    xt = np.array([1.0, 1.0, 2.0, 2.0, 2.0])  # ties
    brute = np.mean(abs(xt - 1.4)) - 0.5 * np.mean(abs(xt[:, None] - xt[None]))
    np.testing.assert_allclose(V.crps_ensemble(xt, 1.4), brute)


def test_fair_crps_is_unbiased_where_the_plain_one_is_not():
    """Members and observation iid N(0, 1): the expected CRPS of the true distribution is 1/sqrt(pi). The plain
    ensemble CRPS expects (1 + 1/m)/sqrt(pi); the fair one 1/sqrt(pi) at every m."""
    rng = np.random.default_rng(42)
    n = 100_000
    for m in (2, 5, 10, 50):
        ens = rng.normal(size=(n, m))
        y = rng.normal(size=n)
        fair = V.crps_fair(ens, y)
        plain = V.crps_ensemble(ens, y)
        se = fair.std() / np.sqrt(n)
        assert abs(fair.mean() - 1 / SQPI) < 4 * se, m
        assert abs(plain.mean() - (1 + 1 / m) / SQPI) < 4 * plain.std() / np.sqrt(n), m
        assert plain.mean() - fair.mean() > 0.8 / (m * SQPI)  # the 1/m inflation, measured
        brute = np.mean(abs(ens[:5, :, None] - ens[:5, None, :]), (1, 2)) * m / (m - 1)
        np.testing.assert_allclose(fair[:5], np.mean(abs(ens[:5] - y[:5, None]), 1) - 0.5 * brute, rtol=1e-10)


def test_fair_crps_weighted_mixture():
    """A 3-member model (weight 0.6, N(0, 1)) and a 5-member model (weight 0.4, N(1, 4)), observation from the
    mixture: the grouped fair CRPS is unbiased for the mixture's expected CRPS, 1/2 E|X - X'|."""
    rng = np.random.default_rng(7)
    n = 200_000
    ens = np.concatenate([rng.normal(0, 1, (n, 3)), rng.normal(1, 2, (n, 5))], 1)
    w = np.r_[np.full(3, 0.6 / 3), np.full(5, 0.4 / 5)]
    g = np.r_[0, 0, 0, 1, 1, 1, 1, 1]
    y = np.where(rng.random(n) < 0.6, rng.normal(0, 1, n), rng.normal(1, 2, n))
    truth = 0.5 * (
        0.36 * _abs_normal(0, np.sqrt(2)) + 0.16 * _abs_normal(0, np.sqrt(8)) + 2 * 0.24 * _abs_normal(1, np.sqrt(5))
    )
    fair = V.crps_fair(ens, y, weights=w, groups=g)
    plain = V.crps_ensemble(ens, y, weights=w)
    assert abs(fair.mean() - truth) < 4 * fair.std() / np.sqrt(n)
    assert plain.mean() - truth > 0.05  # the small-ensemble penalty
    # one group with equal weights is Ferro's fair CRPS; a point-mass member needs no correction
    np.testing.assert_allclose(V.crps_fair(ens[:10], y[:10], weights=np.ones(8)), V.crps_fair(ens[:10], y[:10]))
    det = np.c_[ens[:10, :5], np.zeros(10)]
    wd = np.r_[np.full(5, 0.1), 0.5]
    gd = np.r_[0, 0, 0, 0, 0, 1]
    within = np.mean(abs(det[:, :5, None] - det[:, None, :5]), (1, 2)) * 5 / 4
    cross = np.mean(abs(det[:, :5]), 1)
    ref = np.sum(wd * abs(det - y[:10, None]), 1) - 0.5 * (0.25 * within + 2 * 0.25 * cross)
    np.testing.assert_allclose(V.crps_fair(det, y[:10], weights=wd, groups=gd), ref, rtol=1e-10)


# --------------------------------------------------------------------------
# ranks and PIT
# --------------------------------------------------------------------------
def test_rank_histogram_flat_for_a_calibrated_ensemble():
    from scipy.stats import chisquare

    rng = np.random.default_rng(1)
    n, m = 55_000, 10
    h = V.rank_histogram(rng.normal(size=(n, m)), rng.normal(size=n), rng=0)
    assert h.shape == (m + 1,) and h.sum() == n
    assert chisquare(h).pvalue > 0.001
    # under-dispersive: U shape
    hu = V.rank_histogram(0.5 * rng.normal(size=(n, m)), rng.normal(size=n), rng=0)
    assert hu[0] > 2 * hu[m // 2] and hu[-1] > 2 * hu[m // 2]


def test_ties_are_randomised():
    from scipy.stats import chisquare

    rng = np.random.default_rng(2)
    n, m = 40_000, 9
    ens = rng.poisson(2.0, size=(n, m)).astype(float)
    obs = rng.poisson(2.0, size=n).astype(float)
    h = V.rank_histogram(ens, obs, rng=3)
    assert chisquare(h).pvalue > 0.001
    naive = np.bincount((ens < obs[:, None]).sum(1), minlength=m + 1)  # ties counted as "above"
    assert chisquare(naive).pvalue < 1e-10
    np.testing.assert_array_equal(V.rank_histogram(ens, obs, rng=3), h)  # seedable
    r = V.ranks(np.array([[0.0, 1.0, 2.0], [np.nan, 1.0, 2.0]]), np.array([1.5, 1.5]))
    np.testing.assert_array_equal(r, [2, -1])


def test_pit_uniform():
    from scipy.stats import kstest

    rng = np.random.default_rng(4)
    n, m = 20_000, 20
    u = V.pit(rng.normal(size=(n, m)), rng.normal(size=n), rng=5)
    assert (u > 0).all() and (u < 1).all()
    assert kstest(u, "uniform").pvalue > 0.01
    ut = V.pit(rng.poisson(1.0, (n, m)).astype(float), rng.poisson(1.0, n).astype(float), rng=6)
    assert kstest(ut, "uniform").pvalue > 0.01
    assert np.isnan(V.pit(np.array([[np.nan, 1.0]]), np.array([0.0]))).all()


# --------------------------------------------------------------------------
# Brier
# --------------------------------------------------------------------------
def test_brier_decomposition_identity():
    rng = np.random.default_rng(8)
    for _ in range(5):
        p = rng.uniform(size=3000)
        o = (rng.uniform(size=3000) < p**1.5).astype(float)
        d = V.brier_decomposition(p, o, bins=10)
        assert abs(d["bs"] - (d["reliability"] - d["resolution"] + d["uncertainty"] + d["wbv"] - d["wbc"])) < 1e-12
        assert abs(d["bs"] - V.brier_score(p, o)) < 1e-15 and d["n"] == 3000
    # forecasts on one value per bin: the within-bin terms vanish and Murphy's three-term identity is exact
    p = rng.choice(np.arange(10) / 10 + 0.05, size=5000)
    o = (rng.uniform(size=5000) < p).astype(float)
    d = V.brier_decomposition(p, o)
    assert d["wbv"] < 1e-25 and abs(d["wbc"]) < 1e-15
    assert abs(d["bs"] - (d["reliability"] - d["resolution"] + d["uncertainty"])) < 1e-12
    assert d["reliability"] < 0.002  # reliable by construction
    c = np.full(5000, o.mean())  # climatology: no resolution
    dc = V.brier_decomposition(c, o)
    assert dc["resolution"] < 1e-25 and abs(dc["bs"] - dc["uncertainty"]) < 1e-12


def test_reliability_curve():
    rng = np.random.default_rng(9)
    p = rng.uniform(size=20000)
    o = (rng.uniform(size=20000) < p).astype(float)
    rc = V.reliability_curve(p, o, bins=5)
    assert rc["count"].sum() == 20000 and rc["edges"].size == 6
    np.testing.assert_allclose(rc["observed"], rc["forecast"], atol=0.03)
    rc1 = V.reliability_curve(np.array([1.0, 0.95, 0.0]), np.array([1.0, 1.0, 0.0]), bins=10)
    assert rc1["count"][-1] == 2 and np.isnan(rc1["observed"][5])  # p = 1 closes the last bin
    with pytest.raises(ValueError):
        V.brier_score([0.5, 1.2], [0, 1])


# --------------------------------------------------------------------------
# Diebold-Mariano
# --------------------------------------------------------------------------
def _ar1(rng, reps, n, rho):
    e = rng.normal(size=(reps, n))
    x = np.empty_like(e)
    x[:, 0] = e[:, 0] / np.sqrt(1 - rho**2)
    for i in range(1, n):
        x[:, i] = rho * x[:, i - 1] + e[:, i]
    return x


def test_dm_size_near_nominal_on_autocorrelated_equal_skill_losses():
    """Two equally good forecasts with AR(1) errors (rho 0.8): squared-error loss differentials have lag-k
    autocorrelation 0.64^k. The HAC test holds its size; the naive test does not."""
    rng = np.random.default_rng(10)
    reps, n = 600, 300
    ea, eb = _ar1(rng, reps, n, 0.8), _ar1(rng, reps, n, 0.8)
    rej = np.mean([V.diebold_mariano(ea[k] ** 2, eb[k] ** 2).pvalue < 0.05 for k in range(reps)])
    naive = np.mean(
        [V.diebold_mariano(ea[k] ** 2, eb[k] ** 2, lags=0, prewhiten=False).pvalue < 0.05 for k in range(reps)]
    )
    assert 0.025 < rej < 0.085
    assert naive > 0.25


def test_dm_power_and_direction():
    rng = np.random.default_rng(11)
    reps, n = 300, 500
    ea, eb = _ar1(rng, reps, n, 0.5), 1.25 * _ar1(rng, reps, n, 0.5)
    res = [V.diebold_mariano(ea[k] ** 2, eb[k] ** 2) for k in range(reps)]
    assert np.mean([r.pvalue < 0.05 for r in res]) > 0.9
    assert all(r.mean_diff < 0 for r in res[:20])  # a has the lower loss
    r = V.diebold_mariano(ea[0] ** 2, eb[0] ** 2, alternative="less")
    assert r.pvalue < V.diebold_mariano(ea[0] ** 2, eb[0] ** 2, alternative="greater").pvalue
    assert abs(r.pvalue - res[0].pvalue / 2) < 1e-12
    assert V.diebold_mariano(ea[0], eb[0], h=5, lags=0).lags == 4  # at least h - 1 lags
    with pytest.raises(ValueError):
        V.diebold_mariano([1.0, np.nan, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0])


def test_dm_hln_factor():
    rng = np.random.default_rng(12)
    a, b = rng.normal(size=40), rng.normal(size=40)
    r0 = V.diebold_mariano(a, b, h=3, lags=2, prewhiten=False, hln=False)
    r1 = V.diebold_mariano(a, b, h=3, lags=2, prewhiten=False, hln=True)
    np.testing.assert_allclose(r1.statistic / r0.statistic, np.sqrt((40 + 1 - 6 + 6 / 40) / 40))
    d = a - b
    u = d - d.mean()
    lrv = (u @ u + 2 * (2 / 3) * (u[1:] @ u[:-1]) + 2 * (1 / 3) * (u[2:] @ u[:-2])) / 40
    np.testing.assert_allclose(r0.lrv, lrv)
    np.testing.assert_allclose(r0.statistic, d.mean() / np.sqrt(lrv / 40))


# --------------------------------------------------------------------------
# pairing
# --------------------------------------------------------------------------
def test_paired_keeps_only_days_both_legs_have():
    a = {"2026-09-01": 1.0, "2026-09-02": 2.0, "2026-09-03": None, "2026-09-04": 4.0, "2026-09-05": 5.0}
    b = (["2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04", "2026-09-06"], [10.0, np.nan, 30.0, 40.0, 60.0])
    r = V.paired(a, b)
    np.testing.assert_array_equal(r.dates, np.array(["2026-09-01", "2026-09-04"], dtype="datetime64[D]"))
    np.testing.assert_array_equal(r.a, [1.0, 4.0])
    np.testing.assert_array_equal(r.b, [10.0, 40.0])
    assert r.n == 2 and r.window == 6  # union of dates with any data
    w = V.paired(a, b, start="2026-09-01", end="2026-09-07")
    assert w.n == 2 and w.window == 7  # "2/7 d"
    # the ratio over the window charged with a missing day would be wrong; over paired days it is exact
    assert np.sum(r.a) / np.sum(r.b) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        V.paired((["2026-09-01", "2026-09-01"], [1.0, 2.0]), a)
    h = V.paired(
        (np.arange("2026-09-01T00", "2026-09-01T06", dtype="datetime64[h]"), np.arange(6.0)),
        {np.datetime64("2026-09-01T03"): 3.0},
        unit="h",
    )
    assert h.n == 1 and h.a[0] == 3.0


def test_paired_window_counts_29_february():
    for year, days in ((2023, 28), (2024, 29), (2100, 28), (2000, 29)):
        dates = np.arange(f"{year}-02-01", f"{year}-03-01", dtype="datetime64[D]")
        r = V.paired((dates, np.ones(dates.size)), (dates, np.ones(dates.size)), start=f"{year}-02-01", end=dates[-1])
        assert r.n == days and r.window == days


def test_paired_pandas_series():
    pd = pytest.importorskip("pandas")
    a = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2026-01-01", periods=3))
    b = pd.Series([5.0, np.nan], index=pd.date_range("2026-01-02", periods=2))
    r = V.paired(a, b)
    assert r.n == 1 and r.a[0] == 2.0 and r.b[0] == 5.0
