import datetime as dt

import numpy as np
import pytest

from quantmet import degdays as D


def _sine_hdd(mean, amp, base):
    """Exact time-mean HDD of T = mean + amp sin(t), d = base - mean:
    [d (pi + 2 asin(d/A)) + 2 A sqrt(1 - (d/A)^2)] / (2 pi)."""
    d = base - mean
    x = np.clip(d / amp, -1, 1)
    return (d * (np.pi + 2 * np.arcsin(x)) + 2 * amp * np.sqrt(1 - x * x)) / (2 * np.pi)


def _day(mean, amp=10.0, n=24):
    h = np.arange(n) * 24.0 / n
    return mean + amp * np.sin(2 * np.pi * (h - 9) / 24)


def test_integral_vs_midpoint_on_a_sinusoidal_day():
    # shoulder season: the day swings 58..78 around a 65 base -> midpoint HDD 0, integral HDD 1.83
    t = _day(68.0, n=24 * 60)
    mh, mc = D.midpoint_dd(t.max(), t.min())
    ih, ic = D.integral_dd(t)
    assert mh == 0.0 and mc == pytest.approx(3.0)
    assert ih == pytest.approx(_sine_hdd(68, 10, 65), abs=1e-4) and ih > 1.8
    assert ic == pytest.approx(_sine_hdd(-68, 10, -65), abs=1e-4)
    # hourly samples are already within 0.01; the 'linear' integral of the closed day likewise
    assert abs(D.integral_dd(_day(68.0))[0] - _sine_hdd(68, 10, 65)) < 0.01
    t24 = _day(68.0)
    assert abs(D.integral_dd(np.r_[t24, t24[0]], method="linear")[0] - _sine_hdd(68, 10, 65)) < 0.03
    # deep winter / high summer: the whole day on one side of the base -> the two agree
    for m in (40.0, 90.0):
        t = _day(m)
        np.testing.assert_allclose(D.integral_dd(t), D.midpoint_dd(t.max(), t.min()), atol=1e-12)
    # the integral is never below the midpoint for a symmetric day, and HDD - CDD = base - mean for both
    for m in np.arange(50, 81, 1.5):
        t = _day(m, n=240)
        ih, ic = D.integral_dd(t)
        mh, mc = D.midpoint_dd(t.max(), t.min())
        assert ih >= mh - 1e-9 and ic >= mc - 1e-9
        assert ih - ic == pytest.approx(65 - t.mean())


def test_linear_integral_handles_crossings_exactly():
    # one segment 60 -> 70 across a 65 base: HDD is the triangle 5 * 0.5 / 2 = 1.25 over the unit interval
    h, c = D.integral_dd(np.array([60.0, 70.0]), method="linear")
    assert h == pytest.approx(1.25) and c == pytest.approx(1.25)
    h, c = D.integral_dd(np.array([[50.0, 55.0, 60.0], [70.0, 80.0, 66.0]]), method="linear")
    np.testing.assert_allclose(h, [10.0, 0.0])
    np.testing.assert_allclose(c, [0.0, (np.mean([5.0, 15.0]) + np.mean([15.0, 1.0])) / 2])


def test_degree_hours_blocks():
    t = np.stack([_day(60.0), _day(68.0)])
    hrs = np.arange(24.0)
    hdh, cdh = D.degree_hours(t, hrs)
    assert hdh.shape == (2, 4)
    np.testing.assert_allclose(hdh.sum(-1) / 24, D.integral_dd(t)[0])
    np.testing.assert_allclose(cdh.sum(-1) / 24, D.integral_dd(t)[1])
    dn, _ = D.degree_hours(t, hrs, blocks=((7, 19), (19, 24)))
    assert (dn[:, 1] > 0).all()  # evenings carry heating
    h3, _ = D.degree_hours(t[:, ::3], hrs[::3], step=3.0)  # 3-hourly samples
    assert np.allclose(h3.sum(-1) / 24, D.integral_dd(t[:, ::3])[0])


def test_dd_from_extremes_and_shape():
    shape = np.sin(2 * np.pi * (np.arange(24) - 9) / 24)
    h, c = D.dd_from_extremes(78.0, 58.0, shape)
    np.testing.assert_allclose((h, c), D.integral_dd(_day(68.0)), atol=1e-12)
    np.testing.assert_allclose(D.dd_from_extremes(78.0, 58.0, 3 * shape + 7), (h, c))  # shape rescaled to [0, 1]
    hh, cc = D.dd_from_extremes(np.array([78.0, 50.0]), np.array([58.0, 30.0]), shape)
    assert hh[1] == pytest.approx(25.0) and cc[1] == 0.0
    with pytest.raises(ValueError):
        D.dd_from_extremes(70.0, 60.0, np.ones(24))


def test_expected_dd_gaussian_vs_monte_carlo():
    rng = np.random.default_rng(0)
    for mu, s in [(60.0, 5.0), (65.0, 3.0), (70.0, 8.0), (40.0, 6.0)]:
        x = rng.normal(mu, s, 1_000_000)
        h, c = D.expected_dd(mu, s)
        mh, mc = np.maximum(65 - x, 0), np.maximum(x - 65, 0)
        assert abs(h - mh.mean()) < 4 * mh.std() / 1e3 + 1e-12
        assert abs(c - mc.mean()) < 4 * mc.std() / 1e3 + 1e-12
        assert h - c == pytest.approx(65 - mu)
        assert h >= max(65 - mu, 0) and c >= max(mu - 65, 0)  # Jensen
    h, c = D.expected_dd(65.0, 4.0)
    assert h == pytest.approx(4.0 / np.sqrt(2 * np.pi))  # sigma phi(0) at the base
    np.testing.assert_allclose(D.expected_dd(np.array([60.0, 70.0]), 0.0), ([5.0, 0.0], [0.0, 5.0]))
    np.testing.assert_allclose(
        D.expected_dd(15.0, 2.0, base=18.0)[0],  # units scale through
        D.expected_dd(15 * 1.8 + 32, 3.6, 18 * 1.8 + 32)[0] / 1.8,
    )


def test_kink_first_beats_weight_first():
    rng = np.random.default_rng(1)
    t = rng.normal(64.0, 8.0, size=(500, 30))  # 500 days x 30 locations
    w = rng.uniform(0.1, 5.0, 30)
    kh, kc = D.weighted_dd(t, w)
    tw = t @ w / w.sum()
    fh, fc = np.maximum(65 - tw, 0), np.maximum(tw - 65, 0)
    assert (kh >= fh - 1e-12).all() and (kc >= fc - 1e-12).all()  # weighting temperature first is low
    straddle = (t.min(1) < 65) & (t.max(1) > 65)
    assert (kh[straddle] > fh[straddle]).all()
    assert np.mean(kh - fh) > 1.0
    one_side = t[:, :3] * 0 + np.array([40.0, 50.0, 60.0])
    np.testing.assert_allclose(D.weighted_dd(one_side, w[:3])[0], np.maximum(65 - one_side @ w[:3] / w[:3].sum(), 0))
    # the mild aggregate that annihilates heating: half the weight at 45, half at 85 -> mean 65, HDD 0 vs 10
    h2, c2 = D.weighted_dd(np.array([45.0, 85.0]), np.array([1.0, 1.0]))
    assert (h2, c2) == (10.0, 10.0)
    # missing locations: strict by default, renormalised above a coverage threshold
    tm = np.array([[50.0, np.nan, 60.0]])
    assert np.isnan(D.weighted_dd(tm, [1.0, 1.0, 2.0])[0]).all()
    np.testing.assert_allclose(D.weighted_dd(tm, [1.0, 1.0, 2.0], min_coverage=0.7)[0], [(15 + 2 * 5) / 3])


def test_local_days_dst():
    ny = "America/New_York"
    np.testing.assert_array_equal(D.day_length_hours(["2026-03-08", "2026-11-01", "2026-07-04"], ny), [23, 25, 24])
    np.testing.assert_array_equal(D.day_length_hours(["2026-03-29", "2026-10-25"], "Europe/London"), [23, 25])
    t = np.arange("2026-10-30T00", "2026-11-04T00", dtype="datetime64[h]")
    ld = D.local_days(t, ny)
    np.testing.assert_array_equal(
        ld.dates, np.array(["2026-10-30", "2026-10-31", "2026-11-01", "2026-11-02"], dtype="datetime64[D]")
    )
    np.testing.assert_array_equal(ld.n, [24, 24, 25, 24])
    assert ld.dates[0] != np.datetime64("2026-10-29")  # partial first (local) day dropped
    t2 = np.arange("2026-03-07T00", "2026-03-10T00", dtype="datetime64[h]")
    ld2 = D.local_days(t2, ny)
    assert list(ld2.n) == [24, 23] and list(ld2.hours) == [24, 23]
    # every local day, partial ones included, when complete=False
    assert D.local_days(t2, ny, complete=False).dates.size == 4
    # daily extremes per local day
    vals = np.arange(t.size, dtype=float)
    tmax = ld.reduce(vals, np.max)
    assert tmax.shape == (4,) and (np.diff(tmax) > 0).all()
    np.testing.assert_allclose(ld.reduce(np.stack([vals, -vals]), np.min)[1], -tmax)
    # aware datetimes and datetime64 give the same days
    aware = [dt.datetime(2026, 11, 1, h, tzinfo=dt.timezone.utc) for h in range(24)]
    assert D.local_days(aware, ny, complete=False).dates[0] == np.datetime64("2026-10-31")
    # 3-hourly data: a 23-hour day holds 7 or 8 samples, a complete 24-hour day 8
    t3 = np.arange("2026-03-06T00", "2026-03-11T00", 3, dtype="datetime64[h]")
    ld3 = D.local_days(t3, ny)
    assert set(ld3.expected) <= {7, 8} and (ld3.n >= ld3.expected).all() and ld3.dates.size == 4
    # a gap makes the day incomplete
    gap = np.delete(t, 40)
    assert np.datetime64("2026-10-31") not in D.local_days(gap, ny).dates


def test_doy366():
    d = ["2024-02-28", "2024-02-29", "2024-03-01", "2025-02-28", "2025-03-01", "2025-12-31", "2024-12-31", "2025-01-01"]
    np.testing.assert_array_equal(D.doy366(d), [58, 59, 60, 58, 60, 365, 365, 0])
    days = np.arange("2023-01-01", "2025-01-01", dtype="datetime64[D]")
    s = D.doy366(days)
    assert s.min() == 0 and s.max() == 365 and np.unique(s).size == 366
    assert (s[days.astype("datetime64[Y]") == np.datetime64("2023", "Y")] != 59).all()  # common year skips 29 Feb


def test_doy366_one_slot_per_calendar_day_across_centuries():
    # 1896-2104 covers the century rule: 1900 and 2100 are common years, 2000 is a leap year
    days = np.arange("1896-01-01", "2105-01-01", dtype="datetime64[D]")
    s = D.doy366(days)
    md = np.array([str(x)[5:] for x in days])  # "MM-DD"
    for k in np.unique(s):  # each slot is one calendar day in every year, and each calendar day one slot
        assert np.unique(md[s == k]).size == 1
    assert np.unique(s).size == 366 and np.unique(md).size == 366
    np.testing.assert_array_equal(D.doy366(["1900-03-01", "2000-03-01", "2100-03-01", "2000-02-29"]), [60, 60, 60, 59])
    years = days.astype("datetime64[Y]").astype(int) + 1970
    feb29 = np.unique(years[s == 59])
    assert 1900 not in feb29 and 2100 not in feb29 and 2000 in feb29 and feb29.size == 51


def test_doy366_accepts_times_and_python_dates():
    # sub-daily stamps take their calendar day, before 1970 as after
    t = np.array(
        ["1899-12-31T23:00", "1904-02-29T18:00", "2023-03-01T06:00", "2024-03-01T23:59"], dtype="datetime64[s]"
    )
    np.testing.assert_array_equal(D.doy366(t), [365, 59, 60, 60])
    np.testing.assert_array_equal(D.doy366([dt.date(2023, 3, 1), dt.datetime(2024, 3, 1, 12)]), [60, 60])


def test_local_days_across_29_february():
    t = np.arange("2024-02-28T06", "2024-03-02T06", dtype="datetime64[h]")  # Chicago midnight = 06 UTC
    ld = D.local_days(t, "America/Chicago", step_hours=1.0)
    np.testing.assert_array_equal(ld.dates, np.array(["2024-02-28", "2024-02-29", "2024-03-01"], dtype="datetime64[D]"))
    assert (ld.hours == 24).all() and (ld.n == 24).all()
    np.testing.assert_array_equal(D.doy366(ld.dates), [58, 59, 60])


def test_balance_point_fit_recovers_hinges():
    rng = np.random.default_rng(3)
    n = 3000
    t = rng.uniform(10, 95, n)
    dow = rng.integers(0, 7, n)
    X = (dow[:, None] == np.arange(1, 7)[None]).astype(float)  # day-of-week dummies (Monday base)
    eff = np.r_[0.0, 20.0, 25.0, 25.0, 20.0, -60.0, -90.0]
    y = 1000 + 30 * np.maximum(52 - t, 0) + 45 * np.maximum(t - 67, 0) + eff[dow] + rng.normal(0, 20, n)
    f = D.fit_balance_points(t, y, X=X)
    assert (f.heat_base, f.cool_base) == (52.0, 67.0)
    assert f.heat_slope == pytest.approx(30, rel=0.02) and f.cool_slope == pytest.approx(45, rel=0.02)
    np.testing.assert_allclose(f.extra, eff[1:], atol=4.0)
    assert f.r2 > 0.99 and f.n == n
    np.testing.assert_allclose(f.predict(t, X), y, atol=100)
    # without the day-of-week terms the hinges still come out, the fit is just noisier
    g = D.fit_balance_points(t, y, step=0.5)
    assert abs(g.heat_base - 52) <= 1.0 and abs(g.cool_base - 67) <= 1.0
    # a sloped dead band
    y2 = 500 + 2 * t + 30 * np.maximum(50 - t, 0) + 40 * np.maximum(t - 70, 0) + rng.normal(0, 5, n)
    h = D.fit_balance_points(t, y2, linear=True)
    assert (h.heat_base, h.cool_base) == (50.0, 70.0) and h.middle_slope == pytest.approx(2.0, abs=0.1)
    y3 = y.copy()
    y3[:5] = np.nan
    assert D.fit_balance_points(t, y3, X=X).n == n - 5
