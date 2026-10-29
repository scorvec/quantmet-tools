# quantmet

[![tests](https://github.com/scorvec/quantmet-tools/actions/workflows/tests.yml/badge.svg)](https://github.com/scorvec/quantmet-tools/actions/workflows/tests.yml)
[![build](https://github.com/scorvec/quantmet-tools/actions/workflows/release.yml/badge.svg)](https://github.com/scorvec/quantmet-tools/actions/workflows/release.yml)
[![PyPI](https://img.shields.io/pypi/v/quantmet-tools)](https://pypi.org/project/quantmet-tools/)
[![docs](https://github.com/scorvec/quantmet-tools/actions/workflows/docs.yml/badge.svg)](https://scorvec.com/quantmet-tools/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://github.com/scorvec/quantmet-tools/blob/main/pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/scorvec/quantmet-tools/blob/main/LICENSE)

Documentation: [scorvec.com/quantmet-tools](https://scorvec.com/quantmet-tools/)

<!-- --8<-- [start:overview] -->
Quantitative meteorology tools, extracted from the diagnostics pipelines
behind [scorvec.com](https://scorvec.com), whose daily products import this
package. `dct_spectra` and `tc_detect` come from products since retired and
`conservation` from a one-off study; `verify`, `thermo` and `degdays` (0.4)
are general-purpose implementations written for the package. Each module is self-contained and
documented with the numerical subtleties that matter in practice, and each
has a test file in `tests/`, against analytic cases wherever one exists.

| Module | What it does | Key references |
|---|---|---|
| `quantmet.tem` | Transformed-Eulerian-mean diagnostics: eddy covariances by zonal wavenumber (Nyquist-exact), residual circulation v\*, ω\*, the full primitive-equation E–P flux and its divergence by wave group, residual streamfunction, downward control, tropical upwelling w\*, eddy heat flux [v′T′] by wavenumber | Andrews, Holton & Leovy (1987); Haynes et al. (1991, JAS) |
| `quantmet.epflux` | Quasi-geostrophic Eliassen–Palm flux + divergence as a zonal force, with a per-member-flux ensemble path; zonal-mean static stability, ln-p derivatives and below-ground masking; Charney–Drazin propagation ceiling for stationary planetary waves | Eliassen & Palm (1961); Edmon, Hoskins & McIntyre (1980, JAS); Charney & Drazin (1961, JGR) |
| `quantmet.waf` | Takaya–Nakamura wave-activity flux, per ensemble member, with divergence, sign agreement and a low-passed basic state | Takaya & Nakamura (2001, JAS) |
| `quantmet.pv` | Ertel PV on pressure levels, the dynamic tropopause (θ, p, wind on 2 PVU by an upward search interpolated in PV), PV on isentropes | Hoskins, McIntyre & Robertson (1985, QJRMS); Morgan & Nielsen-Gammon (1998, MWR) |
| `quantmet.packets` | Rossby wave-packet envelope (Hilbert transform along latitude circles) and latitude-band means for packet Hovmöllers | Zimin et al. (2003, MWR) |
| `quantmet.tc` | ECMWF tropical-cyclone track BUFR decoding (every ensemble member), recurvature, the outflow–jet interaction metric (−v_χ·∇PV), Hart cyclone phase space and extratropical-transition timing | Archambault et al. (2013, 2015, MWR); Hart (2003, MWR); Evans & Hart (2003, MWR) |
| `quantmet.ensemble` | Ensemble sensitivity (regression across members) and member-group composites with FDR masking, and a no-lookahead estimator of systematic 00Z/12Z cycle offsets | Ancell & Hakim (2007, MWR); Torn & Hakim (2008, MWR) |
| `quantmet.conservation` | Global dry-air mass, the column water budget (implied evaporation), column total energy on pressure levels, atmospheric angular momentum and its mountain and friction torques | Trenberth & Smith (2005, J. Climate); Peixoto & Oort (1992); Egger et al. (2007, Rev. Geophys.) |
| `quantmet.gill` | Gill model: steady damped shallow-water response to tropical heating by one sparse direct solve; heating from SST anomalies (total-SST convective threshold) or from rainfall | Gill (1980, QJRMS) |
| `quantmet.qbo` | QBO zero-wind line: the latitude where the zonal-mean wind turns easterly equatorward of the winter westerlies, with the dip rule and censoring | Holton & Tan (1980, JAS) |
| `quantmet.stats` | Benjamini–Hochberg FDR, Prais–Winsten regression with AR(1) errors and segment breaks, Bartlett variance inflation, event-block bootstrap composites, Freedman–Lane partial-correlation test, bootstrap r comparison, AUC | Wilks (2016, BAMS); Freedman & Lane (1983) |
| `quantmet.helmholtz` | Global Helmholtz decomposition (velocity potential / streamfunction, divergent / rotational wind) by spherical-harmonic Poisson inversion on a pole-free Gauss–Legendre grid | — |
| `quantmet.wk_filter` | Wheeler–Kiladis space-time filtering of equatorial waves (Kelvin / ER / MJO or any custom band) with per-wavenumber dispersion bounds from the full cubic; Hermitian-safe FFT masking; any number of middle dimensions; soft-landing end padding; Lanczos low-pass | Wheeler & Kiladis (1999, JAS) |
| `quantmet.dct_spectra` | Kinetic-energy power spectra on limited-area grids via 2-D DCT + elliptic-ring binning; spectral slopes and effective resolution | Denis, Côté & Laprise (2002, MWR); Skamarock (2004) |
| `quantmet.tc_detect` | Closed-circulation tropical-cyclone candidate detection with an O(1)-per-gridpoint summed-area-table ring test (C kernel, compiled on first use) | — |
| `quantmet.verify` | Forecast verification: ensemble, fair (unbiased for finite and weighted multi-model ensembles) and Gaussian CRPS; tie-randomised rank histograms and PIT; the Brier score with its exact decomposition and reliability curves; the Diebold–Mariano test with a prewhitened Newey–West variance and the Harvey–Leybourne–Newbold correction; strict pairing of dated series | Gneiting & Raftery (2007, JASA); Ferro (2014, QJRMS); Hamill (2001, MWR); Stephenson et al. (2008, WAF); Diebold & Mariano (1995); Harvey et al. (1997, IJF) |
| `quantmet.thermo` | Moist thermodynamics on SI inputs: saturation vapour pressure (Magnus/Alduchov–Eskridge, Bolton, Murphy–Koop; water and ice), pressure-aware humidity conversions, θ (dry and moist κ), T_v, θ_v, density temperature, θ_e (Bolton pseudo-adiabatic and reversible), θ_es, θ_l, θ_il, θ_w, Stull and psychrometric wet bulb, exact LCL, static energies, enthalpy, the pseudo-adiabat, column water / MSE / enthalpy 0 → p_s, hypsometric thickness; NWS heat index and wind chill; unit helpers | Bolton (1980, MWR); Alduchov & Eskridge (1996, JAM); Murphy & Koop (2005, QJRMS); Emanuel (1994); Davies-Jones (2008, MWR); Romps (2017, JAS); Stull (2011, JAMC); Rothfusz (1990); Osczevski & Bluestein (2005, BAMS) |
| `quantmet.degdays` | Degree days: midpoint and integral conventions, degree hours by time-of-day block, degree days from daily extremes and a diurnal shape, expected degree days of a Gaussian temperature, kink-first weighted aggregation, DST-safe local days from UTC data, a 366-slot day of year, two-hinge balance-point fit | Thom (1954, MWR); Kissock, Haberl & Claridge (2003, ASHRAE RP-1050) |
| `quantmet.constants` | One SI constant set (R_d, R_v, c_pd, c_pv, c_l, c_i, L_v0, L_f0, g, …) used by `thermo` | — |
<!-- --8<-- [end:overview] -->

## Quick starts

<!-- --8<-- [start:quickstarts] -->
**Stratospheric momentum budget and the Brewer–Dobson circulation** (zonal
means and eddy covariances from (plev, lat, lon) snapshots, levels 0.1–200 hPa):

```python
from quantmet.tem import eddy_covariances, tem_terms, downward_control, fill_tropics, upwelling_w

r = eddy_covariances(u, v, t, omega)  # [.] and [a'b'], total and by k1 / k2 / k3+
tt = tem_terms(lat, plev_hpa, r)  # fhat, v*, omega*, EP flux, div F (m/s^2) by wave group
psi = downward_control(lat, plev_hpa, tt["fhat"], tt["epd"] + gwd)  # kg/s; parts add linearly
psi = fill_tropics(psi, lat, edge=15)  # fhat -> 0 near the equator
up = upwelling_w(psi, lat, plev_hpa, lat_s=-25, lat_n=25)  # tropical w* (m/s) by level
```

**Is the planetary wave going up?** (100 hPa eddy heat flux, poleward positive):

```python
from quantmet.tem import eddy_heat_flux

hf = eddy_heat_flux(v100, t100, lat, kmax=72, lat_band=(45, 75))
hf.total, hf.by_k[..., 0], hf.by_k[..., 1]  # all waves, wave-1, wave-2
```

**Stratospheric wave driving** (QG E–P flux, per-member ensemble path) and the
**Charney–Drazin ceiling**:

```python
from quantmet.epflux import ep_flux, ensemble_ep_flux, charney_drazin_excess

r = ep_flux(u, v, t, lat, plev_hpa, kmax=3)  # (plev, lat, lon) snapshots in
r.force  # ∇·F as m/s per day, (plev, lat)
ens = ensemble_ep_flux(((u_m, v_m, t_m) for u_m, v_m, t_m in members), lat, plev_hpa)  # per-member fluxes averaged
lid = charney_drazin_excess(ubar, lat, plev_hpa, theta_profile, k=1)  # u - U_c; 0 < u < U_c propagates
```

**Rossby wave packets** (250 hPa wave-activity flux from an ensemble):

```python
from quantmet.waf import streamfunction_anomaly, ensemble_tn01_flux, lead_smooth

psi_a, lat, lon = streamfunction_anomaly(u250, v250, psi_clim)  # needs quantmet[waf]
w = ensemble_tn01_flux(member_psi_anomalies, U, V, lat, lon, p_hpa=250)
w.wx, w.wy, w.div, w.agree  # mean of per-member fluxes; sign agreement 0.5-1
```

**Gill response to a heating pattern**:

```python
from quantmet.gill import gill_response, heating_from_ssta, heating_from_precip, to_dimensional

Q = heating_from_ssta(ssta, sst_clim, lat)  # shape only; or, in physical units:
Q = heating_from_precip(precip_anom_mm_day)  # 1 mm/day = 28.9 W/m^2 of column heating
u, v, p = gill_response(Q, lat, lon, eps=0.1)  # low-level fields, nondimensional
u_ms, v_ms, phi = to_dimensional(u, v, p)
```

**QBO zero-wind line** (per day, level or member):

```python
from quantmet.qbo import zero_line

lat0, flag = zero_line(ubar_50hpa, lat, hemi="nh")  # flag 0 found, 1 undefined, 2 censored
```

**Significance**:

```python
from quantmet.stats import benjamini_hochberg, prais_winsten, acf_inflation, event_bootstrap

sig = benjamini_hochberg(pvals, alpha=0.10)  # field significance, NaN-aware
fit = prais_winsten(y, X, start=starts)  # trend with AR(1) errors, segment breaks
infl = acf_inflation(daily_residuals)  # variance inflation of a mean (Bartlett)
mean, se = event_bootstrap(composite_days, event_ids)
```

**Equatorial wave decomposition** (time × ... × longitude anomalies):

```python
from quantmet.wk_filter import wk_filter

kelvin = wk_filter(olr_anom, "Kelvin")  # same shape, real
kelvin = wk_filter(chi_anom, dict(k=(1, 14), p=(2.5, 20), h=(8, 90), n=None), pad_end=60)  # custom band, live end
```

**Walker circulation strength** (velocity potential of the divergent wind):

```python
from quantmet.helmholtz import velocity_potential, irrotational_wind

chi, lats, lons = velocity_potential(u200, v200)  # xarray DataArrays in
uchi, vchi = irrotational_wind(chi, lats, lons)
```

**Model effective resolution** (used to compare HRRR vs RRFS):

```python
from quantmet.dct_spectra import ke_spectrum, spectral_slope, effective_resolution

wl, p = ke_spectrum(u10, v10, res_km=3.0)
print(spectral_slope(wl, p, 30, 300))  # ≈ -5/3 in the resolved mesoscale
print(effective_resolution(wl, p))  # where numerics start dissipating
```

**Dynamic tropopause** (θ on 2 PVU; one member — PV does not survive averaging):

```python
from quantmet.pv import ertel_pv, dynamic_tropopause, on_isentrope

pv, theta = ertel_pv(t, u, v, p_pa, lat, lon)  # (lev, lat, lon) in, PVU out
dt = dynamic_tropopause(pv, theta, u, v, p_pa)  # dt["theta"], dt["p"] (hPa), dt["u"], dt["v"]
pv330 = on_isentrope(pv, theta, u, v, 330.0)["pv"]
```

**Rossby wave packets** (a packet Hovmöller):

```python
from quantmet.packets import packet_envelope, band_mean

env = band_mean(packet_envelope(v250, kmin=4, kmax=15), lat, 35, 60)  # v250 (time, lat, lon) -> (time, lon)
```

**Tropical cyclones meeting the jet**:

```python
from quantmet.tc import decode_tracks, recurvature, pv_advection, outflow_index, phase_series, et_times

storms = decode_tracks("20261002000000-360h-enfo-tf.bufr")  # ECMWF open data, every member
ctl = storms[0]["tracks"][0]  # member 0 = control
recurvature(ctl)  # lead (h) of the westward-to-eastward turn
adv = pv_advection(u_chi, v_chi, pv_300_200, lat, lon)  # PVU/day
outflow_index(adv, lat, lon, clat, clon)  # Archambault et al. interaction metric
hart = phase_series(z_steps, levs, lat, lon, ctl, steps)  # B, -VT_L, -VT_U along the track
et_times(hart)  # (onset, completion) leads
```

**Ensemble sensitivity with field significance**:

```python
from quantmet.ensemble import sensitivity

r = sensitivity(day3_storm_latitude, z500_day7)  # (members,) and (members, lat, lon)
r["slope"], r["sig"]  # m per degree; Benjamini-Hochberg mask at FDR 10 %
```

**Do AI weather models conserve mass, water, energy and angular momentum?**

```python
from quantmet import conservation as C

C.dry_air_mass(ps, tcw, lat, lon)  # kg, ~5.135e18
C.column_energy(t, phi, q, p_pa, ps, u, v, phis=g * orography_m)  # J/m2: total = cpT + Lq + KE + Phi_s p_s/g
C.relative_aam(u, p_pa, ps, lat, lon, hemispheres=True)
C.mountain_torque(ps, orography_m, lat, lon)
```

**Ensemble TC tracking building block**:

```python
from quantmet.tc_detect import detect_candidates

clat, clon, cp = detect_candidates(mslp_hpa, lat, lon, ring_deg=2.5, depth_hpa=2.0)
```

**Is the new ensemble better, or just bigger?** (fair CRPS, a significance test, honest pairing):

```python
from quantmet import verify as V

crps_a = V.crps_fair(ens_a, obs)  # (..., m) members last; unbiased whatever m
crps_b = V.crps_fair(ens_b, obs)
blend = V.crps_fair(
    np.concatenate([ens_a, ens_b], -1), obs, weights=w_members, groups=model_of_member
)  # a weighted multi-model mixture
V.crps_gaussian(mu, sigma, obs)  # EMOS / any Gaussian forecast
dm = V.diebold_mariano(daily_crps_a, daily_crps_b)  # dm.statistic, dm.pvalue; mean_diff < 0: a better
pr = V.paired(score_a_by_date, score_b_by_date)  # dates both legs have; report pr.n / pr.window
V.rank_histogram(ens, obs, rng=0)
V.pit(ens, obs, rng=0)  # ties randomised, seedable
d = V.brier_decomposition(p, o, bins=10)  # bs = reliability - resolution + uncertainty + wbv - wbc
```

**Moist thermodynamics** (K, Pa, kg/kg, RH in %):

```python
from quantmet import thermo as T

q = T.specific_humidity_from_dewpoint(td, p)
the = T.equivalent_potential_temperature(t, td, p)  # Bolton (1980) eq. 39, pseudo-adiabatic
thw = T.wet_bulb_potential_temperature(t, td, p)  # Davies-Jones (2008)
t_lcl, p_lcl, z_lcl = T.lcl_romps(t, p, rh)  # exact LCL (Romps 2017)
tw = T.wet_bulb_temperature(t, p, rh=rh)  # psychrometric, at the real pressure
T.wet_bulb_stull(t, rh)  # Stull (2011): sea-level fit, NaN outside its domain
pw = T.precipitable_water(q3d, p_levels, ps)  # kg/m2, integrated 0 -> p_s
dz = T.thickness(t3d, p_levels, 100000.0, 50000.0, q=q3d)  # 1000-500 hPa thickness, m
T.heat_index(T.k_to_f(t2m), rh2m)
T.wind_chill(T.k_to_f(t2m), T.ms_to_mph(wind10))
```

**Degree days** (any units; base 65 by default, pass 18 for °C):

```python
from quantmet import degdays as D

ld = D.local_days(times_utc, "America/Chicago")  # 23/24/25-hour local days, partial days dropped
tmax, tmin = ld.reduce(t_hourly, np.max), ld.reduce(t_hourly, np.min)
hdd_mid, cdd_mid = D.midpoint_dd(tmax, tmin)  # the published convention
hdd_int = ld.reduce(t_hourly, lambda x, axis: D.integral_dd(x, axis=axis)[0])  # time mean of the hinge
hdh, cdh = D.degree_hours(t_day, local_hours, blocks=((7, 19), (19, 24), (0, 7)))
hdd, cdd = D.weighted_dd(t_by_location, population)  # kink per location, THEN weight
hdd, cdd = D.expected_dd(ens_mean, ens_sd)  # E[max(65 - T, 0)] for T ~ N(mu, sigma^2)
fit = D.fit_balance_points(daily_temp, daily_load, X=dow_dummies)  # fit.heat_base, fit.cool_base, slopes
```
<!-- --8<-- [end:quickstarts] -->

## Case studies

<!-- --8<-- [start:casestudies] -->
- [**AIFS single vs AIFS-ENS control: spectral fidelity by lead time**](https://github.com/scorvec/quantmet-tools/tree/main/examples/aifs_spectral_fidelity)
  — the deterministic AIFS blurs progressively (MSE objective) while the
  CRPS-trained ENS control holds a realistic spectrum ten days in; measured
  with `quantmet.dct_spectra` on open data.
<!-- --8<-- [end:casestudies] -->

## Numerical notes worth stealing

<!-- --8<-- [start:notes] -->
- **Quadratics are formed per member, then averaged.** E–P flux, [v′T′],
  the Takaya–Nakamura flux and every other eddy quadratic: the quadratic of
  the ensemble-MEAN fields fades with lead time as the members' phases
  decorrelate, which reads as waves dying when they are merely uncertain.
  The mean of the members' quadratics does not fade.
- **Nyquist halving.** Summing a covariance from the cross-spectrum, each
  wavenumber contributes 2 Re(A_k B_k*)/n², except the Nyquist term of an
  even grid, which has no conjugate partner and contributes once. Doubling
  it is invisible at 0.25° and wrong by exactly the Nyquist variance when
  [v′T′] is summed to k = 72 on a 2.5° grid. With the halving the
  wavenumber sum equals the direct covariance to rounding.
- **Gill by direct solve, heated by total SST.** The steady Gill problem is
  one sparse linear system; an explicitly time-stepped version was unstable
  (the Coriolis term grows with |y|) and needed spin-up anyway. Heating is
  switched on where the *total* SST (climatology + anomaly) supports deep
  convection, not the climatology — that is what lets a warm cold tongue
  convect, and it took the modelled El Niño response from ~50 % to ~70 % of
  the observed regression. The model is checked against the exact damped
  dispersion relation (Kelvin decay = ε; n = 1 Rossby decay without the
  long-wave approximation), and the docstring quantifies the periodic wrap
  of the Kelvin wave at small ε.
- **The zero-line dip rule.** A zero crossing only counts as a critical
  line if the easterlies beyond it reach 1 m/s; shallower dips are stepped
  over. Otherwise a −0.3 m/s wobble in a westerly-QBO autumn reads as a
  critical line and the index jumps 20° for a day.
- **Bartlett, not AR(1), for the noise of a mean.** An AR(1) model fills
  the variance-inflation sum from ρ₁ alone; for an oscillating daily index
  (ρ₁ = 0.59 with a ~6-day oscillation) that gives 3.9 where the Bartlett
  sum of the measured autocorrelations gives ~1.3 — a threefold overstatement
  of the noise of a 30-day mean.
- **DCT over FFT for regional spectra**: no periodicity assumption, so no
  detrending step and no leakage from the domain-scale gradient
  (the difference is not subtle — it's the difference between seeing a
  k⁻⁵/³ range and seeing an artifact).
- **Gauss–Legendre sampling before spherical differentiation**: the GLQ grid
  has no pole points, so divergence/vorticity never touch 1/cos φ = ∞.
- **Fold-to-positive-frequency masking** in space-time filtering keeps the
  spectral mask Hermitian, so the inverse FFT is *exactly* real — no
  imaginary residue to silently discard. A soft-landing pad at the end of
  the record keeps the taper off the most recent days.
- **Summed-area tables** turn a radius-r ring mean into four lookups,
  making closed-circulation tests O(1) per gridpoint.
- **A profile, never a pointwise, static stability** in the QG E–P heat-flux
  term: a local ∂θ/∂p crosses zero in the troposphere and blows the flux up
  by orders of magnitude. Since v0.3 the profile is the ZONAL mean
  [θ](φ, p): the global mean carries tropospheric stability above the polar
  tropopause, 3–4× too small. Take vertical derivatives in **ln p** on the
  actual levels, **mask below ground before differentiating**, and **floor
  cos φ before smoothing** — smoothing first lets the polar 1/cos φ blow-up
  bleed equatorward and bury the real signal.
- **Smooth PV before ∇PV**: the divergent wind from a spherical-harmonic
  inversion is smooth, the 0.25° PV gradient is not.
- **Integrate pressure-level columns 0 → p_s exactly**: layer edges midway,
  top at 0 hPa, bottom clipped to the surface pressure — stopping at the top
  level drops the mass above it, and extrapolated below-ground values must
  get zero weight.
- **Column total energy is ∫(c_pT + Lq + k) dp/g + Φ_s p_s/g, not
  ∫(c_pT + Φ + Lq) dp/g.** Internal plus potential energy is ∫(c_vT + Φ) dp/g;
  integrating p dΦ by parts turns it into ∫c_pT dp/g + Φ_s p_s/g. Adding Φ to
  the enthalpy counts the pressure work R T twice — +28 % of the total
  (quantmet ≤ 0.3.0 did exactly that; fixed in 0.3.1).
- **Recurvature needs a westward phase**: a storm that heads north or east
  from the start has no recurvature in the forecast; a westward wobble of a
  few tenths of a degree is not one either (thresholds −1 and +2 m/s).
- **Filter along named axes, never positional ones.** Longitude is the last
  axis everywhere; a band-pass along "axis 1" silently filters latitude (or
  members) the day a leading axis is added. The tests pin it.
- **The plain ensemble CRPS rewards size, not skill.** For members drawn from
  the true distribution its expectation is the true CRPS plus
  E|X − X′|/(2m): +50 % at m = 2, +10 % at m = 10, +2 % at m = 50. The fair
  CRPS (Ferro 2014) divides the spread term by m(m − 1) and is unbiased at
  every m (tested 2–50); for a weighted multi-model blend the correction is
  applied within each model only.
- **Diebold–Mariano needs a HAC variance — and prewhitening.** Daily loss
  differentials are persistent; on equal-skill AR(1) errors the naive test
  rejects 35 % of the time at a nominal 5 %, Bartlett/Newey–West alone
  8–10 %, AR(1)-prewhitened Newey–West 5–6 %.
- **Pair before you compare.** Two series that stop on different days make
  "the last 7 days" mean two different windows; the error always makes the
  newer, less-published quantity look low. Compare only on dates both legs
  have, and print the count ("6/7 d"). Missing is NaN, never 0.
- **Stull's wet bulb is a sea-level fit.** At 840 hPa, 40 °C and 15 % the
  psychrometric wet bulb is 19.2 °C; Stull (2011) says 20.8 °C — 1.1 °C of the
  difference is pressure alone. Even at sea level it runs up to ~1 °C warm in
  hot, moderately dry air.
- **Heat index: the simple formula first.** The NWS computes Steadman's simple
  formula, averages it with T, and uses the Rothfusz regression (and its two
  adjustments) only when that average reaches 80 °F; Rothfusz everywhere is
  wrong below 80 °F. Tested against the NWS chart.
- **Kink first, weight after.** Degree days are convex in temperature, so the
  degree day of a weighted-mean temperature is never larger than the weighted
  mean of the degree days (Jensen) — and a mild national mean reads zero HDD
  while half the population heats. The same goes for ensemble members.
- **Midpoint and integral degree days are different quantities.** A day
  swinging 58–78 °F around a 65 °F base has a midpoint HDD of 0 and an
  integral HDD of 1.83; they agree only when the whole day sits on one side
  of the base, so shoulder seasons are where they part. Expected degree days
  of N(μ, σ²) exceed the degree days of μ by up to 0.4 σ.
<!-- --8<-- [end:notes] -->

## Install

<!-- --8<-- [start:install] -->
```bash
pip install quantmet-tools                 # core: numpy, scipy
pip install "quantmet-tools[helmholtz]"    # + pyshtools, xarray (helmholtz; the waf streamfunction path)
pip install "quantmet-tools[tc]"           # + eccodes for ECMWF track BUFR, pyshtools/xarray for the outflow wind
```

The package imports as `quantmet` (`from quantmet.tem import tem_terms`). Releases before 0.4.4 are not on
PyPI and carry the old distribution name; install one from its GitHub tag:

```bash
pip install "quantmet @ git+https://github.com/scorvec/quantmet-tools@v0.4.2"
```

`quantmet.degdays.local_days` uses the standard-library `zoneinfo`, which
needs the IANA time-zone database (`pip install tzdata` where the system
has none).

The TC kernel compiles itself with `cc` on first use into a per-user cache
directory (`$QUANTMET_CACHE_DIR`, else `$XDG_CACHE_HOME/quantmet`, else
`~/.cache/quantmet`).
<!-- --8<-- [end:install] -->

## Development

<!-- --8<-- [start:development] -->
```bash
pip install -e ".[test]"   # editable install with pytest
pre-commit install         # ruff lint and format on every commit, as in CI
pytest
```

Documentation builds with `pip install -e ".[docs]"` and `mkdocs serve`.
<!-- --8<-- [end:development] -->

## Citing

If you use quantmet in published work, please cite it: [`CITATION.cff`](https://github.com/scorvec/quantmet-tools/blob/main/CITATION.cff)
holds the metadata, and GitHub's "Cite this repository" button formats it.
