# Quick starts

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
