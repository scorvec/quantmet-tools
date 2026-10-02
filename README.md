# quantmet

Quantitative meteorology tools, extracted from the real-time diagnostics
pipelines behind [scorvec.com](https://scorvec.com). Everything here runs
in production daily; each module is self-contained and documented with the
numerical subtleties that matter in practice, and each is covered by tests
against analytic cases.

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

## Quick starts

**Stratospheric momentum budget and the Brewer–Dobson circulation** (zonal
means and eddy covariances from (plev, lat, lon) snapshots, levels 0.1–200 hPa):

```python
from quantmet.tem import eddy_covariances, tem_terms, downward_control, fill_tropics, upwelling_w
r = eddy_covariances(u, v, t, omega)          # [.] and [a'b'], total and by k1 / k2 / k3+
tt = tem_terms(lat, plev_hpa, r)              # fhat, v*, omega*, EP flux, div F (m/s^2) by wave group
psi = downward_control(lat, plev_hpa, tt["fhat"], tt["epd"] + gwd)   # kg/s; parts add linearly
psi = fill_tropics(psi, lat, edge=15)         # fhat -> 0 near the equator
up = upwelling_w(psi, lat, plev_hpa, lat_s=-25, lat_n=25)            # tropical w* (m/s) by level
```

**Is the planetary wave going up?** (100 hPa eddy heat flux, poleward positive):

```python
from quantmet.tem import eddy_heat_flux
hf = eddy_heat_flux(v100, t100, lat, kmax=72, lat_band=(45, 75))
hf.total, hf.by_k[..., 0], hf.by_k[..., 1]    # all waves, wave-1, wave-2
```

**Stratospheric wave driving** (QG E–P flux, per-member ensemble path) and the
**Charney–Drazin ceiling**:

```python
from quantmet.epflux import ep_flux, ensemble_ep_flux, charney_drazin_excess
r = ep_flux(u, v, t, lat, plev_hpa, kmax=3)   # (plev, lat, lon) snapshots in
r.force                                        # ∇·F as m/s per day, (plev, lat)
ens = ensemble_ep_flux(((u_m, v_m, t_m) for u_m, v_m, t_m in members),
                       lat, plev_hpa)          # per-member fluxes averaged
lid = charney_drazin_excess(ubar, lat, plev_hpa, theta_profile, k=1)   # u - U_c; 0 < u < U_c propagates
```

**Rossby wave packets** (250 hPa wave-activity flux from an ensemble):

```python
from quantmet.waf import streamfunction_anomaly, ensemble_tn01_flux, lead_smooth
psi_a, lat, lon = streamfunction_anomaly(u250, v250, psi_clim)     # needs quantmet[waf]
w = ensemble_tn01_flux(member_psi_anomalies, U, V, lat, lon, p_hpa=250)
w.wx, w.wy, w.div, w.agree                     # mean of per-member fluxes; sign agreement 0.5-1
```

**Gill response to a heating pattern**:

```python
from quantmet.gill import gill_response, heating_from_ssta, heating_from_precip, to_dimensional
Q = heating_from_ssta(ssta, sst_clim, lat)                 # shape only; or, in physical units:
Q = heating_from_precip(precip_anom_mm_day)                # 1 mm/day = 28.9 W/m^2 of column heating
u, v, p = gill_response(Q, lat, lon, eps=0.1)              # low-level fields, nondimensional
u_ms, v_ms, phi = to_dimensional(u, v, p)
```

**QBO zero-wind line** (per day, level or member):

```python
from quantmet.qbo import zero_line
lat0, flag = zero_line(ubar_50hpa, lat, hemi="nh")   # flag 0 found, 1 undefined, 2 censored
```

**Significance**:

```python
from quantmet.stats import benjamini_hochberg, prais_winsten, acf_inflation, event_bootstrap
sig = benjamini_hochberg(pvals, alpha=0.10)          # field significance, NaN-aware
fit = prais_winsten(y, X, start=starts)              # trend with AR(1) errors, segment breaks
infl = acf_inflation(daily_residuals)                # variance inflation of a mean (Bartlett)
mean, se = event_bootstrap(composite_days, event_ids)
```

**Equatorial wave decomposition** (time × ... × longitude anomalies):

```python
from quantmet.wk_filter import wk_filter
kelvin = wk_filter(olr_anom, "Kelvin")                                  # same shape, real
kelvin = wk_filter(chi_anom, dict(k=(1, 14), p=(2.5, 20), h=(8, 90), n=None),
                   pad_end=60)                                          # custom band, live end
```

**Walker circulation strength** (velocity potential of the divergent wind):

```python
from quantmet.helmholtz import velocity_potential, irrotational_wind
chi, lats, lons = velocity_potential(u200, v200)   # xarray DataArrays in
uchi, vchi = irrotational_wind(chi, lats, lons)
```

**Model effective resolution** (used to compare HRRR vs RRFS):

```python
from quantmet.dct_spectra import ke_spectrum, spectral_slope, effective_resolution
wl, p = ke_spectrum(u10, v10, res_km=3.0)
print(spectral_slope(wl, p, 30, 300))       # ≈ -5/3 in the resolved mesoscale
print(effective_resolution(wl, p))          # where numerics start dissipating
```

**Dynamic tropopause** (θ on 2 PVU; one member — PV does not survive averaging):

```python
from quantmet.pv import ertel_pv, dynamic_tropopause, on_isentrope
pv, theta = ertel_pv(t, u, v, p_pa, lat, lon)           # (lev, lat, lon) in, PVU out
dt = dynamic_tropopause(pv, theta, u, v, p_pa)          # dt["theta"], dt["p"] (hPa), dt["u"], dt["v"]
pv330 = on_isentrope(pv, theta, u, v, 330.0)["pv"]
```

**Rossby wave packets** (a packet Hovmöller):

```python
from quantmet.packets import packet_envelope, band_mean
env = band_mean(packet_envelope(v250, kmin=4, kmax=15), lat, 35, 60)   # v250 (time, lat, lon) -> (time, lon)
```

**Tropical cyclones meeting the jet**:

```python
from quantmet.tc import decode_tracks, recurvature, pv_advection, outflow_index, phase_series, et_times
storms = decode_tracks("20261002000000-360h-enfo-tf.bufr")   # ECMWF open data, every member
ctl = storms[0]["tracks"][0]                                  # member 0 = control
recurvature(ctl)                                              # lead (h) of the westward-to-eastward turn
adv = pv_advection(u_chi, v_chi, pv_300_200, lat, lon)        # PVU/day
outflow_index(adv, lat, lon, clat, clon)                      # Archambault et al. interaction metric
hart = phase_series(z_steps, levs, lat, lon, ctl, steps)      # B, -VT_L, -VT_U along the track
et_times(hart)                                                # (onset, completion) leads
```

**Ensemble sensitivity with field significance**:

```python
from quantmet.ensemble import sensitivity
r = sensitivity(day3_storm_latitude, z500_day7)   # (members,) and (members, lat, lon)
r["slope"], r["sig"]                               # m per degree; Benjamini-Hochberg mask at FDR 10 %
```

**Do AI weather models conserve mass, water, energy and angular momentum?**

```python
from quantmet import conservation as C
C.dry_air_mass(ps, tcw, lat, lon)                 # kg, ~5.135e18
C.column_energy(t, phi, q, p_pa, ps, u, v)        # J/m2: cpT, Phi, Lq, KE, total
C.relative_aam(u, p_pa, ps, lat, lon, hemispheres=True)
C.mountain_torque(ps, orography_m, lat, lon)
```

**Ensemble TC tracking building block**:

```python
from quantmet.tc_detect import detect_candidates
clat, clon, cp = detect_candidates(mslp_hpa, lat, lon, ring_deg=2.5, depth_hpa=2.0)
```

## Case studies

- [**AIFS single vs AIFS-ENS control: spectral fidelity by lead time**](examples/aifs_spectral_fidelity/)
  — the deterministic AIFS blurs progressively (MSE objective) while the
  CRPS-trained ENS control holds a realistic spectrum ten days in; measured
  with `quantmet.dct_spectra` on open data.

## Numerical notes worth stealing

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
- **Recurvature needs a westward phase**: a storm that heads north or east
  from the start has no recurvature in the forecast; a westward wobble of a
  few tenths of a degree is not one either (thresholds −1 and +2 m/s).
- **Filter along named axes, never positional ones.** Longitude is the last
  axis everywhere; a band-pass along "axis 1" silently filters latitude (or
  members) the day a leading axis is added. The tests pin it.

## Install

```bash
pip install .                        # core: numpy, scipy
pip install ".[helmholtz]"           # + pyshtools, xarray (helmholtz; the waf streamfunction path)
pip install ".[tc]"                  # + eccodes for ECMWF track BUFR, pyshtools/xarray for the outflow wind
pip install ".[test]" && pytest      # the test suite
```

The TC kernel compiles itself with `cc` on first use into a per-user cache
directory (`$QUANTMET_CACHE_DIR`, else `$XDG_CACHE_HOME/quantmet`, else
`~/.cache/quantmet`); tested on macOS arm64 and Linux x86-64.

## Provenance

Extracted from the pipelines behind scorvec.com's real-time products: the
stratospheric momentum budget, Brewer–Dobson downward control, 100 hPa eddy
heat flux, E–P flux loop and QBO zero-wind-line tracker on
[stratosphere.html](https://scorvec.com/stratosphere.html); the Walker
circulation (and its Gill-model decomposition by ocean basin) and the
wave-activity flux on [circulation.html](https://scorvec.com/circulation.html);
the equatorial-wave Hovmöller and Kelvin-wave tracker on
[enso.html](https://scorvec.com/enso.html); the HRRR/RRFS spectra comparison;
the 101-member ECMWF ensemble TC tracker; the dynamic-tropopause loops and
the tropical-cyclone / jet-stream cards on circulation.html; and the
AI-model conservation study ([aifs-aam-budget.html](https://scorvec.com/aifs-aam-budget.html)).
The v0.3 functions were checked against that production code on a real
AIFS-ENS cycle: identical, or within float32 round-off. See also
[ufs-hafs-on-apple-silicon](https://github.com/scorvec/ufs-hafs-on-apple-silicon)
for the companion project compiling and running NOAA's hurricane model on a
laptop.
