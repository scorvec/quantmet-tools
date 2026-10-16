# Changelog

## Unreleased

Project infrastructure only; the package code is unchanged.

- **Documentation site** at [scorvec.com/quantmet-tools](https://scorvec.com/quantmet-tools/) (MkDocs, built strictly
  and deployed from `main` by `docs.yml`): the overview, quick starts, numerical notes and install sections of the
  README, and one API page per module with every docstring shown exactly as written. `pip install -e ".[docs]"`
  installs the tools.
- **PyPI**: `release.yml` builds the sdist and wheel on every change, checks them (`twine check --strict`, an import
  of the installed wheel in a clean environment) and publishes a GitHub release to PyPI by Trusted Publishing; the
  release tag must match `quantmet.__version__`.
- **Packaging metadata**: SPDX license expression (`license = "MIT"`, setuptools >= 77), supported-Python and audience
  classifiers, documentation and issue-tracker URLs; the sdist also carries the changelog and `CITATION.cff`.
- **`CITATION.cff`**, a security policy (`SECURITY.md`), a `pre-commit` configuration running the CI ruff checks, and
  Dependabot for the workflow actions and the Python dependencies.

## 0.4.2 — 2026-10-15

### Leap years

- **`quantmet.waf.lowpass_anomaly`** calls `clim` with the 1-based day on the 366-slot calendar
  (`quantmet.degdays.doy366` + 1: 29 Feb = 60, 1 March = 61 in every year). It used the plain day of the year, so in
  a common year every day from 1 March was read from the climatology slot of the day before. Climatologies indexed
  by the plain day of the year must be rebuilt on the 366-slot calendar.
- **`quantmet.degdays.doy366`** docstring: how to fill slot 59 (29 Feb) and the cyclic phase on 366 slots.
- **Tests:** 7 new (179 in all): `doy366` over 1896–2104 (one calendar day per slot, the century rule) and on
  sub-daily, pre-1970 and Python dates; local days through 29 Feb; `lowpass_anomaly` on a 366-slot climatology in a
  common and a leap year; calendar-day lags of `acf_inflation`, the window of `paired` and the same-valid-time pairs
  of `cycle_offset` across 29 Feb.

### Maintenance (no numerical change)

- **`quantmet.tc_detect`**: the compiled kernel is cached with `functools.cache` (tests reset it with
  `tc_detect._lib.cache_clear()`), and the shared library is named `.so` on every platform but Windows (`.dll`).
- **Rounding**: `int(np.rint(x))` where values may be NumPy scalars (`ensemble.cycle_offset`, `tc.phase_series`);
  `np.rint` rounds half to even exactly like `round`.
- **Code style**: ruff lint and format (configuration in `pyproject.toml`), checked in CI.
- **Docs**: README install notes without platform specifics; the full MIT license text.

## 0.4.1 — 2026-10-12

### `quantmet.waf`: one truncated-divergence method on every grid (#1)

- **The inconsistency.** With `div_lmax`, `tn01_flux` on a closed grid (the 360° column repeated, e.g. the extended
  Driscoll–Healy grids of `quantmet.helmholtz`) set the masked cells' divergence to zero before the spectral
  truncation; on an open grid it truncated the divergence as computed. Both zero-fill the masked flux, so the step at
  the mask edge is the same. Its centred difference lands half in the first masked cell and half in the first valid
  one, and only the closed path dropped the masked half. `wx`, `wy` and the mask were identical, but the truncated
  divergence was not: 8.5 % rms and 11–25 % of the maximum apart on real 250 hPa AIFS-ENS input (leads 0–10 d,
  20–80°), and 37 % rms on a synthetic field with large mask edges.
- **New `div_method` keyword**, one implementation for both layouts:
  - `"zero"` (default): the 0.4.0 closed-grid behaviour, reproduced bit for bit;
  - `"as_computed"`: the 0.4.0 open-grid behaviour, reproduced bit for bit;
  - `"taper"`: the flux is ramped to zero with a cos² weight over `taper_deg` (default 5°) of great-circle distance
    from the mask (`mask_taper`), its divergence is taken spectrally (`sh_divergence`), then masked.

  **Open-grid results change** unless `div_method="as_computed"` is passed. Without `div_lmax` nothing changes.
- **How the default was chosen.** `tests/test_waf_divergence.py` (run it as a script for the table). Setup: extended
  DH2 grid at 1.41°, T15, 20 paired realisations. Values are medians of the relative rms in the valid cells.

  | case | zero | as_computed | taper 5° |
  |---|---|---|---|
  | (a) known divergence, W = ∇χ + k×∇ψ (T12): error | **0.241** | 0.307 | 0.348 |
  | (a′) divergence concentrated inside the masked caps: error | **0.469** | 0.680 | 0.713 |
  | (b) divergence-free W = k×∇ψ: spurious divergence | **0.085** | 0.163 | 0.183 |
  | (c) real TN01 flux, westerlies falling to zero across a band, rim wind direction ±1.5 m/s: leakage | 0.055 | 0.039 | **0.012** |
  | (c′) the same around a weak-wind cap | 0.007 | 0.005 | **0.002** |

  - `"zero"` was best in every realisation of (a), (a′) and (b).
  - `"taper"` was worst there, and worse the wider the ramp (2° → 8°): it keeps the whole edge term W·∇w, now inside
    the valid ramp. It was best in (c), where all three stay at a few % of the signal.
  - With rougher (T30) fields, (a) ties between zero and as_computed (0.35 vs 0.34), while (b) still halves.
  - A realistic check used the live 250 hPa flux, evaluated everywhere, as the known field, scored on the real mask
    over 11 leads:
    - (a): 0.164 / 0.190 / 0.245;
    - (b), on its rotational part: 0.060 / 0.118 / 0.146;
    - zero was best at every lead;
    - (c) on the real basic state: 0.033 / 0.031 / 0.027.
- **New helpers.**
  - `sh_divergence`: divergence on a Driscoll–Healy grid from the weak form D_lm = −(1/4π)∮ W·∇Y_lm dΩ, so the flux
    is not differenced; exact to round-off for band-limited fields; leading dimensions allowed.
  - `mask_taper`: cos² edge weight from the great-circle distance to the mask (scipy KD-tree; the last mask is
    cached).
  - `masked_divergence`: the divergence stage for any masked flux.
  - `DIV_METHODS` and `TAPER_DEG`.
- **`sh_truncate`** accepts leading dimensions, and an extended grid with only the −90° row or only the 360° column
  (0.4.0 misread both).
- **Docs.** The horizontal TN01 flux is homogeneous of degree zero in (U, V): its magnitude does not grow as |U| → 0,
  but its direction becomes ill-conditioned (a ±1.5 m/s wind error turns it by ~27° at 3 m/s).
- **Tests:** 9 new (172 in all).

## 0.4.0 — 2026-10-12

### New modules
- `quantmet.verify` — forecast verification: `crps_ensemble` (weighted empirical CRPS, O(m log m)), `crps_fair`
  (Ferro 2014; unbiased at any ensemble size; member `weights` and `groups` for a weighted multi-model mixture, with
  the within-model correction W_k²/(W_k² − Σw²) and one-member groups as point masses), `crps_gaussian` (closed form,
  EMOS), `ranks` / `rank_histogram` / `pit` (ties spread uniformly, seedable), `brier_score`, `brier_decomposition`
  (reliability, resolution, uncertainty plus the Stephenson et al. 2008 within-bin terms, so the identity is exact),
  `reliability_curve`, `diebold_mariano` (Newey–West Bartlett HAC after AR(1) prewhitening, Andrews 1991 bandwidth,
  ≥ h − 1 lags, Harvey–Leybourne–Newbold correction with t(n − 1) p-values; size 5–6 % at a nominal 5 % on
  persistent equal-skill losses, where the naive test rejects 35 %), and `paired`, which aligns two dated series on
  the dates both legs have and returns the day count with the window it was meant to cover.
- `quantmet.thermo` — moist thermodynamics on SI inputs (K, Pa, kg/kg, RH in %): saturation vapour pressure over
  water and ice (Magnus with Alduchov–Eskridge 1996 coefficients, Bolton 1980, Murphy–Koop 2005), dewpoint / frost
  point, the full set of pressure-aware RH ↔ dewpoint ↔ specific humidity ↔ mixing ratio conversions, saturation
  mixing ratio and specific humidity, dewpoint depression, vapour pressure deficit, Kirchhoff latent heats;
  potential temperature (dry, or the exact moist Poisson exponent), Exner, virtual and virtual potential temperature,
  density temperature with condensate, air density, equivalent temperature, θ_e (Bolton eq. 39 pseudo-adiabatic and
  Emanuel's reversible form), saturation θ_e, θ_l (Betts 1973), θ_il (Tripoli & Cotton 1981), θ_w (Davies-Jones 2008);
  Stull (2011) wet bulb with its validity mask and the psychrometric (isobaric) wet bulb at any pressure; the exact
  LCL (Romps 2017, reproducing his published value) and Bolton's; dry and moist static energy, specific enthalpy with
  a stated reference state, the pseudo-adiabatic lapse rate and an RK4 pseudo-adiabat; column integrals 0 → p_s on
  pressure or model levels (precipitable water, column vapour / liquid / ice, column MSE, column enthalpy) with
  below-ground levels ignored; ln p-mean layer temperature and hypsometric thickness; the NWS heat index (simple
  formula first, Rothfusz with both adjustments only above 80 °F, tested against the NWS chart) and the NWS 2001
  wind chill (tested against the NWS chart); °C/°F/K and m/s–kt–mph helpers.
- `quantmet.degdays` — `midpoint_dd`, `integral_dd` (sample mean, or the exact integral of the piecewise-linear
  curve), `degree_hours` by time-of-day block, `dd_from_extremes` (daily extremes plus a normalised diurnal shape),
  `expected_dd` (Gaussian temperature, Thom 1954), `weighted_dd` (kink per location, then weight; strict about
  missing locations), `local_days` / `day_length_hours` (DST-safe local calendar days from UTC data via zoneinfo,
  incomplete days dropped), `doy366` (29 Feb in its own slot) and `fit_balance_points` (two-hinge change-point fit
  of load on temperature with optional extra regressors, batched over the hinge grid).
- `quantmet.constants` — one SI constant set (R_d 287.04, R_v 461.5, c_pd 1004.7, c_pv 1846, c_l 4218, c_i 2106,
  L_v0 2.501e6, L_f0 3.337e5, g 9.80665, ε = R_d/R_v), used by `thermo`; older modules keep their own constants.

### Packaging
- Version 0.4.0; description and keywords cover the new modules. No new dependencies (`zoneinfo` is standard
  library; Windows needs the `tzdata` package for the time-zone database).

## 0.3.1 — 2026-10-12

Bug fixes (all four reproduced, each now pinned by a test that fails on 0.3.0):

- `quantmet.conservation.column_energy` — the `total` was ∫(c_pT + Φ + Lq) dp/g, which counts the pressure work
  twice (∫Φ dp/g = Φ_s p_s/g + ∫R T dp/g): a hydrostatic isothermal column (250 K, p_s = 1000 hPa) gave
  3.286e9 J/m² against the true 2.561e9 (+28 %). `total` is now the total energy I + P + LH + K: with the new `phis`
  argument (surface geopotential) ∫(c_pT + Lq + k) dp/g + Φ_s p_s/g (Trenberth 1997; exact for that column), without
  it ∫(c_vT + Φ + Lq + k) dp/g (Peixoto & Oort 1992; the same for a hydrostatic column, −0.26 % there from the
  top layer's log-singular Φ). New components `cvT` and `PhiS`; `cpT`, `Phi`, `Lq`, `KE` unchanged. New constants
  `RD`, `CV`.
- `quantmet.dct_spectra.ke_spectrum` — wavelengths were labelled one wavenumber too long and the longest bin was
  dropped: exact DCT modes of 192, 96, 48 and 24 km came out at 256, 109.7, 51.2 and 24.8 km. Now
  `wavelengths = res_km * min(shape) / k` for k = 1 … n and every bin is returned (one more than before), so
  slopes, `effective_resolution` and `fidelity_resolution` change. The docstring now says the method assumes one
  grid spacing in both directions.
- `quantmet.helmholtz` — interpolation to the Gauss–Legendre grid had no longitude wrap, so on 1.5° and 2.5° grids at
  lmax = 120 (and `quantmet.tc.irrotational_on_grid` at 2.5°, lmax = 106) every GLQ column past the last input
  longitude was NaN and χ / ψ came out 100 % NaN. The input is now padded with a column at lon ± 360 and, if the grid
  stops short of the poles, with pole rows (zonal wavenumber 1 of the nearest row, the only part of a wind component
  that survives at a pole); repeated 360° columns are dropped. Longitude derivatives (divergence, vorticity,
  `irrotational_wind`, `nondivergent_wind`) are periodic centred differences on global grids instead of
  np.gradient's one-sided ones at the seam (seam-column error in u_χ 0.012 → 0.002 m/s on the test field).
- `quantmet.tc_detect` (C kernel) — the ring test was centred one column WEST of the candidate (the summed-area table
  is built on a padded field whose column j + ring_px + 1 holds source column j, but the boxes started at j): on a
  background gradient the depth was off by one column's worth of it. Boxes now start at j + 1. The kernel source
  hash changes, so the cached library is rebuilt on first use. Latitude and longitude spacings may now differ (ring,
  local-minimum and wind windows are converted with dlat for rows and dlon for columns; the kernel takes separate
  row/column half-widths).

Other changes:

- `quantmet.epflux.ep_flux` accepts leading axes, (..., plev, lat, lon) → (..., plev, lat), one flux per member (0.3.0
  raised); `below_ground_share` takes (..., lat, lon). The module docstring now says the static stability is the
  zonal-mean profile by default (it still said global-mean).
- `quantmet.tc_detect` docstrings: candidates are local minima with no LOWER neighbour (ties count), not strict
  minima; the annulus is square in degrees.
- `__version__` is the single source of the version (pyproject reads it); 0.3.0 shipped with `__version__ = "0.2.0"`.
- `examples/aifs_spectral_fidelity` — rerun with the fixed spectra (table, figure and npz regenerated; the 0.3.0
  numbers are listed there), a caveat that one res_km (21.3 km, dx at 40°N) was used for a 0.25° box whose dy is
  27.8 km, and fetching from ECMWF's Google Cloud mirror (`source="google"`, one parameter per request).
- README: the "runs in production daily" and "every module tested" claims made accurate (tests for `dct_spectra`
  added); column-energy numerical note.

## 0.3.0 — 2026-10-02

### New modules
- `quantmet.pv` — Ertel PV on pressure levels (`ertel_pv`), the dynamic tropopause (`dynamic_tropopause`: upward
  search, interpolation in PV, top-level fill where the column never reaches the threshold) and isentropic PV
  (`on_isentrope`).
- `quantmet.packets` — Rossby wave-packet envelope (`packet_envelope`, Zimin et al. 2003) and `band_mean`.
- `quantmet.tc` — ECMWF tropical-cyclone track BUFR decoding for every ensemble member (`decode_tracks`; control =
  member 0, deterministic subset excluded), `recurvature`, the outflow-jet metric (`irrotational_on_grid`,
  `pv_advection`, `outflow_index`; Archambault et al. 2013), Hart (2003) phase space (`hart_parameters`,
  `phase_series`) and Evans & Hart (2003) ET timing (`et_times`); helpers `at`, `within`, `great_circle_km`.
- `quantmet.ensemble` — ensemble `sensitivity` (Torn & Hakim) and `group_composite` with FDR masking (via
  `quantmet.stats`), and `cycle_offset`, a no-lookahead estimator of systematic 00Z/12Z cycle offsets.
- `quantmet.conservation` — dry-air mass, implied evaporation, column total energy, relative and mass AAM, mountain
  and friction torques, with exact 0 → p_s pressure-level integration and hemispheric splits.

### Changed
- `quantmet.epflux` — static stability from the ZONAL-MEAN theta by default (`theta_p="global"` restores the
  earlier global profile); vertical derivatives in ln p on the actual levels; below-ground masking (`psfc`,
  `bg_frac`, `below_ground_share`; members may pass `(u, v, t, psfc)`); NaN-aware latitude smoothing
  (`smooth_deg`). Existing calls keep working; their numbers change through the new default stability and the
  ln-p derivative.

### Packaging
- `tc` and `all` extras; version 0.3.0.

## 0.2.0 — 2026-09-28

### New modules
- `quantmet.tem` — transformed-Eulerian-mean diagnostics on pressure levels:
  `cospectrum` / `eddy_covariances` (by zonal wavenumber, Nyquist term not
  doubled), `tem_terms` (fhat, v\*, ω\*, full primitive-equation E–P flux and
  its divergence, total and by wave group), `residual_streamfunction`,
  `residual_velocities`, `downward_control`, `fill_tropics`, `upwelling_w`,
  `eddy_heat_flux` (by wavenumber, band mean, poleward-positive).
- `quantmet.waf` — Takaya–Nakamura (2001) wave-activity flux on the sphere
  (`tn01_flux`) and on a Cartesian grid (`tn01_flux_cartesian`), the
  per-member ensemble mean with sign agreement (`ensemble_tn01_flux`),
  `lead_smooth`, `lowpass_anomaly` (low-passed basic state),
  `streamfunction_anomaly` (via `quantmet.helmholtz`) and `sh_truncate`.
  Periodic longitude differences; the cross term uses the plain mixed angle
  derivative of TN01 eq. 38 (no metric term). Masking thresholds are
  arguments.
- `quantmet.gill` — Gill (1980) model by sparse direct solve with c, β, ε,
  sponge and smoothing exposed; `heating_from_ssta` (total-SST threshold),
  `column_heating` and `heating_from_precip` (latent heat, 28.9 W/m² per
  mm/day, documented first-baroclinic projection), `rossby_decay_rate`
  (exact damped n = 1 root), `to_dimensional`, `band_mean`.
- `quantmet.qbo` — `zero_line`, the QBO zero-wind-line index with the dip
  rule, censoring and hemispheric mirroring.
- `quantmet.stats` — `fdr_adjust` / `benjamini_hochberg` (NaN-aware),
  `prais_winsten` (AR(1) GLS with segment breaks) and `segment_starts`,
  `acf_inflation` (Bartlett window, optional daily grid), `label_events` and
  `event_bootstrap`, `partial_corr_test` (Freedman–Lane permutation,
  bootstrap CI, F test), `compare_r`, `auc`.

### Changed
- `quantmet.epflux`: new `charney_drazin_uc` and `charney_drazin_excess`
  (latitude-only smoothing, leading member/lead axes carried through).
  Provenance link updated to stratosphere.html.
- `quantmet.wk_filter`: `(time, ..., lon)` input with any number of middle
  dimensions; custom bands as dicts (ER bounds solved for any wavenumber
  range); `pad_end` / `ramp` soft landing at the end of the record; `dt` for
  sub-daily sampling; new `wave_mask`. Two-dimensional results are
  bit-identical to 0.1.0. The time-Nyquist row is never kept.
- `quantmet.tc_detect`: the 0.1.0 wrapper called a `detect` symbol the C
  kernel does not define (the kernel exports `detect_step`), so
  `detect_candidates` failed on first use; it now binds `detect_step` with
  explicit argtypes and exposes its search band, pressure ceiling, local-min
  window and optional wind maximum. The library is built into a per-user
  cache directory with the platform's shared-library suffix, named by a
  hash of the C source, and renamed into place atomically.
- `quantmet.helmholtz`: docstring documents the accuracy (∝ lmax^-1.7) and
  the extended output grid.
- `import quantmet` exposes `__version__` and imports nothing else.

### Packaging
- `[build-system]`, version 0.2.0, license/authors/URLs, package data for
  `tc_detect.c`, extras `helmholtz` and `waf` (pyshtools + xarray) and `test`.
- pytest suite in `tests/` (analytic cases throughout) and a GitHub Actions
  workflow: Python 3.10–3.12 with core dependencies only, plus a job with
  the extras.
- Committed `__pycache__` removed; `.gitignore` added.

## 0.1.0

- `dct_spectra`, `helmholtz`, `wk_filter`, `tc_detect`; `epflux` added later.
