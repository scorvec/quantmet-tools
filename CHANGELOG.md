# Changelog

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
