# Changelog

## 0.2.0 (2026-10-02)

New modules
- `quantmet.pv`: Ertel PV on pressure levels, dynamic tropopause, isentropic PV.
- `quantmet.waf`: Takaya-Nakamura (2001) wave-activity flux (eq. 38 cross term), divergence, ensemble flux.
- `quantmet.packets`: Rossby wave-packet envelope (Zimin et al. 2003) and band means.
- `quantmet.tc`: ECMWF tf BUFR decoding for every ensemble member, recurvature, outflow-jet metric
  (Archambault et al. 2013), Hart (2003) phase space and Evans & Hart (2003) ET timing.
- `quantmet.ensemble`: Benjamini-Hochberg FDR, ensemble sensitivity, group composites, cycle-offset estimator.
- `quantmet.conservation`: dry-air mass, water budget, column energy, AAM and mountain/friction torques.

Changed
- `quantmet.epflux`: static stability now from the ZONAL-MEAN theta by default (`theta_p="global"` restores the
  v0.1 global profile); vertical derivatives in ln p on the actual levels; new `psfc`/`bg_frac` below-ground masking,
  `smooth_deg` NaN-aware latitude smoothing, `below_ground_share`. Existing calls keep working; their numbers change
  through the new default stability and the ln-p derivative.

Infrastructure
- pytest suite with synthetic-field checks for every module; GitHub Actions CI (Python 3.10 and 3.12).
- `pyproject.toml`: build system, `tc`/`test`/`all` extras, package data for the C kernel.
- `__pycache__` untracked; `.gitignore` added.

## 0.1.0
- DCT spectra, Helmholtz decomposition, Wheeler-Kiladis filtering, TC detection; E-P flux.
