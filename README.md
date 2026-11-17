# quantmet

[![tests](https://github.com/scorvec/quantmet-tools/actions/workflows/tests.yml/badge.svg)](https://github.com/scorvec/quantmet-tools/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/quantmet-tools)](https://pypi.org/project/quantmet-tools/)
[![docs](https://github.com/scorvec/quantmet-tools/actions/workflows/docs.yml/badge.svg)](https://scorvec.com/quantmet-tools/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/scorvec/quantmet-tools/blob/main/LICENSE)

Atmospheric diagnostics for Python: plain functions on numpy arrays, each module tested against analytic
cases where they exist. The daily products of [scorvec.com](https://scorvec.com) import it.

Documentation, with a worked example per module and the API reference:
[scorvec.com/quantmet-tools](https://scorvec.com/quantmet-tools/).

## Install

<!-- --8<-- [start:install] -->
```bash
pip install quantmet-tools                 # core: numpy, scipy
pip install "quantmet-tools[helmholtz]"    # + pyshtools, xarray (helmholtz; the waf streamfunction path)
pip install "quantmet-tools[tc]"           # + eccodes for ECMWF track BUFR, pyshtools/xarray for the outflow wind
```

The package imports as `quantmet`. Python 3.10 to 3.12. Releases before 0.4.4 are not on PyPI; install one
from its tag: `pip install "quantmet @ git+https://github.com/scorvec/quantmet-tools@v0.4.2"`.

`quantmet.degdays.local_days` needs the IANA time-zone database (`pip install tzdata` where the system has
none). `quantmet.tc_detect` compiles its C kernel with `cc` on first use into `$QUANTMET_CACHE_DIR`, else
`$XDG_CACHE_HOME/quantmet`, else `~/.cache/quantmet`.
<!-- --8<-- [end:install] -->

## Use

Import the module you need. Fields are arrays with longitude on the last axis; each docstring states its units:

```python
from quantmet.tem import eddy_heat_flux
from quantmet.wk_filter import wk_filter
from quantmet.stats import benjamini_hochberg
from quantmet import verify, thermo

hf = eddy_heat_flux(v100, t100, lat, kmax=72, lat_band=(45, 75))  # 100 hPa [v'T'], total and by wavenumber
kelvin = wk_filter(olr_anom, "Kelvin")                            # (time, ..., lon) in, same shape out
sig = benjamini_hochberg(pvals, alpha=0.10)                       # field significance mask
crps = verify.crps_fair(ens, obs)                                 # members on the last axis
tw = thermo.wet_bulb_temperature(t, p, rh=rh)                     # K, Pa, %
```

More examples: [quick starts](https://scorvec.com/quantmet-tools/quickstarts/). Pitfalls each module avoids:
[numerical notes](https://scorvec.com/quantmet-tools/notes/).

## Modules

| Module | What it computes |
|---|---|
| `tem` | transformed-Eulerian-mean diagnostics: residual circulation, E–P flux and divergence by wavenumber, downward control, eddy heat flux |
| `epflux` | quasi-geostrophic E–P flux with a per-member ensemble path; Charney–Drazin propagation ceiling |
| `waf` | Takaya–Nakamura wave-activity flux per ensemble member |
| `pv` | Ertel PV on pressure levels, the dynamic tropopause, PV on isentropes |
| `packets` | Rossby wave-packet envelope and latitude-band means |
| `helmholtz` | velocity potential, streamfunction, divergent and rotational wind by spherical harmonics |
| `gill` | steady Gill response to tropical heating from SST or rainfall |
| `qbo` | the QBO zero-wind line |
| `wk_filter` | Wheeler–Kiladis space-time filtering of equatorial waves |
| `tc` | ECMWF cyclone-track BUFR decoding, recurvature, outflow–jet interaction, Hart phase space |
| `tc_detect` | closed-circulation cyclone candidates from sea-level pressure |
| `ensemble` | ensemble sensitivity and member-group composites with FDR masking |
| `conservation` | global dry-air mass, column water and energy, angular momentum and its torques |
| `dct_spectra` | kinetic-energy spectra on limited-area grids, spectral slope, effective resolution |
| `stats` | Benjamini–Hochberg FDR, Prais–Winsten regression, variance inflation, block bootstrap, partial-correlation tests |
| `verify` | CRPS (ensemble, fair, Gaussian), rank histograms, Brier decomposition, Diebold–Mariano test, strict pairing |
| `thermo` | moist thermodynamics: vapour pressure, humidity conversions, θ and θ<sub>e</sub> variants, wet bulb, LCL, column integrals, heat index, wind chill |
| `degdays` | degree days: midpoint and integral conventions, weighted aggregation, local days, balance-point fit |
| `constants` | the SI constants `thermo` uses |

## Develop

<!-- --8<-- [start:development] -->
```bash
pip install -e ".[test]"   # editable install with pytest
pre-commit install         # ruff lint and format on every commit, as in CI
pytest
```

The documentation builds with `pip install -e ".[docs]"` and `mkdocs serve`.
<!-- --8<-- [end:development] -->

## Cite

[`CITATION.cff`](https://github.com/scorvec/quantmet-tools/blob/main/CITATION.cff) holds the metadata; GitHub's
"Cite this repository" button formats it.
