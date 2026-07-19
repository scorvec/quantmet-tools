# quantmet

Quantitative meteorology tools, extracted from the real-time diagnostics
pipelines behind [scorvec.com](https://scorvec.com). Everything here runs
in production daily; each module is self-contained and documented with the
numerical subtleties that matter in practice.

| Module | What it does | Key references |
|---|---|---|
| `quantmet.dct_spectra` | Kinetic-energy power spectra on limited-area grids via 2-D DCT + elliptic-ring binning; spectral slopes and effective resolution | Denis, Côté & Laprise (2002, MWR); Skamarock (2004) |
| `quantmet.helmholtz` | Global Helmholtz decomposition (velocity potential / streamfunction, divergent / rotational wind) by spherical-harmonic Poisson inversion on a pole-free Gauss–Legendre grid | — |
| `quantmet.wk_filter` | Wheeler–Kiladis space-time filtering of equatorial waves (Kelvin / ER / MJO) with per-wavenumber dispersion bounds from the full cubic; Hermitian-safe FFT masking; Lanczos low-pass | Wheeler & Kiladis (1999, JAS) |
| `quantmet.tc_detect` | Closed-circulation tropical-cyclone candidate detection with an O(1)-per-gridpoint summed-area-table ring test (C kernel, auto-compiled) | — |

## Quick starts

**Model effective resolution** (used to compare HRRR vs RRFS):

```python
from quantmet.dct_spectra import ke_spectrum, spectral_slope, effective_resolution
wl, p = ke_spectrum(u10, v10, res_km=3.0)
print(spectral_slope(wl, p, 30, 300))       # ≈ -5/3 in the resolved mesoscale
print(effective_resolution(wl, p))          # where numerics start dissipating
```

**Walker circulation strength** (velocity potential of the divergent wind):

```python
from quantmet.helmholtz import velocity_potential, irrotational_wind
chi, lats, lons = velocity_potential(u200, v200)   # xarray DataArrays in
uchi, vchi = irrotational_wind(chi, lats, lons)
```

**Equatorial wave decomposition** (time × longitude OLR anomalies):

```python
from quantmet.wk_filter import wk_filter
kelvin = wk_filter(olr_anom, "Kelvin")     # same shape, band-filtered, real
```

**Ensemble TC tracking building block**:

```python
from quantmet.tc_detect import detect_candidates
clat, clon, cp = detect_candidates(mslp_hpa, lat, lon, ring_deg=2.5, depth_hpa=2.0)
```

## Numerical notes worth stealing

- **DCT over FFT for regional spectra**: no periodicity assumption, so no
  detrending step and no leakage from the domain-scale gradient
  (the difference is not subtle — it's the difference between seeing a
  k⁻⁵/³ range and seeing an artifact).
- **Gauss–Legendre sampling before spherical differentiation**: the GLQ grid
  has no pole points, so divergence/vorticity never touch 1/cos φ = ∞.
- **Fold-to-positive-frequency masking** in space-time filtering keeps the
  spectral mask Hermitian, so the inverse FFT is *exactly* real — no
  imaginary residue to silently discard.
- **Summed-area tables** turn a radius-r ring mean into four lookups,
  making closed-circulation tests O(1) per gridpoint.

## Install

```bash
pip install numpy scipy            # core
pip install pyshtools xarray       # for helmholtz
```

The TC kernel compiles itself with `cc` on first import (tested on macOS
arm64 and Linux x86-64).

## Provenance

Extracted from the pipelines behind scorvec.com's real-time products:
HRRR/RRFS spectra comparison, Walker-circulation monitor, equatorial-wave
Hovmöller, and the 101-member ECMWF ensemble TC tracker. See also
[ufs-hafs-on-apple-silicon](https://github.com/scorvec/ufs-hafs-on-apple-silicon)
for the companion project compiling and running NOAA's hurricane model on a
laptop.
