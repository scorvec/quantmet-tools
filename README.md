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
| `quantmet.epflux` | Quasi-geostrophic Eliassen–Palm flux + divergence as a zonal force (the wave driving of the mean flow), with a per-member-flux ensemble path; zonal-mean static stability, ln-p derivatives, below-ground masking | Eliassen & Palm (1961); Edmon, Hoskins & McIntyre (1980, JAS) |
| `quantmet.pv` | Ertel PV on pressure levels, the dynamic tropopause (θ, p, wind on 2 PVU by an upward search interpolated in PV), PV on isentropes | Hoskins, McIntyre & Robertson (1985, QJRMS); Morgan & Nielsen-Gammon (1998, MWR) |
| `quantmet.waf` | Takaya–Nakamura wave-activity flux for stationary Rossby waves (eq. 38 exactly, periodic in λ), its divergence, and a per-member ensemble flux | Takaya & Nakamura (2001, JAS) |
| `quantmet.packets` | Rossby wave-packet envelope (Hilbert transform along latitude circles) and latitude-band means for packet Hovmöllers | Zimin et al. (2003, MWR) |
| `quantmet.tc` | ECMWF tropical-cyclone track BUFR decoding (every ensemble member), recurvature, the outflow–jet interaction metric (−v_χ·∇PV), Hart cyclone phase space and extratropical-transition timing | Archambault et al. (2013, 2015, MWR); Hart (2003, MWR); Evans & Hart (2003, MWR) |
| `quantmet.ensemble` | Benjamini–Hochberg FDR masking, ensemble sensitivity (regression across members), member-group composites, and a no-lookahead estimator of systematic 00Z/12Z cycle offsets | Benjamini & Hochberg (1995); Ancell & Hakim (2007); Torn & Hakim (2008, MWR) |
| `quantmet.conservation` | Global dry-air mass, the column water budget (implied evaporation), column total energy on pressure levels, atmospheric angular momentum and its mountain and friction torques | Trenberth & Smith (2005, J. Climate); Peixoto & Oort (1992); Egger et al. (2007, Rev. Geophys.) |

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

**Stratospheric wave driving** (the SSW forcing diagnostic, live on the
Atmospheric-Response page):

```python
from quantmet.epflux import ep_flux, ensemble_ep_flux
r = ep_flux(u, v, t, lat, plev_hpa, kmax=3)   # (plev, lat, lon) snapshots in
r.force                                        # ∇·F as m/s per day, (plev, lat)
ens = ensemble_ep_flux(((u_m, v_m, t_m) for u_m, v_m, t_m in members),
                       lat, plev_hpa)          # per-member fluxes averaged
```

**Dynamic tropopause** (θ on 2 PVU; one member — PV does not survive averaging):

```python
from quantmet.pv import ertel_pv, dynamic_tropopause, on_isentrope
pv, theta = ertel_pv(t, u, v, p_pa, lat, lon)           # (lev, lat, lon) in, PVU out
dt = dynamic_tropopause(pv, theta, u, v, p_pa)          # dt["theta"], dt["p"] (hPa), dt["u"], dt["v"]
pv330 = on_isentrope(pv, theta, u, v, 330.0)["pv"]
```

**Rossby wave packets and stationary-wave activity**:

```python
from quantmet.packets import packet_envelope, band_mean
env = band_mean(packet_envelope(v250, kmin=4, kmax=15), lat, 35, 60)   # (time, lon) Hovmöller
from quantmet.waf import tn01_flux, flux_divergence
wx, wy = tn01_flux(psi_anom, U_basic, V_basic, lat, lon, p_hpa=250)
```

**Tropical cyclones meeting the jet**:

```python
from quantmet.tc import decode_tracks, recurvature, pv_advection, outflow_index, phase_series, et_times
storms = decode_tracks("20261002000000-360h-enfo-tf.bufr")   # ECMWF open data, every member
ctl = storms[0]["tracks"][0]                                  # member 0 = control
recurvature(ctl)                                              # lead (h) of the westward-to-eastward turn
adv = pv_advection(u_chi, v_chi, pv_300_200, lat, lon)        # PVU/day
outflow_index(adv, lat, lon, *ctl_position)                   # Archambault et al. interaction metric
hart = phase_series(z_steps, levs, lat, lon, ctl, steps)      # B, -VT_L, -VT_U along the track
```

**Ensemble sensitivity with field significance**:

```python
from quantmet.ensemble import sensitivity
r = sensitivity(day3_storm_latitude, z500_day7)   # (members,) and (members, lat, lon)
r["slope"], r["sig"]                               # m per degree, Benjamini-Hochberg mask at FDR 10 %
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
- **Global-mean static stability** in the E–P heat-flux term: a local
  ∂θ/∂p crosses zero in the troposphere and blows the flux up by orders of
  magnitude; and **floor cos φ before smoothing** — smoothing first lets the
  polar 1/cos φ blow-up bleed equatorward and bury the real signal.
- **Ensemble quadratics**: E–P flux and wave-activity flux are quadratic in
  the eddies, so average per-member fluxes; the flux of the ensemble-mean
  fields fades with lead time as averaging damps the waves. PV likewise:
  compute it per member.
- **Zonal-mean, not global-mean, static stability above the polar
  tropopause** (E–P flux): the global profile carries tropospheric stability
  there, 3–4× too small. Take vertical derivatives in **ln p** on the actual
  levels, and **mask below ground before differentiating**.
- **TN01's cross term is the plain ψ_λφ**: writing it as ∂/∂y of
  ψ_λ/(a cos φ) adds ψ_λ tan φ/a² and breaks the flux's phase independence.
- **Smooth PV before ∇PV**: the divergent wind from a spherical-harmonic
  inversion is smooth, the 0.25° PV gradient is not.
- **Integrate pressure-level columns 0 → p_s exactly**: layer edges midway,
  top at 0 hPa, bottom clipped to the surface pressure — stopping at the top
  level drops the mass above it, and extrapolated below-ground values must
  get zero weight.
- **Field significance**: thousands of grid points tested at 5 % give
  hundreds of false positives — control the false-discovery rate.

## Install

```bash
pip install -e .                   # core (numpy, scipy)
pip install -e .[helmholtz]        # + pyshtools, xarray
pip install -e .[tc]               # + eccodes for track BUFR decoding
pip install -e .[test] && pytest   # synthetic-field tests for every module
```

The TC kernel compiles itself with `cc` on first import (tested on macOS
arm64 and Linux x86-64).

## Provenance

Extracted from the pipelines behind scorvec.com's real-time products:
HRRR/RRFS spectra comparison, Walker-circulation monitor, equatorial-wave
Hovmöller, the 101-member ECMWF ensemble TC tracker, the E–P flux &
wave-driving forecast loop, the dynamic-tropopause and wave-activity-flux
loops, the tropical-cyclone / jet-stream cards, and the AI-model
conservation study. Each ported function was checked against the
production code on a real AIFS-ENS cycle (identical, or within float32
round-off). See also
[ufs-hafs-on-apple-silicon](https://github.com/scorvec/ufs-hafs-on-apple-silicon)
for the companion project compiling and running NOAA's hurricane model on a
laptop.
