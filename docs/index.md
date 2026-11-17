# quantmet

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
