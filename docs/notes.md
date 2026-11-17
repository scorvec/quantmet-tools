# Numerical notes

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
