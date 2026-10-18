"""Moist thermodynamics, column integrals and apparent temperature, numpy-first.

UNITS. Thermodynamic functions take temperature in K, pressure in Pa, specific humidity q and the condensate mass
fractions ql, qi in kg/kg of moist air, mixing ratios r in kg/kg of dry air, and relative humidity in PERCENT (0-100),
and return SI. The two apparent-temperature indices keep the units their agencies define them in (heat index:
degF and %; wind chill: degF and mph); the unit helpers at the end convert. Every function broadcasts.

CONSTANTS come from ``quantmet.constants``: RD 287.04, RV 461.5, EPSILON = RD/RV, CPD 1004.7, CPV 1846, CL 4218,
CI 2106, LV0 2.501e6, LF0 3.337e5, G 9.80665, T0 273.15, P0 1e5. Coefficients that belong to a published fit
(Magnus, Bolton's 0.2854, Romps's constant set) stay with their formula so that it reproduces the paper.

Numerical notes that matter in practice:
- MAGNUS IS A FIT, AND WHICH FIT MATTERS. Alduchov & Eskridge (1996) coefficients (the default) stay within 0.4 % of
  the reference vapour pressure over -40..50 degC; Bolton's (1980) are within 0.3 % over -30..35 degC and are the ones
  his theta_e formula was fitted with, so ``equivalent_potential_temperature`` uses them internally. Over ice the
  Magnus-type fit uses its own coefficients; "auto" switches at 0 degC. Murphy & Koop (2005) is the reference-quality
  option (and what the Magnus accuracy claims are tested against).
- PRESSURE-AWARE HUMIDITY. q = eps e / (p - (1 - eps) e) and r = eps e / (p - e) are exact; the familiar
  q ~ r ~ eps e/p is low by (1 - eps) e/p for q and e/p for r: 1.6 % and 4.2 % at a 30 degC dewpoint and 1000 hPa,
  more at lower pressure. Where e >= p (a hot parcel at stratospheric pressure) the saturation humidities are NaN,
  not negative.
- STULL'S WET BULB ASSUMES SEA-LEVEL PRESSURE. Stull (2011) is a fit at 1013.25 hPa, valid for RH 5-99 % and
  T -20..50 degC except where it is both cold and dry (``wet_bulb_stull`` returns NaN outside RH 5-99 %,
  T -20..50 degC unless strict=False). The psychrometric ``wet_bulb_temperature`` solves the isobaric enthalpy
  balance at the actual pressure (it agrees with the WMO psychrometer formula and with Normand's rule to 0.1-0.3 degC).
  Even at sea level Stull runs up to ~1 degC warm in hot, moderately dry air (+0.8 degC at 40 degC and 30 %); on a
  high plateau the pressure adds to that: at 840 hPa, 40 degC and 15 % the wet bulb is 19.2 degC against Stull's
  20.8 degC (20.4 degC at sea level, so 1.1 degC of the 1.5 degC is pressure alone) -- in heat-stress work the
  difference between a dangerous and a survivable wet bulb.
- HEAT INDEX: SIMPLE FORMULA FIRST. The NWS computes Steadman's simple formula, averages it with T, and only if that
  average is >= 80 degF uses the Rothfusz regression with its two adjustments. Rothfusz everywhere is wrong below
  ~80 degF, where it was never fitted: 76.9 degF at 70 degF and 50 % (NWS: 69.1), 78.7 at 60 degF and 30 % (57.1).
- COLUMN INTEGRALS ARE 0 -> p_s. Layer edges halfway between levels, the top edge at 0 Pa (or ``top``), the bottom at
  the surface pressure; levels below ground get zero weight, so the values a reanalysis extrapolates (or NaN) there
  never enter. Without p_s the lowest level is taken as the surface (half a layer below it).

References: Alduchov & Eskridge (1996, J. Appl. Meteor. 35, 601-609); Bolton (1980, Mon. Wea. Rev. 108, 1046-1053);
Murphy & Koop (2005, Q. J. R. Meteorol. Soc. 131, 1539-1565); Emanuel (1994, Atmospheric Convection, ch. 4);
Betts (1973, Q. J. R. Meteorol. Soc. 99, 178-196); Tripoli & Cotton (1981, Mon. Wea. Rev. 109, 1094-1102);
Davies-Jones (2008, Mon. Wea. Rev. 136, 2764-2785); Romps (2017, J. Atmos. Sci. 74, 3891-3900); Stull (2011,
J. Appl. Meteor. Climatol. 50, 2267-2269); AMS Glossary, "pseudoadiabatic lapse rate"; Rothfusz (1990, NWS SR
Technical Attachment 90-23) and the NWS heat index equation page; Steadman (1979, J. Appl. Meteor. 18, 861-873);
Osczevski & Bluestein (2005, Bull. Amer. Meteor. Soc. 86, 1453-1458) for the 2001 wind chill.

Requires: numpy, scipy (Lambert W for the Romps LCL).
"""

from __future__ import annotations

import numpy as np

from .constants import CI, CL, CPD, CPV, EPSILON, G, KAPPA, LF0, LV0, P0, RD, RV, T0

__all__ = [
    # vapour pressure and humidity
    "saturation_vapour_pressure",
    "vapour_pressure",
    "dewpoint",
    "relative_humidity",
    "dewpoint_from_rh",
    "mixing_ratio",
    "specific_humidity",
    "vapour_pressure_from_mixing_ratio",
    "vapour_pressure_from_specific_humidity",
    "specific_humidity_from_mixing_ratio",
    "mixing_ratio_from_specific_humidity",
    "specific_humidity_from_dewpoint",
    "dewpoint_from_specific_humidity",
    "relative_humidity_from_specific_humidity",
    "specific_humidity_from_relative_humidity",
    "saturation_mixing_ratio",
    "saturation_specific_humidity",
    "dewpoint_depression",
    "vapour_pressure_deficit",
    "latent_heat",
    # temperatures
    "exner",
    "potential_temperature",
    "temperature_from_potential_temperature",
    "virtual_temperature",
    "virtual_potential_temperature",
    "density_temperature",
    "air_density",
    "equivalent_temperature",
    "equivalent_potential_temperature",
    "saturation_equivalent_potential_temperature",
    "liquid_water_potential_temperature",
    "ice_liquid_potential_temperature",
    "wet_bulb_potential_temperature",
    "wet_bulb_stull",
    "stull_valid",
    "wet_bulb_temperature",
    # LCL, energies, adiabats
    "lcl_romps",
    "lcl_bolton",
    "dry_static_energy",
    "moist_static_energy",
    "enthalpy",
    "moist_lapse_rate",
    "moist_adiabat",
    # columns
    "column_integral",
    "precipitable_water",
    "column_water",
    "column_moist_static_energy",
    "column_enthalpy",
    "layer_mean_temperature",
    "thickness",
    # apparent temperature and units
    "heat_index",
    "wind_chill",
    "c_to_f",
    "f_to_c",
    "c_to_k",
    "k_to_c",
    "f_to_k",
    "k_to_f",
    "ms_to_kt",
    "kt_to_ms",
    "ms_to_mph",
    "mph_to_ms",
    "kt_to_mph",
    "mph_to_kt",
]

# --------------------------------------------------------------------------
# Saturation vapour pressure
# --------------------------------------------------------------------------
# Magnus form e_s = c exp(a t / (b + t)), t in degC, c in Pa
_MAGNUS = {
    ("magnus", "liquid"): (610.94, 17.625, 243.04),  # Alduchov & Eskridge (1996) AERK
    ("magnus", "ice"): (611.21, 22.587, 273.86),  # Alduchov & Eskridge (1996) AERKi
    ("bolton", "liquid"): (611.2, 17.67, 243.5),
}  # Bolton (1980) eq. 10


def _as(x):
    return np.asarray(x, dtype=float)


def _es_phase(t, phase, formula):
    if formula == "murphy-koop":
        lt = np.log(t)
        if phase == "liquid":  # Murphy & Koop (2005) eq. 10, 123 < T < 332 K
            return np.exp(
                54.842763
                - 6763.22 / t
                - 4.210 * lt
                + 0.000367 * t
                + np.tanh(0.0415 * (t - 218.8)) * (53.878 - 1331.22 / t - 9.44523 * lt + 0.014025 * t)
            )
        if phase == "ice":  # Murphy & Koop (2005) eq. 7, T > 110 K
            return np.exp(9.550426 - 5723.265 / t + 3.53068 * lt - 0.00728332 * t)
        raise ValueError(f"phase must be 'liquid', 'ice' or 'auto', not {phase!r}")
    try:
        c, a, b = _MAGNUS[(formula, phase)]
    except KeyError:
        raise ValueError(
            f"no {formula!r} formula over {phase!r} (formulas: magnus [liquid, ice], "
            "bolton [liquid], murphy-koop [liquid, ice])"
        ) from None
    tc = t - T0
    return c * np.exp(a * tc / (b + tc))


def saturation_vapour_pressure(t, phase: str = "liquid", formula: str = "magnus"):
    """Saturation vapour pressure (Pa) at temperature t (K).

    phase   : "liquid", "ice", or "auto" (ice below T0 = 273.15 K, liquid above)
    formula : "magnus" (Alduchov & Eskridge 1996: liquid c, a, b = 610.94 Pa, 17.625, 243.04 degC; ice 611.21 Pa,
              22.587, 273.86 degC), "bolton" (Bolton 1980: 611.2 Pa, 17.67, 243.5 degC; liquid only) or
              "murphy-koop" (Murphy & Koop 2005, reference accuracy, liquid 123-332 K, ice > 110 K)."""
    t = _as(t)
    if phase == "auto":
        return np.where(t < T0, _es_phase(t, "ice", formula), _es_phase(t, "liquid", formula))
    return _es_phase(t, phase, formula)


def vapour_pressure(td, formula: str = "magnus"):
    """Vapour pressure (Pa) from the dewpoint td (K): e = e_s,liquid(td)."""
    return saturation_vapour_pressure(td, "liquid", formula)


def dewpoint(e, phase: str = "liquid", formula: str = "magnus"):
    """Dewpoint (K) from vapour pressure e (Pa), the inverse of the saturation vapour pressure; phase="ice" gives the
    frost point. Magnus and Bolton invert analytically; Murphy-Koop by Newton iteration from the Magnus value."""
    e = _as(e)
    if phase not in ("liquid", "ice"):
        raise ValueError("dewpoint inverts over 'liquid' (dewpoint) or 'ice' (frost point)")
    base = "bolton" if formula == "bolton" else "magnus"
    if (base, phase) not in _MAGNUS:
        raise ValueError(f"no {formula!r} formula over {phase!r}")
    c, a, b = _MAGNUS[(base, phase)]
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.log(e / c)
        td = T0 + b * x / (a - x)
    if formula == "murphy-koop":
        for _ in range(6):
            h = 1e-3
            f0 = np.log(_es_phase(td, phase, formula))
            df = (np.log(_es_phase(td + h, phase, formula)) - f0) / h
            td = td - (f0 - np.log(e)) / df
    elif formula not in ("magnus", "bolton"):
        raise ValueError(f"unknown formula {formula!r}")
    return td


def relative_humidity(t, td, phase: str = "liquid", formula: str = "magnus"):
    """Relative humidity (%) = 100 e / e_s(t), with e = e_s,liquid(td). phase selects the saturation reference
    ("liquid", "ice" or "auto"); the dewpoint is always over liquid."""
    return 100.0 * vapour_pressure(td, formula) / saturation_vapour_pressure(t, phase, formula)


def dewpoint_from_rh(t, rh, formula: str = "magnus"):
    """Dewpoint (K) from temperature (K) and relative humidity (%, over liquid)."""
    return dewpoint(_as(rh) / 100.0 * saturation_vapour_pressure(t, "liquid", formula), "liquid", formula)


def mixing_ratio(e, p):
    """Mixing ratio (kg/kg of dry air) r = eps e / (p - e); NaN where e >= p."""
    e = _as(e)
    p = _as(p)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(e < p, EPSILON * e / (p - e), np.nan)


def specific_humidity(e, p):
    """Specific humidity (kg/kg of moist air) q = eps e / (p - (1 - eps) e); NaN where e >= p."""
    e = _as(e)
    p = _as(p)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(e < p, EPSILON * e / (p - (1.0 - EPSILON) * e), np.nan)


def vapour_pressure_from_mixing_ratio(r, p):
    """e = r p / (eps + r) (Pa)."""
    r = _as(r)
    return r * _as(p) / (EPSILON + r)


def vapour_pressure_from_specific_humidity(q, p):
    """e = q p / (eps + (1 - eps) q) (Pa)."""
    q = _as(q)
    return q * _as(p) / (EPSILON + (1.0 - EPSILON) * q)


def specific_humidity_from_mixing_ratio(r):
    """q = r / (1 + r)."""
    r = _as(r)
    return r / (1.0 + r)


def mixing_ratio_from_specific_humidity(q):
    """r = q / (1 - q)."""
    q = _as(q)
    return q / (1.0 - q)


def specific_humidity_from_dewpoint(td, p, formula: str = "magnus"):
    """q (kg/kg) from dewpoint (K) and pressure (Pa)."""
    return specific_humidity(vapour_pressure(td, formula), p)


def dewpoint_from_specific_humidity(q, p, formula: str = "magnus"):
    """Dewpoint (K) from q (kg/kg) and pressure (Pa)."""
    return dewpoint(vapour_pressure_from_specific_humidity(q, p), "liquid", formula)


def relative_humidity_from_specific_humidity(t, q, p, phase: str = "liquid", formula: str = "magnus"):
    """RH (%) = 100 e / e_s(t) from q (kg/kg), t (K), p (Pa)."""
    return 100.0 * vapour_pressure_from_specific_humidity(q, p) / saturation_vapour_pressure(t, phase, formula)


def specific_humidity_from_relative_humidity(t, rh, p, phase: str = "liquid", formula: str = "magnus"):
    """q (kg/kg) from RH (%), t (K), p (Pa)."""
    return specific_humidity(_as(rh) / 100.0 * saturation_vapour_pressure(t, phase, formula), p)


def saturation_mixing_ratio(t, p, phase: str = "liquid", formula: str = "magnus"):
    """Saturation mixing ratio r_s (kg/kg) over liquid, ice or "auto"; NaN where e_s >= p."""
    return mixing_ratio(saturation_vapour_pressure(t, phase, formula), p)


def saturation_specific_humidity(t, p, phase: str = "liquid", formula: str = "magnus"):
    """Saturation specific humidity q_s (kg/kg) over liquid, ice or "auto"; NaN where e_s >= p."""
    return specific_humidity(saturation_vapour_pressure(t, phase, formula), p)


def dewpoint_depression(t, td):
    """T - Td (K)."""
    return _as(t) - _as(td)


def vapour_pressure_deficit(t, td=None, rh=None, formula: str = "magnus"):
    """Vapour pressure deficit e_s(t) - e (Pa), from the dewpoint td (K) or the relative humidity rh (%)."""
    es = saturation_vapour_pressure(t, "liquid", formula)
    if td is not None:
        return es - vapour_pressure(td, formula)
    if rh is not None:
        return es * (1.0 - _as(rh) / 100.0)
    raise ValueError("give td or rh")


def latent_heat(t, kind: str = "vaporisation"):
    """Latent heat (J/kg) at t (K) by Kirchhoff's law with constant heat capacities:
    vaporisation LV0 + (CPV - CL)(t - T0), fusion LF0 + (CL - CI)(t - T0), sublimation the sum."""
    t = _as(t)
    lv = LV0 + (CPV - CL) * (t - T0)
    lf = LF0 + (CL - CI) * (t - T0)
    if kind == "vaporisation":
        return lv
    if kind == "fusion":
        return lf
    if kind == "sublimation":
        return lv + lf
    raise ValueError("kind is 'vaporisation', 'fusion' or 'sublimation'")


# --------------------------------------------------------------------------
# Potential, virtual, density and equivalent temperatures
# --------------------------------------------------------------------------
def _kappa(q):
    if q is None:
        return KAPPA
    q = _as(q)
    return ((1.0 - q) * RD + q * RV) / ((1.0 - q) * CPD + q * CPV)


def exner(p, p0: float = P0, q=None):
    """Exner function (p / p0)^kappa, kappa = RD/CPD, or the moist kappa when q is given."""
    return (_as(p) / p0) ** _kappa(q)


def potential_temperature(t, p, q=None, p0: float = P0):
    """theta = T (p0 / p)^kappa (K). Dry kappa = RD/CPD = 0.2857 by default; with q (kg/kg) the exact Poisson exponent
    of moist air, kappa_m = ((1-q) RD + q RV) / ((1-q) CPD + q CPV), conserved by an unsaturated adiabat of fixed q."""
    return _as(t) * (p0 / _as(p)) ** _kappa(q)


def temperature_from_potential_temperature(theta, p, q=None, p0: float = P0):
    """T = theta (p / p0)^kappa (K), the inverse of potential_temperature."""
    return _as(theta) * (_as(p) / p0) ** _kappa(q)


def virtual_temperature(t, q):
    """T_v = T (1 + (1/eps - 1) q) (K), exact for specific humidity q (no condensate)."""
    return _as(t) * (1.0 + (1.0 / EPSILON - 1.0) * _as(q))


def virtual_potential_temperature(t, p, q, p0: float = P0):
    """theta_v = T_v (p0 / p)^(RD/CPD) (K)."""
    return potential_temperature(virtual_temperature(t, q), p, p0=p0)


def density_temperature(t, q, ql=0.0, qi=0.0):
    """Density temperature T_rho = T (1 - q_t + q/eps) (K), q_t = q + ql + qi: the temperature dry air would need to
    have the density of the cloudy air (condensate loading included). Equals T_v when ql = qi = 0 (Emanuel 1994)."""
    q = _as(q)
    qt = q + _as(ql) + _as(qi)
    return _as(t) * (1.0 - qt + q / EPSILON)


def air_density(p, t, q=0.0, ql=0.0, qi=0.0):
    """Density of moist (and cloudy) air (kg/m3) = p / (RD T_rho)."""
    return _as(p) / (RD * density_temperature(t, q, ql, qi))


def equivalent_temperature(t, q):
    """Isobaric equivalent temperature T_e = T + LV0 r / CPD (K), r = q/(1-q) (AMS Glossary)."""
    return _as(t) + LV0 * mixing_ratio_from_specific_humidity(q) / CPD


def _t_lcl_bolton(t, td):
    """Bolton (1980) eq. 15: temperature (K) at the lifting condensation level from T and Td (K)."""
    return 1.0 / (1.0 / (td - 56.0) + np.log(t / td) / 800.0) + 56.0


def equivalent_potential_temperature(t, td, p, method: str = "bolton", rt=None):
    """Equivalent potential temperature (K) from temperature t (K), dewpoint td (K) and pressure p (Pa).

    method "bolton" -- pseudo-adiabatic, Bolton (1980) eq. 39, the most accurate of his formulas:
        theta_E = T (p0/(p - e))^0.2854 (T/T_L)^(0.28 r) exp[(3036/T_L - 1.78) r (1 + 0.448 r)],
        r in kg/kg, T_L from eq. 15, e from Bolton's own Magnus coefficients (the ones the fit used).
    method "reversible" -- Emanuel (1994, sec. 4.5), conserved in reversible moist adiabatic processes:
        theta_e = T (p0/p_d)^(RD/c) H^(-r RV/c) exp[L_v(T) r / (c T)],  c = CPD + r_t CL,
        H = e/e_s(T), p_d = p - e, r_t the total-water mixing ratio (default r: no condensate).
    The two differ by a few K at high water content: a pseudo-adiabat drops its condensate (and its heat capacity)."""
    t = _as(t)
    td = _as(td)
    p = _as(p)
    if method == "bolton":
        e = saturation_vapour_pressure(td, "liquid", "bolton")
        r = EPSILON * e / (p - e)
        tl = _t_lcl_bolton(t, td)
        theta_dl = t * (P0 / (p - e)) ** 0.2854 * (t / tl) ** (0.28 * r)
        return theta_dl * np.exp((3036.0 / tl - 1.78) * r * (1.0 + 0.448 * r))
    if method == "reversible":
        e = vapour_pressure(td)
        r = EPSILON * e / (p - e)
        rt = r if rt is None else _as(rt)
        c = CPD + rt * CL
        h = e / saturation_vapour_pressure(t)
        return t * (P0 / (p - e)) ** (RD / c) * h ** (-r * RV / c) * np.exp(latent_heat(t) * r / (c * t))
    raise ValueError("method is 'bolton' or 'reversible'")


def saturation_equivalent_potential_temperature(t, p, method: str = "bolton"):
    """theta_e of saturated air at (t, p): equivalent_potential_temperature with td = t."""
    return equivalent_potential_temperature(t, t, p, method)


def liquid_water_potential_temperature(t, p, ql, p0: float = P0):
    """Liquid-water potential temperature (Betts 1973), theta_l = theta - (theta/T)(LV0/CPD) ql (K): conserved to
    first order under reversible condensation and evaporation; the shallow-cloud LES/boundary-layer variable."""
    th = potential_temperature(t, p, p0=p0)
    return th - th / _as(t) * (LV0 / CPD) * _as(ql)


def ice_liquid_potential_temperature(t, p, ql, qi, p0: float = P0):
    """Ice-liquid water potential temperature (Tripoli & Cotton 1981),
    theta_il = theta / (1 + (LV0 ql + LS0 qi) / (CPD max(T, 253 K))) (K); the 253 K floor is theirs."""
    th = potential_temperature(t, p, p0=p0)
    return th / (1.0 + (LV0 * _as(ql) + (LV0 + LF0) * _as(qi)) / (CPD * np.maximum(_as(t), 253.0)))


def wet_bulb_potential_temperature(t, td, p):
    """Wet-bulb potential temperature theta_w (K): the 1000 hPa temperature of the pseudo-adiabat through the parcel,
    by Davies-Jones (2008, eq. 3.8) from the Bolton (1980, eq. 39) theta_E: with X = theta_E / 273.15,
        theta_w = theta_E - exp(A(X) / B(X)), A, B quartics (coefficients below), theta_w = theta_E below 173.15 K.
    Accurate to ~0.001 K against the inversion of Bolton's theta_E (tested)."""
    te = equivalent_potential_temperature(t, td, p, "bolton")
    x = te / 273.15
    a = 7.101574 - 20.68208 * x + 16.11182 * x**2 + 2.574631 * x**3 - 5.205688 * x**4
    b = 1.0 - 3.552497 * x + 3.781782 * x**2 - 0.6899655 * x**3 - 0.5929340 * x**4
    return np.where(te <= 173.15, te, te - np.exp(a / b))


# --------------------------------------------------------------------------
# Wet bulb
# --------------------------------------------------------------------------
def stull_valid(t, rh):
    """Validity domain of Stull (2011): RH 5-99 % and T -20..50 degC (t in K). Outside it the fit's error grows
    quickly (and it never knew about pressure: it is a fit at 1013.25 hPa)."""
    tc = _as(t) - T0
    rh = _as(rh)
    return (rh >= 5.0) & (rh <= 99.0) & (tc >= -20.0) & (tc <= 50.0)


def wet_bulb_stull(t, rh, strict: bool = True):
    """Wet-bulb temperature (K) by Stull (2011) from t (K) and RH (%):
        Tw = T atan(0.151977 (RH + 8.313659)^1/2) + atan(T + RH) - atan(RH - 1.676331)
             + 0.00391838 RH^3/2 atan(0.023101 RH) - 4.686035          (T, Tw in degC)
    Stull quotes -1 to +0.65 degC against his reference; against the isobaric psychrometric solution
    (wet_bulb_temperature) at 1013.25 hPa it is -1.2 to +1.1 degC for T >= 0 degC (mean |error| 0.34 degC), and worse
    in cold dry air. It is a fit AT SEA-LEVEL PRESSURE. strict: NaN outside RH 5-99 %, T -20..50 degC (stull_valid);
    strict=False evaluates the fit everywhere. For any other pressure use wet_bulb_temperature."""
    tc = _as(t) - T0
    rh = _as(rh)
    tw = (
        tc * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
        + np.arctan(tc + rh)
        - np.arctan(rh - 1.676331)
        + 0.00391838 * rh**1.5 * np.arctan(0.023101 * rh)
        - 4.686035
    ) + T0
    return np.where(stull_valid(t, rh), tw, np.nan) if strict else tw


def wet_bulb_temperature(t, p, td=None, rh=None, formula: str = "magnus", tol: float = 1e-6):
    """Isobaric (psychrometric) wet-bulb temperature (K) at pressure p (Pa), from t (K) and the dewpoint td (K) or RH
    (%). Solves the enthalpy balance of adiabatic saturation at constant pressure,
        (CPD + r CPV)(T - Tw) = (r_s(Tw, p) - r) L_v(Tw),
    exact for ideal gases with constant heat capacities; bracketed between Td (where the right side is 0) and T, and
    bisected to `tol` K. Vectorised."""
    t = _as(t)
    p = _as(p)
    if td is None:
        if rh is None:
            raise ValueError("give td or rh")
        td = dewpoint_from_rh(t, rh, formula)
    td = np.minimum(_as(td), t)
    r = mixing_ratio(vapour_pressure(td, formula), p)
    lo, hi = np.broadcast_arrays(td, t)
    lo = lo.copy()
    hi = hi.copy()
    cp = CPD + r * CPV

    def f(tw):
        return cp * (t - tw) - (saturation_mixing_ratio(tw, p, "liquid", formula) - r) * latent_heat(tw)

    n = int(np.ceil(np.log2(max(float(np.nanmax(hi - lo, initial=0.0)), tol) / tol))) + 1
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        pos = f(mid) > 0
        lo = np.where(pos, mid, lo)
        hi = np.where(pos, hi, mid)
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------
# Lifting condensation level
# --------------------------------------------------------------------------
# Romps (2017) constant set: his exact expression is exact for HIS saturation vapour pressure (Rankine-Kirchhoff
# with these constants), so they are kept together to reproduce the paper's values.
ROMPS2017 = dict(
    ttrip=273.16,
    ptrip=611.65,
    e0v=2.3740e6,
    e0s=0.3337e6,
    g=9.81,
    rgasa=287.04,
    rgasv=461.0,
    cva=719.0,
    cvv=1418.0,
    cvl=4119.0,
    cvs=1861.0,
)


def lcl_romps(t, p, rh, phase: str = "liquid"):
    """Exact lifting condensation level (Romps 2017) from t (K), p (Pa) and RH (%).

    Returns (t_lcl K, p_lcl Pa, z_lcl m above the parcel). phase "liquid" gives the LCL; "ice" the lifting
    deposition level (RH then over ice); "auto" ice below the triple point. Uses Romps's own constants (ROMPS2017):
    exactness is relative to his saturation vapour pressure, which RH is taken against (differences from the
    Magnus RH are ~0.1 %); the Lambert W branch -1 is evaluated by scipy.special.lambertw."""
    from scipy.special import lambertw

    c_ = ROMPS2017
    t = _as(t)
    p = _as(p)
    rh = _as(rh) / 100.0
    cpa = c_["cva"] + c_["rgasa"]
    cpv = c_["cvv"] + c_["rgasv"]

    def branch(cvx, e0):
        pstar = (
            c_["ptrip"]
            * (t / c_["ttrip"]) ** ((cpv - cvx) / c_["rgasv"])
            * np.exp((e0 - (c_["cvv"] - cvx) * c_["ttrip"]) / c_["rgasv"] * (1.0 / c_["ttrip"] - 1.0 / t))
        )
        pv = rh * pstar
        qv = c_["rgasa"] * pv / (c_["rgasv"] * p + (c_["rgasa"] - c_["rgasv"]) * pv)
        rgasm = (1.0 - qv) * c_["rgasa"] + qv * c_["rgasv"]
        cpm = (1.0 - qv) * cpa + qv * cpv
        a = cpm / rgasm + (cvx - cpv) / c_["rgasv"]
        b = -(e0 - (c_["cvv"] - cvx) * c_["ttrip"]) / (c_["rgasv"] * t)
        c = b / a
        tl = t * c / np.real(lambertw(rh ** (1.0 / a) * c * np.exp(c), -1))
        return tl, p * (tl / t) ** (cpm / rgasm), cpm * (t - tl) / c_["g"]

    liq = branch(c_["cvl"], c_["e0v"])
    if phase == "liquid":
        return liq
    ice = branch(c_["cvs"], c_["e0v"] + c_["e0s"])
    if phase == "ice":
        return ice
    if phase == "auto":
        cold = t < c_["ttrip"]
        return tuple(np.where(cold, i, w) for i, w in zip(ice, liq))
    raise ValueError("phase is 'liquid', 'ice' or 'auto'")


def lcl_bolton(t, td, p):
    """LCL temperature (Bolton 1980 eq. 15) and pressure (Poisson with Bolton's 0.2854 (1 - 0.28 r)) from t, td (K),
    p (Pa): returns (t_lcl K, p_lcl Pa). Within ~0.1 K of the exact (Romps) T_LCL at ordinary humidities."""
    t = _as(t)
    td = _as(td)
    p = _as(p)
    tl = _t_lcl_bolton(t, td)
    r = mixing_ratio(saturation_vapour_pressure(td, "liquid", "bolton"), p)
    return tl, p * (tl / t) ** (1.0 / (0.2854 * (1.0 - 0.28 * r)))


# --------------------------------------------------------------------------
# Static energy, enthalpy, the pseudo-adiabat
# --------------------------------------------------------------------------
def dry_static_energy(t, z):
    """s = CPD T + G z (J/kg), z the geopotential height (m)."""
    return CPD * _as(t) + G * _as(z)


def moist_static_energy(t, z, q):
    """h = CPD T + G z + LV0 q (J/kg), z the geopotential height (m), q specific humidity: the conventional form,
    conserved to within a few tenths of a per cent in moist adiabatic processes."""
    return CPD * _as(t) + G * _as(z) + LV0 * _as(q)


def enthalpy(t, q=0.0, ql=0.0, qi=0.0, t_ref: float = T0):
    """Specific enthalpy of moist air (J per kg of moist air), constant heat capacities.

    REFERENCE STATE: dry air and liquid water have zero enthalpy at t_ref (default T0 = 273.15 K); vapour then carries
    L_v(t_ref) and ice -L_f(t_ref) (Kirchhoff latent heats), so
        h = q_d CPD (T - Tr) + q (CPV (T - Tr) + L_v(Tr)) + ql CL (T - Tr) + qi (CI (T - Tr) - L_f(Tr)),
    q_d = 1 - q - ql - qi. Differences and fluxes do not depend on the reference when water is conserved; totals do
    -- state it whenever a column total is shown. With q = ql = qi = 0 this is the dry enthalpy CPD (T - Tr)."""
    t = _as(t)
    q = _as(q)
    ql = _as(ql)
    qi = _as(qi)
    dt = t - t_ref
    qd = 1.0 - q - ql - qi
    return (
        qd * CPD * dt
        + q * (CPV * dt + latent_heat(t_ref))
        + ql * CL * dt
        + qi * (CI * dt - latent_heat(t_ref, "fusion"))
    )


def _dt_dlnp_pseudo(t, p, formula):
    """dT/dln p along the pseudo-adiabat (liquid saturation), the AMS Glossary form with L_v(T) and p_d = p - e_s."""
    es = saturation_vapour_pressure(t, "liquid", formula)
    r = EPSILON * es / (p - es)
    lv = latent_heat(t)
    num = RD * t + lv * r
    den = CPD + r * CPV + lv * lv * r * (EPSILON + r) / (RD * t * t)
    return p / (p - es) * num / den


def moist_lapse_rate(t, p, formula: str = "magnus"):
    """Pseudo-adiabatic (saturated) lapse rate -dT/dz (K/m) at t (K), p (Pa) (AMS Glossary):
        G (1 + r)(1 + L_v r/(RD T)) / (CPD + r CPV + L_v^2 r (eps + r)/(RD T^2)),  r = r_s(T, p), L_v = L_v(T).
    9.8 K/km for dry air (r -> 0: G/CPD), ~3.5 K/km in warm saturated boundary layers."""
    t = _as(t)
    p = _as(p)
    r = saturation_mixing_ratio(t, p, "liquid", formula)
    lv = latent_heat(t)
    return G * (1.0 + r) * (1.0 + lv * r / (RD * t)) / (CPD + r * CPV + lv * lv * r * (EPSILON + r) / (RD * t * t))


def moist_adiabat(t_start, p_start, p, formula: str = "magnus", dlnp: float = 0.002):
    """Temperature (K) along the pseudo-adiabat through (t_start K, p_start Pa) at the pressures p (Pa, 1-D, any
    order, either side of p_start). RK4 in ln p with steps of at most `dlnp`; t_start and p_start broadcast, giving
    output (len(p),) + their broadcast shape."""
    t0, p0 = np.broadcast_arrays(_as(t_start), _as(p_start))
    p = np.atleast_1d(_as(p))
    out = np.empty((p.size, *t0.shape))
    tc = t0.astype(float).copy()
    lnp = np.log(p0.astype(float))
    # The pseudo-adiabat is a first-order ODE in ln p, so the levels can be visited in any order (each leg may run up
    # or down, point by point); descending pressure keeps the legs short.
    for k in np.argsort(-p):
        target = np.log(p[k])
        n = max(1, int(np.ceil(np.max(np.abs(target - lnp)) / dlnp)))
        h = (target - lnp) / n
        for _ in range(n):
            pp = np.exp(lnp)
            k1 = _dt_dlnp_pseudo(tc, pp, formula)
            k2 = _dt_dlnp_pseudo(tc + 0.5 * h * k1, pp * np.exp(0.5 * h), formula)
            k3 = _dt_dlnp_pseudo(tc + 0.5 * h * k2, pp * np.exp(0.5 * h), formula)
            k4 = _dt_dlnp_pseudo(tc + h * k3, pp * np.exp(h), formula)
            tc = tc + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            lnp = lnp + h
        out[k] = tc
    return out


# --------------------------------------------------------------------------
# Column integrals on pressure (or model) levels
# --------------------------------------------------------------------------
def _layer_dp(p, ps, top, shape):
    """dp (Pa) per level, level axis first, broadcastable to `shape`: edges midway between levels, the top edge at
    `top`, the bottom at ps (clipped) -- or at the lowest level when ps is None."""
    p = _as(p)
    if p.ndim == 1:
        p = p.reshape((-1,) + (1,) * (len(shape) - 1))
    o = np.argsort(p, axis=0)
    q = np.take_along_axis(p, o, 0)
    mid = 0.5 * (q[1:] + q[:-1])
    top_e = np.full_like(q[:1], float(top))
    bot_e = q[-1:].copy() if ps is None else np.full_like(q[:1], np.inf)
    edges = np.concatenate([top_e, mid, bot_e], 0)
    if ps is not None:
        edges = np.minimum(edges, _as(ps)[None])
    dps = np.clip(np.diff(edges, axis=0), 0.0, None)
    dp = np.empty(np.broadcast_shapes(dps.shape, o.shape))
    np.put_along_axis(dp, np.broadcast_to(o, dp.shape), np.broadcast_to(dps, dp.shape), 0)
    return dp


def column_integral(x, p, ps=None, axis: int = 0, top: float = 0.0):
    """Mass-weighted vertical integral  sum_k x_k dp_k / G  (x units times kg/m2).

    x  : (..., lev, ...) with the level axis `axis`
    p  : level pressures (Pa), 1-D (pressure levels) or x's shape (model levels; any order along the axis)
    ps : surface pressure (Pa), x's shape without the level axis; None = the lowest level is the surface.
    Edges are midway between levels, the top edge at `top` (Pa), the bottom at ps; levels below ground get zero weight
    and whatever they hold (extrapolated values, NaN) is ignored. NaN above ground propagates."""
    x0 = np.moveaxis(_as(x), axis, 0)
    pp = _as(p)
    if pp.ndim > 1:
        pp = np.moveaxis(pp, axis, 0)
    dp = np.broadcast_to(_layer_dp(pp, ps, top, x0.shape), x0.shape)
    with np.errstate(invalid="ignore"):
        return np.where(dp > 0, x0 * dp, 0.0).sum(0) / G


def precipitable_water(q, p, ps=None, axis: int = 0, top: float = 0.0):
    """Total column water vapour / precipitable water (kg/m2 = mm), the integral of q dp/G from `top` to p_s."""
    return column_integral(q, p, ps, axis, top)


def column_water(p, ps=None, q=None, ql=None, qi=None, axis: int = 0, top: float = 0.0) -> dict:
    """Column vapour, liquid and ice (kg/m2) for whichever of q, ql, qi are given, plus their 'total'."""
    out = {}
    for name, v in (("vapour", q), ("liquid", ql), ("ice", qi)):
        if v is not None:
            out[name] = column_integral(v, p, ps, axis, top)
    if not out:
        raise ValueError("give at least one of q, ql, qi")
    out["total"] = sum(out.values())
    return out


def column_moist_static_energy(t, z, q, p, ps=None, axis: int = 0, top: float = 0.0):
    """Column-integrated moist static energy (J/m2), the integral of (CPD T + G z + LV0 q) dp/G."""
    return column_integral(moist_static_energy(t, z, q), p, ps, axis, top)


def column_enthalpy(t, p, ps=None, q=0.0, ql=0.0, qi=0.0, axis: int = 0, top: float = 0.0, t_ref: float = T0):
    """Column-integrated moist enthalpy (J/m2) with the reference state of `enthalpy` (zero for dry air and liquid
    water at t_ref)."""
    return column_integral(enthalpy(t, q, ql, qi, t_ref), p, ps, axis, top)


def _lnp_weights(p, p_bottom, p_top):
    """Weights c_k with sum_k c_k f_k = integral of f d(ln p) from p_top to p_bottom, f linear in ln p between
    levels. No extrapolation."""
    lp = np.log(_as(p))
    o = np.argsort(lp)
    x = lp[o]
    a, b = np.log(p_top), np.log(p_bottom)
    if not (b > a):
        raise ValueError("p_bottom must exceed p_top")
    if a < x[0] - 1e-12 or b > x[-1] + 1e-12:
        raise ValueError("the layer must lie within the levels given (no extrapolation)")
    c = np.zeros(x.size)
    for k in range(x.size - 1):
        u, v = max(a, x[k]), min(b, x[k + 1])
        if v <= u:
            continue
        h = x[k + 1] - x[k]
        c[k] += ((x[k + 1] - u) ** 2 - (x[k + 1] - v) ** 2) / (2 * h)
        c[k + 1] += ((v - x[k]) ** 2 - (u - x[k]) ** 2) / (2 * h)
    out = np.empty_like(c)
    out[o] = c
    return out, b - a


def layer_mean_temperature(t, p, p_bottom: float, p_top: float, q=None, axis: int = 0):
    """ln p-weighted mean temperature of the layer p_top..p_bottom (Pa), the hypsometric mean: the integral of T d ln p
    over ln(p_bottom/p_top), T linear in ln p between the 1-D levels p (Pa). Virtual temperature when q is given."""
    tt = _as(t) if q is None else virtual_temperature(t, q)
    c, span = _lnp_weights(p, p_bottom, p_top)
    return np.tensordot(c, np.moveaxis(tt, axis, 0), axes=(0, 0)) / span


def thickness(t, p, p_bottom: float, p_top: float, q=None, axis: int = 0):
    """Hypsometric thickness (m) of the layer p_top..p_bottom (Pa): (RD/G) <T_v> ln(p_bottom/p_top), with <T_v> from
    layer_mean_temperature (pass q for the virtual correction; without it, the dry thickness)."""
    return RD / G * layer_mean_temperature(t, p, p_bottom, p_top, q, axis) * np.log(p_bottom / p_top)


# --------------------------------------------------------------------------
# Apparent temperature (NWS)
# --------------------------------------------------------------------------
def heat_index(t_f, rh):
    """NWS heat index (degF) from air temperature (degF) and RH (%), exactly as the NWS specifies:
    1. the simple formula HI = 0.5 (T + 61 + 1.2 (T - 68) + 0.094 RH) (Steadman 1979);
    2. if the average of that and T is >= 80 degF, the Rothfusz (1990) regression instead,
       HI = -42.379 + 2.04901523 T + 10.14333127 RH - 0.22475541 T RH - 6.83783e-3 T^2 - 5.481717e-2 RH^2
            + 1.22874e-3 T^2 RH + 8.5282e-4 T RH^2 - 1.99e-6 T^2 RH^2,
       minus ((13 - RH)/4) sqrt((17 - |T - 95|)/17) if RH < 13 % and 80 <= T <= 112 degF,
       plus ((RH - 85)/10) ((87 - T)/5) if RH > 85 % and 80 <= T <= 87 degF."""
    t = _as(t_f)
    r = _as(rh)
    simple = 0.5 * (t + 61.0 + (t - 68.0) * 1.2 + r * 0.094)
    full = (
        -42.379
        + 2.04901523 * t
        + 10.14333127 * r
        - 0.22475541 * t * r
        - 6.83783e-3 * t * t
        - 5.481717e-2 * r * r
        + 1.22874e-3 * t * t * r
        + 8.5282e-4 * t * r * r
        - 1.99e-6 * t * t * r * r
    )
    dry = (r < 13.0) & (t >= 80.0) & (t <= 112.0)
    full = full - np.where(dry, (13.0 - r) / 4.0 * np.sqrt(np.clip(17.0 - np.abs(t - 95.0), 0.0, None) / 17.0), 0.0)
    wet = (r > 85.0) & (t >= 80.0) & (t <= 87.0)
    full = full + np.where(wet, (r - 85.0) / 10.0 * (87.0 - t) / 5.0, 0.0)
    return np.where(0.5 * (simple + t) >= 80.0, full, simple)


def wind_chill(t_f, wind_mph):
    """NWS (2001) wind chill temperature (degF) from air temperature (degF) and 10 m wind (mph):
        WC = 35.74 + 0.6215 T - 35.75 V^0.16 + 0.4275 T V^0.16   for T <= 50 degF and V >= 3 mph, else T
    (Osczevski & Bluestein 2005; the wind is the 10 m value, the formula reduces it to face height itself)."""
    t = _as(t_f)
    v = _as(wind_mph)
    v16 = np.power(np.maximum(v, 0.0), 0.16)
    wc = 35.74 + 0.6215 * t - 35.75 * v16 + 0.4275 * t * v16
    return np.where((t <= 50.0) & (v >= 3.0), wc, t)


# --------------------------------------------------------------------------
# Units
# --------------------------------------------------------------------------
_KT = 1852.0 / 3600.0  # m/s per knot (exact)
_MPH = 0.44704  # m/s per mph (exact)


def c_to_f(x):
    return _as(x) * 9.0 / 5.0 + 32.0


def f_to_c(x):
    return (_as(x) - 32.0) * 5.0 / 9.0


def c_to_k(x):
    return _as(x) + T0


def k_to_c(x):
    return _as(x) - T0


def f_to_k(x):
    return f_to_c(x) + T0


def k_to_f(x):
    return c_to_f(_as(x) - T0)


def ms_to_kt(x):
    return _as(x) / _KT


def kt_to_ms(x):
    return _as(x) * _KT


def ms_to_mph(x):
    return _as(x) / _MPH


def mph_to_ms(x):
    return _as(x) * _MPH


def kt_to_mph(x):
    return _as(x) * _KT / _MPH


def mph_to_kt(x):
    return _as(x) * _MPH / _KT
