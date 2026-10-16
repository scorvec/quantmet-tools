"""Physical constants shared across quantmet (SI units).

    RD      287.04      J kg-1 K-1   gas constant of dry air
    RV      461.5       J kg-1 K-1   gas constant of water vapour
    EPSILON RD / RV     = 0.62197    ratio of the molar masses of water and dry air
    CPD     1004.7      J kg-1 K-1   specific heat of dry air at constant pressure
    CPV     1846.0      J kg-1 K-1   specific heat of water vapour at constant pressure
    CL      4218.0      J kg-1 K-1   specific heat of liquid water
    CI      2106.0      J kg-1 K-1   specific heat of ice
    LV0     2.501e6     J kg-1       latent heat of vaporisation at T0
    LF0     3.337e5     J kg-1       latent heat of fusion at T0
    LS0     LV0 + LF0   J kg-1       latent heat of sublimation at T0
    G       9.80665     m s-2        standard gravity
    T0      273.15      K            0 degC, the reference temperature of the latent heats
    P0      1.0e5       Pa           reference pressure of potential temperatures
    KAPPA   RD / CPD    = 0.28570    Poisson exponent of dry air
    A_EARTH 6.371e6     m            mean Earth radius
    OMEGA   7.292115e-5 s-1          Earth's rotation rate

With constant heat capacities the latent heats follow Kirchhoff's law,
L_v(T) = LV0 + (CPV - CL)(T - T0) (dL_v/dT = -2372 J kg-1 K-1, Bolton's 1980
value is -2370) and L_f(T) = LF0 + (CL - CI)(T - T0); see
``quantmet.thermo.latent_heat``.

``quantmet.thermo`` uses these. Modules older than 0.4 still carry their own module-level constants (some
deliberately different, e.g. the E-P flux's kappa = 0.2854), so that their published numbers do not move.

Formula-specific constants that are part of a published fit (Bolton's 0.2854,
the Magnus coefficients, Romps's 2017 constant set for the exact LCL) stay
with the formula in ``quantmet.thermo`` so that it reproduces the paper.
"""

from __future__ import annotations

RD = 287.04
RV = 461.5
EPSILON = RD / RV
CPD = 1004.7
CPV = 1846.0
CL = 4218.0
CI = 2106.0
LV0 = 2.501e6
LF0 = 3.337e5
LS0 = LV0 + LF0
G = 9.80665
T0 = 273.15
P0 = 1.0e5
KAPPA = RD / CPD
A_EARTH = 6.371e6
OMEGA = 7.292115e-5

__all__ = [
    "RD",
    "RV",
    "EPSILON",
    "CPD",
    "CPV",
    "CL",
    "CI",
    "LV0",
    "LF0",
    "LS0",
    "G",
    "T0",
    "P0",
    "KAPPA",
    "A_EARTH",
    "OMEGA",
]
