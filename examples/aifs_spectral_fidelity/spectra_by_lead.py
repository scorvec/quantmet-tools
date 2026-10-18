#!/usr/bin/env python3
"""Reproduce the AIFS single vs AIFS-ENS control spectral comparison.

Fetches 700 & 250 hPa u/v for both models (steps 0/120/240) from ECMWF open data on its Google Cloud mirror and
computes DCT KE spectra over a 25-55N Pacific-North America box with quantmet.dct_spectra; prints the table in
README.md and redraws aifs_spectral_fidelity.png. ~25 MB of downloads (24 single-field GRIB files), kept in $SPECTRA_DATA (default: ./data) so a
rerun reads them back; adjust DATE to any init the mirror still holds.

Grid caveat: the box is a 0.25 deg latitude-longitude grid, not a uniform one. res_km below is dx at 40N (21.3 km);
dy is 27.8 km and dx runs from 25.2 km at 25N to 15.9 km at 55N, so the wavelength axis is approximate (the short
side of the box is the meridional one, 121 points: its true length is 3336 km, labelled here 2577 km). The
comparison between the two models is unaffected -- both are binned identically.

2026-10-12 (quantmet 0.3.1): rerun after the dct_spectra wavenumber fix (0.3.0 labelled every bin one wavenumber
too long); source="google" (the ECMWF origin server is not used).
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from ecmwf.opendata import Client
from quantmet.dct_spectra import ke_spectrum, spectral_slope, fidelity_resolution

DATE = "2026-08-11"
STEPS = (0, 120, 240)
LEVS = (700, 250)
MODELS = (("aifs-single", "fc", "oper"), ("aifs-ens", "cf", "enfo"))
HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("SPECTRA_DATA", "data"))
DATA.mkdir(parents=True, exist_ok=True)

# one parameter per request: the Google mirror serves a single byte range per request, not several
for model, typ, stream in MODELS:
    c = Client(source="google", model=model)
    for lev in LEVS:
        for step in STEPS:
            for par in ("u", "v"):
                target = DATA / f"{model}_{lev}_s{step}_{par}.grib2"
                if not target.exists() or target.stat().st_size == 0:
                    c.retrieve(
                        date=DATE,
                        time=0,
                        type=typ,
                        stream=stream,
                        levtype="pl",
                        param=par,
                        levelist=lev,
                        step=step,
                        target=str(target),
                    )


def box_uv(model, lev, step):
    def get(par):
        ds = xr.open_dataset(
            DATA / f"{model}_{lev}_s{step}_{par}.grib2", engine="cfgrib", backend_kwargs={"indexpath": ""}
        )
        return ds[par]

    u, v = get("u"), get("v")
    lat, lon = u.latitude.values, np.where(u.longitude.values < 0, u.longitude.values + 360, u.longitude.values)
    order = np.argsort(lon)
    mlat = (lat <= 55) & (lat >= 25)
    mlon = (lon[order] >= 150) & (lon[order] <= 250)
    return (u.values[:, order][np.ix_(mlat, mlon)], v.values[:, order][np.ix_(mlat, mlon)])


res_km = 111.2 * 0.25 * np.cos(np.deg2rad(40.0))  # dx at 40N; see the grid caveat above
spec, ref, rows = {}, {}, {}
for lev in LEVS:
    for step in STEPS:
        for model, _, _ in MODELS:
            u, v = box_uv(model, lev, step)
            wl, p = ke_spectrum(u, v, res_km=res_km)
            spec[(lev, model, step)] = (wl, p)
            if step == 0:
                ref[(lev, model)] = p
                print(f"{lev:>4} hPa day  0 {model:>12}: analysis reference")
                continue
            fr = fidelity_resolution(wl, p, ref[(lev, model)])
            sl = spectral_slope(wl, p, 100, 1000)
            rows[(lev, model, step)] = (sl, fr)
            print(f"{lev:>4} hPa day {step // 24:>2} {model:>12}: slope {sl:+.2f} · fidelity res {fr:.0f} km")

np.savez_compressed(
    HERE / "aifs_spectra.npz",
    res_km=res_km,
    date=DATE,
    **{f"dct_{m}_{lev}_s{s}_{q}": spec[(lev, m, s)][i] for (lev, m, s) in spec for i, q in enumerate(("wl", "p"))},
)


fig, axes = plt.subplots(1, 2, figsize=(17, 6.6))
for ax, lev in zip(axes, LEVS):
    wl, p0 = spec[(lev, "aifs-single", 0)]
    ax.loglog(wl, p0, color="#c62828", ls=":", lw=1.4, label="single · day 0  (=analysis, both)")
    for step, ls, lw in ((120, "--", 1.8), (240, "-", 2.4)):
        for model, col, lab in (("aifs-single", "#c62828", "single"), ("aifs-ens", "#1565c0", "ENS control")):
            wl, p = spec[(lev, model, step)]
            ax.loglog(wl, p, color=col, ls=ls, lw=lw, label=f"{lab} · day {step // 24}")
    w3 = np.geomspace(300, 2000, 20)
    a3 = p0[np.argmin(np.abs(wl - 1000))] * 1000.0**-3
    ax.loglog(w3, 2.5 * a3 * w3**3, color="0.35", ls="--", lw=1)
    ax.text(420, 2.5 * a3 * 420.0**3, "k$^{-3}$", color="0.35", fontsize=10, ha="left", va="top")
    ax.invert_xaxis()
    ax.grid(True, which="both", alpha=0.25)
    ax.set_xlabel("wavelength (km)")
    ax.set_title(f"{lev} hPa", loc="left", fontweight="bold", fontsize=14)
    txt = "slope (100–1000 km) / fidelity res (P<½·analysis):\n" + "\n".join(
        f"d{s // 24}: single {rows[(lev, 'aifs-single', s)][0]:+.2f} / {rows[(lev, 'aifs-single', s)][1]:.0f} km · "
        f"ctrl {rows[(lev, 'aifs-ens', s)][0]:+.2f} / {rows[(lev, 'aifs-ens', s)][1]:.0f} km"
        for s in (120, 240)
    )
    ax.text(
        0.03,
        0.04,
        txt,
        transform=ax.transAxes,
        fontsize=9.5,
        va="bottom",
        bbox=dict(boxstyle="round", fc="#fffdf0", ec="0.6"),
    )
axes[0].set_ylabel("KE spectral density")
axes[0].legend(fontsize=9.5, loc="upper right")
fig.suptitle(
    f"AIFS single vs AIFS-ENS control — DCT KE spectra by lead · 25–55°N, 150°E–110°W · init {DATE} 00Z",
    fontsize=14,
    fontweight="bold",
)
fig.text(
    0.5,
    0.005,
    "quantmet.dct_spectra (0.3.1) · 0.25° ECMWF open data · wavelengths use dx at 40°N (21.3 km); "
    "dy is 27.8 km, so the axis is approximate",
    ha="center",
    fontsize=9,
    color="0.35",
)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(HERE / "aifs_spectral_fidelity.png", dpi=100)
print("wrote", HERE / "aifs_spectral_fidelity.png")
