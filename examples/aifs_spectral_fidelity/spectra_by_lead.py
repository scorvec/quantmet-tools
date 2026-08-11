#!/usr/bin/env python3
"""Reproduce the AIFS single vs AIFS-ENS control spectral comparison.

Fetches 700 & 250 hPa u/v for both models (steps 0/120/240) from ECMWF open
data and computes DCT KE spectra over a 25-55N Pacific-North America box with
quantmet.dct_spectra. ~10 MB of downloads; adjust DATE to any init within the
open-data retention window.
"""
import numpy as np
import xarray as xr
from ecmwf.opendata import Client
from quantmet.dct_spectra import ke_spectrum, spectral_slope, effective_resolution

DATE = "2026-08-11"
STEPS = (0, 120, 240)
LEVS = (700, 250)

for model, typ, stream in (("aifs-single", "fc", "oper"), ("aifs-ens", "cf", "enfo")):
    c = Client(source="ecmwf", model=model)
    for lev in LEVS:
        for step in STEPS:
            c.retrieve(date=DATE, time=0, type=typ, stream=stream, levtype="pl",
                       param=["u", "v"], levelist=lev, step=step,
                       target=f"{model}_{lev}_s{step}.grib2")

def box_uv(path, lev):
    ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
    u, v = ds["u"], ds["v"]
    lat, lon = u.latitude.values, np.where(u.longitude.values < 0,
                                           u.longitude.values + 360, u.longitude.values)
    order = np.argsort(lon)
    mlat = (lat <= 55) & (lat >= 25)
    mlon = (lon[order] >= 150) & (lon[order] <= 250)
    return (u.values[:, order][np.ix_(mlat, mlon)],
            v.values[:, order][np.ix_(mlat, mlon)])

res_km = 111.2 * 0.25 * np.cos(np.deg2rad(40.0))
for lev in LEVS:
    for step in STEPS:
        for model in ("aifs-single", "aifs-ens"):
            u, v = box_uv(f"{model}_{lev}_s{step}.grib2", lev)
            wl, p = ke_spectrum(u, v, res_km=res_km)
            print(f"{lev:>4} hPa day {step//24:>2} {model:>12}: "
                  f"slope {spectral_slope(wl, p, 100, 1000):+.2f} · "
                  f"eff res {effective_resolution(wl, p):.0f} km")
