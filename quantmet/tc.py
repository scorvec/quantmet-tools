"""Tropical cyclones in ensemble forecasts: ECMWF track decoding, recurvature, the outflow-jet interaction metric, and
Hart's cyclone phase space.

Tracks. ECMWF disseminates its tropical-cyclone tracker output as BUFR (open data: ``...-enfo-tf.bufr`` for the
ensembles, ``...-oper-tf.bufr`` for the deterministic runs). One message per storm, one subset per ensemble member;
the ``ensembleForecastType`` is 1 for the unperturbed control, 4 for perturbed members, 0 for the high-resolution
deterministic run carried in the same file (excluded here). The analysis position is ``#2#latitude`` (falling back to
the observed ``#1#latitude``); step i of the delayed replication is ``#i#timePeriod`` with the centre at
``#(2i+2)#latitude`` and MSLP / max wind at ``#(i+1)#``. The tracker also repeats named storms under placeholder
genesis ids (70-99 + basin letter); de-duplicate by position if you merge them.

Recurvature: the lead where the storm's 24-h zonal motion turns from westward (<= -1 m/s) to eastward (>= +2 m/s).
A storm heading north or east from the start has none (it never had a westward phase in the forecast).

Outflow metric (Archambault, Bosart, Keyser & Cordeira 2013, MWR 141; Archambault et al. 2015, MWR 143): negative PV
advection by the irrotational wind, -v_chi . grad(PV), in the 300-200 hPa layer. The divergent outflow of a recurving
storm pushes low-PV air poleward against the PV gradient at the jet, building the downstream ridge. ``outflow_index``
is the area mean of the negative part within 500 km of the centre, sign flipped (PVU/day; 0 = no interaction).
Numerics: v_chi from a spherical-harmonic inversion (quantmet.helmholtz) is smooth by construction, while the PV
gradient on a 0.25 deg grid is grid-noisy -- smooth PV (~1 deg) before differentiating, or the contours speckle.

Hart (2003, MWR 131) phase space, within 500 km of the centre:
  B     = (Z_top - Z_bottom) thickness right of the storm motion minus left (m, NH sign; negate in the SH).
          |B| ~ 0 thermally symmetric (tropical); B > 10 m marks extratropical-transition ONSET (Evans & Hart 2003).
  -VT_L = d(dZ)/d(ln p) over 900-600 hPa, dZ = Zmax - Zmin in the disc; > 0 warm core, < 0 cold core.
  -VT_U = the same over 600-300 hPa.  ET is COMPLETE when -VT_L turns negative after onset.
Hart used 25 hPa levels; with fewer levels (open data: 925/850/700/600/500/400/300) the fit is coarser -- say so.
Parameters are conventionally smoothed with a 24-h running mean.

Powers the tropical-cyclone cards at https://scorvec.com/circulation.html.

Requires: numpy; eccodes for ``decode_tracks``; pyshtools + xarray for ``irrotational_on_grid``.
"""
from __future__ import annotations

import numpy as np

A_EARTH = 6.371e6
MISSING = -1e99
__all__ = ["decode_tracks", "at", "great_circle_km", "within", "recurvature", "irrotational_on_grid",
           "pv_advection", "outflow_index", "hart_parameters", "phase_series", "et_times"]


# ── tracks ────────────────────────────────────────────────────────────────────────────────────────────────────────

def decode_tracks(path) -> list[dict]:
    """All storms in an ECMWF tf BUFR file: [{id, name, named, tracks: {member: track}}]; member 0 = control,
    1..N perturbed; track = dict(steps (h), lat, lon (-180..180), pmsl (hPa), wind (m/s)). The deterministic subset
    (ensembleForecastType 0) is excluded."""
    import eccodes as ec
    out = []
    with open(path, "rb") as f:
        while True:
            h = ec.codes_bufr_new_from_file(f)
            if h is None:
                break
            try:
                ec.codes_set(h, "unpack", 1)
                sid = ec.codes_get(h, "#1#stormIdentifier").strip()
                name = ec.codes_get(h, "#1#longStormName").strip()
                ft = np.atleast_1d(ec.codes_get_array(h, "#1#ensembleForecastType"))
                mem = np.atleast_1d(ec.codes_get_array(h, "#1#ensembleMemberNumber"))
                n = len(ft)

                def arr(key):
                    a = np.atleast_1d(ec.codes_get_array(h, key)).astype(float)
                    return np.repeat(a, n) if a.size == 1 else a

                rep = ec.codes_get(h, "#1#delayedDescriptorReplicationFactor")
                a_la, a_lo = arr("#2#latitude"), arr("#2#longitude")
                o_la, o_lo = arr("#1#latitude"), arr("#1#longitude")
                a_la = np.where(a_la > MISSING, a_la, o_la); a_lo = np.where(a_lo > MISSING, a_lo, o_lo)
                cols = [(np.zeros(n), a_la, a_lo, arr("#1#pressureReducedToMeanSeaLevel"), arr("#1#windSpeedAt10M"))]
                for i in range(1, rep + 1):
                    cols.append((arr(f"#{i}#timePeriod"), arr(f"#{2 * i + 2}#latitude"), arr(f"#{2 * i + 2}#longitude"),
                                 arr(f"#{i + 1}#pressureReducedToMeanSeaLevel"), arr(f"#{i + 1}#windSpeedAt10M")))
                tracks = {}
                for j in range(n):
                    if ft[j] not in (1, 4):
                        continue
                    rows = [(c[0][j], c[1][j], c[2][j], c[3][j], c[4][j]) for c in cols
                            if c[1][j] > MISSING and c[0][j] > MISSING]
                    if not rows:
                        continue
                    st, la, lo, p, w = (np.array(x, float) for x in zip(*rows))
                    m = 0 if ft[j] == 1 else int(mem[j])
                    tracks[m] = dict(steps=st.astype(int), lat=la, lon=(lo + 180) % 360 - 180,
                                     pmsl=np.where(p > MISSING, p / 100.0, np.nan),
                                     wind=np.where(w > MISSING, w, np.nan))
                if tracks:
                    named = not (sid[:2].isdigit() and int(sid[:2]) >= 70)
                    out.append(dict(id=sid, name=name, named=named, tracks=tracks))
            finally:
                ec.codes_release(h)
    return out


def at(track: dict, h: int):
    """(lat, lon) of the track at lead h, or None."""
    k = np.where(np.asarray(track["steps"]) == h)[0]
    return (float(track["lat"][k[0]]), float(track["lon"][k[0]])) if len(k) else None


def great_circle_km(la1, lo1, la2, lo2):
    p1, p2, dl = np.deg2rad(la1), np.deg2rad(la2), np.deg2rad(np.asarray(lo2) - np.asarray(lo1))
    c = np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl)
    return A_EARTH / 1000 * np.arccos(np.clip(c, -1, 1))


def within(lat, lon, clat, clon, r_km) -> np.ndarray:
    """(lat, lon) boolean mask of grid points within r_km of (clat, clon)."""
    return great_circle_km(np.asarray(lat, float)[:, None], np.asarray(lon, float)[None, :], clat, clon) <= r_km


def recurvature(track: dict, west: float = -1.0, east: float = 2.0):
    """Lead (h) of the last westward-moving point (24-h zonal motion <= west m/s) before the motion reaches >= east
    m/s; None if the track never turns. Needs steps every 12 h or finer."""
    st = np.asarray(track["steps"], float); la = np.asarray(track["lat"], float)
    lo = np.rad2deg(np.unwrap(np.deg2rad(np.asarray(track["lon"], float))))
    if len(st) < 5:
        return None
    u = np.full(len(st), np.nan)
    for k in range(len(st)):
        a = np.where(np.abs(st - (st[k] - 12)) < 1e-6)[0]; b = np.where(np.abs(st - (st[k] + 12)) < 1e-6)[0]
        if len(a) and len(b):
            u[k] = (lo[b[0]] - lo[a[0]]) * 111.2e3 * np.cos(np.deg2rad(la[k])) / 86400.0
    w = np.where(u <= west)[0]
    if not len(w):
        return None
    e = np.where((u >= east) & (np.arange(len(u)) > w[0]))[0]
    if not len(e):
        return None
    k0 = int(np.where((u <= 0) & (np.arange(len(u)) < e[0]))[0][-1])
    return int(st[k0])


# ── outflow metric ────────────────────────────────────────────────────────────────────────────────────────────────

def irrotational_on_grid(u2d, v2d, lat, lon, lmax: int = 106):
    """(u_chi, v_chi) from global (lat, lon) xarray DataArrays u2d, v2d (dims latitude/longitude), interpolated to the
    target (lat, lon) grid (degrees; lon in 0..360). Uses quantmet.helmholtz (spherical-harmonic inversion, T lmax)."""
    import xarray as xr
    from .helmholtz import irrotational_wind, velocity_potential
    chi, dlat, dlon_ = velocity_potential(u2d, v2d, lmax=lmax)
    uc, vc = irrotational_wind(chi, dlat, dlon_)
    dl = np.asarray(dlon_, float)
    if dl[-1] < 360.0 - 1e-6:
        dl = np.r_[dl, dl[0] + 360.0]
        uc = np.concatenate([uc, uc[:, :1]], axis=1); vc = np.concatenate([vc, vc[:, :1]], axis=1)
    dla = np.asarray(dlat, float)
    keep = np.r_[True, np.diff(dla) != 0]
    U = xr.DataArray(uc[keep], dims=("lat", "lon"), coords={"lat": dla[keep], "lon": dl}).sortby("lat")
    V = xr.DataArray(vc[keep], dims=("lat", "lon"), coords={"lat": dla[keep], "lon": dl}).sortby("lat")
    tgt = dict(lat=xr.DataArray(np.asarray(lat, float), dims="y"), lon=xr.DataArray(np.asarray(lon, float) % 360, dims="x"))
    return U.interp(**tgt).values, V.interp(**tgt).values


def pv_advection(uchi, vchi, pv_layer, lat, lon, sigma_deg: float = 1.0) -> np.ndarray:
    """-v_chi . grad(PV) in PVU/day on a regular (lat, lon) grid (degrees); PV in PVU, Gaussian-smoothed by sigma_deg
    first (0 = no smoothing). Longitude periodic."""
    from scipy.ndimage import gaussian_filter
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    pv = np.nan_to_num(np.asarray(pv_layer, float), nan=0.0)
    if sigma_deg > 0:
        n = sigma_deg / abs(lat[1] - lat[0])
        pv = gaussian_filter(pv, sigma=(n, n), mode=("nearest", "wrap"))
    phi = np.deg2rad(lat); lam = np.deg2rad(lon)
    cosphi = np.clip(np.cos(phi), 1e-3, None)[:, None]
    dpdx = (np.roll(pv, -1, -1) - np.roll(pv, 1, -1)) / (2 * (lam[1] - lam[0])) / (A_EARTH * cosphi)
    dpdy = np.gradient(pv, phi, axis=0) / A_EARTH
    return -(uchi * dpdx + vchi * dpdy) * 86400.0


def outflow_index(adv, lat, lon, clat, clon, r_km: float = 500.0) -> float:
    """Area (cos lat) mean of max(-adv, 0) within r_km of (clat, clon), PVU/day."""
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    j = np.where(np.abs(lat - clat) <= r_km / 111.0 + 0.5)[0]
    if not len(j):
        return np.nan
    m = within(lat[j], lon, clat, clon, r_km)
    if not m.any():
        return np.nan
    w = np.broadcast_to(np.cos(np.deg2rad(lat[j]))[:, None], m.shape)[m]
    return float(np.sum(np.maximum(-np.asarray(adv)[j][m], 0) * w) / np.sum(w))


# ── Hart phase space ──────────────────────────────────────────────────────────────────────────────────────────────

def _slope(dz, levs):
    return float(np.polyfit(np.log(np.asarray(levs, float)), np.asarray(dz, float), 1)[0])


def hart_parameters(z, levs, lat, lon, clat, clon, motion_deg, r_km: float = 500.0,
                    low=(900, 850, 800, 750, 700, 650, 600), upp=(600, 550, 500, 450, 400, 350, 300),
                    thick=(600, 900)) -> dict:
    """B, -VT_L, -VT_U (m) for one time.

    z : geopotential HEIGHT (m), (lev, lat, lon); levs : hPa for axis 0; motion_deg : storm heading, degrees clockwise
    from north. ``low``/``upp``/``thick`` are matched to the available levels (missing ones are dropped; at least two
    per layer are needed). Southern-hemisphere storms: negate B (Hart's convention keeps B > 0 = frontal)."""
    levs = [int(round(x)) for x in levs]
    low = [p for p in low if p in levs]; upp = [p for p in upp if p in levs]
    if len(low) < 2 or len(upp) < 2:
        raise ValueError("need at least two levels in each layer")
    top, bot = thick
    if top not in levs or bot not in levs:
        bot = max(low); top = min(low)
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    dl = r_km / 111.0 + 0.5
    j = np.where(np.abs(lat - clat) <= dl)[0]
    dlon = ((lon - clon + 180) % 360) - 180
    i = np.where(np.abs(dlon) <= dl / max(np.cos(np.deg2rad(min(abs(clat) + dl, 80))), 0.2))[0]
    la = np.deg2rad(lat[j])[:, None]; lo = np.deg2rad(lon[i])[None, :]
    c0, l0 = np.deg2rad(clat), np.deg2rad(clon)
    m = great_circle_km(lat[j][:, None], lon[i][None, :], clat, clon) <= r_km
    brg = np.arctan2(np.sin(lo - l0) * np.cos(la), np.cos(c0) * np.sin(la) - np.sin(c0) * np.cos(la) * np.cos(lo - l0))
    rel = ((brg - np.deg2rad(motion_deg) + np.pi) % (2 * np.pi)) - np.pi
    right, left = m & (rel > 0) & (rel < np.pi), m & (rel < 0) & (rel > -np.pi)
    Z = np.asarray(z, float)[:, j][:, :, i]
    th = Z[levs.index(top)] - Z[levs.index(bot)]
    B = float(th[right].mean() - th[left].mean()) if right.any() and left.any() else np.nan
    dz = {p: float(Z[levs.index(p)][m].max() - Z[levs.index(p)][m].min()) for p in set(low) | set(upp)}
    return dict(B=B, VTL=_slope([dz[p] for p in low], low), VTU=_slope([dz[p] for p in upp], upp))


def phase_series(z_steps, levs, lat, lon, track: dict, steps, smooth_h: float = 24.0, **kw) -> dict:
    """Hart parameters along a track: z_steps (step, lev, lat, lon) at the given steps (h). Storm motion from the
    neighbouring track points; a ``smooth_h`` running mean (Hart: 24 h). Returns lists h, lat, lon, B, VTL, VTU."""
    out = {"h": [], "lat": [], "lon": [], "B": [], "VTL": [], "VTU": []}
    ts = np.asarray(track["steps"])
    for k, h in enumerate(steps):
        pos = at(track, int(h))
        if pos is None:
            continue
        kk = np.where(ts == h)[0][0]
        a, b = max(kk - 1, 0), min(kk + 1, len(ts) - 1)
        if a == b:
            continue
        dlo = ((track["lon"][b] - track["lon"][a] + 180) % 360) - 180
        mot = np.rad2deg(np.arctan2(np.deg2rad(dlo) * np.cos(np.deg2rad(pos[0])),
                                    np.deg2rad(track["lat"][b] - track["lat"][a])))
        r = hart_parameters(z_steps[k], levs, lat, lon, pos[0], pos[1], mot, **kw)
        if pos[0] < 0:
            r["B"] = -r["B"]
        out["h"].append(int(h)); out["lat"].append(pos[0]); out["lon"].append(pos[1])
        for key in ("B", "VTL", "VTU"):
            out[key].append(r[key])
    if smooth_h:
        hh = np.array(out["h"], float)
        for key in ("B", "VTL", "VTU"):
            v = np.array(out[key], float)
            out[key] = [float(np.nanmean(v[np.abs(hh - x) <= smooth_h / 2])) for x in hh]
    return out


def et_times(series: dict, b_onset: float = 10.0):
    """Evans & Hart (2003): onset = first B > b_onset; completion = first -VT_L < 0 at or after onset."""
    onset = next((h for h, b in zip(series["h"], series["B"]) if b > b_onset), None)
    comp = next((h for h, v in zip(series["h"], series["VTL"]) if v < 0 and (onset is None or h >= onset)), None)
    return onset, comp
