#!/usr/bin/env python3
"""Sanity-check plot: T and S time series at several fixed locations and
depths, over all 11 steps of the smoke test (model days 0-10).

Reads from run_alldump/ (local dumpFreq override -- see
plot_ssh_map.py's docstring and HANDOFF.md for why this separate run
directory exists; the committed run/ only has step-0/step-10 output).

Locations chosen to span different regimes: equatorial Pacific (warm,
well-mixed), Weddell Sea (cold, polar), North Atlantic (mid-latitude,
seasonally forced), and the Mediterranean outflow region visible in
plot_depth_level_maps.py's S maps (distinct salty signature at depth).

Phase A (adjoint-experiment plan, see HANDOFF.md): this config now sets
useSingleCpuIO=.TRUE. (matching real V4r6), so run_alldump/ writes one
GLOBAL file per field instead of 13 per-tile files -- the nearest-cell
lookup now searches each facet's own (ny,nx) XC/YC array directly (no
tile-corner stitching needed for a single-point lookup/timeseries).
"""
import os

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.abspath(os.path.join(HERE, "..", "run_alldump"))

NSTEPS = 11
NLEV = 15

FACET_DIMS = {
    1: (90, 30),
    2: (90, 30),
    3: (30, 30),
    4: (30, 90),
    5: (30, 90),
}

LOCATIONS = [
    ("Equatorial Pacific", 0.0, -160.0),
    ("Weddell Sea", -65.0, -45.0),
    ("North Atlantic", 45.0, -30.0),
    ("Mediterranean outflow", 36.0, -6.0),
]
DEPTH_LEVELS = [0, 4, 8]  # surface, 455 m, 1615 m (see RC.data)


def read_global_facets(path, nlev=None):
    """Read a GLOBAL run_alldump/<name>.data file (real*4 big-endian) and
    split into its 5 native-shaped facets -- flat concatenation of each
    facet's own native (ny,nx) (or (nlev,ny,nx)) row-major block, same
    convention as plot_bathy_sanity.py's read_bathy_facets."""
    raw = np.fromfile(path, dtype=">f4")
    per_level = 390 * 30
    if nlev is None:
        assert raw.size == per_level, (path, raw.size, per_level)
        levels = raw.reshape(1, per_level)
    else:
        assert raw.size == nlev * per_level, (path, raw.size, nlev, per_level)
        levels = raw.reshape(nlev, per_level)

    facets = {}
    offset = 0
    for facet in (1, 2, 3, 4, 5):
        ny, nx = FACET_DIMS[facet]
        n = ny * nx
        block = levels[:, offset:offset + n].reshape((levels.shape[0], ny, nx))
        facets[facet] = block[0] if nlev is None else block
        offset += n
    assert offset == per_level
    return facets


def find_facet_cell(xc_facets, yc_facets, lat0, lon0):
    """Return (facet, j, i) of the LLC30 cell nearest (lat0, lon0),
    searching each facet's own GLOBAL XC/YC array."""
    best = None
    for facet in (1, 2, 3, 4, 5):
        xc, yc = xc_facets[facet], yc_facets[facet]
        dist = (yc - lat0) ** 2 + (((xc - lon0 + 180) % 360 - 180)) ** 2
        j, i = np.unravel_index(np.argmin(dist), dist.shape)
        d = dist[j, i]
        if best is None or d < best[0]:
            best = (d, facet, j, i)
    _, facet, j, i = best
    return facet, j, i


def read_timeseries(name, facet, j, i, k):
    vals = []
    for step in range(NSTEPS):
        path = os.path.join(RUN_DIR, f"{name}.{step:010d}.data")
        facets = read_global_facets(path, nlev=NLEV)
        vals.append(facets[facet][k, j, i])
    return np.array(vals)


def is_wet(hfacc_facets, facet, j, i, k):
    return hfacc_facets[facet][k, j, i] != 0.0


def make_timeseries_plot(xc_facets, yc_facets, hfacc_facets, name, units, out_filename):
    rc = np.fromfile(os.path.join(RUN_DIR, "RC.data"), dtype=">f4")
    days = np.arange(NSTEPS)

    fig, axes = plt.subplots(len(LOCATIONS), 1, figsize=(9, 3 * len(LOCATIONS)), sharex=True)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(DEPTH_LEVELS)))

    for ax, (label, lat0, lon0) in zip(axes, LOCATIONS):
        facet, j, i = find_facet_cell(xc_facets, yc_facets, lat0, lon0)
        for k, color in zip(DEPTH_LEVELS, colors):
            if not is_wet(hfacc_facets, facet, j, i, k):
                continue
            series = read_timeseries(name, facet, j, i, k)
            depth_m = -rc[k]
            ax.plot(days, series, "-o", color=color, label=f"{depth_m:.0f} m")
        ax.set_title(f"{label} ({lat0:+.0f},{lon0:+.0f}) -- facet {facet}, j={j}, i={i}",
                     fontsize=10)
        ax.set_ylabel(f"{name} ({units})")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Model time (days since 1992-01-01)")
    axes[-1].set_xticks(days)
    fig.suptitle(f"LLC30 smoke test, real V4r6-derived IC: {name} time series at fixed locations")
    fig.tight_layout()

    out_path = os.path.join(HERE, out_filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")


def main():
    xc_facets = read_global_facets(os.path.join(RUN_DIR, "XC.data"))
    yc_facets = read_global_facets(os.path.join(RUN_DIR, "YC.data"))
    hfacc_facets = read_global_facets(os.path.join(RUN_DIR, "hFacC.data"), nlev=NLEV)

    print("Locations resolved to (facet, j, i):")
    for label, lat0, lon0 in LOCATIONS:
        facet, j, i = find_facet_cell(xc_facets, yc_facets, lat0, lon0)
        print(f"  {label}: facet={facet}, j={j}, i={i}")

    make_timeseries_plot(xc_facets, yc_facets, hfacc_facets, "T", "degC", "T_timeseries_locations.png")
    make_timeseries_plot(xc_facets, yc_facets, hfacc_facets, "S", "psu", "S_timeseries_locations.png")


if __name__ == "__main__":
    main()
