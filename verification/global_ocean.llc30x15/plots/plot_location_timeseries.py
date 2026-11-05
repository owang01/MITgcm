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
"""
import os

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.abspath(os.path.join(HERE, "..", "run_alldump"))

NTILES = 13
TILE_NY, TILE_NX, TILE_NZ = 30, 30, 15
NSTEPS = 11

LOCATIONS = [
    ("Equatorial Pacific", 0.0, -160.0),
    ("Weddell Sea", -65.0, -45.0),
    ("North Atlantic", 45.0, -30.0),
    ("Mediterranean outflow", 36.0, -6.0),
]
DEPTH_LEVELS = [0, 4, 8]  # surface, 455 m, 1615 m (see RC.data)


def find_tile_cell(lat0, lon0):
    """Return (tile, j, i) of the LLC30 cell nearest (lat0, lon0), using
    each tile's own XC/YC (same lookup as the other plotting scripts)."""
    best = None
    for tile in range(1, NTILES + 1):
        xc = np.fromfile(os.path.join(RUN_DIR, f"XC.{tile:03d}.001.data"), dtype=">f4").reshape(
            TILE_NY, TILE_NX
        )
        yc = np.fromfile(os.path.join(RUN_DIR, f"YC.{tile:03d}.001.data"), dtype=">f4").reshape(
            TILE_NY, TILE_NX
        )
        dist = (yc - lat0) ** 2 + (((xc - lon0 + 180) % 360 - 180)) ** 2
        j, i = np.unravel_index(np.argmin(dist), dist.shape)
        d = dist[j, i]
        if best is None or d < best[0]:
            best = (d, tile, j, i)
    _, tile, j, i = best
    return tile, j, i


def read_timeseries(name, tile, j, i, k):
    vals = []
    for step in range(NSTEPS):
        path = os.path.join(RUN_DIR, f"{name}.{step:010d}.{tile:03d}.001.data")
        raw = np.fromfile(path, dtype=">f4")
        cube = raw.reshape(TILE_NZ, TILE_NY, TILE_NX)
        vals.append(cube[k, j, i])
    return np.array(vals)


def is_wet(tile, j, i, k):
    raw = np.fromfile(os.path.join(RUN_DIR, f"hFacC.{tile:03d}.001.data"), dtype=">f4")
    cube = raw.reshape(TILE_NZ, TILE_NY, TILE_NX)
    return cube[k, j, i] != 0.0


def make_timeseries_plot(name, units, out_filename):
    rc = np.fromfile(os.path.join(RUN_DIR, "RC.data"), dtype=">f4")
    days = np.arange(NSTEPS)

    fig, axes = plt.subplots(len(LOCATIONS), 1, figsize=(9, 3 * len(LOCATIONS)), sharex=True)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(DEPTH_LEVELS)))

    for ax, (label, lat0, lon0) in zip(axes, LOCATIONS):
        tile, j, i = find_tile_cell(lat0, lon0)
        for k, color in zip(DEPTH_LEVELS, colors):
            if not is_wet(tile, j, i, k):
                continue
            series = read_timeseries(name, tile, j, i, k)
            depth_m = -rc[k]
            ax.plot(days, series, "-o", color=color, label=f"{depth_m:.0f} m")
        ax.set_title(f"{label} ({lat0:+.0f},{lon0:+.0f}) -- tile {tile}, j={j}, i={i}",
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
    print("Locations resolved to (tile, j, i):")
    for label, lat0, lon0 in LOCATIONS:
        tile, j, i = find_tile_cell(lat0, lon0)
        print(f"  {label}: tile={tile}, j={j}, i={i}")

    make_timeseries_plot("T", "degC", "T_timeseries_locations.png")
    make_timeseries_plot("S", "psu", "S_timeseries_locations.png")


if __name__ == "__main__":
    main()
