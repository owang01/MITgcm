#!/usr/bin/env python3
"""Sanity-check plot: surface T (top vertical level) at step 0 vs. step 10
of the LLC30 smoke test run with the REAL V4r6-derived initial condition
(see HANDOFF.md's "V4r6-faithfulness audit" section) -- reassembled from
the 13 per-tile model-output files using each tile's own XG/YG corner
coordinates.

This is the same reassembly method as plot_T_snapshot.py (same per-tile
corner-reading + dateline-seam-masking fix), but against the CURRENT
run/ output, which now reflects the real coarsened V4r6 Theta field at
step 0, not the old flat tRef=20 analytic IC. Written as a new file
(not overwriting plot_T_snapshot.py/T_step0_vs_step10.png) since the
old plot still correctly describes the run it was made from -- it's
just now a plot of a superseded initial condition, not a wrong plot.

Step 0 here should already show real ocean structure (unlike the old
flat-IC plot, which started perfectly uniform at step 0 by construction)
-- that's the main visual difference to look for.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.abspath(os.path.join(HERE, "..", "run"))

NTILES = 13
TILE_NY, TILE_NX, TILE_NZ = 30, 30, 15


# Which tiles stack north-south (y) or west-east (x) within the same LLC
# facet (tile 7, the Arctic cap, has no same-facet neighbor). Needed
# because reading each tile's XG/YG alone only gives its south-west
# corners, (ny,nx) extent -- the missing "+1" north row (y-stacked
# facets) or east column (x-stacked facets) must come from the adjacent
# tile's own true edge, NOT a linear extrapolation: this is a
# curvilinear grid, so extrapolating from a tile's own last two rows/
# columns measurably overshoots/undershoots the true neighboring edge
# (confirmed empirically: up to ~0.36 deg off at the tile2/tile3 seam),
# leaving a thin real gap between tiles that shows up as a faint white
# line through the rendered map -- not a plotting/antialiasing artifact.
Y_STACK_NEXT = {1: 2, 2: 3, 4: 5, 5: 6}   # tile -> its northward same-facet neighbor
X_STACK_NEXT = {9: 10, 11: 12, 12: 13}    # tile -> its eastward same-facet neighbor


def read_tile_corners(tile):
    path = os.path.join(RUN_DIR, f"XG.{tile:03d}.001.data")
    xg = np.fromfile(path, dtype=">f4").reshape(TILE_NY, TILE_NX)
    yg = np.fromfile(os.path.join(RUN_DIR, f"YG.{tile:03d}.001.data"), dtype=">f4").reshape(
        TILE_NY, TILE_NX
    )
    xg_ext = np.pad(xg, ((0, 1), (0, 1)), mode="edge")
    yg_ext = np.pad(yg, ((0, 1), (0, 1)), mode="edge")

    if tile in Y_STACK_NEXT:
        nxg = np.fromfile(
            os.path.join(RUN_DIR, f"XG.{Y_STACK_NEXT[tile]:03d}.001.data"), dtype=">f4"
        ).reshape(TILE_NY, TILE_NX)
        nyg = np.fromfile(
            os.path.join(RUN_DIR, f"YG.{Y_STACK_NEXT[tile]:03d}.001.data"), dtype=">f4"
        ).reshape(TILE_NY, TILE_NX)
        xg_ext[-1, :-1] = nxg[0, :]
        yg_ext[-1, :-1] = nyg[0, :]
    else:
        xg_ext[-1, :-1] = xg[-1, :] + (xg[-1, :] - xg[-2, :])
        yg_ext[-1, :-1] = yg[-1, :] + (yg[-1, :] - yg[-2, :])

    if tile in X_STACK_NEXT:
        nxg = np.fromfile(
            os.path.join(RUN_DIR, f"XG.{X_STACK_NEXT[tile]:03d}.001.data"), dtype=">f4"
        ).reshape(TILE_NY, TILE_NX)
        nyg = np.fromfile(
            os.path.join(RUN_DIR, f"YG.{X_STACK_NEXT[tile]:03d}.001.data"), dtype=">f4"
        ).reshape(TILE_NY, TILE_NX)
        xg_ext[:-1, -1] = nxg[:, 0]
        yg_ext[:-1, -1] = nyg[:, 0]
    else:
        xg_ext[:-1, -1] = xg[:, -1] + (xg[:, -1] - xg[:, -2])
        yg_ext[:-1, -1] = yg[:, -1] + (yg[:, -1] - yg[:, -2])

    xg_ext[-1, -1] = xg_ext[-1, -2] + (xg_ext[-1, -2] - xg_ext[-1, -3])
    yg_ext[-1, -1] = yg_ext[-1, -2] + (yg_ext[-1, -2] - yg_ext[-1, -3])
    return xg_ext, yg_ext


def read_surface_T(step):
    step_str = f"{step:010d}"
    fields = []
    for tile in range(1, NTILES + 1):
        path = os.path.join(RUN_DIR, f"T.{step_str}.{tile:03d}.001.data")
        raw = np.fromfile(path, dtype=">f4")
        assert raw.size == TILE_NY * TILE_NX * TILE_NZ, (
            f"T step {step} tile {tile}: expected "
            f"{TILE_NY * TILE_NX * TILE_NZ}, got {raw.size}"
        )
        cube = raw.reshape(TILE_NZ, TILE_NY, TILE_NX)
        fields.append(cube[0])
    return fields


def dateline_seam_mask(xg):
    corners = np.stack([xg[:-1, :-1], xg[:-1, 1:], xg[1:, :-1], xg[1:, 1:]])
    return (corners.max(axis=0) - corners.min(axis=0)) > 180.0


def plot_panel(ax, surface_T, land_mask_tiles, vmin, vmax, title):
    ax.set_global()
    ax.coastlines(resolution="110m", linewidth=0.5, color="black", zorder=3)
    mesh = None
    for tile in range(1, NTILES + 1):
        xg, yg = read_tile_corners(tile)
        t = surface_T[tile - 1]
        land = land_mask_tiles[tile - 1]
        seam = dateline_seam_mask(xg)
        data = np.ma.masked_where(land | seam, t)
        mesh = ax.pcolormesh(
            xg, yg, data,
            transform=ccrs.PlateCarree(),
            cmap="RdYlBu_r",
            vmin=vmin, vmax=vmax,
            shading="flat",
            zorder=1,
        )
    ax.set_title(title)
    return mesh


def main():
    land_mask_tiles = []
    for tile in range(1, NTILES + 1):
        path = os.path.join(RUN_DIR, f"hFacC.{tile:03d}.001.data")
        raw = np.fromfile(path, dtype=">f4")
        cube = raw.reshape(TILE_NZ, TILE_NY, TILE_NX)
        land_mask_tiles.append(cube[0] == 0.0)

    T0 = read_surface_T(0)
    T10 = read_surface_T(10)

    all_vals = np.concatenate(
        [t[~m].ravel() for t, m in zip(T0 + T10, land_mask_tiles * 2)]
    )
    vmin, vmax = all_vals.min(), all_vals.max()
    print(f"Surface T range across step 0 and step 10: [{vmin:.4f}, {vmax:.4f}] degC")

    fig = plt.figure(figsize=(16, 7))
    ax0 = fig.add_subplot(1, 2, 1, projection=ccrs.Robinson())
    ax1 = fig.add_subplot(1, 2, 2, projection=ccrs.Robinson())

    plot_panel(ax0, T0, land_mask_tiles, vmin, vmax,
               "Surface T, step 0 (1992-01-01)\n(real V4r6-derived IC)")
    mesh = plot_panel(ax1, T10, land_mask_tiles, vmin, vmax, "Surface T, step 10")

    cbar = fig.colorbar(mesh, ax=[ax0, ax1], orientation="horizontal", pad=0.08, shrink=0.5)
    cbar.set_label("Temperature (degC)")
    fig.suptitle("LLC30 smoke test, REAL V4r6-derived IC: surface T, step 0 vs. step 10")

    out_path = os.path.join(HERE, "T_step0_vs_step10_v4r6ic.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")

    max_abs_diff = 0.0
    for t0, t10, land in zip(T0, T10, land_mask_tiles):
        diff = np.abs(t10 - t0)
        diff[land] = 0.0
        max_abs_diff = max(max_abs_diff, diff.max())
    print(f"Max |T(step10) - T(step0)| over wet surface cells: {max_abs_diff:.6f} degC")


if __name__ == "__main__":
    main()
