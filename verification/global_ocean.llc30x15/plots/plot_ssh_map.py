#!/usr/bin/env python3
"""Sanity-check plot: sea surface height (Eta) maps at step 0 vs. step 10,
reassembled from the 13 per-tile model-output files using each tile's own
XG/YG corner coordinates (same method/gotcha-fix as the other plotting
scripts in this directory).

Reads from run_alldump/ (a separate run with dumpFreq overridden to dump
every step -- see run_alldump/data's local-override comment and
HANDOFF.md -- NOT the committed run/, which only has step-0/step-10
output under this config's committed dumpFreq). Physics/IC are identical
to run/; only output frequency differs.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.abspath(os.path.join(HERE, "..", "run_alldump"))

NTILES = 13
TILE_NY, TILE_NX = 30, 30


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
    xg = np.fromfile(os.path.join(RUN_DIR, f"XG.{tile:03d}.001.data"), dtype=">f4").reshape(
        TILE_NY, TILE_NX
    )
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


def read_eta(step):
    step_str = f"{step:010d}"
    fields = []
    for tile in range(1, NTILES + 1):
        path = os.path.join(RUN_DIR, f"Eta.{step_str}.{tile:03d}.001.data")
        raw = np.fromfile(path, dtype=">f4")
        assert raw.size == TILE_NY * TILE_NX, (
            f"Eta step {step} tile {tile}: expected {TILE_NY * TILE_NX}, got {raw.size}"
        )
        fields.append(raw.reshape(TILE_NY, TILE_NX))
    return fields


def dateline_seam_mask(xg):
    corners = np.stack([xg[:-1, :-1], xg[:-1, 1:], xg[1:, :-1], xg[1:, 1:]])
    return (corners.max(axis=0) - corners.min(axis=0)) > 180.0


def read_land_mask_tiles():
    land_mask_tiles = []
    for tile in range(1, NTILES + 1):
        raw = np.fromfile(os.path.join(RUN_DIR, f"hFacC.{tile:03d}.001.data"), dtype=">f4")
        cube = raw.reshape(15, TILE_NY, TILE_NX)
        land_mask_tiles.append(cube[0] == 0.0)
    return land_mask_tiles


def plot_panel(ax, eta_tiles, land_mask_tiles, vmin, vmax, title):
    ax.set_global()
    ax.coastlines(resolution="110m", linewidth=0.5, color="black", zorder=3)
    mesh = None
    for tile in range(1, NTILES + 1):
        xg, yg = read_tile_corners(tile)
        e = eta_tiles[tile - 1]
        land = land_mask_tiles[tile - 1]
        seam = dateline_seam_mask(xg)
        data = np.ma.masked_where(land | seam, e)
        mesh = ax.pcolormesh(
            xg, yg, data,
            transform=ccrs.PlateCarree(),
            cmap="RdBu_r",
            vmin=vmin, vmax=vmax,
            shading="flat",
            zorder=1,
        )
    ax.set_title(title)
    return mesh


def main():
    land_mask_tiles = read_land_mask_tiles()
    eta0 = read_eta(0)
    eta10 = read_eta(10)

    all_vals = np.concatenate(
        [e[~m].ravel() for e, m in zip(eta0 + eta10, land_mask_tiles * 2)]
    )
    vmax = np.abs(all_vals).max()
    vmin = -vmax
    print(f"Eta range across step 0 and step 10: [{all_vals.min():.4f}, {all_vals.max():.4f}] m")

    fig = plt.figure(figsize=(16, 7))
    ax0 = fig.add_subplot(1, 2, 1, projection=ccrs.Robinson())
    ax1 = fig.add_subplot(1, 2, 2, projection=ccrs.Robinson())

    plot_panel(ax0, eta0, land_mask_tiles, vmin, vmax, "SSH (Eta), step 0 (1992-01-01)")
    mesh = plot_panel(ax1, eta10, land_mask_tiles, vmin, vmax, "SSH (Eta), step 10")

    cbar = fig.colorbar(mesh, ax=[ax0, ax1], orientation="horizontal", pad=0.08, shrink=0.5)
    cbar.set_label("Sea surface height (m)")
    fig.suptitle("LLC30 smoke test, real V4r6-derived IC: SSH, step 0 vs. step 10")

    out_path = os.path.join(HERE, "ssh_map.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
