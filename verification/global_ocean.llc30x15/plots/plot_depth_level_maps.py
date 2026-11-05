#!/usr/bin/env python3
"""Sanity-check plots: T and S maps at several depth levels, step 0 vs.
step 10, reassembled from the 13 per-tile model-output files (same
per-tile corner-reading + dateline-seam-masking method as the other
plotting scripts here).

Reads from run_alldump/ (local dumpFreq override -- see
plot_ssh_map.py's docstring and HANDOFF.md for why this separate run
directory exists instead of the committed run/).

Depth levels plotted: surface (k=0), and 3 more spread through the
water column (chosen from RC.data's real cell-center depths, printed at
runtime) -- not just the surface, to check vertical structure too.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.abspath(os.path.join(HERE, "..", "run_alldump"))

NTILES = 13
TILE_NY, TILE_NX, TILE_NZ = 30, 30, 15

# Chosen to spread through the column: surface, upper-mid, mid, deep.
LEVEL_INDICES = [0, 4, 8, 12]


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


def read_field_level(name, step, k):
    step_str = f"{step:010d}"
    fields = []
    for tile in range(1, NTILES + 1):
        path = os.path.join(RUN_DIR, f"{name}.{step_str}.{tile:03d}.001.data")
        raw = np.fromfile(path, dtype=">f4")
        assert raw.size == TILE_NY * TILE_NX * TILE_NZ, (
            f"{name} step {step} tile {tile}: expected "
            f"{TILE_NY * TILE_NX * TILE_NZ}, got {raw.size}"
        )
        cube = raw.reshape(TILE_NZ, TILE_NY, TILE_NX)
        fields.append(cube[k])
    return fields


def read_land_mask_tiles(k):
    land_mask_tiles = []
    for tile in range(1, NTILES + 1):
        raw = np.fromfile(os.path.join(RUN_DIR, f"hFacC.{tile:03d}.001.data"), dtype=">f4")
        cube = raw.reshape(TILE_NZ, TILE_NY, TILE_NX)
        land_mask_tiles.append(cube[k] == 0.0)
    return land_mask_tiles


def dateline_seam_mask(xg):
    corners = np.stack([xg[:-1, :-1], xg[:-1, 1:], xg[1:, :-1], xg[1:, 1:]])
    return (corners.max(axis=0) - corners.min(axis=0)) > 180.0


def plot_panel(ax, field_tiles, land_mask_tiles, vmin, vmax, cmap, title):
    ax.set_global()
    ax.coastlines(resolution="110m", linewidth=0.5, color="black", zorder=3)
    mesh = None
    for tile in range(1, NTILES + 1):
        xg, yg = read_tile_corners(tile)
        f = field_tiles[tile - 1]
        land = land_mask_tiles[tile - 1]
        seam = dateline_seam_mask(xg)
        data = np.ma.masked_where(land | seam, f)
        mesh = ax.pcolormesh(
            xg, yg, data,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            shading="flat",
            antialiased=False,
            zorder=1,
        )
    ax.set_title(title, fontsize=10)
    return mesh


def make_depth_level_plot(name, cmap, units, out_filename):
    rc = np.fromfile(os.path.join(RUN_DIR, "RC.data"), dtype=">f4")

    nlev = len(LEVEL_INDICES)
    fig = plt.figure(figsize=(16, 4 * nlev))
    axes = [
        [fig.add_subplot(nlev, 2, 2 * row + col + 1, projection=ccrs.Robinson())
         for col in range(2)]
        for row in range(nlev)
    ]

    meshes_per_row = [None] * nlev
    for row, k in enumerate(LEVEL_INDICES):
        depth_m = -rc[k]
        f0 = read_field_level(name, 0, k)
        f10 = read_field_level(name, 10, k)
        land_mask_tiles = read_land_mask_tiles(k)

        all_vals = np.concatenate(
            [f[~m].ravel() for f, m in zip(f0 + f10, land_mask_tiles * 2)]
        )
        if all_vals.size == 0:
            print(f"  level {k} (depth {depth_m:.0f} m): no wet cells, skipping")
            continue
        vmin, vmax = all_vals.min(), all_vals.max()
        print(f"  level {k} (depth {depth_m:.0f} m): {name} range "
              f"[{vmin:.3f}, {vmax:.3f}] {units}")

        plot_panel(axes[row][0], f0, land_mask_tiles, vmin, vmax, cmap,
                   f"{name}, depth {depth_m:.0f} m, step 0")
        mesh = plot_panel(axes[row][1], f10, land_mask_tiles, vmin, vmax, cmap,
                           f"{name}, depth {depth_m:.0f} m, step 10")
        meshes_per_row[row] = mesh

    fig.suptitle(f"LLC30 smoke test, real V4r6-derived IC: {name} at several depths, "
                 f"step 0 vs. step 10", y=1.0)
    # Reserve the right margin for colorbars BEFORE reading axes positions
    # below -- get_position() reflects the layout as of the last draw, so
    # subplots_adjust must run first, or a stale pre-adjustment bbox gets
    # used and the colorbar ends up overlapping the right-column panel
    # (the original bug: fig.colorbar(mesh, ax=axes[row]) stole width from
    # that row's right-hand axis instead of using separately reserved
    # margin space).
    fig.subplots_adjust(left=0.02, right=0.90, top=0.95, bottom=0.02,
                         wspace=0.05, hspace=0.3)

    divider_pad = 0.015
    for row, mesh in enumerate(meshes_per_row):
        if mesh is None:
            continue
        bbox = axes[row][1].get_position()
        cax = fig.add_axes([bbox.x1 + divider_pad, bbox.y0 + 0.1 * bbox.height,
                             0.012, 0.8 * bbox.height])
        cbar = fig.colorbar(mesh, cax=cax, orientation="vertical")
        cbar.set_label(f"{name} ({units})", fontsize=8)

    out_path = os.path.join(HERE, out_filename)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"Wrote {out_path}")


def main():
    print("Theta depth-level maps:")
    make_depth_level_plot("T", "RdYlBu_r", "degC", "T_depth_levels.png")
    print("Salt depth-level maps:")
    make_depth_level_plot("S", "viridis", "psu", "S_depth_levels.png")


if __name__ == "__main__":
    main()
