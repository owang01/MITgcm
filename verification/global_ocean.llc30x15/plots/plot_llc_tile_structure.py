#!/usr/bin/env python3
"""Sanity-check plot: the LLC30 domain decomposition itself -- 5 facets /
13 tiles -- so a human can see the LLC structure (tall facets 1/2, square
Arctic cap facet 3, wide facets 4/5, and how they connect) rather than any
physical field.

Tile numbering confirmed via code/SIZE.h's comment (facet tile counts:
3+3+1+3+3=13, in facet order 1,2,3,4,5) and cross-checked against each
run/ tile's own XC/YC center coordinates:
  facet 1 (tall,  30x90): tiles 1,2,3
  facet 2 (tall,  30x90): tiles 4,5,6
  facet 3 (square,30x30): tile  7   (Arctic cap)
  facet 4 (wide,  90x30): tiles 8,9,10
  facet 5 (wide,  90x30): tiles 11,12,13

Same dateline-seam-masking + corner-coordinate approach as
plot_bathy_sanity.py / plot_T_snapshot.py.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.abspath(os.path.join(HERE, "..", "run"))

NTILES = 13
TILE_NY, TILE_NX = 30, 30

TILE_TO_FACET = {
    1: 1, 2: 1, 3: 1,
    4: 2, 5: 2, 6: 2,
    7: 3,
    8: 4, 9: 4, 10: 4,
    11: 5, 12: 5, 13: 5,
}

FACET_COLORS = {
    1: "#4c72b0",
    2: "#dd8452",
    3: "#55a868",
    4: "#c44e52",
    5: "#8172b2",
}


def read_tile_centers(tile):
    xc = np.fromfile(os.path.join(RUN_DIR, f"XC.{tile:03d}.001.data"), dtype=">f4").reshape(
        TILE_NY, TILE_NX
    )
    yc = np.fromfile(os.path.join(RUN_DIR, f"YC.{tile:03d}.001.data"), dtype=">f4").reshape(
        TILE_NY, TILE_NX
    )
    return xc, yc


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
    # XG/YG model output is the south-west corner of each cell, (ny,nx)
    # extent -- pad on the high-index side. Where a same-facet neighbor
    # exists, use ITS true edge (see Y_STACK_NEXT/X_STACK_NEXT above);
    # only a genuinely open facet boundary uses a linear extrapolation.
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


def dateline_seam_mask(xg):
    corners = np.stack([xg[:-1, :-1], xg[:-1, 1:], xg[1:, :-1], xg[1:, 1:]])
    return (corners.max(axis=0) - corners.min(axis=0)) > 180.0


def robust_center(xc, yc):
    """Median center lon/lat, skipping any dateline-straddling columns so
    tiles 8/9/10 (facet 4, dateline-crossing) still get a sane label spot
    instead of landing on the +/-180 discontinuity."""
    col_span = np.abs(xc[:, 1:] - xc[:, :-1]).max(axis=0) if xc.shape[1] > 1 else np.zeros(1)
    straddles = np.concatenate([[False], col_span > 180.0])
    good_cols = ~straddles
    if good_cols.sum() == 0:
        good_cols[:] = True
    lon = np.median(xc[:, good_cols])
    lat = np.median(yc[:, good_cols])
    return lon, lat


def main():
    fig = plt.figure(figsize=(14, 7))
    ax = plt.axes(projection=ccrs.Robinson())
    ax.set_global()
    ax.coastlines(resolution="110m", linewidth=0.5, color="black", zorder=3)

    legend_handles = {}
    for tile in range(1, NTILES + 1):
        facet = TILE_TO_FACET[tile]
        color = FACET_COLORS[facet]
        xg, yg = read_tile_corners(tile)
        seam = dateline_seam_mask(xg)
        fill = np.ma.masked_where(seam, np.ones((TILE_NY, TILE_NX)))

        cmap = mcolors.ListedColormap([color])
        mesh = ax.pcolormesh(
            xg, yg, fill,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=0, vmax=1,
            shading="flat",
            edgecolors="black",
            linewidth=0.15,
            alpha=0.55,
            zorder=1,
        )
        legend_handles[facet] = mesh

        xc, yc = read_tile_centers(tile)
        lon, lat = robust_center(xc, yc)
        ax.text(
            lon, lat, str(tile),
            transform=ccrs.PlateCarree(),
            ha="center", va="center",
            fontsize=13, fontweight="bold", color="black", zorder=4,
            bbox=dict(boxstyle="circle,pad=0.2", facecolor="white", edgecolor="black", alpha=0.85),
        )

    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=FACET_COLORS[f], edgecolor="black", alpha=0.55, label=f"Facet {f}")
        for f in sorted(FACET_COLORS)
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=9, framealpha=0.9)

    ax.set_title(
        "LLC30 domain decomposition: 5 facets / 13 tiles\n"
        "(tile numbers annotated; color = facet)"
    )

    out_path = os.path.join(HERE, "llc_tile_structure.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")
    print(f"Total facets: {len(FACET_COLORS)}, total tiles: {NTILES}")


if __name__ == "__main__":
    main()
