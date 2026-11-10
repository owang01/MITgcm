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

Phase A (adjoint-experiment plan, see HANDOFF.md): this config now sets
useSingleCpuIO=.TRUE. (matching real V4r6), so run/ writes one GLOBAL
file per field instead of 13 per-tile files. Per-tile corner/center
grids are now sliced directly out of each facet's own GLOBAL (ny+1,nx+1)
extended-corner array (built once per facet, see
plot_ssh_map.py's extended_facet_corners design) -- since tiles within a
facet are contiguous, this gives every internal tile-to-tile edge for
free (no neighbor-tile lookup needed at all, unlike the old per-tile
Y_STACK_NEXT/X_STACK_NEXT workaround); only each FACET's own outer edge
still needs the facet-to-facet neighbor-borrow / extrapolation logic.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.abspath(os.path.join(HERE, "..", "run"))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "preprocess")))
import coarsen_llc90_to_llc30 as base  # noqa: E402

NTILES = 13
TILE_NY, TILE_NX = 30, 30

FACET_DIMS = {
    1: (90, 30),
    2: (90, 30),
    3: (30, 30),
    4: (30, 90),
    5: (30, 90),
}
FACET_TILE_RANGE = {1: (1, 3), 2: (4, 6), 3: (7, 7), 4: (8, 10), 5: (11, 13)}
FACET_STACK_AXIS = {1: "y", 2: "y", 3: None, 4: "x", 5: "x"}

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


def read_global_facets(path):
    """Read a GLOBAL run/<name>.data 2-D field (real*4 big-endian) and
    split into its 5 native-shaped facets -- flat concatenation of each
    facet's own native (ny,nx) row-major block, same convention as
    plot_bathy_sanity.py's read_bathy_facets."""
    raw = np.fromfile(path, dtype=">f4")
    per_level = 390 * 30
    assert raw.size == per_level, (path, raw.size, per_level)
    facets = {}
    offset = 0
    for facet in (1, 2, 3, 4, 5):
        ny, nx = FACET_DIMS[facet]
        n = ny * nx
        facets[facet] = raw[offset:offset + n].reshape(ny, nx)
        offset += n
    assert offset == per_level
    return facets


def _edge_adjacent_line(arr, edge):
    if edge == "N":
        return arr[-1, :]
    if edge == "S":
        return arr[0, :]
    if edge == "E":
        return arr[:, -1]
    if edge == "W":
        return arr[:, 0]
    raise ValueError(edge)


def extended_facet_corners(xg_facets, yg_facets, facet):
    """See plot_ssh_map.py's identical helper for the full design
    rationale -- returns (ny+1,nx+1) corner arrays spanning the WHOLE
    facet."""
    xg, yg = xg_facets[facet], yg_facets[facet]
    xg_ext = np.pad(xg, ((0, 1), (0, 1)), mode="edge")
    yg_ext = np.pad(yg, ((0, 1), (0, 1)), mode="edge")

    res_n = base._facet_neighbor(facet, "N")
    if res_n is None:
        xg_ext[-1, :-1] = xg[-1, :] + (xg[-1, :] - xg[-2, :])
        yg_ext[-1, :-1] = yg[-1, :] + (yg[-1, :] - yg[-2, :])
    else:
        nf, ne, rev = res_n
        nxg_line = _edge_adjacent_line(xg_facets[nf], ne)
        nyg_line = _edge_adjacent_line(yg_facets[nf], ne)
        if rev:
            nxg_line = nxg_line[::-1]
            nyg_line = nyg_line[::-1]
        xg_ext[-1, :-1] = nxg_line
        yg_ext[-1, :-1] = nyg_line

    res_e = base._facet_neighbor(facet, "E")
    if res_e is None:
        xg_ext[:-1, -1] = xg[:, -1] + (xg[:, -1] - xg[:, -2])
        yg_ext[:-1, -1] = yg[:, -1] + (yg[:, -1] - yg[:, -2])
    else:
        nf, ne, rev = res_e
        nxg_line = _edge_adjacent_line(xg_facets[nf], ne)
        nyg_line = _edge_adjacent_line(yg_facets[nf], ne)
        if rev:
            nxg_line = nxg_line[::-1]
            nyg_line = nyg_line[::-1]
        xg_ext[:-1, -1] = nxg_line
        yg_ext[:-1, -1] = nyg_line

    xg_ext[-1, -1] = xg_ext[-1, -2] + (xg_ext[-1, -2] - xg_ext[-1, -3])
    yg_ext[-1, -1] = yg_ext[-1, -2] + (yg_ext[-1, -2] - yg_ext[-1, -3])
    return xg_ext, yg_ext


def tile_corner_block(ext_corners, facet, tile):
    """ext_corners: (ny+1,nx+1) full-facet extended corner array. Returns
    the (31,31) corner sub-block for `tile` -- tiles within a facet are
    contiguous along FACET_STACK_AXIS, so adjacent tiles' corner blocks
    naturally share their boundary row/column (a 1-cell overlap), no
    neighbor lookup needed."""
    t0, t1 = FACET_TILE_RANGE[facet]
    axis = FACET_STACK_AXIS[facet]
    if axis is None:
        return ext_corners
    i = tile - t0
    if axis == "y":
        return ext_corners[i * TILE_NY:(i + 1) * TILE_NY + 1, :]
    return ext_corners[:, i * TILE_NX:(i + 1) * TILE_NX + 1]


def tile_center_block(center_facet, facet, tile):
    t0, t1 = FACET_TILE_RANGE[facet]
    axis = FACET_STACK_AXIS[facet]
    if axis is None:
        return center_facet
    i = tile - t0
    if axis == "y":
        return center_facet[i * TILE_NY:(i + 1) * TILE_NY, :]
    return center_facet[:, i * TILE_NX:(i + 1) * TILE_NX]


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
    xg_facets = read_global_facets(os.path.join(RUN_DIR, "XG.data"))
    yg_facets = read_global_facets(os.path.join(RUN_DIR, "YG.data"))
    xc_facets = read_global_facets(os.path.join(RUN_DIR, "XC.data"))
    yc_facets = read_global_facets(os.path.join(RUN_DIR, "YC.data"))

    ext_corners = {
        f: extended_facet_corners(xg_facets, yg_facets, f) for f in (1, 2, 3, 4, 5)
    }

    fig = plt.figure(figsize=(14, 7))
    ax = plt.axes(projection=ccrs.Robinson())
    ax.set_global()
    ax.coastlines(resolution="110m", linewidth=0.5, color="black", zorder=3)

    legend_handles = {}
    for tile in range(1, NTILES + 1):
        facet = TILE_TO_FACET[tile]
        color = FACET_COLORS[facet]
        xg_ext, yg_ext = ext_corners[facet]
        xg = tile_corner_block(xg_ext, facet, tile)
        yg = tile_corner_block(yg_ext, facet, tile)
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

        xc = tile_center_block(xc_facets[facet], facet, tile)
        yc = tile_center_block(yc_facets[facet], facet, tile)
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
