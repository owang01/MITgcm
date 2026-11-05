#!/usr/bin/env python3
"""Reproduce the ECCOv4 Python Tutorial's native-tile-layout figure
(fields.html's first figure, https://github.com/ECCO-GROUP/ECCOv4-py
figures/llc90_0.png) for LLC30 instead of LLC90.

That figure places the 13 LLC tiles in native tile-index (x,y) space (NOT
geographic lon/lat) in a staircase matching how the tiles are actually
indexed in the model: the two "tall" facets stacked in the lower-left two
columns, the Arctic cap directly above the second column, and the two
"wide" facets fanning out to the upper right. This is exactly the
`tile_order` staircase used by `ecco_v4_py.plot_tiles(layout='llc')`
(~/git/ECCO-GROUP/ECCOv4-py/ecco_v4_py/tile_plot.py), reproduced here
by hand (land mask only, no subplot borders, big axis arrows) to match
the tutorial figure's illustrative style rather than its default
labeled-subplot-grid style.

Tile numbering: ecco_v4_py's tile 0..12 <-> our LLC30 tile001..013:
  ecco tile 0,1,2  = facet 1 (tall)   <-> our tile 1,2,3
  ecco tile 3,4,5  = facet 2 (tall)   <-> our tile 4,5,6
  ecco tile 6      = facet 3 (Arctic) <-> our tile 7
  ecco tile 7,8,9  = facet 4 (wide)   <-> our tile 8,9,10
  ecco tile 10,11,12 = facet 5 (wide) <-> our tile 11,12,13
"""
import os

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.abspath(os.path.join(HERE, "..", "run"))

TILE_NY, TILE_NX, TILE_NZ = 30, 30, 15

# 5x5 grid slot -> ecco tile index (0..12), -1 = empty. Copied verbatim
# from ecco_v4_py.tile_plot.plot_tiles's layout='llc' tile_order.
TILE_ORDER = np.array([
    -1, -1, 10, 11, 12,
    -1,  6,  7,  8,  9,
     2,  5, -1, -1, -1,
     1,  4, -1, -1, -1,
     0,  3, -1, -1, -1,
])
NROWS, NCOLS = 5, 5

# ecco tile index (0-based) -> our LLC30 tile file number (1-based).
ECCO_TO_OURS = {i: i + 1 for i in range(13)}


def read_land_mask(our_tile):
    raw = np.fromfile(os.path.join(RUN_DIR, f"hFacC.{our_tile:03d}.001.data"), dtype=">f4")
    cube = raw.reshape(TILE_NZ, TILE_NY, TILE_NX)
    return cube[0] == 0.0  # True where land


def main():
    fig_size = 6.5
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    cell = 1.0  # one tile occupies a unit cell in the staircase grid
    pad = 0.08  # gap between adjacent tile boxes

    for slot, ecco_tile in enumerate(TILE_ORDER):
        if ecco_tile < 0:
            continue
        our_tile = ECCO_TO_OURS[ecco_tile]
        land = read_land_mask(our_tile)

        # slot -> (col, row) matching plot_tiles' own index math:
        # colnum = ncols-1-i//ncols ; rownum = i % nrows
        colnum = NCOLS - 1 - slot // NCOLS
        rownum = slot % NROWS

        x0 = rownum * cell
        y0 = colnum * cell

        land_img = np.where(land, 1.0, np.nan)
        ax.imshow(
            land_img,
            origin="lower",
            extent=(x0 + pad / 2, x0 + cell - pad / 2, y0 + pad / 2, y0 + cell - pad / 2),
            cmap="Greens",
            vmin=0, vmax=1,
            zorder=2,
        )
        # tile outline box
        ax.plot(
            [x0 + pad / 2, x0 + cell - pad / 2, x0 + cell - pad / 2, x0 + pad / 2, x0 + pad / 2],
            [y0 + pad / 2, y0 + pad / 2, y0 + cell - pad / 2, y0 + cell - pad / 2, y0 + pad / 2],
            color="black", linewidth=0.8, zorder=3,
        )
        ax.text(
            x0 + cell / 2, y0 + cell - pad * 1.5, f"Tile {ecco_tile}",
            ha="center", va="top", fontsize=9, zorder=5,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.75),
        )
        ax.text(
            x0 + cell / 2, y0 + pad * 1.5, f"{TILE_NX}x{TILE_NY}",
            ha="center", va="bottom", fontsize=7.5, color="black", zorder=5,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.75),
        )

    ax.set_xlim(-0.3, NROWS * cell + 0.2)
    ax.set_ylim(-0.3, NCOLS * cell + 0.2)
    ax.set_aspect("equal")
    ax.axis("off")

    # big axis arrows + labels, matching the tutorial figure's style
    ax.annotate(
        "", xy=(NROWS * cell + 0.1, 0), xytext=(-0.25, 0),
        arrowprops=dict(arrowstyle="-|>", linewidth=2, color="black"),
    )
    ax.annotate(
        "", xy=(0, NCOLS * cell + 0.1), xytext=(0, -0.25),
        arrowprops=dict(arrowstyle="-|>", linewidth=2, color="black"),
    )
    ax.text(NROWS * cell / 2, -0.45, "tile x-axis", ha="center", va="top", fontsize=12)
    ax.text(-0.45, NCOLS * cell / 2, "tile y-axis", ha="center", va="center",
            fontsize=12, rotation=90)

    ntiles = int((TILE_ORDER >= 0).sum())
    ngrid = ntiles * TILE_NX * TILE_NY
    ax.set_title(
        f"LLC30: {ntiles} tiles x {TILE_NX}x{TILE_NY} grid cells each "
        f"= {ngrid:,} horizontal grid cells total",
        fontsize=11,
    )

    fig.tight_layout()
    out_path = os.path.join(HERE, "llc30_tile_layout.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
