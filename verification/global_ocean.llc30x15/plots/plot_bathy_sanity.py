#!/usr/bin/env python3
"""Sanity-check plot: reassemble the LLC30 5-facet bathymetry/land-sea mask
into one global map using real lon/lat from the mitgrid tiles.

See ../preprocess/README.md for the .mitgrid binary layout and the compact
(30,390) facet-flattening convention this script relies on.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.abspath(os.path.join(HERE, "..", "input"))

# 16 fields in exact on-disk order; each shape (ny+1, nx+1), row-major.
MITGRID_FIELDS = [
    "XC", "YC", "DXF", "DYF", "RAC", "XG", "YG", "DXV", "DYU", "RAZ",
    "DXC", "DYC", "RAW", "RAS", "DXG", "DYG",
]

# Per-facet LLC30 native (ny, nx) cell-quantity dims (facets 1,2 tall;
# 3 square/Arctic cap; 4,5 wide) -- from preprocess/README.md's table.
FACET_DIMS = {
    1: (90, 30),
    2: (90, 30),
    3: (30, 30),
    4: (30, 90),
    5: (30, 90),
}


def read_mitgrid_tile(facet):
    ny, nx = FACET_DIMS[facet]
    path = os.path.join(INPUT_DIR, f"tile{facet:03d}.mitgrid")
    raw = np.fromfile(path, dtype=">f8")
    expected = 16 * (ny + 1) * (nx + 1)
    assert raw.size == expected, (
        f"facet {facet}: expected {expected} values, got {raw.size}"
    )
    fields = {}
    per_field = (ny + 1) * (nx + 1)
    for i, name in enumerate(MITGRID_FIELDS):
        chunk = raw[i * per_field:(i + 1) * per_field]
        fields[name] = chunk.reshape(ny + 1, nx + 1)
    return fields


def cell_center_lonlat(facet):
    """Cell-center XC/YC, cropped to (ny,nx) -- last row/col are zero-pad."""
    ny, nx = FACET_DIMS[facet]
    fields = read_mitgrid_tile(facet)
    xc = fields["XC"][:ny, :nx]
    yc = fields["YC"][:ny, :nx]
    return xc, yc


def cell_corner_lonlat(facet):
    """Cell-corner XG/YG, full (ny+1,nx+1) extent -- no padding, real
    corner positions (rule 2 in preprocess/README.md)."""
    ny, nx = FACET_DIMS[facet]
    fields = read_mitgrid_tile(facet)
    return fields["XG"], fields["YG"]


def dateline_seam_mask(xg):
    """True for cells (ny,nx) whose 4 corners straddle the +/-180 seam --
    a naive pcolormesh quad through those corners would otherwise stretch
    across the whole globe. These are a 1-cell-wide sliver; masking them
    out of the plotted data is a cosmetic concession, not a trust issue
    in the underlying grid/bathy data."""
    corners = np.stack([xg[:-1, :-1], xg[:-1, 1:], xg[1:, :-1], xg[1:, 1:]])
    return (corners.max(axis=0) - corners.min(axis=0)) > 180.0


def read_bathy_facets():
    """Split BATHY_LLC30.bin (compact shape (30,390), real*4 big-endian)
    into its 5 native-shaped facets. The compact buffer is the flat
    concatenation of each facet's own native (ny,nx) row-major block --
    NOT a uniform (30,390) 2-D array that can be row/col-sliced, per the
    'Compact-grid layout -- IMPORTANT CORRECTION' section of
    preprocess/README.md.
    """
    path = os.path.join(INPUT_DIR, "BATHY_LLC30.bin")
    raw = np.fromfile(path, dtype=">f4")
    expected = 30 * 390
    assert raw.size == expected, f"expected {expected} values, got {raw.size}"

    facets = {}
    offset = 0
    for facet in (1, 2, 3, 4, 5):
        ny, nx = FACET_DIMS[facet]
        n = ny * nx
        facets[facet] = raw[offset:offset + n].reshape(ny, nx)
        offset += n
    assert offset == raw.size
    return facets


def main():
    bathy = read_bathy_facets()

    fig = plt.figure(figsize=(14, 7))
    ax = plt.axes(projection=ccrs.Robinson())
    ax.set_global()
    ax.coastlines(resolution="110m", linewidth=0.5, color="black", zorder=3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, zorder=3)

    depth = np.concatenate([bathy[f].ravel() for f in (1, 2, 3, 4, 5)])
    vmin, vmax = depth[depth < 0].min(), 0.0

    mesh = None
    for facet in (1, 2, 3, 4, 5):
        xg, yg = cell_corner_lonlat(facet)
        d = bathy[facet]
        seam = dateline_seam_mask(xg)

        ocean = np.ma.masked_where((d == 0.0) | seam, d)
        mesh = ax.pcolormesh(
            xg, yg, ocean,
            transform=ccrs.PlateCarree(),
            cmap="viridis_r",
            vmin=vmin, vmax=vmax,
            shading="flat",
            zorder=1,
        )
        # Land (zero-depth) cells drawn in a flat tan color.
        land = np.ma.masked_where((d != 0.0) | seam, d)
        ax.pcolormesh(
            xg, yg, land * 0,
            transform=ccrs.PlateCarree(),
            cmap=mcolors.ListedColormap(["#c2b280"]),
            shading="flat",
            zorder=2,
        )

    cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.05, shrink=0.6)
    cbar.set_label("Depth (m)")
    ax.set_title(
        "LLC30 bathymetry / land-sea mask sanity check\n"
        "(5 facets reassembled from tile00{1..5}.mitgrid lon/lat)"
    )

    out_path = os.path.join(HERE, "bathy_sanity_check.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
