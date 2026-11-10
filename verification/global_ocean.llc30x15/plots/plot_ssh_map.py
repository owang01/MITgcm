#!/usr/bin/env python3
"""Sanity-check plot: sea surface height (Eta) maps at step 0 vs. step 10,
reassembled from the 5 GLOBAL facet model-output files using each facet's
own XG/YG corner coordinates.

Reads from run_alldump/ (a separate run with dumpFreq overridden to dump
every step -- see run_alldump/data's local-override comment and
HANDOFF.md -- NOT the committed run/, which only has step-0/step-10
output under this config's committed dumpFreq). Physics/IC are identical
to run/; only output frequency differs.

Phase A (adjoint-experiment plan, see HANDOFF.md): this config now sets
useSingleCpuIO=.TRUE. (matching real V4r6), so run_alldump/ writes one
GLOBAL file per field instead of 13 per-tile files. The old per-tile
Y_STACK_NEXT/X_STACK_NEXT "borrow the same-facet neighbor TILE's edge"
workaround is gone -- each facet's own XG/YG now already spans the WHOLE
facet in one read, so the only remaining missing "+1" edge is at each
facet's own OUTER boundary (a genuinely open edge, no neighbor facet to
borrow from at all) -- confirmed via coarsen_llc90_to_llc30._facet_neighbor,
which returns None for exactly 4 of the 20 facet edges (facet 1's S,
facet 2's S, facet 4's E, facet 5's E). Those 4 still extrapolate from
the facet's own last two rows/columns (same formula as before, just at
facet granularity); every other edge borrows the true connected
neighbor FACET's own edge-adjacent row/column directly, with the same
orientation-reversal handling (_facet_neighbor's `reversed` flag) this
project's own preprocessing script (coarsen_llc90_to_llc30.py) already
uses for the identical problem on mitgrid-resolution corner arrays.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.abspath(os.path.join(HERE, "..", "run_alldump"))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "preprocess")))
import coarsen_llc90_to_llc30 as base  # noqa: E402

# Per-facet LLC30 native (ny, nx) dims (facets 1,2 tall; 3 square/Arctic
# cap; 4,5 wide) -- same convention as plot_bathy_sanity.py's FACET_DIMS.
FACET_DIMS = {
    1: (90, 30),
    2: (90, 30),
    3: (30, 30),
    4: (30, 90),
    5: (30, 90),
}
NLEV = 15


def read_global_facets(path, nlev=None):
    """Read a GLOBAL run_alldump/<name>.data file (real*4 big-endian) and
    split into its 5 native-shaped facets -- the on-disk layout is a
    FLAT CONCATENATION of each facet's own native (ny,nx) (or
    (nlev,ny,nx)) row-major block, same convention as
    plot_bathy_sanity.py's read_bathy_facets."""
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


def _edge_adjacent_line(arr, edge):
    """arr: (ny,nx). Returns the row/column directly adjacent to `edge`
    (the TRUE edge line, for corner arrays this is the last row/col of
    centers -- used here only for the neighbor-borrow source)."""
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
    """xg_facets/yg_facets: dict facet -> (ny,nx) SW-corner arrays (one
    corner per tracer cell, from run_alldump/XG.data/YG.data). Returns
    (xg_ext, yg_ext), each (ny+1,nx+1): pads the missing north row and
    east column, borrowing the TRUE connected neighbor facet's own edge
    wherever one exists (per coarsen_llc90_to_llc30._facet_neighbor),
    falling back to linear extrapolation only at a genuinely open outer
    facet edge."""
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


def read_eta(step):
    step_str = f"{step:010d}"
    path = os.path.join(RUN_DIR, f"Eta.{step_str}.data")
    return read_global_facets(path)


def dateline_seam_mask(xg):
    corners = np.stack([xg[:-1, :-1], xg[:-1, 1:], xg[1:, :-1], xg[1:, 1:]])
    return (corners.max(axis=0) - corners.min(axis=0)) > 180.0


def read_land_mask_facets():
    path = os.path.join(RUN_DIR, "hFacC.data")
    hfacc_facets = read_global_facets(path, nlev=NLEV)
    return {f: hfacc_facets[f][0] == 0.0 for f in (1, 2, 3, 4, 5)}


def plot_panel(ax, xg_facets, yg_facets, eta_facets, land_mask_facets, vmin, vmax, title):
    ax.set_global()
    ax.coastlines(resolution="110m", linewidth=0.5, color="black", zorder=3)
    mesh = None
    for facet in (1, 2, 3, 4, 5):
        xg, yg = extended_facet_corners(xg_facets, yg_facets, facet)
        e = eta_facets[facet]
        land = land_mask_facets[facet]
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
    xg_facets = read_global_facets(os.path.join(RUN_DIR, "XG.data"))
    yg_facets = read_global_facets(os.path.join(RUN_DIR, "YG.data"))
    land_mask_facets = read_land_mask_facets()
    eta0 = read_eta(0)
    eta10 = read_eta(10)

    all_vals = np.concatenate([
        eta0[f][~land_mask_facets[f]].ravel() for f in (1, 2, 3, 4, 5)
    ] + [
        eta10[f][~land_mask_facets[f]].ravel() for f in (1, 2, 3, 4, 5)
    ])
    vmax = np.abs(all_vals).max()
    vmin = -vmax
    print(f"Eta range across step 0 and step 10: [{all_vals.min():.4f}, {all_vals.max():.4f}] m")

    fig = plt.figure(figsize=(16, 7))
    ax0 = fig.add_subplot(1, 2, 1, projection=ccrs.Robinson())
    ax1 = fig.add_subplot(1, 2, 2, projection=ccrs.Robinson())

    plot_panel(ax0, xg_facets, yg_facets, eta0, land_mask_facets, vmin, vmax,
               "SSH (Eta), step 0 (1992-01-01)")
    mesh = plot_panel(ax1, xg_facets, yg_facets, eta10, land_mask_facets, vmin, vmax,
                       "SSH (Eta), step 10")

    cbar = fig.colorbar(mesh, ax=[ax0, ax1], orientation="horizontal", pad=0.08, shrink=0.5)
    cbar.set_label("Sea surface height (m)")
    fig.suptitle("LLC30 smoke test, real V4r6-derived IC: SSH, step 0 vs. step 10")

    out_path = os.path.join(HERE, "ssh_map.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
