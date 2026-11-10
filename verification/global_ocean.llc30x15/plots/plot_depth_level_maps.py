#!/usr/bin/env python3
"""Sanity-check plots: T and S maps at several depth levels, step 0 vs.
step 10, reassembled from the 5 GLOBAL facet model-output files (same
facet-neighbor-borrow + dateline-seam-masking method as the other
plotting scripts here, see plot_ssh_map.py's module docstring).

Reads from run_alldump/ (local dumpFreq override -- see
plot_ssh_map.py's docstring and HANDOFF.md for why this separate run
directory exists instead of the committed run/).

Depth levels plotted: surface (k=0), and 3 more spread through the
water column (chosen from RC.data's real cell-center depths, printed at
runtime) -- not just the surface, to check vertical structure too.

Phase A (adjoint-experiment plan, see HANDOFF.md): this config now sets
useSingleCpuIO=.TRUE. (matching real V4r6), so run_alldump/ writes one
GLOBAL file per field instead of 13 per-tile files.
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

FACET_DIMS = {
    1: (90, 30),
    2: (90, 30),
    3: (30, 30),
    4: (30, 90),
    5: (30, 90),
}
NLEV = 15

# Chosen to spread through the column: surface, upper-mid, mid, deep.
LEVEL_INDICES = [0, 4, 8, 12]


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
    rationale."""
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


def read_field_level(name, step, k):
    step_str = f"{step:010d}"
    path = os.path.join(RUN_DIR, f"{name}.{step_str}.data")
    facets = read_global_facets(path, nlev=NLEV)
    return {f: facets[f][k] for f in (1, 2, 3, 4, 5)}


def read_land_mask_facets(k):
    hfacc_facets = read_global_facets(os.path.join(RUN_DIR, "hFacC.data"), nlev=NLEV)
    return {f: hfacc_facets[f][k] == 0.0 for f in (1, 2, 3, 4, 5)}


def dateline_seam_mask(xg):
    corners = np.stack([xg[:-1, :-1], xg[:-1, 1:], xg[1:, :-1], xg[1:, 1:]])
    return (corners.max(axis=0) - corners.min(axis=0)) > 180.0


def plot_panel(ax, xg_facets, yg_facets, field_facets, land_mask_facets, vmin, vmax, cmap, title):
    ax.set_global()
    ax.coastlines(resolution="110m", linewidth=0.5, color="black", zorder=3)
    mesh = None
    for facet in (1, 2, 3, 4, 5):
        xg, yg = extended_facet_corners(xg_facets, yg_facets, facet)
        f = field_facets[facet]
        land = land_mask_facets[facet]
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


def make_depth_level_plot(xg_facets, yg_facets, name, cmap, units, out_filename):
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
        land_mask_facets = read_land_mask_facets(k)

        all_vals = np.concatenate(
            [f0[f][~land_mask_facets[f]].ravel() for f in (1, 2, 3, 4, 5)] +
            [f10[f][~land_mask_facets[f]].ravel() for f in (1, 2, 3, 4, 5)]
        )
        if all_vals.size == 0:
            print(f"  level {k} (depth {depth_m:.0f} m): no wet cells, skipping")
            continue
        vmin, vmax = all_vals.min(), all_vals.max()
        print(f"  level {k} (depth {depth_m:.0f} m): {name} range "
              f"[{vmin:.3f}, {vmax:.3f}] {units}")

        plot_panel(axes[row][0], xg_facets, yg_facets, f0, land_mask_facets, vmin, vmax, cmap,
                   f"{name}, depth {depth_m:.0f} m, step 0")
        mesh = plot_panel(axes[row][1], xg_facets, yg_facets, f10, land_mask_facets, vmin, vmax, cmap,
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
    xg_facets = read_global_facets(os.path.join(RUN_DIR, "XG.data"))
    yg_facets = read_global_facets(os.path.join(RUN_DIR, "YG.data"))

    print("Theta depth-level maps:")
    make_depth_level_plot(xg_facets, yg_facets, "T", "RdYlBu_r", "degC", "T_depth_levels.png")
    print("Salt depth-level maps:")
    make_depth_level_plot(xg_facets, yg_facets, "S", "viridis", "psu", "S_depth_levels.png")


if __name__ == "__main__":
    main()
