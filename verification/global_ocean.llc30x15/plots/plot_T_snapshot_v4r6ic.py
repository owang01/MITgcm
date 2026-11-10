#!/usr/bin/env python3
"""Sanity-check plot: surface T (top vertical level) at step 0 vs. step 10
of the LLC30 smoke test run with the REAL V4r6-derived initial condition
(see HANDOFF.md's "V4r6-faithfulness audit" section) -- reassembled from
the 5 GLOBAL facet model-output files using each facet's own XG/YG
corner coordinates.

This is the same reassembly method as plot_T_snapshot.py (same
facet-neighbor-borrow + dateline-seam-masking fix), but against the
CURRENT run/ output, which now reflects the real coarsened V4r6 Theta
field at step 0, not the old flat tRef=20 analytic IC. Written as a new
file (not overwriting plot_T_snapshot.py/T_step0_vs_step10.png) since
the old plot still correctly describes the run it was made from -- it's
just now a plot of a superseded initial condition, not a wrong plot.

Step 0 here should already show real ocean structure (unlike the old
flat-IC plot, which started perfectly uniform at step 0 by construction)
-- that's the main visual difference to look for.

Phase A (adjoint-experiment plan, see HANDOFF.md): this config now sets
useSingleCpuIO=.TRUE. (matching real V4r6), so run/ writes one GLOBAL
file per field instead of 13 per-tile files -- see plot_ssh_map.py's
module docstring for the full facet-neighbor-borrow design this script
reuses verbatim.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.abspath(os.path.join(HERE, "..", "run"))
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


def read_global_facets(path, nlev=None):
    """Read a GLOBAL run/<name>.data file (real*4 big-endian) and split
    into its 5 native-shaped facets -- flat concatenation of each
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


def read_surface_T(step):
    step_str = f"{step:010d}"
    path = os.path.join(RUN_DIR, f"T.{step_str}.data")
    facets = read_global_facets(path, nlev=NLEV)
    return {f: facets[f][0] for f in (1, 2, 3, 4, 5)}


def dateline_seam_mask(xg):
    corners = np.stack([xg[:-1, :-1], xg[:-1, 1:], xg[1:, :-1], xg[1:, 1:]])
    return (corners.max(axis=0) - corners.min(axis=0)) > 180.0


def plot_panel(ax, xg_facets, yg_facets, surface_T, land_mask_facets, vmin, vmax, title):
    ax.set_global()
    ax.coastlines(resolution="110m", linewidth=0.5, color="black", zorder=3)
    mesh = None
    for facet in (1, 2, 3, 4, 5):
        xg, yg = extended_facet_corners(xg_facets, yg_facets, facet)
        t = surface_T[facet]
        land = land_mask_facets[facet]
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
    xg_facets = read_global_facets(os.path.join(RUN_DIR, "XG.data"))
    yg_facets = read_global_facets(os.path.join(RUN_DIR, "YG.data"))

    hfacc_facets = read_global_facets(os.path.join(RUN_DIR, "hFacC.data"), nlev=NLEV)
    land_mask_facets = {f: hfacc_facets[f][0] == 0.0 for f in (1, 2, 3, 4, 5)}

    T0 = read_surface_T(0)
    T10 = read_surface_T(10)

    all_vals = np.concatenate(
        [T0[f][~land_mask_facets[f]].ravel() for f in (1, 2, 3, 4, 5)] +
        [T10[f][~land_mask_facets[f]].ravel() for f in (1, 2, 3, 4, 5)]
    )
    vmin, vmax = all_vals.min(), all_vals.max()
    print(f"Surface T range across step 0 and step 10: [{vmin:.4f}, {vmax:.4f}] degC")

    fig = plt.figure(figsize=(16, 7))
    ax0 = fig.add_subplot(1, 2, 1, projection=ccrs.Robinson())
    ax1 = fig.add_subplot(1, 2, 2, projection=ccrs.Robinson())

    plot_panel(ax0, xg_facets, yg_facets, T0, land_mask_facets, vmin, vmax,
               "Surface T, step 0 (1992-01-01)\n(real V4r6-derived IC)")
    mesh = plot_panel(ax1, xg_facets, yg_facets, T10, land_mask_facets, vmin, vmax, "Surface T, step 10")

    cbar = fig.colorbar(mesh, ax=[ax0, ax1], orientation="horizontal", pad=0.08, shrink=0.5)
    cbar.set_label("Temperature (degC)")
    fig.suptitle("LLC30 smoke test, REAL V4r6-derived IC: surface T, step 0 vs. step 10")

    out_path = os.path.join(HERE, "T_step0_vs_step10_v4r6ic.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")

    max_abs_diff = 0.0
    for facet in (1, 2, 3, 4, 5):
        diff = np.abs(T10[facet] - T0[facet])
        diff[land_mask_facets[facet]] = 0.0
        max_abs_diff = max(max_abs_diff, diff.max())
    print(f"Max |T(step10) - T(step0)| over wet surface cells: {max_abs_diff:.6f} degC")


if __name__ == "__main__":
    main()
