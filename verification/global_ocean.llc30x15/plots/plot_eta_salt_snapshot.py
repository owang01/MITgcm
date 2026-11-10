#!/usr/bin/env python3
"""Sanity-check plots: sea-surface height (Eta) and surface salinity (S,
top vertical level) at step 0 vs. step 10 of the LLC30 smoke test, same
facet reassembly + dateline-seam-masking approach as
plot_bathy_sanity.py / plot_T_snapshot.py / plot_ssh_map.py.

Both fields are real, spatially-varying V4r6-derived initial conditions,
NOT flat/uniform (this docstring previously said otherwise -- stale as
of the real-pickup-file session, see HANDOFF.md): Salt has been real
since coarsen_initial_ts_to_llc30.py; Eta became real in the same session
that switched this config's cold start over to reading a real V4r6
pickup file via pickupSuff (preprocess/build_llc30_pickup.py) -- see
plot_ssh_map.py for the dedicated Eta/SSH map. See T_step0_vs_step10 for
the matching temperature plot.

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
    rationale: borrows the TRUE connected neighbor facet's edge wherever
    one exists (coarsen_llc90_to_llc30._facet_neighbor), falling back to
    linear extrapolation only at a genuinely open outer facet edge."""
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


def read_surface_field(name, step, is_3d):
    """Read a GLOBAL model-output field at the given step, split into 5
    facets, surface level only. 3-D fields (T, S, ...) are stored
    (nlev,390,30); 2-D fields (Eta, ...) are (390,30)."""
    step_str = f"{step:010d}"
    path = os.path.join(RUN_DIR, f"{name}.{step_str}.data")
    if is_3d:
        facets = read_global_facets(path, nlev=NLEV)
        return {f: facets[f][0] for f in (1, 2, 3, 4, 5)}
    return read_global_facets(path)


def dateline_seam_mask(xg):
    corners = np.stack([xg[:-1, :-1], xg[:-1, 1:], xg[1:, :-1], xg[1:, 1:]])
    return (corners.max(axis=0) - corners.min(axis=0)) > 180.0


def read_land_mask_facets():
    path = os.path.join(RUN_DIR, "hFacC.data")
    hfacc_facets = read_global_facets(path, nlev=NLEV)
    return {f: hfacc_facets[f][0] == 0.0 for f in (1, 2, 3, 4, 5)}


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
            zorder=1,
        )
    ax.set_title(title)
    return mesh


def make_comparison_plot(xg_facets, yg_facets, land_mask_facets,
                          name, is_3d, cmap, units, out_filename,
                          label_step0, label_step10):
    f0 = read_surface_field(name, 0, is_3d)
    f10 = read_surface_field(name, 10, is_3d)

    all_vals = np.concatenate(
        [f0[f][~land_mask_facets[f]].ravel() for f in (1, 2, 3, 4, 5)] +
        [f10[f][~land_mask_facets[f]].ravel() for f in (1, 2, 3, 4, 5)]
    )
    vmin, vmax = all_vals.min(), all_vals.max()
    print(f"{name} range across step 0 and step 10: [{vmin:.5f}, {vmax:.5f}] {units}")

    fig = plt.figure(figsize=(16, 7))
    ax0 = fig.add_subplot(1, 2, 1, projection=ccrs.Robinson())
    ax1 = fig.add_subplot(1, 2, 2, projection=ccrs.Robinson())

    plot_panel(ax0, xg_facets, yg_facets, f0, land_mask_facets, vmin, vmax, cmap, label_step0)
    mesh = plot_panel(ax1, xg_facets, yg_facets, f10, land_mask_facets, vmin, vmax, cmap, label_step10)

    cbar = fig.colorbar(mesh, ax=[ax0, ax1], orientation="horizontal", pad=0.08, shrink=0.5)
    cbar.set_label(f"{name} ({units})")
    fig.suptitle(f"LLC30 smoke-test {name}: step 0 vs. step 10")

    out_path = os.path.join(HERE, out_filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")

    max_abs_diff = 0.0
    for facet in (1, 2, 3, 4, 5):
        diff = np.abs(f10[facet] - f0[facet])
        diff[land_mask_facets[facet]] = 0.0
        max_abs_diff = max(max_abs_diff, diff.max())
    print(f"Max |{name}(step10) - {name}(step0)| over wet surface cells: {max_abs_diff:.6f} {units}")


def main():
    xg_facets = read_global_facets(os.path.join(RUN_DIR, "XG.data"))
    yg_facets = read_global_facets(os.path.join(RUN_DIR, "YG.data"))
    land_mask_facets = read_land_mask_facets()

    make_comparison_plot(
        xg_facets, yg_facets, land_mask_facets,
        name="Eta",
        is_3d=False,
        cmap="RdBu_r",
        units="m",
        out_filename="Eta_step0_vs_step10.png",
        label_step0="Sea surface height, step 0 (1992-01-01)",
        label_step10="Sea surface height, step 10",
    )
    print()
    make_comparison_plot(
        xg_facets, yg_facets, land_mask_facets,
        name="S",
        is_3d=True,
        cmap="viridis",
        units="psu",
        out_filename="S_step0_vs_step10.png",
        label_step0="Surface salinity, step 0 (1992-01-01)",
        label_step10="Surface salinity, step 10",
    )


if __name__ == "__main__":
    main()
