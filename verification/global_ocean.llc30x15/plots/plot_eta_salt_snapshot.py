#!/usr/bin/env python3
"""Sanity-check plots: sea-surface height (Eta) and surface salinity (S,
top vertical level) at step 0 vs. step 10 of the LLC30 smoke test, same
facet/tile reassembly + dateline-seam-masking approach as
plot_bathy_sanity.py / plot_T_snapshot.py.

Both fields are real, spatially-varying V4r6-derived initial conditions,
NOT flat/uniform (this docstring previously said otherwise -- stale as
of the real-pickup-file session, see HANDOFF.md): Salt has been real
since coarsen_initial_ts_to_llc30.py; Eta became real in the same session
that switched this config's cold start over to reading a real V4r6
pickup file via pickupSuff (preprocess/build_llc30_pickup.py) -- see
plot_ssh_map.py for the dedicated Eta/SSH map. See T_step0_vs_step10 for
the matching temperature plot.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.abspath(os.path.join(HERE, "..", "run"))

NTILES = 13
TILE_NY, TILE_NX, TILE_NZ = 30, 30, 15


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


def read_surface_field(name, step, is_3d):
    """Read a per-tile model-output field at the given step. 3-D fields
    (T, S, ...) are stored (nz,ny,nx); 2-D fields (Eta, ...) are (ny,nx)."""
    step_str = f"{step:010d}"
    fields = []
    for tile in range(1, NTILES + 1):
        path = os.path.join(RUN_DIR, f"{name}.{step_str}.{tile:03d}.001.data")
        raw = np.fromfile(path, dtype=">f4")
        if is_3d:
            expected = TILE_NY * TILE_NX * TILE_NZ
            assert raw.size == expected, (
                f"{name} step {step} tile {tile}: expected {expected}, got {raw.size}"
            )
            cube = raw.reshape(TILE_NZ, TILE_NY, TILE_NX)
            fields.append(cube[0])
        else:
            expected = TILE_NY * TILE_NX
            assert raw.size == expected, (
                f"{name} step {step} tile {tile}: expected {expected}, got {raw.size}"
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
        cube = raw.reshape(TILE_NZ, TILE_NY, TILE_NX)
        land_mask_tiles.append(cube[0] == 0.0)
    return land_mask_tiles


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
            zorder=1,
        )
    ax.set_title(title)
    return mesh


def make_comparison_plot(name, is_3d, cmap, units, out_filename, label_step0, label_step10):
    land_mask_tiles = read_land_mask_tiles()

    f0 = read_surface_field(name, 0, is_3d)
    f10 = read_surface_field(name, 10, is_3d)

    all_vals = np.concatenate(
        [f[~m].ravel() for f, m in zip(f0 + f10, land_mask_tiles * 2)]
    )
    vmin, vmax = all_vals.min(), all_vals.max()
    print(f"{name} range across step 0 and step 10: [{vmin:.5f}, {vmax:.5f}] {units}")

    fig = plt.figure(figsize=(16, 7))
    ax0 = fig.add_subplot(1, 2, 1, projection=ccrs.Robinson())
    ax1 = fig.add_subplot(1, 2, 2, projection=ccrs.Robinson())

    plot_panel(ax0, f0, land_mask_tiles, vmin, vmax, cmap, label_step0)
    mesh = plot_panel(ax1, f10, land_mask_tiles, vmin, vmax, cmap, label_step10)

    cbar = fig.colorbar(mesh, ax=[ax0, ax1], orientation="horizontal", pad=0.08, shrink=0.5)
    cbar.set_label(f"{name} ({units})")
    fig.suptitle(f"LLC30 smoke-test {name}: step 0 vs. step 10")

    out_path = os.path.join(HERE, out_filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")

    max_abs_diff = 0.0
    for a, b, land in zip(f0, f10, land_mask_tiles):
        diff = np.abs(b - a)
        diff[land] = 0.0
        max_abs_diff = max(max_abs_diff, diff.max())
    print(f"Max |{name}(step10) - {name}(step0)| over wet surface cells: {max_abs_diff:.6f} {units}")


def main():
    make_comparison_plot(
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
