#!/usr/bin/env python3
"""Build profiles_llc30_demo.nc: 3-5 REAL extracted 1992-era CTD profiles
(core observational content -- date, lon, lat, depth levels, T, S -- not
synthetic placeholder values), from a real WOD CTD/OSD file, re-targeted
onto LLC30's OWN grid (profilesDoGenGrid=.TRUE. interpolation fields
regenerated from scratch -- the source file's own prof_interp_* fields
are computed against LLC90's tile geometry, not directly usable here).

Source data (read-only, never modified):
  /ecco_nfs_1/owang/projects/ECCO/llc90/v4/release6/PO.DAAC/ancillary_data/
  ancillary_data_orig/data_constraints/input_insitu/
  WOD_WO_CTD_OSD_20231103_92to05.nc
  Confirmed real structure (direct netCDF4 inspection): iPROF=461371
  (unlimited), iDEPTH=94, iINTERP=1 (confirms production ALSO uses simple
  nearest-point interpolation, not 4-point bilinear). prof_T/prof_S
  (iPROF,iDEPTH) are the actual observed values (masked array, fill
  -9999.0); prof_depth (iDEPTH,) is SHARED across all profiles (not
  per-profile); prof_YYYYMMDD/prof_HHMMSS/prof_lon/prof_lat (iPROF,).

Selection: 5 real profiles whose prof_YYYYMMDD lands in January 1992
(near this config's own data.cal startDate_1=19920101), picked from the
real file's own Southern-Ocean-heavy early records (confirmed 1993
profiles fall in that window; the first 5 found are used, not cherry-
picked for any particular location).

Interpolation fields (profilesDoGenGrid=.TRUE., per profiles_init_fixed.F
requirements -- see module docstring in docs/plans or HANDOFF.md for the
full citation): iINTERP=1, weight=1.0 (matches production's own
simplification), nearest LLC30 cell found via the same tile*.mitgrid
XC/YC lookup pattern coarsen_initial_ts_to_llc30.py's
spot_check_known_locations already uses. prof_interp_XC11/YC11/XCNINJ/
YCNINJ are set to the OWNING LLC30 tile's own xC(1,1)/xC(sNx,sNy) corner
values exactly (profiles_init_fixed.F checks these to 0.0001 tolerance).

Run:
    export PATH=$HOME/micromamba/envs/py3p11_mamba/bin:$PATH
    cd MITgcm/verification/global_ocean.llc30x15/preprocess
    python3 build_llc30_profiles_demo.py
"""

import os

import netCDF4 as nc
import numpy as np

import coarsen_llc90_to_llc30 as base

SRC_PROFILE_FILE = (
    "/ecco_nfs_1/owang/projects/ECCO/llc90/v4/release6/PO.DAAC/"
    "ancillary_data/ancillary_data_orig/data_constraints/input_insitu/"
    "WOD_WO_CTD_OSD_20231103_92to05.nc"
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MITGRID_DIR = os.path.join(SCRIPT_DIR, "..", "input")
OUT_DIR = os.path.join(SCRIPT_DIR, "..", "input_ad")
OUT_PATH = os.path.join(OUT_DIR, "profiles_llc30_demo.nc")

N_PROFILES_WANTED = 5
JAN_1992_LO = 19920101
JAN_1992_HI = 19920201
TOL = 0.0001  # profiles_init_fixed.F's own consistency-check tolerance


# ===========================================================================
# Step 1: extract real profiles from the source file
# ===========================================================================

def extract_real_profiles():
    with nc.Dataset(SRC_PROFILE_FILE) as f:
        yyyymmdd = f.variables["prof_YYYYMMDD"][:]
        mask = (yyyymmdd >= JAN_1992_LO) & (yyyymmdd < JAN_1992_HI)
        idx = np.nonzero(mask)[0][:N_PROFILES_WANTED]
        assert idx.size == N_PROFILES_WANTED, (
            f"expected {N_PROFILES_WANTED} Jan-1992 profiles, found {idx.size}"
        )
        print(f"  Selected {idx.size} real profiles (source indices {idx}).")

        depth = np.asarray(f.variables["prof_depth"][:], dtype=np.float64)
        T = np.ma.filled(f.variables["prof_T"][idx], -9999.0).astype(np.float64)
        S = np.ma.filled(f.variables["prof_S"][idx], -9999.0).astype(np.float64)
        lon = np.asarray(f.variables["prof_lon"][idx], dtype=np.float64)
        lat = np.asarray(f.variables["prof_lat"][idx], dtype=np.float64)
        prof_date = np.asarray(yyyymmdd[idx], dtype=np.float64)
        prof_hhmmss = np.asarray(f.variables["prof_HHMMSS"][idx], dtype=np.float64)

    assert T.shape == S.shape == (N_PROFILES_WANTED, depth.size)
    print(f"  depth levels: {depth.size}, lon range "
          f"[{lon.min():.2f},{lon.max():.2f}], lat range "
          f"[{lat.min():.2f},{lat.max():.2f}]")
    return depth, T, S, lon, lat, prof_date, prof_hhmmss


# ===========================================================================
# Step 2: regenerate prof_interp_* fields against LLC30's OWN grid
# ===========================================================================

def load_llc30_tile_corners():
    """Return dict tile(1..13) -> (xc11, yc11, xcnini, ycnini, XC, YC),
    i.e. the tile's own xC(1,1)/xC(sNx,sNy) corner values (profiles_
    init_fixed.F's exact consistency-check targets) plus the full
    (30,30) XC/YC arrays for nearest-cell lookup. Mirrors
    coarsen_initial_ts_to_llc30.py's spot_check_known_locations pattern
    (read per-facet tile*.mitgrid, split facets into their LLC30-native
    tile blocks the same way build_llc30_pickup.py's
    split_facet_into_tiles does).
    """
    FACET_TILE_RANGE = {1: (1, 3), 2: (4, 6), 3: (7, 7), 4: (8, 10), 5: (11, 13)}
    FACET_STACK_AXIS = {1: "y", 2: "y", 3: None, 4: "x", 5: "x"}

    tiles = {}
    for facet in range(1, 6):
        nx30, ny30 = base.LLC30_FACET_DIMS[facet]
        path = os.path.join(MITGRID_DIR, f"tile{facet:03d}.mitgrid")
        fields = base.read_mitgrid_tile(path, nx30, ny30)
        xc_facet = fields["XC"][:ny30, :nx30]
        yc_facet = fields["YC"][:ny30, :nx30]

        t0, t1 = FACET_TILE_RANGE[facet]
        axis = FACET_STACK_AXIS[facet]
        if axis is None:
            tile_xc = {t0: xc_facet}
            tile_yc = {t0: yc_facet}
        else:
            n_tiles = t1 - t0 + 1
            split_axis = -2 if axis == "y" else -1
            xc_chunks = np.split(xc_facet, n_tiles, axis=split_axis)
            yc_chunks = np.split(yc_facet, n_tiles, axis=split_axis)
            tile_xc = {t0 + i: xc_chunks[i] for i in range(n_tiles)}
            tile_yc = {t0 + i: yc_chunks[i] for i in range(n_tiles)}

        for tile in range(t0, t1 + 1):
            XC, YC = tile_xc[tile], tile_yc[tile]
            assert XC.shape == (30, 30), (tile, XC.shape)
            tiles[tile] = {
                "XC": XC,
                "YC": YC,
                "xc11": XC[0, 0],
                "yc11": YC[0, 0],
                "xcnini": XC[29, 29],
                "ycnini": YC[29, 29],
            }
    assert len(tiles) == 13
    return tiles


def find_nearest_llc30_cell(lon0, lat0, tiles):
    """Nearest LLC30 (tile, i, j) [1-indexed i,j within tile, matching
    MITgcm's own sNx/sNy 1-indexing convention] to (lon0,lat0), searched
    across all 13 tiles -- same simple nearest-cell-center distance
    metric as coarsen_initial_ts_to_llc30.py's spot_check_known_locations
    (dateline-safe via the same wrapped-longitude-difference form)."""
    best = None
    for tile, info in tiles.items():
        xc, yc = info["XC"], info["YC"]
        dlon = ((xc - lon0 + 180.0) % 360.0) - 180.0
        dist = dlon ** 2 + (yc - lat0) ** 2
        j, i = np.unravel_index(np.argmin(dist), dist.shape)
        d = dist[j, i]
        if best is None or d < best[0]:
            best = (d, tile, i + 1, j + 1)  # 1-indexed i,j
    _, tile, i, j = best
    return tile, i, j, tiles[tile]


# ===========================================================================
# Step 3: write profiles_llc30_demo.nc in pkg/profiles format
# ===========================================================================

def write_profiles_nc(depth, T, S, lon, lat, prof_date, prof_hhmmss, tiles):
    n_prof = lon.size
    n_depth = depth.size

    prof_point = np.empty(n_prof, dtype=np.float64)
    prof_interp_xc11 = np.empty(n_prof, dtype=np.float64)
    prof_interp_yc11 = np.empty(n_prof, dtype=np.float64)
    prof_interp_xcnini = np.empty(n_prof, dtype=np.float64)
    prof_interp_ycnini = np.empty(n_prof, dtype=np.float64)
    prof_interp_i = np.empty((n_prof, 1), dtype=np.float64)
    prof_interp_j = np.empty((n_prof, 1), dtype=np.float64)
    prof_interp_weights = np.ones((n_prof, 1), dtype=np.float64)

    for p in range(n_prof):
        tile, i, j, info = find_nearest_llc30_cell(lon[p], lat[p], tiles)
        prof_point[p] = tile
        prof_interp_xc11[p] = info["xc11"]
        prof_interp_yc11[p] = info["yc11"]
        prof_interp_xcnini[p] = info["xcnini"]
        prof_interp_ycnini[p] = info["ycnini"]
        prof_interp_i[p, 0] = i
        prof_interp_j[p, 0] = j
        print(f"  profile {p}: (lon={lon[p]:.2f}, lat={lat[p]:.2f}) -> "
              f"tile {tile}, local (i={i}, j={j})")

    os.makedirs(OUT_DIR, exist_ok=True)
    with nc.Dataset(OUT_PATH, "w", format="NETCDF4") as out:
        out.createDimension("iPROF", n_prof)
        out.createDimension("iDEPTH", n_depth)
        out.createDimension("iINTERP", 1)

        v = out.createVariable("prof_depth", "f8", ("iDEPTH",))
        v[:] = depth
        v = out.createVariable("prof_YYYYMMDD", "f8", ("iPROF",))
        v[:] = prof_date
        v = out.createVariable("prof_HHMMSS", "f8", ("iPROF",))
        v[:] = prof_hhmmss
        v = out.createVariable("prof_lon", "f8", ("iPROF",))
        v[:] = lon
        v = out.createVariable("prof_lat", "f8", ("iPROF",))
        v[:] = lat
        v = out.createVariable("prof_point", "f8", ("iPROF",))
        v[:] = prof_point

        v = out.createVariable("prof_T", "f8", ("iPROF", "iDEPTH"),
                                fill_value=-9999.0)
        v[:] = T
        v = out.createVariable("prof_S", "f8", ("iPROF", "iDEPTH"),
                                fill_value=-9999.0)
        v[:] = S

        v = out.createVariable("prof_interp_XC11", "f8", ("iPROF",))
        v[:] = prof_interp_xc11
        v = out.createVariable("prof_interp_YC11", "f8", ("iPROF",))
        v[:] = prof_interp_yc11
        v = out.createVariable("prof_interp_XCNINJ", "f8", ("iPROF",))
        v[:] = prof_interp_xcnini
        v = out.createVariable("prof_interp_YCNINJ", "f8", ("iPROF",))
        v[:] = prof_interp_ycnini
        v = out.createVariable("prof_interp_weights", "f8",
                                ("iPROF", "iINTERP"))
        v[:] = prof_interp_weights
        v = out.createVariable("prof_interp_i", "f8", ("iPROF", "iINTERP"))
        v[:] = prof_interp_i
        v = out.createVariable("prof_interp_j", "f8", ("iPROF", "iINTERP"))
        v[:] = prof_interp_j

    print(f"  Wrote {OUT_PATH} ({n_prof} profiles, {n_depth} depth levels).")


# ===========================================================================
# Verification
# ===========================================================================

def verify_consistency(tiles):
    """Re-read the written file and re-check profiles_init_fixed.F's own
    exact consistency conditions (xC11/yC11/xCNINJ/yCNINJ match the
    owning tile's own corners to TOL; weights sum to 1 within TOL) --
    same discipline as every other check in this codebase: verify the
    written artifact, don't just trust the generation logic."""
    with nc.Dataset(OUT_PATH) as f:
        xc11 = f.variables["prof_interp_XC11"][:]
        yc11 = f.variables["prof_interp_YC11"][:]
        xcnini = f.variables["prof_interp_XCNINJ"][:]
        ycnini = f.variables["prof_interp_YCNINJ"][:]
        weights = f.variables["prof_interp_weights"][:]
        i_idx = f.variables["prof_interp_i"][:]
        j_idx = f.variables["prof_interp_j"][:]
        points = f.variables["prof_point"][:]

    n_prof = xc11.size
    for p in range(n_prof):
        tile = int(points[p])
        info = tiles[tile]
        assert abs(xc11[p] - info["xc11"]) < TOL, (p, "xc11")
        assert abs(yc11[p] - info["yc11"]) < TOL, (p, "yc11")
        assert abs(xcnini[p] - info["xcnini"]) < TOL, (p, "xcnini")
        assert abs(ycnini[p] - info["ycnini"]) < TOL, (p, "ycnini")
        assert abs(weights[p].sum() - 1.0) < TOL, (p, "weights sum")
        i, j = int(i_idx[p, 0]), int(j_idx[p, 0])
        assert 1 <= i <= 30 and 1 <= j <= 30, (p, "i/j out of tile range")
    print(f"  Consistency check PASSED for all {n_prof} profiles "
          f"(tolerance {TOL}).")


def main():
    print("=" * 70)
    print("Step 1/3: extract real profiles")
    print("=" * 70)
    depth, T, S, lon, lat, prof_date, prof_hhmmss = extract_real_profiles()

    print("=" * 70)
    print("Step 2/3: load LLC30 tile corners + find nearest cells")
    print("=" * 70)
    tiles = load_llc30_tile_corners()

    print("=" * 70)
    print("Step 3/3: write profiles_llc30_demo.nc")
    print("=" * 70)
    write_profiles_nc(depth, T, S, lon, lat, prof_date, prof_hhmmss, tiles)

    print("=" * 70)
    print("Verification")
    print("=" * 70)
    verify_consistency(tiles)

    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
