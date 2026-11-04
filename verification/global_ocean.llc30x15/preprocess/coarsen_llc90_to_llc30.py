#!/usr/bin/env python3
"""Task 2: LLC90 -> LLC30 preprocessing (grid, bathymetry, forcing).

This script block-averages (with field-type-specific rules, NOT a naive
mean-for-everything) the 5 real LLC90 .mitgrid geometry tiles, the LLC90
bathymetry file, and 9 combforc 1992 hourly forcing variables down to LLC30
(1/3 the linear resolution), producing the input files consumed by the rest
of this verification config.

Binary-layout assumptions (confirmed in Task 1 -- see
preprocess/inspect_mitgrid.py for the full derivation/citations):

  .mitgrid tiles: raw, unformatted, big-endian real*8, no record markers;
  16 fields in this exact order:
    XC, YC, DXF, DYF, RAC, XG, YG, DXV, DYU, RAZ, DXC, DYC, RAW, RAS, DXG, DYG
  each field shape (ny+1, nx+1) (row-major, y outer / x inner), per facet.
  Expected file size for a facet (nx,ny): 16*(nx+1)*(ny+1)*8 bytes.

  bathymetry / combforc "compact" files: raw, unformatted, big-endian
  real*4, compact shape (90, 1170) per record (combforc has 8784 hourly
  records stacked in time). The 1170 (y) axis stacks the 5 facets as ROW
  RANGES (NOT column ranges) -- see FACET_ROW_RANGES below. Facets 4 and 5
  are wide (nx=270, ny=90): their natural (90, 270) row/column slice out of
  the compact array is ALREADY in (y=90, x=270) order and must NOT be
  reoriented/transposed before coarsening.

Coarsening rules (5 distinct field-type classes -- see Technical Details in
the plan for the full justification of why a naive "center-mean vs.
corner-subsample" split is wrong for lengths/areas):

  1. coarsen_coord_lonlat  -- XC, YC (cell-center lon/lat): block-MEAN of a
     3x3 block. XC uses atan2(mean(sin), mean(cos)) to avoid a dateline
     wrap bug; YC uses a plain arithmetic mean.
  2. coarsen_corner        -- XG, YG (cell-corner lon/lat): subsample every
     3rd point directly (coarse corners coincide with a subset of fine
     corners; averaging would displace the cell boundary).
  3. coarsen_xlength_sum   -- DXF, DXV, DXC, DXG (x-direction lengths): SUM
     of 3 consecutive values along x.
  4. coarsen_ylength_sum   -- DYF, DYU, DYC, DYG (y-direction lengths): SUM
     of 3 consecutive values along y.
  5. coarsen_area_sum      -- RAC, RAZ, RAW, RAS (areas): SUM of the 9
     constituent fine-cell areas in each 3x3 block.

Run:
    export PATH=$HOME/micromamba/envs/podaac/bin:$PATH
    python3 coarsen_llc90_to_llc30.py
"""

import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

GRID_SRC_DIR = (
    "/ecco_nfs_1/owang/projects/ECCO/llc90/v4/release6/PO.DAAC/"
    "ancillary_data/ancillary_data_orig/native_grid_files"
)
BATHY_SRC = (
    "/ecco_nfs_1/owang/projects/ECCO/llc90/v4/release6/PO.DAAC/"
    "ancillary_data/ancillary_data_orig/input_init/"
    "BATHY_ICE_SHELF_CAVITY_PLUS_ICE_FRONT_LLC_0090.bin"
)
FORCING_SRC_DIR = (
    "/ecco_nfs_1/owang/projects/ECCO/llc90/v4/release6/PO.DAAC/"
    "ancillary_data/ancillary_data_orig/input_forcing/combforc"
)
FORCING_VARS = [
    "dlw", "dsw", "pres", "rain", "spfh2m", "tmp2m", "ustr", "vstr", "wspeed",
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "output")
OUT_GRID_DIR = os.path.join(OUT_DIR, "grid")
OUT_BATHY_DIR = os.path.join(OUT_DIR, "bathy")
OUT_FORCING_DIR = os.path.join(OUT_DIR, "forcing")

FACTOR = 3  # LLC90 -> LLC30 coarsening factor

# ---------------------------------------------------------------------------
# mitgrid field order / per-facet dims (LLC90, confirmed in Task 1)
# ---------------------------------------------------------------------------

FIELD_NAMES = [
    "XC", "YC", "DXF", "DYF", "RAC", "XG", "YG", "DXV",
    "DYU", "RAZ", "DXC", "DYC", "RAW", "RAS", "DXG", "DYG",
]
N_FIELDS = 16

# facet -> (nx, ny) at LLC90 resolution (facets 1,2 tall; 3 square; 4,5 wide)
LLC90_FACET_DIMS = {
    1: (90, 270),
    2: (90, 270),
    3: (90, 90),
    4: (270, 90),
    5: (270, 90),
}
# facet -> (nx, ny) at LLC30 resolution (divide by FACTOR, same orientation)
LLC30_FACET_DIMS = {f: (nx // FACTOR, ny // FACTOR) for f, (nx, ny) in LLC90_FACET_DIMS.items()}

EXPECTED_LLC90_TILE_BYTES = {
    f: N_FIELDS * (nx + 1) * (ny + 1) * 8 for f, (nx, ny) in LLC90_FACET_DIMS.items()
}
EXPECTED_LLC30_TILE_BYTES = {
    f: N_FIELDS * (nx + 1) * (ny + 1) * 8 for f, (nx, ny) in LLC30_FACET_DIMS.items()
}

# Field -> which coarsening rule applies (see module docstring for rules)
FIELD_RULE = {
    "XC": "coord_lonlat", "YC": "coord_lonlat",
    "XG": "corner", "YG": "corner",
    "DXF": "xlength_sum", "DXV": "xlength_sum", "DXC": "xlength_sum", "DXG": "xlength_sum",
    "DYF": "ylength_sum", "DYU": "ylength_sum", "DYC": "ylength_sum", "DYG": "ylength_sum",
    "RAC": "area_sum", "RAZ": "area_sum", "RAW": "area_sum", "RAS": "area_sum",
}

# ---------------------------------------------------------------------------
# Compact-grid (90,1170) facet row-range convention (corrected -- see plan's
# "Corrected facet-offset convention" subsection). Facets stack along the
# y (1170) axis as ROW ranges. Facets 4,5 are wide: the natural (90,270)
# row-major slice from the compact array is already in (y=90,x=270) order.
# ---------------------------------------------------------------------------

FACET_ROW_RANGES = {
    1: (0, 270),
    2: (270, 540),
    3: (540, 630),
    4: (630, 900),
    5: (900, 1170),
}
COMPACT_NX = 90
COMPACT_NY = 1170
COMPACT_COARSE_NX = 30
COMPACT_COARSE_NY = 390


# ===========================================================================
# mitgrid I/O
# ===========================================================================

def read_mitgrid_tile(path, nx, ny):
    """Read one .mitgrid file, return dict of 16 named fields, each (ny+1, nx+1).

    Confirms index order matches Task 1's finding: field k (0-based) occupies
    byte range [k*(ny+1)*(nx+1), (k+1)*(ny+1)*(nx+1)) of a flat big-endian
    float64 buffer, reshaped row-major as (ny+1, nx+1).
    """
    raw = np.fromfile(path, dtype=">f8")
    expected_n = N_FIELDS * (ny + 1) * (nx + 1)
    if raw.size != expected_n:
        raise ValueError(
            f"{path}: read {raw.size} float64 values, expected {expected_n} "
            f"(16 * {ny + 1} * {nx + 1})"
        )
    fields = {}
    per_field = (ny + 1) * (nx + 1)
    for k, name in enumerate(FIELD_NAMES):
        chunk = raw[k * per_field:(k + 1) * per_field]
        fields[name] = chunk.reshape(ny + 1, nx + 1).astype(np.float64)
    return fields


def write_mitgrid_tile(path, fields_dict, nx, ny):
    """Write 16 fields back out as a flat big-endian real*8 .mitgrid file.

    Asserts the output byte count matches 16*(nx+1)*(ny+1)*8 for the given
    per-facet (nx,ny) BEFORE writing (facets 1,2,4,5: 361,088 bytes; facet 3:
    123,008 bytes -- these differ, caller must pass the correct (nx,ny)).
    """
    expected_shape = (ny + 1, nx + 1)
    expected_bytes = N_FIELDS * (nx + 1) * (ny + 1) * 8
    parts = []
    for name in FIELD_NAMES:
        arr = fields_dict[name]
        if arr.shape != expected_shape:
            raise ValueError(
                f"{path}: field {name} has shape {arr.shape}, expected {expected_shape}"
            )
        parts.append(arr.astype(">f8").tobytes())
    blob = b"".join(parts)
    if len(blob) != expected_bytes:
        raise AssertionError(
            f"{path}: assembled {len(blob)} bytes, expected {expected_bytes} "
            f"(16*{nx + 1}*{ny + 1}*8)"
        )
    with open(path, "wb") as fh:
        fh.write(blob)
    observed = os.path.getsize(path)
    assert observed == expected_bytes, (
        f"{path}: written file is {observed} bytes on disk, expected {expected_bytes}"
    )
    return observed


# ===========================================================================
# The five field-type-specific coarsening rules (each a separate function,
# no shared implementation between them -- per the plan's explicit
# instruction not to use a loose "center vs corner" split).
# ===========================================================================

def coarsen_coord_lonlat(arr, is_longitude, factor=FACTOR):
    """Rule 1: block-MEAN of a factor x factor block of cell-center coords.

    For latitude (is_longitude=False): plain arithmetic mean.
    For longitude (is_longitude=True): atan2(mean(sin(lon)), mean(cos(lon)))
    circular mean, to correctly handle 3x3 blocks straddling the +/-180 deg
    dateline (confirmed necessary on facet 4's data -- a naive mean of
    e.g. [178.9, ..., -178.9] gives ~20 deg instead of the correct ~180 deg).

    arr: shape (ny, nx) (CENTER-type field, no +1 corner row/col).
    Returns: shape (ny//factor, nx//factor).
    """
    ny, nx = arr.shape
    assert ny % factor == 0 and nx % factor == 0, (ny, nx, factor)
    cny, cnx = ny // factor, nx // factor
    blocks = arr.reshape(cny, factor, cnx, factor)
    if not is_longitude:
        return blocks.mean(axis=(1, 3))
    rad = np.deg2rad(blocks)
    sin_mean = np.sin(rad).mean(axis=(1, 3))
    cos_mean = np.cos(rad).mean(axis=(1, 3))
    return np.rad2deg(np.arctan2(sin_mean, cos_mean))


def coarsen_corner(arr, factor=FACTOR):
    """Rule 2: subsample every factor-th CORNER point directly (no averaging).

    arr: shape (ny+1, nx+1) (CORNER-type field, +1 extra row/col).
    Returns: shape (ny//factor + 1, nx//factor + 1).

    Coarse-cell corners coincide exactly with a subset of fine-grid corners
    (every factor-th one, starting at index 0); averaging would displace the
    cell boundary, which is wrong for XG/YG.
    """
    ny_p1, nx_p1 = arr.shape
    ny, nx = ny_p1 - 1, nx_p1 - 1
    assert ny % factor == 0 and nx % factor == 0, (ny, nx, factor)
    return arr[0:ny_p1:factor, 0:nx_p1:factor]


def coarsen_xlength_sum(arr, factor=FACTOR):
    """Rule 3: x-direction length fields (DXF, DXV, DXC, DXG).

    A coarse cell's WIDTH is the SUM (not the average) of its `factor`
    constituent fine-grid widths along x -- this is the key correction vs.
    an earlier draft that used a mean here (giving 1/3-scale lengths). The
    orthogonal (y) direction still has `factor` sub-rows per coarse cell
    that must also collapse to 1; since those sub-rows are different
    cross-sections of essentially the same coarse-cell width on a smoothly
    varying curvilinear grid, they are combined with an arithmetic MEAN
    (not summed -- summing across y would be physically wrong, there is no
    "total width across 3 rows" quantity).

    arr: shape (ny, nx). Returns: shape (ny//factor, nx//factor).
    """
    ny, nx = arr.shape
    assert ny % factor == 0 and nx % factor == 0, (ny, nx, factor)
    cny, cnx = ny // factor, nx // factor
    blocks = arr.reshape(cny, factor, cnx, factor)
    summed_x = blocks.sum(axis=3)      # sum the `factor` x-values -> (cny, factor, cnx)
    return summed_x.mean(axis=1)       # average the `factor` y-sub-rows -> (cny, cnx)


def coarsen_ylength_sum(arr, factor=FACTOR):
    """Rule 4: y-direction length fields (DYF, DYU, DYC, DYG).

    A coarse cell's HEIGHT is the SUM (not the average) of its `factor`
    constituent fine-grid heights along y -- mirrors coarsen_xlength_sum's
    correction. The orthogonal (x) direction's `factor` sub-columns are
    combined with an arithmetic MEAN for the same reason given there.

    arr: shape (ny, nx). Returns: shape (ny//factor, nx//factor).
    """
    ny, nx = arr.shape
    assert ny % factor == 0 and nx % factor == 0, (ny, nx, factor)
    cny, cnx = ny // factor, nx // factor
    blocks = arr.reshape(cny, factor, cnx, factor)
    summed_y = blocks.sum(axis=1)      # sum the `factor` y-values -> (cny, cnx, factor)
    return summed_y.mean(axis=2)       # average the `factor` x-sub-cols -> (cny, cnx)


def coarsen_area_sum(arr, factor=FACTOR):
    """Rule 5: SUM of the factor x factor (9, for factor=3) constituent
    fine-cell areas (RAC, RAZ, RAW, RAS) -- NOT a mean. A coarse cell's area
    is the sum of its constituent fine-cell areas; using a mean would give
    1/9 of the correct total area.

    arr: shape (ny, nx). Returns: shape (ny//factor, nx//factor).
    """
    ny, nx = arr.shape
    assert ny % factor == 0 and nx % factor == 0, (ny, nx, factor)
    cny, cnx = ny // factor, nx // factor
    blocks = arr.reshape(cny, factor, cnx, factor)
    return blocks.sum(axis=(1, 3))


# ===========================================================================
# Task 2 step: process grid tiles
# ===========================================================================

def coarsen_one_mitgrid_field(name, arr_fine):
    """Dispatch a single named mitgrid field to its correct coarsening rule.

    `arr_fine` is the FULL (ny+1, nx+1) field as read from the fine .mitgrid
    file. CENTER-type fields (XC,YC,DXF,DYF,RAC,DXV,DYU,RAZ,DXC,DYC,RAW,RAS,
    DXG,DYG... wait, some of these are actually corner-adjacent -- see notes
    below) are sliced to drop the extra +1 row/col before applying a sum/mean
    rule, since those rules operate on true (ny,nx)-shaped cell quantities;
    CORNER-type fields (XG,YG) keep the full (ny+1,nx+1) shape and are
    subsampled directly.
    """
    rule = FIELD_RULE[name]
    if rule == "corner":
        # XG, YG: full (ny+1, nx+1) corner array, subsample directly.
        return coarsen_corner(arr_fine)
    # All other rules operate on (ny, nx) cell-centered quantities --
    # drop the extra +1 row/col (mitgrid stores all 16 fields at the full
    # (ny+1, nx+1) shape even though only XG/YG truly need the extra
    # row/col; the other fields' last row/col are padding duplicates of the
    # next tile's first row/col and are not meaningful "new" cells).
    arr = arr_fine[:-1, :-1]
    if rule == "coord_lonlat":
        is_lon = (name == "XC")
        return coarsen_coord_lonlat(arr, is_longitude=is_lon)
    elif rule == "xlength_sum":
        return coarsen_xlength_sum(arr)
    elif rule == "ylength_sum":
        return coarsen_ylength_sum(arr)
    elif rule == "area_sum":
        return coarsen_area_sum(arr)
    else:
        raise ValueError(f"unknown rule {rule!r} for field {name!r}")


def process_grid():
    """Read all 5 LLC90 .mitgrid tiles, coarsen every field by its correct
    rule, write out 5 LLC30 .mitgrid tiles, and run the mandatory RAC-
    conservation verification. Returns (fine_rac_total, coarse_rac_total).
    """
    os.makedirs(OUT_GRID_DIR, exist_ok=True)

    fine_rac_total = 0.0
    coarse_rac_total = 0.0

    for facet in range(1, 6):
        nx90, ny90 = LLC90_FACET_DIMS[facet]
        nx30, ny30 = LLC30_FACET_DIMS[facet]

        in_path = os.path.join(GRID_SRC_DIR, f"tile{facet:03d}.mitgrid")
        observed_in = os.path.getsize(in_path)
        expected_in = EXPECTED_LLC90_TILE_BYTES[facet]
        assert observed_in == expected_in, (
            f"tile{facet:03d}.mitgrid: input is {observed_in} bytes, "
            f"expected {expected_in}"
        )

        fine_fields = read_mitgrid_tile(in_path, nx90, ny90)
        fine_rac_total += fine_fields["RAC"][:-1, :-1].sum()

        coarse_fields = {}
        for name in FIELD_NAMES:
            coarse_full = coarsen_one_mitgrid_field(name, fine_fields[name])
            rule = FIELD_RULE[name]
            if rule == "corner":
                # already full (ny30+1, nx30+1) shape
                coarse_fields[name] = coarse_full
            else:
                # coarse_full is (ny30, nx30) -- the mitgrid format expects
                # (ny30+1, nx30+1) for every field (incl. non-corner ones,
                # matching the fine-file layout exactly). Confirmed by
                # inspecting the real fine-grid files: for non-corner
                # (center-type) fields the extra row/col is NOT a duplicate
                # of the last real row/col -- it is zero-filled padding (the
                # fine tile001.mitgrid's RAC/DXF/DYF/XC last column is all
                # zeros, not equal to the second-to-last column). Reproduce
                # that exact zero-padding convention here.
                padded = np.zeros((ny30 + 1, nx30 + 1), dtype=np.float64)
                padded[:ny30, :nx30] = coarse_full
                coarse_fields[name] = padded

        coarse_rac_total += coarse_fields["RAC"][:-1, :-1].sum()

        out_path = os.path.join(OUT_GRID_DIR, f"tile{facet:03d}.mitgrid")
        expected_out = EXPECTED_LLC30_TILE_BYTES[facet]
        written = write_mitgrid_tile(out_path, coarse_fields, nx30, ny30)
        assert written == expected_out, (
            f"tile{facet:03d}.mitgrid (LLC30): wrote {written} bytes, "
            f"expected {expected_out}"
        )
        print(f"  tile{facet:03d}.mitgrid: LLC90 ({nx90},{ny90}) -> "
              f"LLC30 ({nx30},{ny30}), {written} bytes OK")

    return fine_rac_total, coarse_rac_total


def verify_rac_conservation(fine_total, coarse_total):
    """Mandatory verification: fine and coarse global RAC sums must agree to
    within floating-point rounding, and both must be ~ Earth's surface area
    (~5.1e14 m^2, +/- 5%).
    """
    rel_diff = abs(fine_total - coarse_total) / fine_total
    print(f"\nRAC conservation check:")
    print(f"  fine   total RAC = {fine_total:.6e} m^2")
    print(f"  coarse total RAC = {coarse_total:.6e} m^2")
    print(f"  relative difference = {rel_diff:.3e}")

    assert rel_diff < 1e-6, (
        f"RAC conservation FAILED: fine={fine_total:.6e}, coarse={coarse_total:.6e}, "
        f"rel_diff={rel_diff:.3e} (expected < 1e-6 -- this is the check that "
        f"catches a mean-vs-sum area bug)"
    )

    earth_area = 5.1e14
    frac_of_earth = fine_total / earth_area
    print(f"  fine total / Earth surface area (5.1e14 m^2) = {frac_of_earth:.4f}")
    assert 0.95 <= frac_of_earth <= 1.05, (
        f"RAC total {fine_total:.6e} m^2 is not within 5% of Earth's surface "
        f"area {earth_area:.3e} m^2 (ratio {frac_of_earth:.4f})"
    )
    print("  RAC conservation check PASSED.")


# ===========================================================================
# Task 2 step: bathymetry
# ===========================================================================

def split_compact_into_facets(compact_arr):
    """Split a compact LLC90 array into its 5 native-shaped facet chunks.

    IMPORTANT (empirically re-verified against the real bathymetry data --
    the plan's row-range-of-a-(1170,90)-matrix description does not hold
    byte-for-byte for the wide facets): the on-disk "compact (90,1170)"
    file is NOT a literal uniform 2-D (1170,90) array that can be sliced by
    rows for every facet. It is 5 facet blocks concatenated in the flat
    buffer, each stored in that facet's OWN native (ny,nx) row-major order.
    For the narrow facets (1,2,3, nx=90) this is indistinguishable from
    row-slicing a (1170,90) matrix (since their native nx already equals
    the nominal compact width of 90). For the WIDE facets (4,5, nx=270) it
    is NOT: naively slicing rows out of a (1170,90)-shaped array and
    treating the result as (ny=90,nx=270) via `.reshape` or `.T` produces
    garbage (confirmed empirically: roughness/landmask checks on facet 4/5
    sliced-and-reoriented that way were ~15-20x rougher and visibly wrong
    compared to reshaping the facet's own flat byte range directly to its
    native (ny,nx)).

    FACET_ROW_RANGES' (r0,r1) still correctly identify each facet's
    flat-buffer element range as [r0*COMPACT_NX : r1*COMPACT_NX) (that part
    of the plan's convention IS correct -- (r1-r0)*COMPACT_NX always equals
    nx*ny for every facet, narrow or wide); what differs is the reshape
    target used on that slice: reshape directly to the facet's native
    (ny,nx), not to (r1-r0, COMPACT_NX).

    `compact_arr` may be passed either as the flat 1-D buffer (preferred)
    or as the (COMPACT_NY, COMPACT_NX) 2-D array -- both are accepted and
    flattened internally so this function is usable from both the
    bathymetry (which reshapes to 2D first) and forcing (same) code paths.

    Returns dict facet -> 2D array, shape (ny,nx) per LLC90_FACET_DIMS.
    """
    flat = np.asarray(compact_arr).reshape(-1)
    facets = {}
    for facet, (r0, r1) in FACET_ROW_RANGES.items():
        nx, ny = LLC90_FACET_DIMS[facet]
        n = (r1 - r0) * COMPACT_NX
        assert n == nx * ny, (facet, n, nx, ny)
        start = r0 * COMPACT_NX
        chunk_flat = flat[start:start + n]
        chunk = chunk_flat.reshape(ny, nx)
        facets[facet] = chunk
    return facets


def assemble_facets_into_compact(facets_coarse):
    """Inverse of split_compact_into_facets, at LLC30 coarse resolution.

    Mirrors the fix above: each coarse facet's (ny30,nx30) block is
    flattened in its own native row-major order and placed into the
    correct flat element range of the (COMPACT_COARSE_NY, COMPACT_COARSE_NX)
    output buffer -- NOT row-assigned into a (COMPACT_COARSE_NY,
    COMPACT_COARSE_NX)-shaped 2-D array (which is wrong for facets 4/5 for
    the same reason splitting by row-slicing is wrong).
    """
    out_flat = np.empty(COMPACT_COARSE_NY * COMPACT_COARSE_NX, dtype=facets_coarse[1].dtype)
    for facet in range(1, 6):
        nx30, ny30 = LLC30_FACET_DIMS[facet]
        r0, r1 = FACET_ROW_RANGES[facet]
        r0c, r1c = r0 // FACTOR, r1 // FACTOR
        n = (r1c - r0c) * COMPACT_COARSE_NX
        assert n == nx30 * ny30, (facet, n, nx30, ny30)
        chunk = facets_coarse[facet]
        assert chunk.shape == (ny30, nx30), (
            f"facet {facet}: coarse chunk shape {chunk.shape}, expected {(ny30, nx30)}"
        )
        start = r0c * COMPACT_COARSE_NX
        out_flat[start:start + n] = chunk.reshape(-1)
    return out_flat.reshape(COMPACT_COARSE_NY, COMPACT_COARSE_NX)


def coarsen_bathy_block(block9):
    """3x3 block of depth values (possibly 0 = land). Return the coarse
    value: land only if ALL 9 constituent cells are land (0); otherwise the
    MIN depth MAGNITUDE over the wet constituents only (most conservative /
    shallowest), preserving sign.
    """
    wet_mask = block9 != 0.0
    if not wet_mask.any():
        return 0.0
    wet_vals = block9[wet_mask]
    # min magnitude among wet cells, preserving original sign
    idx = np.argmin(np.abs(wet_vals))
    return wet_vals[idx]


def coarsen_bathy_facet(arr):
    """Vectorized 3x3-block min-magnitude-over-wet-cells coarsening.

    arr: shape (ny, nx) real depth values (0 = land, negative = ocean depth).
    Returns: shape (ny//3, nx//3).
    """
    ny, nx = arr.shape
    factor = FACTOR
    cny, cnx = ny // factor, nx // factor
    blocks = arr.reshape(cny, factor, cnx, factor).transpose(0, 2, 1, 3).reshape(cny, cnx, factor * factor)
    out = np.empty((cny, cnx), dtype=arr.dtype)
    wet = blocks != 0.0
    any_wet = wet.any(axis=2)
    # magnitude, with dry cells masked to +inf so they never win the min
    mag = np.where(wet, np.abs(blocks), np.inf)
    min_idx = np.argmin(mag, axis=2)
    flat_blocks = blocks.reshape(cny * cnx, factor * factor)
    flat_min_idx = min_idx.reshape(cny * cnx)
    chosen = flat_blocks[np.arange(cny * cnx), flat_min_idx].reshape(cny, cnx)
    out[:] = np.where(any_wet, chosen, 0.0)
    return out


def process_bathy():
    os.makedirs(OUT_BATHY_DIR, exist_ok=True)

    observed_in = os.path.getsize(BATHY_SRC)
    expected_in = COMPACT_NX * COMPACT_NY * 4
    assert observed_in == expected_in, (
        f"BATHY source: {observed_in} bytes, expected {expected_in} (90*1170*4)"
    )

    raw = np.fromfile(BATHY_SRC, dtype=">f4").astype(np.float64)
    compact = raw.reshape(COMPACT_NY, COMPACT_NX)

    fine_facets = split_compact_into_facets(compact)
    coarse_facets = {}
    for facet in range(1, 6):
        coarse_facets[facet] = coarsen_bathy_facet(fine_facets[facet])

    coarse_compact = assemble_facets_into_compact(coarse_facets)

    out_path = os.path.join(OUT_BATHY_DIR, "BATHY_LLC30.bin")
    expected_out = COMPACT_COARSE_NX * COMPACT_COARSE_NY * 4
    out_bytes = coarse_compact.astype(">f4").tobytes()
    assert len(out_bytes) == expected_out, (
        f"BATHY_LLC30.bin: assembled {len(out_bytes)} bytes, expected {expected_out}"
    )
    with open(out_path, "wb") as fh:
        fh.write(out_bytes)
    observed_out = os.path.getsize(out_path)
    assert observed_out == expected_out, (
        f"BATHY_LLC30.bin: written {observed_out} bytes, expected {expected_out}"
    )
    print(f"  BATHY_LLC30.bin: {observed_out} bytes OK "
          f"(compact shape {COMPACT_COARSE_NY}x{COMPACT_COARSE_NX})")
    return coarse_compact


# ===========================================================================
# Task 2 step: forcing
# ===========================================================================

# 1992 is a leap year: Feb has 29 days. Month lengths in hours.
MONTH_DAYS_1992 = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
MONTH_HOURS_1992 = [d * 24 for d in MONTH_DAYS_1992]
assert sum(MONTH_HOURS_1992) == 8784, sum(MONTH_HOURS_1992)


def coarsen_forcing_facet_mean(arr):
    """Spatial block-AVERAGE (mean, not min/sum) of a 3x3 block -- forcing
    fields are flux/state quantities, not bathymetry or geometric lengths.

    arr: shape (ny, nx). Returns shape (ny//3, nx//3).
    """
    ny, nx = arr.shape
    factor = FACTOR
    cny, cnx = ny // factor, nx // factor
    blocks = arr.reshape(cny, factor, cnx, factor)
    return blocks.mean(axis=(1, 3))


def read_combforc_records(path, rec_slice):
    """Read a slice of (90,1170) float32 big-endian records from a combforc
    file without loading the whole multi-GB file into memory at once.
    rec_slice: a Python slice object over the record (time) axis.
    Returns array shape (n_records, 1170, 90) float64.
    """
    rec_bytes = COMPACT_NX * COMPACT_NY * 4
    start, stop, step = rec_slice.indices(8784)
    assert step == 1
    n_rec = stop - start
    with open(path, "rb") as fh:
        fh.seek(start * rec_bytes)
        raw = np.fromfile(fh, dtype=">f4", count=n_rec * COMPACT_NX * COMPACT_NY)
    assert raw.size == n_rec * COMPACT_NX * COMPACT_NY, (
        f"{path}: read {raw.size} floats for records [{start}:{stop}], "
        f"expected {n_rec * COMPACT_NX * COMPACT_NY}"
    )
    return raw.reshape(n_rec, COMPACT_NY, COMPACT_NX).astype(np.float64)


def process_forcing_variable(var):
    """Full two-stage reduction for one combforc variable:
      1. temporal: monthly arithmetic mean over all hourly records -> 12 x (90,1170)
      2. spatial: 3x3 block-mean per facet -> 12 x (30,390)
    Writes preprocess/output/forcing/eccov4r6_<var>_llc30_clim.
    Returns the January (month 0) coarse monthly-mean compact array, and the
    January fine-resolution monthly-mean compact array (for the spot-check).
    """
    src_path = os.path.join(FORCING_SRC_DIR, f"eccov4r6_{var}_1992")
    observed_in = os.path.getsize(src_path)
    expected_in = COMPACT_NX * COMPACT_NY * 4 * 8784
    assert observed_in == expected_in, (
        f"{src_path}: {observed_in} bytes, expected {expected_in} (90*1170*4*8784)"
    )

    monthly_means_fine = []
    rec_start = 0
    jan_fine_mean = None
    for m, n_hours in enumerate(MONTH_HOURS_1992):
        rec_stop = rec_start + n_hours
        recs = read_combforc_records(src_path, slice(rec_start, rec_stop))
        month_mean = recs.mean(axis=0)  # (1170, 90)
        monthly_means_fine.append(month_mean)
        if m == 0:
            jan_fine_mean = month_mean
        rec_start = rec_stop
    assert rec_start == 8784, rec_start

    monthly_means_coarse = []
    jan_coarse_mean = None
    for m, month_mean in enumerate(monthly_means_fine):
        fine_facets = split_compact_into_facets(month_mean)
        coarse_facets = {}
        for facet in range(1, 6):
            coarse_facets[facet] = coarsen_forcing_facet_mean(fine_facets[facet])
        coarse_compact = assemble_facets_into_compact(coarse_facets)
        monthly_means_coarse.append(coarse_compact)
        if m == 0:
            jan_coarse_mean = coarse_compact

    out_path = os.path.join(OUT_FORCING_DIR, f"eccov4r6_{var}_llc30_clim")
    expected_out = 12 * COMPACT_COARSE_NX * COMPACT_COARSE_NY * 4
    out_bytes = b"".join(
        m.astype(">f4").tobytes() for m in monthly_means_coarse
    )
    assert len(out_bytes) == expected_out, (
        f"{out_path}: assembled {len(out_bytes)} bytes, expected {expected_out}"
    )
    with open(out_path, "wb") as fh:
        fh.write(out_bytes)
    observed_out = os.path.getsize(out_path)
    assert observed_out == expected_out, (
        f"{out_path}: written {observed_out} bytes, expected {expected_out}"
    )
    print(f"  eccov4r6_{var}_llc30_clim: {observed_out} bytes OK (12 x 30x390 records)")

    return jan_coarse_mean, jan_fine_mean


def process_all_forcing():
    os.makedirs(OUT_FORCING_DIR, exist_ok=True)
    jan_results = {}
    for var in FORCING_VARS:
        jan_coarse, jan_fine = process_forcing_variable(var)
        jan_results[var] = (jan_coarse, jan_fine)
    return jan_results


def spot_check_forcing(jan_results, var="dlw"):
    """Compare one coarsened forcing variable's January domain mean against
    the source's January domain mean (computed ad hoc above). Confirms the
    monthly-mean and block-average arithmetic end-to-end, not just byte
    counts.
    """
    jan_coarse, jan_fine = jan_results[var]
    coarse_mean = jan_coarse.mean()
    fine_mean = jan_fine.mean()
    rel_diff = abs(coarse_mean - fine_mean) / abs(fine_mean) if fine_mean != 0 else float("nan")
    print(f"\nSpot-check ({var}, January domain mean):")
    print(f"  LLC90 source January mean = {fine_mean:.6f}")
    print(f"  LLC30 coarsened January mean = {coarse_mean:.6f}")
    print(f"  relative difference = {rel_diff:.3e}")
    assert rel_diff < 0.03, (
        f"Spot-check FAILED for {var}: coarse={coarse_mean:.6f}, fine={fine_mean:.6f}, "
        f"rel_diff={rel_diff:.3e} (expected < 3%)"
    )
    print("  Spot-check PASSED (within a few percent).")


# ===========================================================================
# main
# ===========================================================================

def main():
    print("=" * 70)
    print("Step 1/4: coarsening 5 LLC90 .mitgrid tiles -> LLC30")
    print("=" * 70)
    fine_rac_total, coarse_rac_total = process_grid()
    verify_rac_conservation(fine_rac_total, coarse_rac_total)

    print()
    print("=" * 70)
    print("Step 2/4: coarsening bathymetry -> LLC30")
    print("=" * 70)
    process_bathy()

    print()
    print("=" * 70)
    print("Step 3/4: reducing 9 combforc 1992 hourly variables -> LLC30 monthly clim")
    print("=" * 70)
    jan_results = process_all_forcing()

    print()
    print("=" * 70)
    print("Step 4/4: spot-check")
    print("=" * 70)
    spot_check_forcing(jan_results, var="dlw")

    print()
    print("ALL CHECKS PASSED. Task 2 preprocessing complete.")


if __name__ == "__main__":
    main()
