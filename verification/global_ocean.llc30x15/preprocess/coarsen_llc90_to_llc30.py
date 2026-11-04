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

Coarsening rules (6 distinct field-type classes -- see Technical Details in
the plan for the full justification of why a naive "center-mean vs.
corner-subsample" split is wrong for lengths/areas):

  1. coarsen_coord_lonlat  -- XC, YC (cell-center lon/lat): block-MEAN of a
     3x3 block. XC uses atan2(mean(sin), mean(cos)) to avoid a dateline
     wrap bug; YC uses a plain arithmetic mean.
  2. coarsen_corner        -- XG, YG (cell-corner lon/lat): subsample every
     3rd point directly (coarse corners coincide with a subset of fine
     corners; averaging would displace the cell boundary).
  3. coarsen_xlength_sum   -- DXF, DXC, DXG (x-direction lengths): SUM
     of 3 consecutive values along x.
  4. coarsen_ylength_sum   -- DYF, DYC, DYG (y-direction lengths): SUM
     of 3 consecutive values along y.
  5. coarsen_area_sum      -- RAC, RAW, RAS (areas): SUM of the 9
     constituent fine-cell areas in each 3x3 block.
  6. coarsen_corner_indexed_windowed_sum -- DXV, DYU, RAZ: these are
     corner-INDEXED (full (ny+1,nx+1) extent, no padding at all, same as
     XG/YG) but are PHYSICALLY length/area quantities, not positions --
     unlike XG/YG, subsampling them is WRONG (confirmed empirically: it
     silently deflates RAZ by ~9x relative to the correct coarse-cell-area
     magnitude, since RAZ needs to represent the AREA of a coarse corner
     cell, not copy one fine corner cell's area -- this caused a real
     numerical blow-up in mom_vecinv's vorticity term, 1/rAz, during this
     fix's own smoke-test rebuild). Each coarse value (beyond the shared
     boundary point at index 0, which is a direct copy) is the SUM (DXV:
     along x; DYU: along y; RAZ: along both) of the 3 (or 3x3) fine values
     immediately following it, with the ORTHOGONAL axis combined by MEAN
     (DXV/DYU only; RAZ sums both axes) -- see
     coarsen_corner_indexed_windowed_sum's docstring for the exact window
     convention (it is NOT a simple reshape-into-blocks like the other
     sum rules, because corner-indexed arrays have one extra point that
     doesn't fit evenly into 3x3 blocks the way center/face quantities do).

Padding/boundary handling (separate from the above rule -- see
PADDING_CLASS): a fine-grid field's extra "+1" row/column is NOT uniformly
zero-padding for every field. There are 4 distinct patterns, confirmed by
direct inspection of the real tile001.mitgrid data:
  - "both_pad" (XC, YC, DXF, DYF, RAC): last row AND column are padding.
  - "y_pad_x_real" (DXC, DYG, RAW): only the last ROW is padding; the last
    COLUMN holds real boundary data that must be coarsened, not discarded.
  - "x_pad_y_real" (DYC, DXG, RAS): mirror -- only the last COLUMN is
    padding, the last ROW is real.
  - "corner" (XG, YG): neither row nor column is padding -- both are real
    corner-POSITION data over the full (ny+1,nx+1) extent, subsampled.
  - "corner_indexed_real" (DXV, DYU, RAZ): also neither row nor column is
    padding (full real extent, like XG/YG) but these are length/area
    QUANTITIES, not positions, so they use rule 6 above (windowed sum),
    not plain subsampling.

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
#
# IMPORTANT (fix for a real data-integrity defect found in code review -- see
# PADDING_CLASS below and coarsen_one_mitgrid_field's docstring): this maps
# each field to its SUM/MEAN/SUBSAMPLE rule family only. It does NOT by
# itself say which of the field's +1 corner row/column is zero-padding vs.
# real boundary data -- that is a SEPARATE, per-field classification
# (PADDING_CLASS below), confirmed empirically against tile001.mitgrid.
FIELD_RULE = {
    "XC": "coord_lonlat", "YC": "coord_lonlat",
    "XG": "corner", "YG": "corner",
    # DXV, DYU, RAZ: corner-INDEXED (full extent, like XG/YG) but are
    # length/area QUANTITIES, not positions -- plain subsampling ("corner"
    # rule) deflates them by ~factor^(ndims) since it picks one fine value
    # instead of summing the `factor` (or factor^2) fine values a coarse
    # corner-cell length/area actually spans. Confirmed by an earlier
    # attempt at this exact fix: classifying these as "corner" (subsample)
    # caused a real numerical blow-up (uvel_max growing >100x within 9
    # timesteps, CALC_R_STAR abort) in this config's smoke-test rebuild,
    # traced to mom_vecinv's vorticity term dividing by a RAZ that was ~9x
    # too small. Use the dedicated windowed-sum rule instead (rule 6 in
    # the module docstring).
    "DXV": "corner_indexed_windowed_sum",
    "DYU": "corner_indexed_windowed_sum",
    "RAZ": "corner_indexed_windowed_sum",
    "DXF": "xlength_sum", "DXC": "xlength_sum", "DXG": "xlength_sum",
    "DYF": "ylength_sum", "DYC": "ylength_sum", "DYG": "ylength_sum",
    "RAC": "area_sum", "RAW": "area_sum", "RAS": "area_sum",
}

# ---------------------------------------------------------------------------
# Per-field padding/boundary classification (fix for code-review finding:
# the previous code assumed a single blanket pattern -- "last row AND last
# column are always zero-padding" -- for every non-corner field. Empirically
# false for 9 of the 14 non-corner fields, confirmed by direct inspection of
# tile001.mitgrid's real data (compare each field's last-row/last-column
# mean magnitude against its interior mean magnitude -- see the progress
# log entry for this fix for the exact numbers). There are actually 4
# distinct patterns:
#
#   "both_pad"    : last ROW and last COLUMN are both zero-padding.
#                   XC, YC, DXF, DYF, RAC.
#   "y_pad_x_real": last ROW is zero-padding; last COLUMN holds REAL,
#                   non-zero boundary data (a genuine extra wall/face
#                   position in X that must be preserved, not discarded).
#                   DXC, DYG, RAW.
#   "x_pad_y_real": mirror of the above -- last COLUMN is zero-padding;
#                   last ROW holds REAL boundary data in Y.
#                   DYC, DXG, RAS.
#   "corner"      : NEITHER row nor column is padding -- both hold REAL
#                   corner-POSITION data over the full (ny+1,nx+1) extent.
#                   XG, YG.
#   "corner_indexed_real": also neither row nor column is padding (full
#                   real extent, same as "corner"), but these are
#                   length/area QUANTITIES, not positions -- handled by
#                   the dedicated windowed-sum rule, not subsampling.
#                   DXV, DYU, RAZ.
# ---------------------------------------------------------------------------
PADDING_CLASS = {
    "XC": "both_pad", "YC": "both_pad", "DXF": "both_pad", "DYF": "both_pad",
    "RAC": "both_pad",
    "DXC": "y_pad_x_real", "DYG": "y_pad_x_real", "RAW": "y_pad_x_real",
    "DYC": "x_pad_y_real", "DXG": "x_pad_y_real", "RAS": "x_pad_y_real",
    "XG": "corner", "YG": "corner",
    "DXV": "corner_indexed_real", "DYU": "corner_indexed_real",
    "RAZ": "corner_indexed_real",
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
    """Rule 3: x-direction length fields (DXF, DXC, DXG -- DXV uses the
    dedicated coarsen_corner_indexed_windowed_sum rule instead, see there).

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
    """Rule 4: y-direction length fields (DYF, DYC, DYG -- DYU uses the
    dedicated coarsen_corner_indexed_windowed_sum rule instead, see there).

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
    fine-cell areas (RAC, RAW, RAS -- RAZ uses the dedicated
    coarsen_corner_indexed_windowed_sum rule instead, see there) -- NOT a
    mean. A coarse cell's area is the sum of its constituent fine-cell
    areas; using a mean would give 1/9 of the correct total area.

    arr: shape (ny, nx). Returns: shape (ny//factor, nx//factor).
    """
    ny, nx = arr.shape
    assert ny % factor == 0 and nx % factor == 0, (ny, nx, factor)
    cny, cnx = ny // factor, nx // factor
    blocks = arr.reshape(cny, factor, cnx, factor)
    return blocks.sum(axis=(1, 3))


def _windowed_reduce_1d(arr_1d, how, factor=FACTOR):
    """Helper for coarsen_corner_indexed_windowed_sum: collapse a 1-D array
    of length (n+1) (n % factor == 0, so n+1 is NOT evenly divisible by
    factor -- this is the key difference from reduce_1d_group, which
    assumes even divisibility) into (n//factor + 1) values:

      out[0]    = arr_1d[0]                              (shared boundary
                   point, copied directly -- this is the same physical
                   corner point the NEIGHBORING coarse cell's own out[0]
                   or out[-1] will also reference, exactly like coarsen_
                   corner's direct subsample of XG/YG)
      out[k]    = reduce(arr_1d[(k-1)*factor+1 : k*factor+1])   for k=1..n//factor

    i.e. index 0 is a standalone shared point, and each subsequent coarse
    index summarizes the `factor` fine points immediately following it.
    This is the corner-indexed analogue of reduce_1d_group, needed because
    DXV/DYU/RAZ have one MORE point than a clean multiple of `factor`
    (they are corner-indexed, full (ny+1,nx+1) extent, not (ny,nx)).
    """
    n = arr_1d.shape[0] - 1
    assert n % factor == 0, (arr_1d.shape, factor)
    cn = n // factor
    out = np.empty(cn + 1, dtype=arr_1d.dtype)
    out[0] = arr_1d[0]
    for k in range(1, cn + 1):
        seg = arr_1d[(k - 1) * factor + 1: k * factor + 1]
        out[k] = seg.sum() if how == "sum" else seg.mean()
    return out


def coarsen_corner_indexed_windowed_sum(arr, own_axes, factor=FACTOR):
    """Rule 6: DXV, DYU, RAZ -- corner-INDEXED (full (ny+1,nx+1) extent, no
    padding, same shape convention as XG/YG) length/area QUANTITIES.

    Confirmed WRONG to subsample these like XG/YG (positions): subsampling
    picks one fine-grid value where the coarse quantity should represent
    the length/area spanning `factor` (or `factor`x`factor`) fine cells,
    deflating DXV/DYU by ~factor and RAZ by ~factor^2. This was caught by
    a real smoke-test rebuild blow-up (see FIELD_RULE's comment) -- an
    empirical confirmation, not just a priori reasoning.

    arr: shape (ny+1, nx+1), full real corner-indexed data (no padding).
    own_axes: tuple of axes (0=y, 1=x) along which the field's own
      physical quantity is summed:
        DXV (x-length):  own_axes=(1,)   -- sum along x, mean along y
        DYU (y-length):  own_axes=(0,)   -- sum along y, mean along x
        RAZ (area):      own_axes=(0,1)  -- sum along BOTH axes

    Returns: shape (ny//factor + 1, nx//factor + 1), using the same
    "shared boundary point at index 0, windowed-sum of `factor` points for
    each subsequent index" convention as _windowed_reduce_1d on each axis
    in turn (this is NOT a simple reshape-into-blocks like coarsen_
    xlength_sum/coarsen_area_sum, because a corner-indexed array's extent
    is ny+1/nx+1, one more than an even multiple of `factor`).
    """
    ny_p1, nx_p1 = arr.shape
    how_x = "sum" if 1 in own_axes else "mean"
    how_y = "sum" if 0 in own_axes else "mean"
    nx = nx_p1 - 1
    cnx = nx // factor
    # Step 1: reduce along x for every fine row -> shape (ny+1, cnx+1)
    tmp = np.empty((ny_p1, cnx + 1), dtype=arr.dtype)
    for j in range(ny_p1):
        tmp[j, :] = _windowed_reduce_1d(arr[j, :], how_x, factor)
    # Step 2: reduce along y for every (already x-reduced) column
    ny = ny_p1 - 1
    cny = ny // factor
    out = np.empty((cny + 1, cnx + 1), dtype=arr.dtype)
    for i in range(cnx + 1):
        out[:, i] = _windowed_reduce_1d(tmp[:, i], how_y, factor)
    return out


def reduce_1d_group(arr_1d, how, factor=FACTOR):
    """Collapse a 1-D array of length n (n % factor == 0) into n//factor
    values by summing or averaging each consecutive group of `factor`
    elements. Used by coarsen_one_mitgrid_field to reduce the single real
    "extra" boundary row (for "x_pad_y_real" fields) or column (for
    "y_pad_x_real" fields) along its remaining fine dimension.
    """
    n = arr_1d.shape[0]
    assert n % factor == 0, (n, factor)
    g = arr_1d.reshape(n // factor, factor)
    if how == "sum":
        return g.sum(axis=1)
    elif how == "mean":
        return g.mean(axis=1)
    else:
        raise ValueError(f"unknown reduction {how!r}")


# For each of the 6 fields with a lone REAL extra boundary row/column
# (PADDING_CLASS "y_pad_x_real" or "x_pad_y_real"), the reduction (sum vs
# mean) to apply when collapsing that extra strip's `factor` fine values
# per coarse value. Determined by whether the strip's remaining fine
# dimension is the field's own summed direction (-> "sum") or its
# orthogonal, mean-combined direction (-> "mean"), per coarsen_xlength_sum/
# coarsen_ylength_sum's own documented convention; area fields always sum.
#   DXC (y_pad_x_real, xlength_sum): extra column, reduce along y (DXC's
#       orthogonal/mean axis) -> "mean".
#   DYG (y_pad_x_real, ylength_sum): extra column, reduce along y (DYG's
#       own/sum axis) -> "sum".
#   RAW (y_pad_x_real, area_sum): extra column, reduce along y -> "sum".
#   DYC (x_pad_y_real, ylength_sum): extra row, reduce along x (DYC's
#       orthogonal/mean axis) -> "mean".
#   DXG (x_pad_y_real, xlength_sum): extra row, reduce along x (DXG's
#       own/sum axis) -> "sum".
#   RAS (x_pad_y_real, area_sum): extra row, reduce along x -> "sum".
EXTRA_STRIP_REDUCTION = {
    "DXC": "mean", "DYG": "sum", "RAW": "sum",
    "DYC": "mean", "DXG": "sum", "RAS": "sum",
}


# ===========================================================================
# Task 2 step: process grid tiles
# ===========================================================================

def coarsen_one_mitgrid_field(name, arr_fine):
    """Dispatch a single named mitgrid field to its correct coarsening rule,
    AND to its correct padding-handling path (PADDING_CLASS).

    `arr_fine` is the FULL (ny+1, nx+1) field as read from the fine .mitgrid
    file. A previous version of this function assumed a single blanket
    pattern -- "every non-corner field's last row AND last column are
    zero-padding" -- and truncated `arr_fine[:-1, :-1]` for all of them.
    That is empirically WRONG for 9 of the 14 non-"corner"-rule fields
    (confirmed on tile001.mitgrid's real data -- see PADDING_CLASS's
    docstring/comment above): it silently discarded real boundary-adjacent
    grid-metric data and replaced it with zeros at facet seams.

    There are 5 distinct cases, handled separately here:

      "corner"       (XG, YG): the FULL (ny+1,nx+1) array is all real
          corner-POSITION data, no padding at all; subsampled directly
          (coarsen_corner) -- summing/averaging would displace the cell
          boundary.
      "corner_indexed_real" (DXV, DYU, RAZ): also the FULL (ny+1,nx+1)
          array, no padding, but these are length/area QUANTITIES, not
          positions (V-point/U-point separations and the corner-cell
          area). Subsampling them like XG/YG is WRONG -- confirmed by a
          real numerical blow-up (mom_vecinv's 1/rAz vorticity term
          exploding within 9 smoke-test timesteps) when an earlier
          version of this fix classified them as plain "corner": a
          coarse RAZ needs to represent the AREA of a coarse corner cell
          (summing ~9 fine areas), not copy one fine corner cell's area.
          Handled by coarsen_corner_indexed_windowed_sum (rule 6).
      "both_pad"     (XC, YC, DXF, DYF, RAC): confirmed last ROW and last
          COLUMN are both zero-padding; truncate both via [:-1,:-1] before
          applying the field's sum/mean rule, then the caller re-pads a
          zero row AND column.
      "y_pad_x_real" (DXC, DYG, RAW): only the last ROW is zero-padding --
          the last COLUMN holds real, non-zero boundary data of the same
          magnitude as the interior (confirmed empirically) and must be
          included in the coarsening, not discarded: the real (ny,1) extra
          column is reduced along y to (ny//factor,1) by EXTRA_STRIP_
          REDUCTION[name] and placed as the coarse output's real last
          column; only the (now-)last ROW is zero-padded.
      "x_pad_y_real" (DYC, DXG, RAS): mirror of the above -- only the last
          COLUMN is zero-padding, the last ROW is real: the real (1,nx)
          extra row is reduced along x to (1,nx//factor) and placed as the
          coarse output's real last row; only the last COLUMN is
          zero-padded.

    Returns the FULL (ny30+1, nx30+1) coarse field, ready to write
    directly (caller does not need to re-pad anything -- all 5 classes'
    padding/real-boundary handling is done here).
    """
    rule = FIELD_RULE[name]
    pad_class = PADDING_CLASS[name]
    ny_p1, nx_p1 = arr_fine.shape
    ny, nx = ny_p1 - 1, nx_p1 - 1
    cny, cnx = ny // FACTOR, nx // FACTOR

    if pad_class == "corner":
        # XG, YG: full (ny+1,nx+1) array, no padding at all, subsample
        # every 3rd corner point -- already returns the full
        # (cny+1, cnx+1) target shape.
        assert rule == "corner", (name, rule)
        return coarsen_corner(arr_fine)

    if pad_class == "corner_indexed_real":
        # DXV, DYU, RAZ: full (ny+1,nx+1) array, no padding, but a
        # length/area quantity -- windowed sum/mean, not subsample.
        assert rule == "corner_indexed_windowed_sum", (name, rule)
        own_axes = {"DXV": (1,), "DYU": (0,), "RAZ": (0, 1)}[name]
        return coarsen_corner_indexed_windowed_sum(arr_fine, own_axes=own_axes)

    def apply_rule(arr):
        if rule == "coord_lonlat":
            return coarsen_coord_lonlat(arr, is_longitude=(name == "XC"))
        elif rule == "xlength_sum":
            return coarsen_xlength_sum(arr)
        elif rule == "ylength_sum":
            return coarsen_ylength_sum(arr)
        elif rule == "area_sum":
            return coarsen_area_sum(arr)
        else:
            raise ValueError(f"unknown rule {rule!r} for field {name!r}")

    if pad_class == "both_pad":
        # XC, YC, DXF, DYF, RAC: last row AND column are zero-padding in
        # both the fine input and (by construction) the coarse output.
        interior = apply_rule(arr_fine[:-1, :-1])
        out = np.zeros((cny + 1, cnx + 1), dtype=arr_fine.dtype)
        out[:cny, :cnx] = interior
        return out

    elif pad_class == "y_pad_x_real":
        # DXC, DYG, RAW: interior (ny,nx) coarsens normally; the real
        # extra column (ny,1) reduces along y to (cny,1) and becomes the
        # coarse output's real last column; only the last ROW is padding.
        interior = apply_rule(arr_fine[:-1, :-1])
        extra_col = arr_fine[:-1, -1]  # shape (ny,), real boundary data
        extra_col_coarse = reduce_1d_group(extra_col, EXTRA_STRIP_REDUCTION[name])
        out = np.zeros((cny + 1, cnx + 1), dtype=arr_fine.dtype)
        out[:cny, :cnx] = interior
        out[:cny, cnx] = extra_col_coarse
        return out

    elif pad_class == "x_pad_y_real":
        # DYC, DXG, RAS: mirror of the above -- real extra row (1,nx)
        # reduces along x to (1,cnx) and becomes the coarse output's real
        # last row; only the last COLUMN is padding.
        interior = apply_rule(arr_fine[:-1, :-1])
        extra_row = arr_fine[-1, :-1]  # shape (nx,), real boundary data
        extra_row_coarse = reduce_1d_group(extra_row, EXTRA_STRIP_REDUCTION[name])
        out = np.zeros((cny + 1, cnx + 1), dtype=arr_fine.dtype)
        out[:cny, :cnx] = interior
        out[cny, :cnx] = extra_row_coarse
        return out

    else:
        raise ValueError(f"unknown padding class {pad_class!r} for field {name!r}")


def process_grid():
    """Read all 5 LLC90 .mitgrid tiles, coarsen every field by its correct
    rule, write out 5 LLC30 .mitgrid tiles, and run the mandatory RAC-
    conservation verification. Returns (fine_rac_total, coarse_rac_total,
    fine_xg_by_facet, fine_yg_by_facet, coarse_xg_by_facet,
    coarse_yg_by_facet) -- the XG/YG dicts feed the decisive facet-
    boundary-continuity check (check_facet_boundary_continuity), run on
    both resolutions in main().
    """
    os.makedirs(OUT_GRID_DIR, exist_ok=True)

    fine_rac_total = 0.0
    coarse_rac_total = 0.0
    fine_xg_by_facet = {}
    fine_yg_by_facet = {}
    coarse_xg_by_facet = {}
    coarse_yg_by_facet = {}

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
        fine_xg_by_facet[facet] = fine_fields["XG"]
        fine_yg_by_facet[facet] = fine_fields["YG"]

        coarse_fields = {}
        for name in FIELD_NAMES:
            # coarsen_one_mitgrid_field now returns the FULL (ny30+1,nx30+1)
            # shape directly for every field, with the correct one of 4
            # padding/real-boundary treatments already applied internally
            # (PADDING_CLASS: "corner" / "both_pad" / "y_pad_x_real" /
            # "x_pad_y_real" -- see that function's docstring). No
            # additional blanket zero-padding step is needed or correct
            # here anymore: that was the bug (a single blanket pattern
            # applied to all non-corner fields, which silently zeroed out
            # real boundary data for 6 of the 14 non-corner fields).
            coarse_fields[name] = coarsen_one_mitgrid_field(name, fine_fields[name])

        coarse_rac_total += coarse_fields["RAC"][:-1, :-1].sum()
        coarse_xg_by_facet[facet] = coarse_fields["XG"]
        coarse_yg_by_facet[facet] = coarse_fields["YG"]

        out_path = os.path.join(OUT_GRID_DIR, f"tile{facet:03d}.mitgrid")
        expected_out = EXPECTED_LLC30_TILE_BYTES[facet]
        written = write_mitgrid_tile(out_path, coarse_fields, nx30, ny30)
        assert written == expected_out, (
            f"tile{facet:03d}.mitgrid (LLC30): wrote {written} bytes, "
            f"expected {expected_out}"
        )
        print(f"  tile{facet:03d}.mitgrid: LLC90 ({nx90},{ny90}) -> "
              f"LLC30 ({nx30},{ny30}), {written} bytes OK")

    return (fine_rac_total, coarse_rac_total, fine_xg_by_facet, fine_yg_by_facet,
            coarse_xg_by_facet, coarse_yg_by_facet)


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


# ---------------------------------------------------------------------------
# Permanent, non-tautological facet-orientation regression check (fix for
# code-review findings #1/#4/#5): the plan's only prior checks (RAC-
# conservation, forcing spot-check) are invariant to facet orientation/
# permutation -- a wrong facet-4/5 reshape still conserves RAC and the
# domain mean. This check is NOT invariant to orientation: it directly
# compares grid-corner (XG/YG) values across every facetEdgeLink-implied
# connection. XG/YG are exact (not block-averaged/interpolated) corner
# positions, so two facets that are genuinely adjacent per facetEdgeLink
# MUST share identical corner coordinates along their common edge -- any
# orientation bug (wrong transpose, reversed index order, wrong facet
# pairing) breaks this exactly, not approximately.
#
# facetEdgeLink copied unchanged from the production LLC90 reference (see
# input/data.exch2's own copy of this table) -- resolution-independent,
# same for LLC90 and LLC30.
# ---------------------------------------------------------------------------
FACET_EDGE_LINK = {
    1: ['3.4', '0.', '2.4', '5.1'],
    2: ['3.2', '0.', '4.2', '1.3'],
    3: ['5.4', '2.1', '4.4', '1.1'],
    4: ['5.2', '2.3', '0.', '3.3'],
    5: ['1.4', '4.1', '0.', '3.1'],
}
EDGE_NAME = {1: 'N', 2: 'S', 3: 'E', 4: 'W'}
_SAME_ORIENT_EDGE_PAIRS = {('N', 'S'), ('S', 'N'), ('E', 'W'), ('W', 'E')}


def _parse_edge_link(token):
    """Parse a facetEdgeLink token like '3.4' -> (jj=3, ii=4); '0.' -> (0,0)."""
    token = token.strip()
    if token in ('0.', '0'):
        return 0, 0
    whole, frac = token.split('.')
    return int(whole), int(frac)


def _get_corner_edge_line(arr_corner, edge):
    """arr_corner: shape (ny+1, nx+1). Return the 1-D line of corner values
    along the given edge (N/S/E/W), ordered by increasing index."""
    if edge == 'N':
        return arr_corner[-1, :]
    elif edge == 'S':
        return arr_corner[0, :]
    elif edge == 'E':
        return arr_corner[:, -1]
    elif edge == 'W':
        return arr_corner[:, 0]
    raise ValueError(edge)


def _angular_diff_deg(a, b):
    """Shortest angular difference (degrees), dateline-safe."""
    return (a - b + 180.0) % 360.0 - 180.0


def check_facet_boundary_continuity(xg_by_facet, yg_by_facet, label, tol_deg=1e-6):
    """Decisive, non-permutation-invariant facet-orientation check (resolves
    code-review finding #4): for every facetEdgeLink-implied connection,
    confirm the two facets' XG/YG corner lines along the shared edge are
    IDENTICAL (within floating-point rounding) -- the only check in this
    script that would catch a facet-4/5 orientation bug of the exact kind
    previously disputed between reviewers.

    xg_by_facet/yg_by_facet: dict facet(1..5) -> full (ny+1,nx+1) XG/YG array
    (works at either LLC90 fine or LLC30 coarse resolution).

    Returns the max |dXG|, |dYG| found (both should be ~0). Raises
    AssertionError if any connection exceeds tol_deg.
    """
    max_dxg = 0.0
    max_dyg = 0.0
    n_checked = 0
    print(f"\nFacet boundary-continuity check ({label}):")
    for j in range(1, 6):
        for i in range(1, 5):
            jj, ii = _parse_edge_link(FACET_EDGE_LINK[j][i - 1])
            if jj == 0:
                continue  # disconnected outer edge
            ei, eii = EDGE_NAME[i], EDGE_NAME[ii]
            yg1 = _get_corner_edge_line(yg_by_facet[j], ei)
            yg2 = _get_corner_edge_line(yg_by_facet[jj], eii)
            xg1 = _get_corner_edge_line(xg_by_facet[j], ei)
            xg2 = _get_corner_edge_line(xg_by_facet[jj], eii)
            if (ei, eii) not in _SAME_ORIENT_EDGE_PAIRS:
                # orientation change (e.g. N<->W): index order reverses
                yg2 = yg2[::-1]
                xg2 = xg2[::-1]
            assert yg1.shape == yg2.shape, (
                f"facet{j}.{ei} <-> facet{jj}.{eii}: shape mismatch "
                f"{yg1.shape} vs {yg2.shape}"
            )
            ydiff = np.abs(yg1 - yg2)
            xdiff = np.abs(_angular_diff_deg(xg1, xg2))
            max_dxg = max(max_dxg, xdiff.max())
            max_dyg = max(max_dyg, ydiff.max())
            n_checked += 1
            print(f"  facet{j}.{ei} <-> facet{jj}.{eii}: "
                  f"max|dXG|={xdiff.max():.3e}  max|dYG|={ydiff.max():.3e}")
    print(f"  Checked {n_checked} connected edges. "
          f"max|dXG|={max_dxg:.3e}  max|dYG|={max_dyg:.3e}")
    assert max_dxg < tol_deg and max_dyg < tol_deg, (
        f"Facet boundary-continuity check FAILED ({label}): max|dXG|={max_dxg:.3e}, "
        f"max|dYG|={max_dyg:.3e} (tol={tol_deg:.1e}) -- this indicates a facet "
        f"orientation/reshape bug (the exact bug class this check exists to catch)"
    )
    print(f"  PASSED ({label}): all {n_checked} facet connections are "
          f"corner-continuous to within {tol_deg:.1e} deg.")
    return max_dxg, max_dyg


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
    """3x3-block min-magnitude-over-wet-cells coarsening, built directly on
    top of coarsen_bathy_block (one call per 3x3 block) so the rule is
    implemented exactly once -- an earlier version of this function
    reimplemented the same rule independently in vectorized form, which
    coarsen_bathy_block (unused dead code at the time) risked silently
    drifting out of sync with.

    arr: shape (ny, nx) real depth values (0 = land, negative = ocean depth).
    Returns: shape (ny//3, nx//3).
    """
    ny, nx = arr.shape
    factor = FACTOR
    cny, cnx = ny // factor, nx // factor
    blocks = arr.reshape(cny, factor, cnx, factor).transpose(0, 2, 1, 3).reshape(cny, cnx, factor * factor)
    out = np.empty((cny, cnx), dtype=arr.dtype)
    for jy in range(cny):
        for jx in range(cnx):
            out[jy, jx] = coarsen_bathy_block(blocks[jy, jx])
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

    verify_bathy_correctness(compact, coarse_compact)

    return coarse_compact


def verify_bathy_correctness(fine_compact, coarse_compact):
    """Mandatory correctness verification beyond byte-size checks
    (code-review finding #6): process_bathy previously had zero numerical
    verification of its own coarsening rule. Two cheap, decisive checks:

      1. Land-fraction-preserved-within-tolerance: the coarsened land
         fraction should be close to the fine land fraction, not wildly
         different. A coarse cell is land only if ALL 9 constituents are
         land, so the coarse land fraction should be <= the fine land
         fraction (coarsening can only turn some land into ocean, never
         the reverse) -- and not implausibly far below it.
      2. Depth-bounds-preserved: every coarsened (wet) depth value must
         fall within [fine_min, fine_max] -- min-magnitude-over-wet-cells
         coarsening selects an existing fine-grid value, so it cannot
         manufacture a depth outside the source range. A violation here
         would indicate a real bug (e.g. sign error, wrong facet feeding
         the wrong output cell, or a stray float32 overflow).
    """
    fine_land_frac = (fine_compact == 0.0).mean()
    coarse_land_frac = (coarse_compact == 0.0).mean()
    print(f"\nBathymetry correctness check:")
    print(f"  fine land fraction   = {fine_land_frac:.4f}")
    print(f"  coarse land fraction = {coarse_land_frac:.4f}")
    assert coarse_land_frac <= fine_land_frac + 1e-9, (
        f"Bathy check FAILED: coarse land fraction ({coarse_land_frac:.4f}) "
        f"exceeds fine land fraction ({fine_land_frac:.4f}) -- a coarse cell "
        f"should only be land if ALL 9 constituents are land, so coarsening "
        f"can only reduce (not increase) the land fraction"
    )
    # The land fraction should not collapse implausibly (e.g. almost all
    # land vanishing) -- bound the drop to within 50% of the fine fraction
    # as a sanity check (block-averaging land into ocean at the coastline
    # is expected and fine; losing most land entirely would indicate a bug).
    assert coarse_land_frac >= 0.5 * fine_land_frac, (
        f"Bathy check FAILED: coarse land fraction ({coarse_land_frac:.4f}) "
        f"is less than half the fine land fraction ({fine_land_frac:.4f}) -- "
        f"implausible for a 3x3 min-magnitude-over-wet-cells coarsening"
    )
    print("  land-fraction check PASSED.")

    fine_wet = fine_compact[fine_compact != 0.0]
    coarse_wet = coarse_compact[coarse_compact != 0.0]
    fine_min, fine_max = fine_wet.min(), fine_wet.max()
    coarse_min, coarse_max = coarse_wet.min(), coarse_wet.max()
    print(f"  fine wet-depth range   = [{fine_min:.2f}, {fine_max:.2f}]")
    print(f"  coarse wet-depth range = [{coarse_min:.2f}, {coarse_max:.2f}]")
    assert coarse_min >= fine_min - 1e-6 and coarse_max <= fine_max + 1e-6, (
        f"Bathy check FAILED: coarse wet-depth range [{coarse_min:.2f}, "
        f"{coarse_max:.2f}] exceeds fine range [{fine_min:.2f}, {fine_max:.2f}] -- "
        f"min-magnitude-over-wet-cells coarsening selects an existing fine value "
        f"and cannot produce a depth outside the source range"
    )
    print("  depth-bounds-preserved check PASSED.")


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
    """Compare one coarsened forcing variable's January mean against the
    source's January mean, PER FACET (not just the domain-wide mean).

    IMPORTANT LIMITATION (code-review finding #1 -- read before trusting
    this check for anything beyond arithmetic sanity): a per-facet mean
    is STILL invariant to any reordering/transpose/flip of cells WITHIN a
    correctly-identified facet, since a mean does not care about element
    order. This check can therefore catch (a) gross temporal/spatial
    block-average arithmetic bugs, and (b) a facet MISLABELING bug (e.g.
    facet 4's data ending up tagged as facet 5), because that changes
    which facet's fine cells feed which facet's coarse mean -- but it
    CANNOT catch a within-facet orientation/transpose bug (the facet-4/5
    bug class this plan previously had). That orientation question is
    answered decisively and independently by
    check_facet_boundary_continuity (run in main() on the grid's XG/YG),
    which IS sensitive to within-facet orientation because it compares
    specific edge-indexed corner values, not an order-invariant mean.
    """
    jan_coarse, jan_fine = jan_results[var]
    # jan_coarse is compact-shaped (COMPACT_COARSE_NY, COMPACT_COARSE_NX) at
    # LLC30 resolution; split it into its 5 facets using the same flat-range
    # convention as assemble_facets_into_compact (coarse row ranges = fine
    # row ranges // FACTOR).
    coarse_facets = {}
    flat = jan_coarse.reshape(-1)
    for facet in range(1, 6):
        r0, r1 = FACET_ROW_RANGES[facet]
        r0c, r1c = r0 // FACTOR, r1 // FACTOR
        nx30, ny30 = LLC30_FACET_DIMS[facet]
        start = r0c * COMPACT_COARSE_NX
        n = (r1c - r0c) * COMPACT_COARSE_NX
        coarse_facets[facet] = flat[start:start + n].reshape(ny30, nx30)
    fine_facets = split_compact_into_facets(jan_fine)

    print(f"\nSpot-check ({var}, January mean, PER FACET -- not domain-wide):")
    worst_rel_diff = 0.0
    for facet in range(1, 6):
        coarse_mean = coarse_facets[facet].mean()
        fine_mean = fine_facets[facet].mean()
        rel_diff = abs(coarse_mean - fine_mean) / abs(fine_mean) if fine_mean != 0 else float("nan")
        worst_rel_diff = max(worst_rel_diff, rel_diff)
        print(f"  facet{facet}: LLC90 mean={fine_mean:.6f}  LLC30 mean={coarse_mean:.6f}  "
              f"rel_diff={rel_diff:.3e}")
        assert rel_diff < 0.03, (
            f"Spot-check FAILED for {var} facet {facet}: coarse={coarse_mean:.6f}, "
            f"fine={fine_mean:.6f}, rel_diff={rel_diff:.3e} (expected < 3%)"
        )
    print(f"  Spot-check PASSED for all 5 facets (worst rel_diff={worst_rel_diff:.3e}, "
          f"within a few percent). NOTE: this checks arithmetic + facet "
          f"identity only, not within-facet orientation -- see "
          f"check_facet_boundary_continuity for that.")


# ===========================================================================
# main
# ===========================================================================

def main():
    print("=" * 70)
    print("Step 1/5: coarsening 5 LLC90 .mitgrid tiles -> LLC30")
    print("=" * 70)
    (fine_rac_total, coarse_rac_total, fine_xg, fine_yg,
     coarse_xg, coarse_yg) = process_grid()
    verify_rac_conservation(fine_rac_total, coarse_rac_total)

    print()
    print("=" * 70)
    print("Step 2/5: decisive facet-orientation check (XG/YG boundary continuity)")
    print("=" * 70)
    # Permanent regression check (fixes findings #1/#4/#5): run on BOTH the
    # LLC90 fine source grid (ground truth) and the LLC30 coarsened output,
    # so a future facet-orientation regression in either the source-data
    # understanding or the coarsening code itself is caught. Unlike RAC
    # conservation or the forcing spot-check, this is NOT invariant to
    # facet orientation/permutation -- it is the exact test that resolves
    # the two reviewers' conflicting conclusions on facet 4/5 orientation
    # (see the progress log for this fix's [decision] entry with the
    # concrete numbers from running this check).
    check_facet_boundary_continuity(fine_xg, fine_yg, label="LLC90 fine source grid")
    check_facet_boundary_continuity(coarse_xg, coarse_yg, label="LLC30 coarsened output")

    print()
    print("=" * 70)
    print("Step 3/5: coarsening bathymetry -> LLC30")
    print("=" * 70)
    process_bathy()

    print()
    print("=" * 70)
    print("Step 4/5: reducing 9 combforc 1992 hourly variables -> LLC30 monthly clim")
    print("=" * 70)
    jan_results = process_all_forcing()

    print()
    print("=" * 70)
    print("Step 5/5: spot-check (per-facet)")
    print("=" * 70)
    spot_check_forcing(jan_results, var="dlw")

    print()
    print("ALL CHECKS PASSED. Task 2 preprocessing complete.")


if __name__ == "__main__":
    main()
